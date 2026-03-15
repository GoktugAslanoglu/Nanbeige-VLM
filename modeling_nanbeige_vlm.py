"""
NanbeigeVLM — SigLIP vision encoder + MLP projector + Nanbeige4.1-3B LLM.

Usage:
    from transformers import AutoModel, AutoTokenizer
    from PIL import Image

    model     = AutoModel.from_pretrained("SkyAsl/Nanbeige4.1-VLM-Base", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("SkyAsl/Nanbeige4.1-VLM-Base", trust_remote_code=True)
    model.set_tokenizer(tokenizer)

    image  = Image.open("photo.jpg")
    result = model.describe(image)
    print(result)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    SiglipVisionModel,
    SiglipImageProcessor,
)
from transformers.utils import logging
from .configuration_nanbeige_vlm import NanbeigeVLMConfig

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pooled projector  (729 → 196 tokens via 2×2 avg-pool then linear)
# ---------------------------------------------------------------------------

class PooledProjector(nn.Module):
    def __init__(self, vision_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        B, N, C = image_features.shape
        x = image_features.permute(0, 2, 1).reshape(B, C, 27, 27)
        x = F.pad(x, (0, 1, 0, 1), mode="replicate")
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = x.flatten(2).permute(0, 2, 1)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class NanbeigeVLMModel(PreTrainedModel):
    config_class = NanbeigeVLMConfig
    _no_split_modules = ["SiglipVisionModel", "NanbeigeForCausalLM"]

    def __init__(self, config: NanbeigeVLMConfig, image_token_id: int = None):
        super().__init__(config)
        # Sub-models are NOT loaded here — from_pretrained() handles this.
        # This prevents the meta-device conflict with nested from_pretrained calls.
        self.vision_tower   = None
        self.language_model = None
        self.mm_projector   = None
        self.image_token_id = image_token_id

    # ------------------------------------------------------------------
    # Override from_pretrained to handle nested model loading correctly
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Custom loader that:
          1. Loads sub-models (SigLIP, LLM) normally — outside any meta context.
          2. Loads only projector weights from the checkpoint.
          3. Returns a fully initialised NanbeigeVLMModel.
        """
        import safetensors.torch
        from huggingface_hub import hf_hub_download

        config    = kwargs.pop("config", None)
        token     = kwargs.pop("token", None)
        cache_dir = kwargs.pop("cache_dir", None)

        if config is None:
            config = NanbeigeVLMConfig.from_pretrained(
                pretrained_model_name_or_path, token=token, cache_dir=cache_dir
            )

        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)

        # ── 1. Build empty shell ──────────────────────────────────────
        model = cls(config)

        # ── 2. Load SigLIP ────────────────────────────────────────────
        logger.info("Loading vision tower...")
        model.vision_tower = SiglipVisionModel.from_pretrained(
            config.vision_model_id,
            torch_dtype=torch_dtype,
            device_map=None,
        )
        model.vision_tower.requires_grad_(False)
        vision_hidden_size = model.vision_tower.config.hidden_size

        # ── 3. Load LLM ───────────────────────────────────────────────
        logger.info("Loading language model...")
        try:
            model.language_model = AutoModelForCausalLM.from_pretrained(
                config.llm_model_id,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                device_map=None,
            )
        except (ImportError, ValueError):
            model.language_model = AutoModelForCausalLM.from_pretrained(
                config.llm_model_id,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=None,
            )
        model.language_model.requires_grad_(False)
        llm_hidden_size = model.language_model.config.hidden_size

        # ── 4. Build projector, load trained weights ──────────────────
        model.mm_projector = PooledProjector(
            vision_hidden_size, llm_hidden_size
        ).to(torch_dtype)

        logger.info("Loading projector weights from checkpoint...")
        weights_path = hf_hub_download(
            repo_id=pretrained_model_name_or_path,
            filename="model.safetensors",
            token=token,
            cache_dir=cache_dir,
        )
        all_weights  = safetensors.torch.load_file(weights_path)
        proj_weights = {k: v for k, v in all_weights.items() if "mm_projector" in k}

        # Strip "mm_projector." prefix for load_state_dict
        proj_weights_clean = {k.replace("mm_projector.", "", 1): v
                              for k, v in proj_weights.items()}
        model.mm_projector.load_state_dict(proj_weights_clean)

        logger.info("NanbeigeVLM loaded successfully.")
        return model

    # ------------------------------------------------------------------
    # forward (used by Trainer during Stage 2)
    # ------------------------------------------------------------------

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        assert self.image_token_id is not None, \
            "image_token_id must be set before calling forward()."

        with torch.inference_mode():
            image_features = self.vision_tower(
                pixel_values=pixel_values
            ).last_hidden_state

        image_embeds     = self.mm_projector(image_features)
        num_image_tokens = image_embeds.shape[1]

        with torch.inference_mode():
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = inputs_embeds.detach()

        batch_size = input_ids.shape[0]
        merged_embeds, merged_mask, merged_labels = [], [], []

        for i in range(batch_size):
            positions = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]

            if len(positions) == 0:
                merged_embeds.append(inputs_embeds[i])
                if attention_mask is not None:
                    merged_mask.append(attention_mask[i])
                if labels is not None:
                    merged_labels.append(labels[i])
                continue

            pos = positions[0].item()
            merged_embeds.append(
                torch.cat([inputs_embeds[i, :pos], image_embeds[i],
                           inputs_embeds[i, pos+1:]], dim=0)
            )
            if attention_mask is not None:
                img_mask = torch.ones(
                    num_image_tokens,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                merged_mask.append(
                    torch.cat([attention_mask[i, :pos], img_mask,
                               attention_mask[i, pos+1:]])
                )
            if labels is not None:
                img_labels = torch.full(
                    (num_image_tokens,), -100,
                    device=labels.device, dtype=labels.dtype,
                )
                merged_labels.append(
                    torch.cat([labels[i, :pos], img_labels,
                               labels[i, pos+1:]])
                )

        combined_embeds = torch.stack(merged_embeds, dim=0)
        combined_mask   = torch.stack(merged_mask,   dim=0) if attention_mask is not None else None
        combined_labels = torch.stack(merged_labels, dim=0) if labels         is not None else None

        return self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
        )

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def describe(
        self,
        image,
        prompt: str = "Describe the image.",
        tokenizer=None,
        max_new_tokens: int = 200,
        do_sample: bool = False,
        temperature: float = 0.7,
        repetition_penalty: float = 1.3,
    ) -> str:
        tok = tokenizer or getattr(self, "_tokenizer", None)
        assert tok is not None, \
            "Pass tokenizer=... to describe() or call model.set_tokenizer(tokenizer) first."

        device    = next(self.mm_projector.parameters()).device
        processor = SiglipImageProcessor.from_pretrained(self.config.vision_model_id)

        pixel_values   = processor(images=image, return_tensors="pt").pixel_values.to(device, dtype=torch.bfloat16)
        image_features = self.vision_tower(pixel_values=pixel_values).last_hidden_state
        image_embeds   = self.mm_projector(image_features)

        input_ids     = tok(f"<image>\n{prompt}", return_tensors="pt").input_ids.to(device)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        pos    = (input_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0][0].item()
        merged = torch.cat(
            [inputs_embeds[0, :pos], image_embeds[0], inputs_embeds[0, pos+1:]], dim=0
        ).unsqueeze(0)

        generate_kwargs = dict(
            inputs_embeds=merged,
            attention_mask=torch.ones(merged.shape[:2], device=device),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            eos_token_id=tok.eos_token_id,
        )
        if do_sample:
            generate_kwargs["temperature"] = temperature

        output_ids = self.language_model.generate(**generate_kwargs)
        return tok.decode(output_ids[0], skip_special_tokens=True)

    def set_tokenizer(self, tokenizer):
        """Attach tokenizer to model so you don't have to pass it to describe()."""
        self._tokenizer  = tokenizer
        self.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
