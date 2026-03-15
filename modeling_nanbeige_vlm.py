"""
NanbeigeVLM — SigLIP vision encoder + MLP projector + Nanbeige4.1-3B LLM.

Usage:
    from transformers import AutoModel, AutoTokenizer
    from PIL import Image

    model     = AutoModel.from_pretrained("SkyAsl/Nanbeige4.1-VLM-Base", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("SkyAsl/Nanbeige4.1-VLM-Base", trust_remote_code=True)

    image  = Image.open("photo.jpg")
    result = model.describe(image)
    print(result)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    SiglipVisionModel,
    SiglipImageProcessor,
)
from .configuration_nanbeige_vlm import NanbeigeVLMConfig


# ---------------------------------------------------------------------------
# Pooled projector  (729 → 196 tokens via 2×2 avg-pool then linear)
# ---------------------------------------------------------------------------

class PooledProjector(nn.Module):
    """
    Reduces SigLIP's 729 patch tokens to 196 via spatial average pooling,
    then projects to the LLM hidden dimension.

        (B, 729, D_vision)
          → reshape  (B, D_vision, 27, 27)
          → pad      (B, D_vision, 28, 28)
          → avgpool  (B, D_vision, 14, 14)
          → flatten  (B, 196, D_vision)
          → linear   (B, 196, D_llm)
    """

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
        x = F.pad(x, (0, 1, 0, 1), mode="replicate")   # 27×27 → 28×28
        x = F.avg_pool2d(x, kernel_size=2, stride=2)    # 28×28 → 14×14
        x = x.flatten(2).permute(0, 2, 1)               # (B, 196, C)
        return self.proj(x)                              # (B, 196, D_llm)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class NanbeigeVLMModel(PreTrainedModel):
    """
    SigLIP so400m  →  PooledProjector (729→196 tokens)  →  Nanbeige4.1-3B

    Stage 1 pretrain: only mm_projector is trained.
    Vision tower and LLM are frozen.
    """

    config_class = NanbeigeVLMConfig

    # Tells transformers which keys belong to sub-models that are loaded
    # separately — prevents 'unexpected key' warnings.
    _no_split_modules = ["SiglipVisionModel", "NanbeigeForCausalLM"]

    def __init__(self, config: NanbeigeVLMConfig, image_token_id: int = None):
        super().__init__(config)

        # ── Vision tower (frozen at Stage 1) ──────────────────────────────
        self.vision_tower = SiglipVisionModel.from_pretrained(
            config.vision_model_id, torch_dtype=torch.bfloat16
        )
        self.vision_tower.requires_grad_(False)
        vision_hidden_size = self.vision_tower.config.hidden_size

        # ── Language model (frozen at Stage 1) ────────────────────────────
        try:
            self.language_model = AutoModelForCausalLM.from_pretrained(
                config.llm_model_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        except (ImportError, ValueError):
            self.language_model = AutoModelForCausalLM.from_pretrained(
                config.llm_model_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        self.language_model.requires_grad_(False)
        llm_hidden_size = self.language_model.config.hidden_size

        # ── Projector ──────────────────────────────────────────────────────
        self.mm_projector = PooledProjector(
            vision_hidden_size, llm_hidden_size
        ).to(torch.bfloat16)

        # Set after resize_token_embeddings if needed
        self.image_token_id = image_token_id
        self.post_init()

    # ------------------------------------------------------------------
    # Core forward (used by Trainer)
    # ------------------------------------------------------------------

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        assert self.image_token_id is not None, \
            "image_token_id must be set before calling forward()."

        with torch.inference_mode():
            image_features = self.vision_tower(
                pixel_values=pixel_values
            ).last_hidden_state                             # (B, 729, D_vision)

        image_embeds     = self.mm_projector(image_features)  # (B, 196, D_llm)
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
                torch.cat([inputs_embeds[i, :pos], image_embeds[i], inputs_embeds[i, pos+1:]], dim=0)
            )
            if attention_mask is not None:
                img_mask = torch.ones(num_image_tokens, device=attention_mask.device, dtype=attention_mask.dtype)
                merged_mask.append(
                    torch.cat([attention_mask[i, :pos], img_mask, attention_mask[i, pos+1:]])
                )
            if labels is not None:
                img_labels = torch.full((num_image_tokens,), -100, device=labels.device, dtype=labels.dtype)
                merged_labels.append(
                    torch.cat([labels[i, :pos], img_labels, labels[i, pos+1:]])
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
    # High-level inference helpers
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
        """
        Convenience method: pass a PIL image, get a text description back.

        Args:
            image:              PIL.Image
            prompt:             Instruction string
            tokenizer:          Pass tokenizer if not set on model
            max_new_tokens:     Max output length
            do_sample:          True for creative outputs, False for deterministic
            temperature:        Sampling temperature (only used if do_sample=True)
            repetition_penalty: Penalise repeated tokens

        Returns:
            str: generated description
        """
        assert self.image_token_id is not None, \
            "Set model.image_token_id before calling describe()."

        tok = tokenizer or getattr(self, "_tokenizer", None)
        assert tok is not None, \
            "Pass tokenizer=... to describe() or call model.set_tokenizer(tokenizer) first."

        device    = next(self.parameters()).device
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
        self._tokenizer = tokenizer
        self.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
