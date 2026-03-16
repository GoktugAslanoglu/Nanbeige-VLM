from transformers import PretrainedConfig


class NanbeigeVLMConfig(PretrainedConfig):

    model_type = "nanbeige_vlm"

    def __init__(
        self,
        vision_model_id: str = "google/siglip-so400m-patch14-384",
        llm_model_id: str = "Nanbeige/Nanbeige4.1-3B",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_model_id = vision_model_id
        self.llm_model_id = llm_model_id
