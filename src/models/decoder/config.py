from typing import Optional
from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING

from ...config import Config

class DecoderConfig(Config, PretrainedConfig):
    model_type='span_decoder'
    is_composition=True

    def __init__(self, backbone_model: str,
                    backbone_config: Optional[dict] = None,
                    initializer_range: Optional[float] = 0.01,
                    **kwargs,
                    ):
        super().__init__(**kwargs)
        self.backbone_model = backbone_model
        self.initializer_range = initializer_range
        if isinstance(backbone_config, dict):
            backbone_config["model_type"] = (backbone_config["model_type"] 
                                                if "model_type" in backbone_config 
                                                else "openai-community/gpt2")
            backbone_config = CONFIG_MAPPING[backbone_config["model_type"]](**backbone_config)
        self.backbone_config = backbone_config

    @property
    def hidden_size(self):
      return self.backbone_config.hidden_size