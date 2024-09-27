
from typing import Optional
import torch
from torch import nn
from transformers import PreTrainedModel, AutoConfig, AutoModelForCausalLM
from transformers.models.auto import CONFIG_MAPPING

from ...span_modeling import LstmSeq2SeqEncoder, SpanRepLayer
from .config import DecoderConfig

class SpanDecoderPreTrainedModel(PreTrainedModel):
    config_class = DecoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )
        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            # Initialize LSTM weights and biases
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    # Input-hidden weights initialization
                    nn.init.normal_(param.data, mean=0.0, std=std)
                elif 'weight_hh' in name:
                    # Hidden-hidden weights initialization
                    nn.init.orthogonal_(param.data, gain=std * 1.0)
                elif 'bias' in name:
                    # Bias initialization: set biases to zero
                    param.data.zero_()
                    # Set forget gate biases to 1 for better training stability
                    n = param.size(0)
                    param.data[n // 4:n // 2].fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.decoder._supports_sdpa

    @property
    def hidden_size(self):
        return self.backbone_config.hidden_size

class SpanDecoder(SpanDecoderPreTrainedModel):
    def __init__(self, config: DecoderConfig, from_pretrained=False):
        super().__init__(config)

        backbone_config = AutoConfig.from_pretrained(config.backbone_model)

        if from_pretrained:
            self.decoder = AutoModelForCausalLM.from_pretrained(config.backbone_model)
        else:
            self.decoder = AutoModelForCausalLM.from_config(backbone_config)

        if self.config.has_rnn:
            self.rnn = LstmSeq2SeqEncoder(config)

        self.span_rep = SpanRepLayer(config.hidden_size, config.max_width, span_mode=config.span_mode)
        
        self.model_parallel = False
        self.device_map = None
        
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        span_idx: Optional[torch.LongTensor] = None,
        text_length: Optional[torch.LongTensor] = None,
        output_input_ids: Optional[torch.LongTensor] = None,
        output_attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        label_smoothing: float =  0.0,
        reduction: str = 'mean',
        **kwargs):
        pass