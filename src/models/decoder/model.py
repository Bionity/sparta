
from typing import Optional, Union, List
from dataclasses import dataclass
import torch
from torch import nn
from transformers import PreTrainedModel, AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache

from ...span_modeling import LstmSeq2SeqEncoder, SpanRepLayer
from .config import DecoderConfig

@dataclass
class CausalLMOutputWithPast(CausalLMOutputWithPast):
    span_embeddings: Optional[torch.FloatTensor] = None
    span_logits: Optional[torch.FloatTensor] = None

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
        
        if config.add_spec_token_span:
            self.spec_span = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        else:
            self.spec_span = None

        self.model_parallel = False
        self.device_map = None
        
        self.post_init()

    def run_decoder(self, input_ids: Optional[torch.LongTensor] = None,
                    attention_mask: Optional[torch.FloatTensor] = None,
                    **kwargs):
        
        model_attr_names = ['model', 'decoder', 'transformer', 'generator']
        
        decoder_model = self._get_decoder_attribute(self.decoder, model_attr_names)
        
        if decoder_model is None:
            raise AttributeError("Decoder model attribute not found. "
                                 "Please specify a custom method for running the decoder.")
        
        decoder_outputs = decoder_model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        **kwargs)
        return decoder_outputs
    
    def run_lm_head(self, hidden_states: torch.FloatTensor):
        lm_head_attr_names = ['lm_head', 'output_layer', 'head', 'classifier']

        lm_head = self._get_decoder_attribute(self.decoder, lm_head_attr_names)
        
        if lm_head is None:
            raise AttributeError("LM head attribute not found. "
                                 "Please specify a custom method for running the language model head.")
        
        lm_logits = lm_head(hidden_states)
        return lm_logits
    
    def _get_decoder_attribute(self, obj, attr_names):
        for name in attr_names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        span_embeddings: Optional[torch.FloatTensor] = None,
        span_idx: Optional[torch.LongTensor] = None,
        text_length: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        label_smoothing: float =  0.0,
        reduction: str = 'mean',
        **kwargs):

        decoder_outputs = self.run_decoder(input_ids=input_ids, 
                                        attention_mask=attention_mask,
                                        past_key_values=past_key_values,
                                        use_cache=use_cache,
                                        **kwargs)

        hidden_states = decoder_outputs[0]

        batch_size, seq_length, hidden_size = hidden_states.shape

        span_mask = (span_idx[:,:,-1]<text_length).unsqueeze(2)
        span_idx = span_mask*span_idx

        if span_embeddings is None:
            if self.config.has_rnn:
                post_rnn_hidden_states = self.rnn(hidden_states, attention_mask)
                span_embeddings = self.span_rep(post_rnn_hidden_states, span_idx)
            else:
                span_embeddings = self.span_rep(hidden_states, span_idx)
            span_embeddings = span_embeddings.view(batch_size, -1, hidden_size)

            if self.spec_span is not None:
                spec_span = self.spec_span.expand(batch_size, 1, hidden_size)
                span_embeddings = torch.cat([spec_span, span_embeddings], dim=1)

        lm_logits = self.decoder.lm_head(hidden_states)

        span_logits = torch.einsum('bld, bkd->blk', hidden_states, span_embeddings)

        span_loss = None
        token_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, 
                                        label_smoothing = label_smoothing,
                                        reduction = reduction)
            
            labels = labels.to(span_logits.device)
            span_loss = loss_fct(span_logits.view(-1, span_logits.size(-1)), labels.view(-1))
            
            lm_logits = lm_logits[:, :-1,:].contiguous()
            token_labels = input_ids[:,1:].contiguous()
            token_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), token_labels.view(-1))

        loss = (span_loss, token_loss)


        if not return_dict:
            output = (span_logits, lm_logits, ) + decoder_outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            span_logits=span_logits,
            span_embeddings=span_embeddings,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


    def generate(self, model_inputs, max_length=512, max_new_tokens=None):
        model_inputs = {key: value.to(self.device) for key, value in model_inputs.items()}

        input_ids = model_inputs.pop('input_ids')

        if max_new_tokens is not None:
            new_tokens_count = max_new_tokens
        else:
            new_tokens_count = max(0, max_length - input_ids.shape[1])

        batch_size = input_ids.shape[0]

        token_count = 0

        past_key_values = None
        span_embeddings = None

        span_idx = model_inputs.get('span_idx')

        curr_input_ids = input_ids
        attention_mask = model_inputs.pop('attention_mask')
        generated_tokens = []
        while token_count<new_tokens_count:
            outputs = self.forward(input_ids = curr_input_ids,
                                    attention_mask = attention_mask,
                                    span_embeddings=span_embeddings,
                                    past_key_values=past_key_values,
                                    **model_inputs, use_cache=True, return_dict=True)

            next_span_ids = outputs.span_logits.argmax(dim=-1)[:, -1]
            
            next_token_ids = outputs.logits.argmax(dim=-1)[:, -1]

            batch_range = torch.arange(batch_size)

            span_indices = span_idx[batch_range, next_span_ids]

            span_lengths = (span_indices[:,1] - span_indices[:, 0])+1

            max_span_length = span_lengths.max(-1).values
            token_count += max_span_length.item()

            curr_input_ids = torch.zeros((batch_size, max_span_length), dtype=span_idx.dtype, device=span_idx.device)
            curr_attention_mask = torch.zeros((batch_size, max_span_length), dtype=span_idx.dtype, device=span_idx.device)

            input_ids_range = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(batch_size, input_ids.shape[1])
            new_tokens_range = torch.arange(max_span_length, device=input_ids.device).unsqueeze(0).expand(batch_size, max_span_length)

            batch_indices, new_token_idx = torch.where(new_tokens_range>=(max_span_length-span_lengths).unsqueeze(1))

            new_tokens_mask = (input_ids_range >= span_indices[:, 0].unsqueeze(1)) & (input_ids_range <= span_indices[:, 1].unsqueeze(1))
            batch_indices, input_token_idx = torch.where(new_tokens_mask)

            curr_attention_mask[batch_indices, new_token_idx ] = 1
            curr_input_ids[batch_indices, new_token_idx ] = input_ids[batch_indices, input_token_idx]

            if self.config.add_spec_token_span:
                spec_token_idx = torch.where(next_span_ids==0)
                if spec_token_idx[0].shape[0] != 0:
                    curr_input_ids[spec_token_idx][-1] = next_token_ids[spec_token_idx]

            attention_mask = torch.cat([attention_mask, curr_attention_mask], dim=-1)
            if span_embeddings is None:
                span_embeddings = outputs.span_embeddings

            past_key_values = outputs.past_key_values
            generated_tokens.append(curr_input_ids)

        output_ids = torch.cat(generated_tokens, dim=-1)

        return output_ids