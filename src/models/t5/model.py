import copy

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.utils import logging
from transformers.models.t5.modeling_t5 import (T5Stack, T5PreTrainedModel, T5DenseActDense, T5LayerNorm,
                                                T5DenseGatedActDense, T5Attention, __HEAD_MASK_WARNING_MSG)

from .config import T5Config

from ...span_modeling import SpanRepLayer, LstmSeq2SeqEncoder

logger = logging.get_logger(__name__)

@dataclass
class Seq2SeqLMOutput(Seq2SeqLMOutput):
    span_logits: Optional[torch.FloatTensor] = None
    span_embeddings: Optional[torch.FloatTensor] = None

class T5PreTrainedModel(T5PreTrainedModel):
    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(
            module, T5ForConditionalGeneration,
        ):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedActDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))
    
        elif isinstance(module, nn.LSTM):
            # Initialize LSTM weights and biases
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    # Input-hidden weights initialization
                    nn.init.normal_(param.data, mean=0.0, std=factor * (module.input_size ** -0.5))
                elif 'weight_hh' in name:
                    # Hidden-hidden weights initialization
                    nn.init.orthogonal_(param.data, gain=factor * 1.0)
                elif 'bias' in name:
                    # Bias initialization: set biases to zero
                    param.data.zero_()
                    # Set forget gate biases to 1 for better training stability
                    n = param.size(0)
                    param.data[n // 4:n // 2].fill_(1.0)
        elif isinstance(module, nn.Linear):
            # Initialize Linear weights and biases
            nn.init.normal_(module.weight.data, mean=0.0, std=factor * (module.weight.size(1) ** -0.5))
            if module.bias is not None:
                module.bias.data.zero_()
                
class T5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.span_rep = SpanRepLayer(config.d_model, config.max_width, span_mode=config.span_mode)

        if config.add_spec_token_span:
            self.spec_span = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        else:
            self.spec_span = None

        if self.config.has_rnn:
            self.rnn = LstmSeq2SeqEncoder(config)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers, you"
            " should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also"
            " provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'encoder.block.0': 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def run_encoder(self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
        ):
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        if return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        return encoder_outputs
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        span_idx: Optional[torch.LongTensor] = None,
        text_length: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        span_embeddings: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        label_smoothing: float =  0.0,
        reduction: str = 'mean'
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        
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

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        # if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
        #     # get decoder inputs from shifting lm labels to the right
        #     decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        span_logits = torch.einsum('bld, bkd->blk', sequence_output, span_embeddings)

        span_loss = None
        token_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100, 
                                        label_smoothing = label_smoothing,
                                        reduction = reduction)
            # move labels to correct device to enable PP
            labels = labels.to(span_logits.device)
            span_logits_shifted = span_logits[:, :-1,:].contiguous()
            labels_shifted = labels[:, 1:].contiguous()
            span_loss = loss_fct(span_logits_shifted.view(-1, span_logits_shifted.size(-1)), labels_shifted.view(-1))
            
            shifted_logits = lm_logits[:, :-1,:].contiguous()
            token_labels = decoder_input_ids[:,1:].contiguous()
            token_loss = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), token_labels.view(-1))

        loss = (span_loss, token_loss)
        if not return_dict:
            output = (span_logits, lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            span_logits=span_logits,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def generate(self, model_inputs, max_length=512, max_new_tokens=None):
        model_inputs = {key: value.to(self.device) for key, value in model_inputs.items()}

        input_ids = model_inputs.pop('input_ids')

        if max_new_tokens is not None:
            new_tokens_count = max_new_tokens
        else:
            new_tokens_count = max(0, max_length - 1)

        batch_size = input_ids.shape[0]

        token_count = 0

        past_key_values = None
        span_embeddings = None

        encoder_outputs = self.run_encoder(input_ids, **model_inputs)

        span_idx = model_inputs.get('span_idx')

        decoder_input_ids = model_inputs.pop("decoder_input_ids")
        decoder_attention_mask = model_inputs.pop("decoder_attention_mask")
        
        curr_input_ids = decoder_input_ids
        attention_mask = model_inputs.pop('attention_mask')

        generated_tokens = []
        while token_count<new_tokens_count:
            outputs = self.forward(input_ids = curr_input_ids,
                                    attention_mask = attention_mask,
                                    decoder_input_ids = curr_input_ids,
                                    decoder_attention_mask = decoder_attention_mask,
                                    span_embeddings=span_embeddings,
                                    encoder_outputs=encoder_outputs,
                                    past_key_values=past_key_values,
                                    **model_inputs, use_cache=True, return_dict=True)

            next_span_ids = outputs.span_logits.argmax(dim=-1)[:, -1]
            next_token_ids = outputs.logits.argmax(dim=-1)[:, -1]

            span_indices = span_idx[torch.arange(batch_size), next_span_ids]

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
                    curr_input_ids[spec_token_idx, -1] = next_token_ids[spec_token_idx]

            decoder_attention_mask = torch.cat([decoder_attention_mask, curr_attention_mask], dim=-1)
            if span_embeddings is None:
                span_embeddings = outputs.span_embeddings

            past_key_values = outputs.past_key_values
            generated_tokens.append(curr_input_ids)

        output_ids = torch.cat(generated_tokens, dim=-1)
        return output_ids