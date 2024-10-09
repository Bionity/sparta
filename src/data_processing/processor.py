from abc import ABC, abstractmethod

import torch


class BaseProcessor(ABC):
    def __init__(self, tokenizer, tokenization_args={}, max_width=12, spec_token_span=0):
        self.tokenizer = tokenizer
        self.tokenization_args = tokenization_args
        self.max_width = max_width
        self.spec_token_span = spec_token_span

    @abstractmethod
    def tokenize_input(self, text):
        pass
    
    @abstractmethod
    def construct_labels(self, tokenized_output, spans):
        pass

    @abstractmethod
    def prepare_model_inputs(self, tokenized_prompt, tokenized_output, spans):
        pass
    
    def init_attention_mask(self, size):
        attention_mask = torch.ones((1, size), dtype=torch.int64)
        return attention_mask
    
    def get_token2start(self, tokens):
        token2start_id = {}
        for id, token in enumerate(tokens):
            if token not in token2start_id:
                token2start_id[token] = [id]
            else:
                token2start_id[token].append(id)
        return token2start_id

    def find_longest(self, spans):
        if not len(spans):
            return (-1, -1)
        
        longest = spans[0]
        max_len = longest[-1]-longest[0]
        for span in spans:
            curr_len = span[-1]-span[0]
            if curr_len>max_len:
                longest = span
                max_len = curr_len
        return longest

    def recursive_search(self, curr_o, start, end, output, token2start, depth=0, curr_spans=[]):
        if curr_o == len(output) or depth>=self.max_width:
            return curr_spans
        if output[curr_o] in token2start:
            for curr_start in token2start[output[curr_o]]:
                if start==-1:
                    start = curr_start
                    end = curr_start
                if curr_start==end or curr_start==(end+1):
                    self.recursive_search(curr_o+1, start, curr_start, output, token2start, depth+1, curr_spans)
        curr_spans.append((start, end))
        return curr_spans

    def construct_spans(self, tokenized_prompt, tokenized_output):
        token2start_id = self.get_token2start(tokenized_prompt)

        spans = []

        curr_o = 0
        while (curr_o<len(tokenized_output)):
            start = -1
            end = -1
            curr_spans = self.recursive_search(curr_o, start, end, tokenized_output, token2start_id, curr_spans=[])
            span = self.find_longest(curr_spans)
            spans.append(span)
            curr_o+=span[-1]-span[0]+1
        return spans

    def tokenize_input_output(self, prompt, output=None):
        tokenized_prompt = self.tokenize_input(prompt)
        if output is not None:
          tokenized_output = self.tokenize_input(output)
        else:
          tokenized_output = None
        return tokenized_prompt,tokenized_output 
    
    def process_example(self, prompt, output=None):
        tokenized_prompt, tokenized_output = self.tokenize_input_output(prompt, output)
        
        spans_idx = [(i, i + j) for i in range(len(tokenized_prompt)) for j in range(self.max_width)]

        if self.spec_token_span:
            spans_idx.insert(0, (0, 0))

        model_inputs = self.prepare_model_inputs(tokenized_prompt, tokenized_output)

        if output is not None:
          output_spans = self.construct_spans(tokenized_prompt, tokenized_output)
          labels = self.construct_labels(output_spans)

          to_decode = []
          for label in labels:
            if label != -100:
              start = label//12
              end = start+label%12
              to_decode.extend(tokenized_prompt[start:end+1])
          
          print('Decoder labels: ', self.tokenizer.decode(to_decode))

          model_inputs['labels'] = torch.tensor(labels).unsqueeze(0)

        model_inputs['span_idx'] = torch.tensor(spans_idx, dtype=torch.int64).unsqueeze(0)

        model_inputs['text_length'] = torch.tensor([len(tokenized_prompt)]).unsqueeze(0)
        return model_inputs


class TokenLevelEncoderDecoderProcessor(BaseProcessor):
    def __init__(self, tokenizer, tokenizer_args, decoder_start_token_id, max_width=12, **kwargs):
        super(TokenLevelEncoderDecoderProcessor, self).__init__(tokenizer, tokenizer_args, max_width)
        self.decoder_start_token_id = decoder_start_token_id

    def tokenize_input(self, text):
        tokens = self.tokenizer.encode(text, **self.tokenization_args)
        return tokens
    
    def tokenize_output(self, text):
        with self.tokenizer.as_target_tokenizer():
            tokens = self.tokenizer.encode(text, **self.tokenization_args)
        return tokens
    
    def construct_labels(self, spans):
        labels = []
        for span in spans:
            if span[0]==-1 and span[1]==-1:
                if self.spec_token_span:
                    labels.append(0)
                else:
                    labels.append(-100)
                continue
            span_id = span[0]*self.max_width+span[1]-span[0]+self.spec_token_span
            labels.append(span_id)
            for i in range(span[1]-span[0]):
                labels.append(-100)
        return labels

    def prepare_model_inputs(self, tokenized_prompt, tokenized_output):
        decoder_input_ids = [self.decoder_start_token_id]
        if tokenized_output is not None:
          decoder_input_ids.extend(tokenized_output[:-1])

        model_inputs = {"input_ids": torch.tensor(tokenized_prompt).unsqueeze(0),
                        "decoder_input_ids": torch.tensor(decoder_input_ids).unsqueeze(0)}
        attention_mask = self.init_attention_mask(len(tokenized_prompt))
        decoder_attention_mask = self.init_attention_mask(len(decoder_input_ids))

        model_inputs['attention_mask'] = attention_mask
        model_inputs['decoder_attention_mask'] = decoder_attention_mask

        return model_inputs
    

class TokenLevelDecoderProcessor(BaseProcessor):
    def tokenize_input(self, text):
        tokens = self.tokenizer.encode(text, **self.tokenization_args)
        return tokens
    
    def construct_labels(self, spans):
        labels = []
        for span in spans:
            if span[0]==-1 and span[1]==-1:
                if self.spec_token_span:
                    labels.append(0)
                else:
                    labels.append(-100)
                continue
            span_id = span[0]*self.max_width+span[1]-span[0]+self.spec_token_span
            labels.append(span_id)
            for i in range(span[1]-span[0]):
                labels.append(-100)
        return labels

    def prepare_model_inputs(self, tokenized_prompt, tokenized_output=None):
        if tokenized_output is not None:
          input_ids = tokenized_prompt+tokenized_output[:-1]
        else:
          input_ids = tokenized_prompt
        model_inputs = {"input_ids": torch.tensor(input_ids).unsqueeze(0)}
        attention_mask = self.init_attention_mask(len(input_ids))

        model_inputs['attention_mask'] = attention_mask
        return model_inputs
    

    def process_example(self, prompt, output=None):
        tokenized_prompt = self.tokenize_input(prompt)

        if output is not None:
          tokenized_output = self.tokenize_input(output)
        else:
          tokenized_output = None
        
        spans_idx = [(i, i + j) for i in range(len(tokenized_prompt)) for j in range(self.max_width)]

        if self.spec_token_span:
            spans_idx.insert(0, (-1, -1))

        model_inputs = self.prepare_model_inputs(tokenized_prompt, tokenized_output)

        if output is not None:
          output_spans = self.construct_spans(tokenized_prompt, tokenized_output)
          labels = self.construct_labels(output_spans)

          blank_labels = [-100 for i in range(len(tokenized_prompt)-1)]
          labels = blank_labels+labels
          model_inputs['labels'] = torch.tensor(labels).unsqueeze(0)

        model_inputs['span_idx'] = torch.tensor(spans_idx, dtype=torch.int64).unsqueeze(0)

        model_inputs['text_length'] = torch.tensor([len(tokenized_prompt)]).unsqueeze(0)
        return model_inputs