from transformers import PretrainedConfig

class Config(PretrainedConfig):
    model_type = "gliner"
    is_composition = True
    def __init__(self, 
                 name: str = "span level gliner",
                 max_width: int = 12,
                 dropout: float = 0.4,
                 subtoken_pooling: str = "first",
                 span_mode: str = "markerV0",
                 max_len: int = 384,
                 words_splitter_type: str = "whitespace",
                 has_rnn: bool = True,
                 add_spec_token_span: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.max_width = max_width
        self.dropout = dropout
        self.subtoken_pooling = subtoken_pooling
        self.span_mode = span_mode
        self.max_len = max_len
        self.words_splitter_type = words_splitter_type
        self.has_rnn = has_rnn
        self.add_spec_token_span = add_spec_token_span