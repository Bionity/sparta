from transformers.utils import logging
from transformers import T5Config

from ...config import Config


logger = logging.get_logger(__name__)


class T5Config(T5Config, Config):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
    @property
    def hidden_size(self):
        return self.d_model
