import logging
import torch

from typing import Union, Optional
from transformers import PreTrainedModel, AutoModelForCausalLM

from src.lens import BaseLens
from src.unembed import Unembed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogitLens(BaseLens):
    """
    The logit lens (nostalgebraist, 2020): decode intermediate hidden states directly with the
    model's output head, without any learned transformation.

    It has no parameters, so it works as the training-free baseline for the other lenses.
    """

    def __init__(self, unembed: Unembed):
        super().__init__(unembed)
        logger.info("Initializing LogitLens.")

    @classmethod
    def from_model(cls, model: PreTrainedModel) -> "LogitLens":
        """Build the lens from the model's output head, on the model's device and in its dtype."""
        logger.info("Creating LogitLens from model.")

        unembed = Unembed(model, device=model.device, dtype=model.dtype)

        return cls(unembed)

    @property
    def device(self) -> torch.device:
        return self.unembed.device

    @property
    def dtype(self) -> torch.dtype:
        return self.unembed.dtype

    def transform_hidden(self, hs: torch.Tensor, idx: int) -> Optional[torch.Tensor]:
        """Identity: the hidden state is used as is, for every layer."""
        return hs

    def forward(self, hs: torch.Tensor, idx: int) -> Optional[torch.Tensor]:
        """Decode `hs` with the output head. `idx` is ignored, since every layer is treated the same."""

        logits = self.unembed(hs)

        return logits
