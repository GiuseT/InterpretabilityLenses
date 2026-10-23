import torch
import torch.nn as nn

from copy import deepcopy
from transformers import PreTrainedModel


class Unembed(nn.Module):
    """
    The model's output head (final norm + unembedding matrix), detached from the rest of the model.

    Every lens ends by calling this module, so that decoding an intermediate hidden state goes
    through exactly the same final norm, projection and logit post-processing the model applies
    to its last layer. The weights are copied from the model and frozen.

    Model-specific post-processing is handled here as well: Gemma2 soft-caps the logits and
    Granite divides them by a constant. Other architectures use the raw projection.

    Attributes:
        final_norm: Copy of the model's last normalization layer (`model.base_model.norm`).
        unembedding: Copy of the output embedding (LM head), mapping hidden_size -> vocab_size.
        final_logit_softcapping: Soft-capping threshold, set only for Gemma2 models.
        logits_scaling: Logit divisor, set only for Granite models.
    """

    def __init__(self, model: PreTrainedModel, device: torch.device, dtype: torch.dtype):
        """
        Args:
            model: The Hugging Face causal LM to take the output head from. It must expose its
                final norm as `model.base_model.norm` (true for Llama-like architectures).
            device: Where to place the copied layers.
            dtype: Dtype of the unembedding matrix, and therefore of the logits.
        """
        super().__init__()

        # The norm keeps the model's dtype: its output is cast to `dtype` in forward(), right before the projection
        self.final_norm = deepcopy(model.base_model.norm).to(device=device)
        self.unembedding = deepcopy(model.get_output_embeddings()).to(device=device, dtype=dtype)

        self.class_name = model.__class__.__name__
        if "Gemma2" in self.class_name:
            self.final_logit_softcapping = model.config.final_logit_softcapping
        elif "Granite" in self.class_name:
            self.logits_scaling = model.config.logits_scaling

        # The output head is part of the model being inspected, never something we train
        self.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return self.unembedding.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.unembedding.weight.dtype

    def normalize(self, hs: torch.Tensor) -> torch.Tensor:
        """Apply the model's final norm to `hs`."""
        return self.final_norm(hs)

    def unembed(self, hs: torch.Tensor) -> torch.Tensor:
        """
        Project already-normalized hidden states to vocabulary logits.

        Applies the same logit post-processing as the original model (soft-capping for Gemma2,
        scaling for Granite), so the result is comparable with the model's own output logits.
        """
        logits = self.unembedding(hs)

        if "Gemma2" in self.class_name:
            # Soft-capping: cap * tanh(logits / cap) squashes logits smoothly into (-cap, cap)
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping
        elif "Granite" in self.class_name:
            logits = logits / self.logits_scaling

        return logits

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        """
        Decode hidden states into logits: final norm, then projection onto the vocabulary.

        Args:
            hs: Hidden states of shape (..., hidden_size).

        Returns:
            Logits of shape (..., vocab_size), in `self.dtype`.
        """
        hs = self.normalize(hs).to(self.dtype)
        logits = self.unembed(hs)

        return logits
