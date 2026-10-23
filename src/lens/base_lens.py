import abc
import torch
import torch.nn as nn

from src.unembed import Unembed


class BaseLens(nn.Module, abc.ABC):
    """
    Common interface of all lenses.

    A lens decodes the hidden state of an intermediate layer into next-token logits. It does so in
    two steps: `transform_hidden` maps the hidden state to something that looks like a final-layer
    hidden state, and the model's own output head (`Unembed`) turns that into logits.

    Layers are indexed by `idx`, where `idx = i` is the output of the i-th transformer block
    (i.e. `output.hidden_states[i + 1]` in Hugging Face). The embeddings and the last block are
    not decoded: the last block's output is what the model itself unembeds.
    """

    def __init__(self, unembed: Unembed):
        """
        Args:
            unembed: The frozen output head of the model, shared by every layer.
        """
        super().__init__()

        self.unembed = unembed

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """
        Same as nn.Module.state_dict, but without the Unembed weights.

        They are a frozen copy of the model's output head, which gets rebuilt from the model every
        time a lens is created, so storing them would only make each checkpoint bigger.
        """
        state = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        for key in [k for k in state if k.startswith(f"{prefix}unembed.")]:
            del state[key]
        return state

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """
        Same as nn.Module.load_state_dict, but the Unembed weights are taken from the lens itself.

        This lets strict loading work on checkpoints saved without them. Older checkpoints that do
        include them still load, since they are a copy of the same weights.
        """
        unembed_state = {f"unembed.{k}": v for k, v in self.unembed.state_dict().items()}
        return super().load_state_dict({**unembed_state, **state_dict}, strict=strict, assign=assign)

    @abc.abstractmethod
    def transform_hidden(self, hs: torch.Tensor, idx: int) -> torch.Tensor:
        """
        Map the hidden state of layer `idx` into the space of the final hidden state, before the final norm.

        Args:
            hs: Hidden states of shape (batch, seq_len, hidden_size).
            idx: Layer index (see the class docstring).

        Returns:
            Tensor of the same shape as `hs`.
        """
        ...

    def forward(self, hs: torch.Tensor, idx: int) -> torch.Tensor:
        """
        Decode the hidden states of layer `idx` into logits of shape (batch, seq_len, vocab_size).

        Subclasses implement this, usually as `self.unembed(self.transform_hidden(hs, idx))`.
        """
        ...
