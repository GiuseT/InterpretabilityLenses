import json
import logging
import os
import torch
import torch.nn as nn

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union
from copy import deepcopy
from transformers import PreTrainedModel, AutoModelForCausalLM
from tqdm import tqdm

from src.lens import BaseLens
from src.unembed import Unembed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TunedLensConfig:
    """
    Hyperparameters of a TunedLens. Saved as config.json next to every checkpoint.

    Attributes:
        device: Device the lens lives on.
        dtype: Dtype of the translators (and of the logits).
        base_model_name_or_path: The model the lens was trained on. Also used to build the checkpoint path.
        hidden_size: Hidden size of that model.
        num_translators: One translator per decoded layer, i.e. `num_hidden_layers - 1`.
        bias: Whether the translators have a bias term.
        opt: Name of the optimizer used for training. Only used to tell checkpoints apart.
    """
    device: torch.device
    dtype: torch.dtype
    base_model_name_or_path: str
    hidden_size: int
    num_translators: int
    bias: bool
    opt: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TunedLensConfig":
        return cls(**config_dict)


class TunedLens(BaseLens):
    """
    The tuned lens (Belrose et al., 2023): one learned affine translator per layer.

    Layer `idx` is decoded as `unembed(h + W_idx h + b_idx)`. The translators are full
    hidden_size x hidden_size matrices, so this is the most expressive (and the largest) of the
    learned lenses here. Since W and b start at zero, an untrained TunedLens is exactly the logit lens.
    """

    def __init__(self, unembed: Unembed, config: TunedLensConfig):
        super().__init__(unembed)

        self.config = config

        logger.info(f"Initializing TunedLens with config: {config}")

        # Zero init + the residual in transform_hidden: training starts from the logit lens
        translator = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias, device=self.device, dtype=self.dtype)
        nn.init.zeros_(translator.weight)
        if config.bias:
            nn.init.zeros_(translator.bias)

        self.translators = nn.ModuleList(deepcopy(translator) for _ in range(config.num_translators))

        logger.info(f"TunedLens initialized with {len(self.translators)} translators.")

        # Log memory occupation
        num_parameters = sum(p.nelement() for p in self.translators.parameters())
        logger.info(f"TunedLens number of parameters: {num_parameters:,}")

        param_size = sum(p.nelement() * p.element_size() for p in self.translators.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.translators.buffers())
        size_all_mb = (param_size + buffer_size) / (1024 * 1024)
        logger.info(f"TunedLens parameters memory footprint: {size_all_mb:,.2f} MB")

    def get_translators(self):
        return self.translators

    def get_path(self):
        """Checkpoint folder of this lens: checkpoints/<model>/TunedLens-<opt>."""
        base_model_name = self.config.base_model_name_or_path.split("/")[-1].replace(".", "_")
        path = os.path.join("checkpoints", base_model_name, f"TunedLens-{self.config.opt}")

        return path

    def __len__(self) -> int:
        return self.config.num_translators

    @property
    def device(self) -> torch.device:
        return self.config.device

    @property
    def dtype(self) -> torch.dtype:
        return self.config.dtype

    @classmethod
    def from_model(cls, model: PreTrainedModel, config: Dict[str, Any]) -> "TunedLens":
        """
        Build a fresh (untrained) TunedLens for `model`.

        Args:
            model: The model to decode.
            config: TunedLensConfig fields. `device` and `dtype` can be given as strings; `hidden_size`,
                `num_translators` and `base_model_name_or_path` are read from the model when missing.
                Note that the dict is modified in place.
        """
        logger.info("Creating TunedLens from model and config dictionary.")

        config["device"] = torch.device(config["device"])
        config["dtype"] = torch.bfloat16 if "bfloat16" in config.get("dtype", "bfloat16") else torch.float32

        if "hidden_size" not in config:
            config["hidden_size"] = model.config.hidden_size
        if "num_translators" not in config:
            # The last block is not decoded (see BaseLens)
            config["num_translators"] = model.config.num_hidden_layers - 1
        if "base_model_name_or_path" not in config:
             config["base_model_name_or_path"] = model.config.name_or_path

        unembed = Unembed(model, device=config["device"], dtype=config["dtype"])

        lens_config = TunedLensConfig.from_dict(config)
        return cls(unembed, lens_config)

    @classmethod
    def from_model_and_pretrained(cls, model: PreTrainedModel, path: str) -> "TunedLens":
        """
        Load a trained TunedLens for `model`.

        Args:
            model: The model the lens was trained on.
            path: A checkpoint folder written by `save`, i.e. containing config.json and params.pt.
        """
        logger.info(f"Loading TunedLens from {path}")
        config_path = os.path.join(path, "config.json")
        params_path = os.path.join(path, "params.pt")

        if not os.path.exists(config_path) or not os.path.exists(params_path):
            raise FileNotFoundError(f"Could not find config.json or params.pt in {path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        lens = cls.from_model(model, config_dict)

        state_dict = torch.load(params_path, map_location=lens.device)
        lens.load_state_dict(state_dict)

        logger.info("TunedLens loaded successfully.")
        return lens

    def save(self, steps: int) -> None:
        """
        Save config.json and params.pt to `<get_path()>/<steps>_steps`.

        Args:
            steps: Training step of this checkpoint, used to name its folder.
        """
        path = os.path.join(self.get_path(), f"{steps}_steps")

        logger.info(f"Saving TunedLens to {path}")
        os.makedirs(path, exist_ok=True)

        config_path = os.path.join(path, "config.json")
        params_path = os.path.join(path, "params.pt")

        # torch.device and torch.dtype are not JSON-serializable; from_model parses the strings back
        config = self.config.to_dict()
        config["dtype"] = str(config["dtype"])
        config["device"] = str(config["device"])

        # Check for non-serializable items and remove them
        keys_to_remove = []
        for key, value in config.items():
            try:
                json.dumps(value)
            except (TypeError, OverflowError):
                logger.warning(f"Key '{key}' in config is not JSON serializable and will be removed.")
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del config[key]

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        torch.save(self.state_dict(), params_path)
        logger.info("TunedLens saved.")

    def transform_hidden(self, hs: torch.Tensor, idx: int) -> Optional[torch.Tensor]:
        """Apply the translator of layer `idx` as a residual: h + W_idx h + b_idx."""
        hs = hs + self.translators[idx](hs)

        return hs

    def forward(self, hs: torch.Tensor, idx: int) -> Optional[torch.Tensor]:
        # Hidden states come from the model, which may sit on another device and/or use another dtype
        hs = hs.to(device=self.device, dtype=self.dtype)

        hs = self.transform_hidden(hs, idx)

        logits = self.unembed(hs)

        return logits

    @torch.no_grad()
    def prepare_for_global_ablation(self) -> None:
        """
        Ablation: replace every translator with the average of all of them.

        This turns the per-layer lens into a single shared transformation, to measure how much of
        the tuned lens performance actually depends on having a different translator per layer.
        The change is in place and cannot be undone, so reload the lens to get the original back.
        """
        logger.info("Preparing TunedLens for global ablation: computing average translator...")

        avg_weight = torch.stack([t.weight.data for t in self.translators]).mean(dim=0)
        if self.config.bias:
            avg_bias = torch.stack([t.bias.data for t in self.translators]).mean(dim=0)

        for t in self.translators:
            t.weight.data.copy_(avg_weight)
            if self.config.bias:
                t.bias.data.copy_(avg_bias)

        logger.info(f"Successfully replaced {len(self.translators)} translators with their global average.")

    @torch.no_grad()
    def prepare_for_local_ablation(self, k_factor: int = 32) -> None:
        """
        Ablation: replace each translator's weight with its best rank-k approximation (truncated SVD).

        With k = hidden_size // k_factor, this gives a TunedLens with the same rank budget as a
        LoRALens with rank_factor = k_factor, so the two can be compared directly. Biases are left
        untouched. Like the global ablation, this overwrites the weights in place.

        Args:
            k_factor: The rank kept is hidden_size // k_factor.
        """
        k = self.config.hidden_size // k_factor
        logger.info(f"Preparing TunedLens for local ablation: applying low-rank SVD approximation (k={k}) to translators...")

        for t in tqdm(self.translators):
            W = t.weight.data
            original_dtype = W.dtype

            # SVD is not implemented for bfloat16, and is more stable in float32 anyway
            W_fp32 = W.to(torch.float32)

            # Perform SVD: W = U * S * Vh
            U, S, Vh = torch.linalg.svd(W_fp32, full_matrices=False)

            # Keeping the top-k singular values gives the best rank-k approximation (Eckart-Young)
            U_k = U[:, :k]
            S_k = S[:k]
            Vh_k = Vh[:k, :]

            # Reconstruct the low-rank approximation
            W_approx = U_k @ torch.diag(S_k) @ Vh_k

            # Cast back to original dtype and overwrite
            t.weight.data.copy_(W_approx.to(original_dtype))

        logger.info(f"Successfully applied low-rank approximation to {len(self.translators)} translators.")
