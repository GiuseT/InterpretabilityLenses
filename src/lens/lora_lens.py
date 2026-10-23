import json
import logging
import os
import torch
import torch.nn as nn

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union
from copy import deepcopy
from transformers import PreTrainedModel, AutoModelForCausalLM

from src.lens import BaseLens
from src.unembed import Unembed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoRALensConfig:
    """
    Hyperparameters of a LoRALens. Saved as config.json next to every checkpoint.

    Attributes:
        device: Device the lens lives on.
        dtype: Dtype of the adapters (and of the logits).
        base_model_name_or_path: The model the lens was trained on. Also used to build the checkpoint path.
        hidden_size: Hidden size of that model.
        rank_factor: The adapters have rank hidden_size // rank_factor.
        num_translators: One adapter per decoded layer, i.e. `num_hidden_layers - 1`.
        bias: Whether each layer also gets its own (full-size) bias vector.
        opt: Name of the optimizer used for training. Only used to tell checkpoints apart.
    """
    device: torch.device
    dtype: torch.dtype
    base_model_name_or_path: str
    hidden_size: int
    rank_factor: int
    num_translators: int
    bias: bool
    opt: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LoRALensConfig":
        return cls(**config_dict)


class LoRALens(BaseLens):
    """
    A tuned lens whose translators are low-rank, LoRA style.

    Layer `idx` is decoded as `unembed(h + B_idx A_idx h + b_idx)`, where A_idx projects down to
    rank r = hidden_size // rank_factor and B_idx projects back up. Compared to the TunedLens this
    cuts the parameters per layer from hidden_size^2 to 2 * hidden_size * r.
    """

    def __init__(self, unembed: Unembed, config: LoRALensConfig):
        super().__init__(unembed)
        self.config = config

        logger.info(f"Initializing LoRALens with config:")
        for key, value in config.to_dict().items():
            logger.info(f"  - {key}: {value}")

        # Standard LoRA init: A gets nn.Linear's default init, B starts at zero.
        # The adapter output is then zero, so training starts from the logit lens.
        down_proj = nn.Linear(config.hidden_size, config.hidden_size // config.rank_factor, bias=False, dtype=config.dtype)
        nn.init.kaiming_uniform_(down_proj.weight, a=5 ** 0.5)

        up_proj = nn.Linear(config.hidden_size // config.rank_factor, config.hidden_size, bias=False, dtype=config.dtype)
        nn.init.zeros_(up_proj.weight)

        self.translators = nn.ModuleList(
            nn.Sequential(deepcopy(down_proj), deepcopy(up_proj))
            for _ in range(self.config.num_translators)
        )

        if config.bias:
            self.bias = nn.ParameterList(
                nn.Parameter(torch.zeros(config.hidden_size, dtype=config.dtype))
                for _ in range(config.num_translators)
            )
        else:
            self.bias = None

        # Move everything to device
        self.to(self.config.device)

        # Log memory occupation
        self._log_memory_usage()

    def _log_memory_usage(self):
        """Log parameter counts and memory footprint, both total and trainable only."""
        total_params = sum(p.nelement() for p in self.parameters() if p is not None)
        learnable_params = sum(p.nelement() for p in self.parameters() if p is not None and p.requires_grad)
        non_learnable_params = total_params - learnable_params

        logger.info(f"LoRALens total parameters: {total_params:,}")
        logger.info(f"LoRALens learnable parameters: {learnable_params:,}")
        logger.info(f"LoRALens non-learnable parameters: {non_learnable_params:,}")

        param_size = sum(p.nelement() * p.element_size() for p in self.parameters() if p is not None)
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers() if b is not None)
        size_all_mb = (param_size + buffer_size) / (1024 * 1024)
        logger.info(f"LoRALens total parameters memory footprint: {size_all_mb:,.2f} MB")

        param_size = sum(p.nelement() * p.element_size() for p in self.parameters() if p is not None and p.requires_grad)
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers() if b is not None and b.requires_grad)
        size_all_mb = (param_size + buffer_size) / (1024 * 1024)
        logger.info(f"LoRALens learnable parameters memory footprint: {size_all_mb:,.2f} MB")

    def get_config(self):
        return self.config

    def get_translators(self):
        """
        Return the adapters and the per-layer biases, as two sequences that can be zipped together.

        Each adapter is an nn.Sequential(down_proj, up_proj). Without bias, the second sequence is all None.
        """
        biases = self.bias if self.bias is not None else [None] * self.config.num_translators

        return self.translators, biases

    def get_path(self):
        """Checkpoint folder of this lens: checkpoints/<model>/LoRALens-rf_<rank_factor>-<opt>."""
        base_model_name = self.config.base_model_name_or_path.split("/")[-1].replace(".", "_")
        setting = f"LoRALens-rf_{self.config.rank_factor}-{self.config.opt}"
        path = os.path.join("checkpoints", base_model_name, setting)
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
    def from_model(cls, model: PreTrainedModel, config: Dict[str, Any]) -> "LoRALens":
        """
        Build a fresh (untrained) LoRALens for `model`.

        Args:
            model: The model to decode.
            config: LoRALensConfig fields. `device` and `dtype` can be given as strings; `hidden_size`,
                `num_translators` and `base_model_name_or_path` are read from the model when missing.
                Note that the dict is modified in place.
        """
        logger.info("Creating LoRALens from model and config dictionary.")

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

        lens_config = LoRALensConfig.from_dict(config)
        return cls(unembed, lens_config)

    @classmethod
    def from_model_and_pretrained(cls, model: PreTrainedModel, path: str) -> "LoRALens":
        """
        Load a trained LoRALens for `model`.

        Args:
            model: The model the lens was trained on.
            path: A checkpoint folder written by `save`, i.e. containing config.json and params.pt.
        """
        logger.info(f"Loading LoRALens from {path}")
        config_path = os.path.join(path, "config.json")
        params_path = os.path.join(path, "params.pt")

        if not os.path.exists(config_path) or not os.path.exists(params_path):
            raise FileNotFoundError(f"Could not find config.json or params.pt in {path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        lens = cls.from_model(model, config_dict)

        state_dict = torch.load(params_path, map_location=lens.device)
        lens.load_state_dict(state_dict)

        logger.info("LoRALens loaded successfully.")
        return lens

    def save(self, steps: int) -> None:
        """
        Save config.json and params.pt to `<get_path()>/<steps>_steps`.

        Args:
            steps: Training step of this checkpoint, used to name its folder.
        """
        path = os.path.join(self.get_path(), f"{steps}_steps")

        logger.info(f"Saving LoRALens to {path}")
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
        logger.info("LoRALens saved.")

    def transform_hidden(self, hs: torch.Tensor, idx: int) -> Optional[torch.Tensor]:
        """Apply the adapter of layer `idx` as a residual, plus its bias: h + B_idx A_idx h + b_idx."""
        hs = hs + self.translators[idx](hs)

        if self.bias is not None:
            hs = hs + self.bias[idx]

        return hs

    def forward(self, hs: torch.Tensor, idx: int) -> Optional[torch.Tensor]:
        # Hidden states come from the model, which may sit on another device and/or use another dtype
        hs = hs.to(device=self.device, dtype=self.dtype)

        hs = self.transform_hidden(hs, idx)

        logits = self.unembed(hs)

        return logits
