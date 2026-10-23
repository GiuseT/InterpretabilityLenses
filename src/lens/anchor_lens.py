import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple, Optional, Union
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from src.lens import BaseLens
from src.unembed import Unembed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnchorLensConfig:
    """
    Hyperparameters of an AnchorLens. Saved as config.json next to every checkpoint.

    Attributes:
        device: Device the lens lives on.
        dtype: Dtype of the lens parameters (and of the logits).
        base_model_name_or_path: The model the lens was trained on. Also used to build the checkpoint paths.
        hidden_size: Hidden size of that model.
        rank_factor: The local adapters have rank hidden_size // rank_factor. None when use_locals is False.
        num_translators: One local adapter per decoded layer, i.e. `num_hidden_layers - 1`.
        bias: Adds a bias to the global transform and, if present, one bias per local adapter.
        opt: Name of the optimizer used for training. It is part of the checkpoint paths, so the
            global + locals stage looks for a global-only checkpoint trained with the same optimizer.
        use_global: Use the global transform, shared by all layers.
        use_locals: Use the per-layer low-rank adapters.
        global_init: Which global-only checkpoint initializes the global transform when both
            use_global and use_locals are True: "latest", or a number of training steps such as "100".
    """
    device: torch.device
    dtype: torch.dtype
    base_model_name_or_path: str
    hidden_size: int
    rank_factor: int
    num_translators: int
    bias: bool
    opt: str
    use_global: bool
    use_locals: bool
    global_init: str = "latest"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AnchorLensConfig":
        return cls(**config_dict)


class AnchorLens(BaseLens):
    """
    A lens made of one global transform shared by all layers (the "anchor") and, optionally,
    a small low-rank correction per layer.

    Layer `idx` is decoded as `unembed(g + B_idx A_idx g + b_idx)`, with `g = G h + c`, where G and c
    are the same for every layer. The idea is that most of what separates an intermediate hidden
    state from the final one is shared across layers, so a single full-rank matrix can carry most of
    it, and the per-layer adapters only need to fix what is left.

    Training happens in two stages:
      1. Global only (use_global=True, use_locals=False): G starts at the identity, i.e. from the
         logit lens, and is trained on all layers at once.
      2. Global + locals (both True): G is loaded from a stage-1 checkpoint (see `global_init`) and
         frozen, and only the local adapters are trained.
    Locals only (use_global=False) is also possible, and is then essentially a LoRALens.
    """

    def __init__(self, unembed: Unembed, config: AnchorLensConfig):
        super().__init__(unembed)
        self.config = config

        logger.info(f"Initializing AnchorLens with config:")
        for key, value in config.to_dict().items():
            logger.info(f"  - {key}: {value}")

        # Training steps of the global-only checkpoint used to initialize the global transform (if any)
        self.global_init_step = None

        if config.use_global:
            # Not a residual like the local adapters: G replaces h, so identity is the "no change" init
            self.global_transform = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias, dtype=config.dtype)
            nn.init.eye_(self.global_transform.weight)
        else:
            self.global_transform = None

        if config.use_locals:
            # Same init as LoRALens: B = 0, so the adapters start as a no-op
            down_proj = nn.Linear(config.hidden_size, config.hidden_size // config.rank_factor, bias=False, dtype=config.dtype)
            nn.init.kaiming_uniform_(down_proj.weight, a=5 ** 0.5)

            up_proj = nn.Linear(config.hidden_size // config.rank_factor, config.hidden_size, bias=False, dtype=config.dtype)
            nn.init.zeros_(up_proj.weight)

            self.locals = nn.ModuleList(
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

            if self.global_transform is not None:
                # Stage 2: start from a trained global transform and keep it fixed
                self._load_initialization_weights()

                self.disable_global_gradients()
        else:
            self.locals = None
            self.bias = None

        # Move everything to device
        self.to(self.config.device)

        # Log memory occupation
        self._log_memory_usage()

    def _load_initialization_weights(self):
        """
        Copy the global transform from a global-only AnchorLens checkpoint.

        The checkpoint is looked up in checkpoints/<model>/AnchorLens-G-<opt>/ and picked according to
        `config.global_init`: "latest" takes the `<N>_steps` folder with the highest N, a number takes
        that exact folder. The step actually loaded is stored in `self.global_init_step`, which
        `get_path` uses to keep runs started from different checkpoints apart.
        """

        base_model_name = self.config.base_model_name_or_path.split("/")[-1].replace(".", "_")
        init_dir = os.path.join("checkpoints", base_model_name, f"AnchorLens-G-{self.config.opt}")
        filename = "params.pt"
        global_init = str(self.config.global_init).strip()

        if global_init == "latest":
            step_dirs = []
            if os.path.isdir(init_dir):
                step_dirs = [d for d in os.listdir(init_dir)
                             if d.endswith("_steps") and d[:-len("_steps")].isdigit()
                             and os.path.exists(os.path.join(init_dir, d, filename))]
            if step_dirs:
                latest_dir = max(step_dirs, key=lambda d: int(d[:-len("_steps")]))
                filepath = os.path.join(init_dir, latest_dir, filename)
                init_step = int(latest_dir[:-len("_steps")])
            else:
                # Old layout, from before checkpoints were saved per step
                filepath = os.path.join(init_dir, filename)
                init_step = None
        elif global_init.isdigit():
            filepath = os.path.join(init_dir, f"{int(global_init)}_steps", filename)
            init_step = int(global_init)
        else:
            raise ValueError(f"Invalid global_init '{self.config.global_init}': expected \"latest\" or a number of steps (e.g. \"100\").")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Initialization weights of the global transformation not found at {filepath}. Please ensure the global-only AnchorLens has been trained first.")

        logger.info(f"Loading init weights from {filepath}...")

        state = torch.load(filepath)

        # Only the global transform is taken from the checkpoint
        self.global_transform.weight.data.copy_(state["global_transform.weight"])
        if self.global_transform.bias is not None:
            self.global_transform.bias.data.copy_(state["global_transform.bias"])
        self.global_init_step = init_step
        logger.info("Global transformation initialized from pre-trained.")

    def _log_memory_usage(self):
        """Log parameter counts and memory footprint, both total and trainable only."""
        total_params = sum(p.nelement() for p in self.parameters() if p is not None)
        learnable_params = sum(p.nelement() for p in self.parameters() if p is not None and p.requires_grad)
        non_learnable_params = total_params - learnable_params

        logger.info(f"AnchorLens total parameters: {total_params:,}")
        logger.info(f"AnchorLens learnable parameters: {learnable_params:,}")
        logger.info(f"AnchorLens non-learnable parameters: {non_learnable_params:,}")

        param_size = sum(p.nelement() * p.element_size() for p in self.parameters() if p is not None)
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers() if b is not None)
        size_all_mb = (param_size + buffer_size) / (1024 * 1024)
        logger.info(f"AnchorLens total parameters memory footprint: {size_all_mb:,.2f} MB")

        param_size = sum(p.nelement() * p.element_size() for p in self.parameters() if p is not None and p.requires_grad)
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers() if b is not None and b.requires_grad)
        size_all_mb = (param_size + buffer_size) / (1024 * 1024)
        logger.info(f"AnchorLens learnable parameters memory footprint: {size_all_mb:,.2f} MB")

    def disable_locals_gradients(self):
        """Freeze the local adapters and their biases."""
        if self.locals is not None:
            for local in self.locals:
                local.requires_grad_(False)
            if self.bias is not None:
                for b in self.bias:
                    b.requires_grad_(False)
        logger.info("Disabled gradients for local components.")
        self._log_active_params()

    def enable_locals_gradients(self):
        """Unfreeze the local adapters and their biases."""
        if self.locals is not None:
            for local in self.locals:
                local.requires_grad_(True)
            if self.bias is not None:
                for b in self.bias:
                    b.requires_grad_(True)
        logger.info("Enabled gradients for local components.")
        self._log_active_params()

    def disable_global_gradients(self):
        """Freeze the global transform (weight and bias)."""
        if self.global_transform is not None:
            self.global_transform.requires_grad_(False)
        logger.info("Disabled gradients for global components.")
        self._log_active_params()

    def enable_global_gradients(self):
        """Unfreeze the global transform (weight and bias)."""
        if self.global_transform is not None:
            self.global_transform.requires_grad_(True)
        logger.info("Enabled gradients for global components.")
        self._log_active_params()

    def _log_active_params(self):
        active_params = sum(p.nelement() for p in self.parameters() if p is not None and p.requires_grad)
        logger.info(f"Current active (learnable) parameters: {active_params:,}")

    def get_config(self):
        return self.config

    def get_global_transform(self):
        """The global nn.Linear, or None if use_global is False."""
        return self.global_transform

    def get_bias(self):
        """The per-layer biases of the local adapters, or None. The global bias lives in the global transform."""
        return self.bias

    def get_locals(self):
        """The local adapters, each an nn.Sequential(down_proj, up_proj), or None if use_locals is False."""
        return self.locals

    def get_path(self):
        """
        Checkpoint folder of this lens, under checkpoints/<model>/. The folder name lists the components in use:
            AnchorLens-G-<opt>                       global only
            AnchorLens-G_<step>-L-rf_<rf>-<opt>      global + locals, global taken from stage 1 at <step>
            AnchorLens-L-rf_<rf>-<opt>               locals only
        """
        base_model_name = self.config.base_model_name_or_path.split("/")[-1].replace(".", "_")
        setting = ("AnchorLens"
                   f"{'-G' if self.config.use_global else ''}{f'_{self.global_init_step}' if self.global_init_step is not None else ''}"
                   f"{f'-L-rf_{self.config.rank_factor}' if self.config.use_locals else ''}"
                   f"-{self.config.opt}")
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
    def from_model(cls, model: PreTrainedModel, config: Dict[str, Any]) -> "AnchorLens":
        """
        Build a fresh AnchorLens for `model`. With both use_global and use_locals, the global
        transform is loaded from a stage-1 checkpoint, which must therefore already exist.

        Args:
            model: The model to decode.
            config: AnchorLensConfig fields. `device` and `dtype` can be given as strings; `hidden_size`,
                `num_translators` and `base_model_name_or_path` are read from the model when missing.
                Note that the dict is modified in place.
        """
        logger.info("Creating AnchorLens from model and config dictionary.")

        if config.get("use_global") is False and config.get("use_locals") is False:
            raise ValueError("At least one of use_global or use_locals must be True.")

        config["device"] = torch.device(config["device"])
        config["dtype"] = torch.bfloat16 if "bfloat16" in config.get("dtype", "bfloat16") else torch.float32

        if not config.get("use_locals", True):
            config["rank_factor"] = None

        if "hidden_size" not in config:
            config["hidden_size"] = model.config.hidden_size
        if "num_translators" not in config:
            # The last block is not decoded (see BaseLens)
            config["num_translators"] = model.config.num_hidden_layers - 1
        if "base_model_name_or_path" not in config:
            config["base_model_name_or_path"] = model.config.name_or_path

        unembed = Unembed(model, device=config["device"], dtype=config["dtype"])

        lens_config = AnchorLensConfig.from_dict(config)
        return cls(unembed, lens_config)

    @classmethod
    def from_model_and_pretrained(cls, model: PreTrainedModel, path: str) -> "AnchorLens":
        """
        Load a trained AnchorLens for `model`.

        For a global + locals lens, building it still goes through stage-1 initialization, so the
        global-only checkpoint must be on disk too, although its weights are then overwritten
        by the ones in `path`.

        Args:
            model: The model the lens was trained on.
            path: A checkpoint folder written by `save`, i.e. containing config.json and params.pt.
        """
        logger.info(f"Loading AnchorLens from {path}")
        config_path = os.path.join(path, "config.json")
        params_path = os.path.join(path, "params.pt")

        if not os.path.exists(config_path) or not os.path.exists(params_path):
            raise FileNotFoundError(f"Could not find config.json or params.pt in {path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        lens = cls.from_model(model, config_dict)

        state_dict = torch.load(params_path, map_location=lens.device)
        lens.load_state_dict(state_dict)

        logger.info("AnchorLens loaded successfully.")
        return lens

    def save(self, steps: int) -> None:
        """
        Save config.json and params.pt to `<get_path()>/<steps>_steps`.

        Args:
            steps: Training step of this checkpoint, used to name its folder.
        """
        path = os.path.join(self.get_path(), f"{steps}_steps")

        logger.info(f"Saving AnchorLens to {path}")
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
        logger.info("AnchorLens saved.")

    def transform_hidden(self, hs: torch.Tensor, idx: int) -> Optional[torch.Tensor]:
        """Apply the global transform first, then the local adapter of layer `idx` on its output."""
        if self.global_transform is not None:
            hs = self.global_transform(hs)

        if self.locals is not None:
            hs = hs + self.locals[idx](hs)

            if self.bias is not None:
                hs = hs + self.bias[idx]

        return hs

    def forward(self, hs: torch.Tensor, idx: int) -> Optional[torch.Tensor]:
        # Hidden states come from the model, which may sit on another device and/or use another dtype
        hs = hs.to(device=self.device, dtype=self.dtype)

        hs = self.transform_hidden(hs, idx)

        logits = self.unembed(hs)

        return logits
