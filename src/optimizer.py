import logging
import torch.nn as nn
import torch.optim as optim

from typing import Optional, Literal, List
from dataclasses import dataclass
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


@dataclass
class Optimizer:
    """
    Optimizer and LR schedule used to train the lenses.

    Attributes:
        weight_decay: Weight decay coefficient.
        lr_scale: Multiplier on the base LR, which is 1e-3 for AdamW and 1e-2 for Muon.
        momentum: Momentum for Muon, beta1 for AdamW.
        optimizer: "AdamW", or "Muon" (Muon for the 2D weight matrices, AdamW for everything else).
        warmup_steps: Linear warmup length. If None, min(1000, num_steps // 10).
    """

    weight_decay: float = 0.01
    lr_scale: float = 1.0
    momentum: float = 0.9
    optimizer: Literal["AdamW", "Muon"] = "AdamW"
    warmup_steps: Optional[int] = None

    def create_scheduler(self, opt: optim.Optimizer, num_steps: int) -> optim.lr_scheduler.LambdaLR:
        """Linear warmup to the base LR, then linear decay to zero at `num_steps`."""
        logger.info("Creating scheduler...")
        self.warmup_steps = self.warmup_steps or min(1000, num_steps // 10)
        logger.info(f"Using {self.warmup_steps} LR warmup steps for {self.optimizer}.")

        # num_training_steps is the total, warmup included: the LR reaches zero exactly at the last step
        scheduler = get_linear_schedule_with_warmup(opt, self.warmup_steps, num_steps)

        return scheduler

    def create_optim(self, params: List[nn.Parameter]) -> List[optim.Optimizer]:
        """
        Build the optimizer(s) for `params`.

        Returns a list because Muon only works on matrices: with "Muon" there is one Muon optimizer
        for the 2D parameters and one AdamW for the rest (biases), with a 10x smaller LR. The
        training loop steps all of them.
        """
        logger.info(f"Creating {self.optimizer} optimizer...")
        if self.optimizer == "AdamW":
            lr = self.lr_scale * 1e-3
            opt = optim.AdamW(params, amsgrad=True, betas=(self.momentum, 0.999), lr=lr, weight_decay=self.weight_decay)
            return [opt]
        elif self.optimizer == "Muon":
            # The default Muon LR is 1e-3, but we find we can go 10x higher
            lr = self.lr_scale * 0.01

            # Muon only handles 2D params, AdamW handles the rest
            params_1d = [p for p in params if p.ndim != 2 and p.requires_grad]
            params_2d = [p for p in params if p.ndim == 2 and p.requires_grad]
            # The AdamW part keeps the AdamW-only setup: the same 10x ratio between the two base LRs, and amsgrad
            opts = [
                optim.Muon(params_2d, lr=lr, momentum=self.momentum, weight_decay=self.weight_decay),
                optim.AdamW(params_1d, amsgrad=True, betas=(self.momentum, 0.999), lr=lr / 10, weight_decay=self.weight_decay)
            ]
            return opts
        else:
            raise ValueError(f"Unknown optimizer '{self.optimizer}'.")
