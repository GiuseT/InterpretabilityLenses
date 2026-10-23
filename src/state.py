import logging
import os
import torch

from dataclasses import dataclass
from typing import Optional, List, Union
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from src.lens.base_lens import BaseLens


@dataclass
class State:
    """
    Everything needed to resume an interrupted training run.

    Snapshots of this state (`save`/`load`) are separate from the lens checkpoints written by
    `lens.save`: they also hold the optimizer and scheduler states and the wandb run id, so a
    resumed run continues on the same LR schedule and logs to the same wandb run.

    Attributes:
        lens: The lens being trained.
        dataloader: Batches of tokenized chunks.
        opts: The optimizers (more than one with Muon, see `Optimizer.create_optim`).
        schedulers: One LR scheduler per optimizer.
        wandb_id: Id of the wandb run.
        seed: Random seed of the run.
        nats_to_bpb: Factor converting the losses from nats per token to bits per byte.
        num_steps: Total number of optimizer steps.
        step: Optimizer steps taken so far.
    """
    lens: BaseLens
    dataloader: DataLoader
    opts: List[Optimizer]
    schedulers: List[LambdaLR]
    wandb_id: Optional[str]
    seed: int
    nats_to_bpb: float
    num_steps: int
    step: int = 0

    def load(self, logger: logging.Logger, path: Union[Path, str]) -> None:
        """
        Restore the state in place from a snapshot written by `save`.

        Anything missing from the file is logged and left as it is.

        Args:
            logger: Where to log progress.
            path: The snapshot file (.pth).
        """
        logger.info(f"Loading state from {path}")
        if not os.path.exists(path):
            logger.error(f"Checkpoint file not found at {path}")
            raise FileNotFoundError(f"Checkpoint file not found at {path}")

        checkpoint = torch.load(path, map_location=self.lens.device)

        if "lens_state_dict" in checkpoint:
            self.lens.load_state_dict(checkpoint["lens_state_dict"])
            logger.info("Lens state loaded.")
        else:
            logger.warning("Lens state dict not found in checkpoint.")

        # Optimizers and schedulers are matched by position, so they must be created in the same order as when saved
        if "optimizer_state_dicts" in checkpoint:
            for opt, state in zip(self.opts, checkpoint["optimizer_state_dicts"]):
                opt.load_state_dict(state)
            logger.info("Optimizer states loaded.")
        else:
            logger.warning("Optimizer state dicts not found in checkpoint.")

        if "scheduler_state_dicts" in checkpoint:
            for scheduler, state in zip(self.schedulers, checkpoint["scheduler_state_dicts"]):
                scheduler.load_state_dict(state)
            logger.info("Scheduler states loaded.")
        else:
            logger.warning("Scheduler state dicts not found in checkpoint.")

        self.wandb_id = checkpoint.get("wandb_id", self.wandb_id)
        self.seed = checkpoint.get("seed", self.seed)
        self.nats_to_bpb = checkpoint.get("nats_to_bpb", self.nats_to_bpb)
        self.num_steps = checkpoint.get("num_steps", self.num_steps)
        self.step = checkpoint.get("step", self.step)

        logger.info(f"State loaded successfully. Resuming from step {self.step}.")

    def save(self, logger: logging.Logger, path: Union[Path, str]) -> None:
        """
        Write a snapshot to `path`, creating its folder if needed. The dataloader is not saved.

        Args:
            logger: Where to log progress.
            path: The snapshot file (.pth).
        """
        logger.info(f"Saving state to {path}")

        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        checkpoint = {
            "lens_state_dict": self.lens.state_dict(),
            "optimizer_state_dicts": [opt.state_dict() for opt in self.opts],
            "scheduler_state_dicts": [scheduler.state_dict() for scheduler in self.schedulers],
            "wandb_id": self.wandb_id,
            "seed": self.seed,
            "nats_to_bpb": self.nats_to_bpb,
            "num_steps": self.num_steps,
            "step": self.step,
        }

        torch.save(checkpoint, path)
        logger.info("State saved successfully.")
