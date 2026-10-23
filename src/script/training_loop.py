import logging
import re
from dataclasses import dataclass, asdict
from typing import Optional, Union, Tuple, List, Dict, Any, get_args
from pathlib import Path
from collections import defaultdict
from itertools import islice

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from wandb.sdk.lib import runid
from transformers import set_seed, PreTrainedModel
from tqdm import trange

from src.lens import *
from src import Dataset, Optimizer, TransformersModel

from src.state import State
from src.script.utils import LENS_TYPE, log_train_step

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trainer:
    """
    Trains a lens to match the model's own next-token distribution, at every layer.

    For each layer, the loss is KL(p_final || p_lens): the divergence between the model's
    output distribution and the one the lens decodes from that layer's hidden state. The model
    stays frozen; only the lens is trained.

    Attributes:
        model: The model to train the lens on.
        data: The training corpus.
        opt: Optimizer settings.
        lens_type: Which lens to train.
        lens_config: Config of the lens (see the lens' Config class). Ignored when resuming from
            `lens_pretrained_path`.
        num_steps: Optimizer steps. Lowered automatically if the dataset is too small.
        tokens_per_step: Tokens per optimizer step. Must be a multiple of the chunk length;
            gradient accumulation is derived from it and from `batch_size`.
        batch_size: Chunks per forward pass.
        seed: Random seed.
        wandb_project: wandb project name.
        wandb_name: wandb run name. If None, derived from the lens checkpoint path.
        lens_pretrained_path: Start from a saved lens instead of a fresh one.
        checkpoint_freq: Write a resumable snapshot every this many steps. None disables snapshots.
        save_steps: Save a lens checkpoint after this many steps (e.g. [25, 50] gives 25_steps and
            50_steps), besides the final one.
    """
    model: TransformersModel
    data: Dataset
    opt: Optimizer
    lens_type: LENS_TYPE
    lens_config: Dict[str, Any]
    num_steps: int
    tokens_per_step: int
    batch_size: int
    seed: int
    wandb_project: str
    wandb_name: Optional[str] = None
    lens_pretrained_path: Optional[str] = None
    checkpoint_freq: Optional[int] = None
    save_steps: Optional[List[int]] = None

    def _init_logging(self, model_name: str, lens: BaseLens, wandb_id: Optional[str]):
        """Start (or resume, if `wandb_id` already exists) the wandb run, grouped by model."""
        if not self.wandb_name:
            # Drop the leading "checkpoints/" from the lens path
            self.wandb_name = "/".join(lens.get_path().split("/")[1:])

        logger.info("Initializing Weights & Biases ...")
        wandb.init(
            config=asdict(self),
            project=self.wandb_project,
            group=model_name,
            name=self.wandb_name,
            id=wandb_id,
            resume="allow",
        )
        wandb.watch(lens)

    def get_lens(self, model: PreTrainedModel):
        """Create a new lens from `lens_config`, or load the one at `lens_pretrained_path`."""
        if self.lens_type not in get_args(LENS_TYPE):
            raise ValueError(f"Unknown lens type: '{self.lens_type}'")

        # The optimizer name is part of the lens checkpoint path
        self.lens_config["opt"] = self.opt.optimizer

        lens_class = {"TunedLens": TunedLens, "LoRALens": LoRALens, "AnchorLens": AnchorLens}[self.lens_type]
        if self.lens_pretrained_path is None:
            lens = lens_class.from_model(model, self.lens_config)
        else:
            lens = lens_class.from_model_and_pretrained(model, self.lens_pretrained_path)

        return lens

    def snapshot(self, state: State):
        """Save a resumable snapshot to `<checkpoint_dir>/snapshot_<step>.pth`."""
        if self.checkpoint_dir is None:
            raise ValueError("Checkpoint directory not set, cannot save snapshot.")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state.save(logger, self.checkpoint_dir / f"snapshot_{state.step}.pth")

    def load_recent_snapshot(self, state: State) -> None:
        """Resume `state` from the snapshot with the highest step, if any. Otherwise leave it untouched."""
        if self.checkpoint_dir is None:
            raise ValueError("Checkpoint directory not set, cannot check for recent snapshot.")

        if not self.checkpoint_dir.exists():
            logger.warning("No checkpoint directory found. Snapshotting is disabled.")
            return None

        # Sort by the step number in the file name (as an int, so that 1000 comes after 900)
        def sort_key_from_path(p: Path):
            if match := re.match(r".*snapshot_(\d+)\.pth", str(p)):
                return int(match.group(1))
            else:
                return -1

        snapshot_location = max(self.checkpoint_dir.glob("snapshot_*.pth"), key=sort_key_from_path, default=None)

        if snapshot_location is None:
            return None

        state.load(logger, snapshot_location)

    def calculate_gradient_accumulation_steps(self, tokens_per_sample: int, total_samples: int) -> int:
        """
        Number of forward/backward passes (of `batch_size` chunks each) per optimizer step.

        Also caps `num_steps` at what the dataset can provide in a single pass. If the chunks
        per step are not a multiple of `batch_size`, rounds up and uses slightly more tokens
        per step than requested.
        """
        # chunk_and_tokenize ensures the samples are all the same length
        samples_per_step, rem = divmod(self.tokens_per_step, tokens_per_sample)
        if rem:
            raise ValueError(
                f"Number of tokens per step ({self.tokens_per_step:,}) must be "
                f"divisible by the number of tokens per sample ({tokens_per_sample:,})."
            )

        if total_samples / samples_per_step < self.num_steps:
            logger.warning(
                f"Can only take {total_samples / samples_per_step:.2f} steps on "
                f"dataset with tokens_per_step={self.tokens_per_step:,}. "
                f"Requested {self.num_steps:,} steps."
            )
            self.num_steps = total_samples // samples_per_step
            logger.warning(f"Setting {self.num_steps:,} steps.")

        grad_acc_steps, rem = divmod(samples_per_step, self.batch_size)
        if rem:
            grad_acc_steps += 1
            adjusted_count = grad_acc_steps * self.batch_size * tokens_per_sample
            logger.warning(f"Increasing grad acc steps from {grad_acc_steps - 1} to {grad_acc_steps}")
            logger.warning(f"Using {adjusted_count:,} tokens per step ({self.tokens_per_step:,} requested).")
        else:
            logger.info(f"Gradient accumulation steps: {grad_acc_steps}.")
            logger.info(f"Using {self.tokens_per_step:,} tokens per training step.")
        return grad_acc_steps

    def setup(self) -> Tuple[State, PreTrainedModel, int]:
        """
        Load model, data and lens, create the optimizers, and resume from a snapshot if one exists.

        Returns:
            The training state, the frozen model, and the gradient accumulation steps.
        """
        logger.info(f"Setting seed {self.seed}...")
        set_seed(self.seed)

        model, tokenizer = self.model.load()

        data, nats_to_bpb = self.data.load(tokenizer)

        lens = self.get_lens(model)

        self.output = Path(lens.get_path())
        logger.info(f"Output dir set to: {self.output}")

        self.checkpoint_dir = self.output / "ckpts"
        logger.info(f"Checkpoints dir set to: {self.checkpoint_dir}")

        logger.info(f"Creating data loader and setting seed to {self.seed}...")
        num_workers = min(self.batch_size, 4)

        # Own generator, so the shuffle order depends only on the seed: a resumed run can then skip
        # exactly the batches the previous run already used (see execute)
        dataloader = DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(self.seed)
        )

        tokens_per_sample = len(data[0]["input_ids"])
        grad_acc_steps = self.calculate_gradient_accumulation_steps(tokens_per_sample, len(data))

        logger.info("Creating optimizers and schedulers...")
        # Skips frozen parameters, e.g. the AnchorLens global transform when training the local adapters
        params = [p for p in lens.parameters() if p.requires_grad]
        opts = self.opt.create_optim(params)
        schedulers = [self.opt.create_scheduler(opt, self.num_steps) for opt in opts]

        state = State(
            lens=lens,
            dataloader=dataloader,
            opts=opts,
            schedulers=schedulers,
            wandb_id=runid.generate_id(),
            seed=self.seed,
            nats_to_bpb=nats_to_bpb,
            num_steps=self.num_steps,
            step=0,
        )

        # Overrides the lens weights, optimizer states, step and wandb id if a snapshot exists
        self.load_recent_snapshot(state)

        self._init_logging(model_name=self.model.name_or_path, lens=state.lens, wandb_id=state.wandb_id)

        logger.info("Setup completed.")

        return state, model, grad_acc_steps

    def _optimizer_step(self, state: State, losses: Dict[str, List[float]], step: int):
        """Clip, log, update the weights and the LR, then write a snapshot or lens checkpoint if one is due."""
        nn.utils.clip_grad_norm_(state.lens.parameters(), 1.0)

        # Before opt.zero_grad, otherwise the gradient norms would all be zero
        log_train_step(state, losses)

        for opt in state.opts:
            opt.step()
            opt.zero_grad(set_to_none=False)
        for scheduler in state.schedulers:
            scheduler.step()

        state.step = step + 1

        if self.checkpoint_freq and step % self.checkpoint_freq == self.checkpoint_freq - 1:
            self.snapshot(state)

        # state.step counts the steps done, so it matches both save_steps and the final save
        if self.save_steps and state.step in self.save_steps:
            state.lens.save(state.step)

    def execute(self):
        """Run the whole training, then save the final lens to `<lens path>/<num_steps>_steps`."""
        state, model, grad_acc_steps = self.setup()

        # Counted in forward passes, so a resumed run starts its progress bar where the previous one stopped
        init_batches = state.step * grad_acc_steps
        total_batches = self.num_steps * grad_acc_steps

        logger.info("Starting training.")

        pbar = trange(
            init_batches,
            total_batches,
            desc="Training...",
            initial=init_batches,
            total=total_batches,
        )
        # When resuming, skip the batches already used. Batches are still loaded, but not run through the model.
        batches = islice(state.dataloader, init_batches, None)

        # One list per layer, filled by every forward pass of the current step and emptied after the optimizer step
        losses = defaultdict(list)
        for batch_idx, batch in zip(pbar, batches):
            with torch.no_grad():
                batch["input_ids"] = batch["input_ids"].to(model.device)
                output = model(**batch, output_hidden_states=True)

            # Target distribution: the model's own log-probs at the last layer
            final_logits = output.logits.to(state.lens.device)
            labels = final_logits.float().log_softmax(dim=-1)

            # Drop the embeddings and the last block's output (see BaseLens for the layer indexing)
            hidden_states = output.hidden_states[1:-1]

            for layer_idx, hidden_state in enumerate(hidden_states):
                logits = state.lens(hidden_state, layer_idx)

                # KL(p_final || p_lens) = sum_v p_final * (log p_final - log p_lens), averaged over positions
                loss = torch.sum(labels.exp() * (labels - logits.log_softmax(-1)), dim=-1).mean()

                logging_loss = loss.detach().item()
                losses[f"{str(layer_idx).zfill(3)}.hidden"].append(logging_loss)

                # Backward right away, layer by layer, so only one layer's graph is in memory at a time
                scaled_loss = loss / grad_acc_steps
                scaled_loss.backward()

            step, rem = divmod(batch_idx, grad_acc_steps)
            if rem == grad_acc_steps - 1:
                self._optimizer_step(state, losses, step)
                losses = defaultdict(list)

        logger.info(f"Training completed. Saving lens to {self.output}...")

        state.lens.save(self.num_steps)
        logger.info("Lens saved.")
