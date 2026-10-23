import os
import json
import logging
import random
from dataclasses import dataclass
from tqdm import tqdm
from typing import Optional, Union, Tuple, List, Dict, Literal
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import set_seed, PreTrainedModel

from src.lens import *
from src import Dataset, TransformersModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_kl_divergence(lens_logits: torch.Tensor, model_probs: torch.Tensor) -> Tuple[float, int]:
    """
    KL(p_final || p_lens) at every position, summed over the batch (the training objective, see Trainer).

    Args:
        lens_logits: (batch, seq_len, vocab) logits decoded by the lens.
        model_probs: (batch, seq_len, vocab) output probabilities of the model.

    Returns:
        The summed KL in nats and the number of positions it covers, so that results can be
        averaged over the whole dataset rather than per batch.
    """
    lens_log_probs = lens_logits.log_softmax(dim=-1)

    # kl_div(input=log q, target=p) computes p * (log p - log q), i.e. KL(p || q) once summed over the vocabulary
    kl_div = nn.functional.kl_div(
        lens_log_probs,
        model_probs,
        reduction='none',
        log_target=False
    ).sum(dim=-1)  # Sum over vocab, shape: (batch_size, seq_len)

    return kl_div.sum().item(), kl_div.numel()


def compute_ce_and_entropy(lens_logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, int]:
    """
    Next-token cross-entropy of the lens against the actual text, and entropy of the lens distribution.

    Args:
        lens_logits: (batch, seq_len, vocab) logits decoded by the lens.
        targets: (batch, seq_len - 1) the input ids shifted by one, i.e. the token following each position.

    Returns:
        Summed cross-entropy and summed entropy, both in nats, and the number of positions they cover.
    """
    # The last position has no next token to predict
    shift_lens_logits = lens_logits[:, :-1, :].contiguous()
    shift_targets = targets.contiguous()

    ce_loss = nn.functional.cross_entropy(
        shift_lens_logits.view(-1, shift_lens_logits.size(-1)),
        shift_targets.view(-1),
        reduction='none'
    )
    ce_loss = ce_loss.view(shift_targets.shape)  # (batch_size, seq_len - 1)

    shift_lens_probs = shift_lens_logits.softmax(dim=-1)
    shift_lens_log_probs = shift_lens_logits.log_softmax(dim=-1)

    entropy = -torch.sum(shift_lens_probs * shift_lens_log_probs, dim=-1)  # (batch_size, seq_len - 1)

    return ce_loss.sum().item(), entropy.sum().item(), ce_loss.numel()


def save_metrics(metrics: Dict[str, List[float]], model_name_or_path: str, lens_pretrained_path: Optional[str]):
    """
    Write the metrics to outputs/eval/<model>/<file>.json.

    The file is named after the last two parts of the lens path, e.g. "TunedLens-AdamW-100_steps.json",
    or "LogitLens.json" when there is no trained lens.
    """
    model_name = model_name_or_path.split('/')[-1].replace(".", "_")
    output_dir = Path(os.path.join("outputs", "eval", model_name))
    output_dir.mkdir(parents=True, exist_ok=True)

    if lens_pretrained_path is not None:
        filename = f"{'-'.join(lens_pretrained_path.split(os.path.sep)[-2:])}.json"
    else:
        filename = "LogitLens.json"

    output_path = output_dir / filename

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info("Evaluation Results saved to %s", output_path)
    logger.info(json.dumps(metrics, indent=4))


@dataclass
class Evaluator:
    """
    Measures how well a lens decodes each layer on held-out text.

    For every layer it reports, in bits per byte:
      - Kullback-Leibler: KL(p_final || p_lens), how far the lens is from the model's own prediction;
      - Cross-Entropy: how well the lens predicts the actual next token;
      - Entropy: how spread out the lens distribution is.

    Attributes:
        model: The model the lens was trained on.
        data: The text corpus to evaluate on.
        lens_type: Which lens to evaluate.
        batch_size: Chunks per forward pass.
        seed: Seed for picking the evaluation chunks.
        num_eval_tokens: Approximate number of tokens to evaluate on, rounded down to whole chunks.
        lens_pretrained_path: Path of the lens checkpoint relative to checkpoints/<model>/,
            e.g. "TunedLens-AdamW/100_steps". Required for every lens except the LogitLens.
    """
    model: TransformersModel
    data: Dataset
    lens_type: Literal["LogitLens", "TunedLens", "LoRALens", "AnchorLens"]
    batch_size: int
    seed: int
    num_eval_tokens: int
    lens_pretrained_path: Optional[str] = None

    def get_lens(self, model: PreTrainedModel) -> Union[LogitLens, TunedLens, LoRALens, AnchorLens]:
        """Build the LogitLens, or load the trained lens from checkpoints/<model>/<lens_pretrained_path>."""
        if self.lens_type == "LogitLens":
            # So that save_metrics names the output file "LogitLens.json"
            self.lens_pretrained_path = None
            logger.info(f"Using LogitLens.")
            return LogitLens.from_model(model)

        if self.lens_type in ["TunedLens", "LoRALens", "AnchorLens"]:
            if self.lens_pretrained_path is None:
                raise ValueError(f"lens_pretrained_path is required for {self.lens_type}")

            model_str = self.model.name_or_path.split('/')[-1].replace(".", "_")
            self.lens_pretrained_path = os.path.join("checkpoints", model_str, self.lens_pretrained_path)

            lens_class = {"TunedLens": TunedLens, "LoRALens": LoRALens, "AnchorLens": AnchorLens}[self.lens_type]
            return lens_class.from_model_and_pretrained(model, self.lens_pretrained_path)

        raise ValueError(f"Unknown lens type: '{self.lens_type}'")

    def setup(self) -> Tuple[Union[LogitLens, TunedLens, LoRALens, AnchorLens], PreTrainedModel, DataLoader, float]:
        """
        Load model, lens and data, and pick a random subset of chunks of about `num_eval_tokens` tokens.

        Returns:
            The lens, the model, a dataloader over the subset, and the nats-to-bits-per-byte factor.
        """
        logger.info(f"Setting seed {self.seed}...")
        set_seed(self.seed)

        model, tokenizer = self.model.load()
        data, nats_to_bpb = self.data.load(tokenizer)

        lens = self.get_lens(model)

        # All chunks have the same length (see chunk_and_tokenize)
        max_seq_len = len(data[0]["input_ids"])
        n_samples = self.num_eval_tokens // max_seq_len

        if n_samples > len(data):
            logger.warning(f"Requested {n_samples} samples but dataset only has {len(data)}. Using all available data.")
            n_samples = len(data)

        logger.info(f"Randomly selecting {n_samples} samples to reach ~{self.num_eval_tokens} tokens...")

        # Same seed, same subset: every lens of a model is evaluated on exactly the same chunks
        random.seed(self.seed)
        indices = random.sample(range(len(data)), n_samples)
        eval_data = data.select(indices)

        logger.info("Creating data loader for the evaluation subset...")
        num_workers = min(self.batch_size, 4)
        dataloader = DataLoader(eval_data, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        logger.info("Setup completed.")

        return lens, model, dataloader, nats_to_bpb

    def evaluate(self) -> Dict[str, List[float]]:
        """
        Run the evaluation and save the results (see `save_metrics`).

        Returns:
            {"Kullback-Leibler": [...], "Cross-Entropy": [...], "Entropy": [...]}, one value per
            layer in bits per byte, ordered by layer index.
        """
        lens, model, dataloader, nats_to_bpb = self.setup()

        lens.eval()
        model.eval()

        # Sums and token counts per layer, accumulated over all batches and divided only at the end
        total_kl = defaultdict(float)
        total_ce = defaultdict(float)
        total_entropy = defaultdict(float)

        total_tokens_kl = defaultdict(float)
        total_tokens_ce = defaultdict(float)

        logger.info("Starting evaluation.")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # The chunked corpus has no padding, so every position counts
                for key in batch.keys():
                    batch[key] = batch[key].to(model.device)

                output = model(**batch, output_hidden_states=True)

                model_probs = output.logits.float().softmax(dim=-1)

                # Drop the embeddings and the last block's output (see BaseLens for the layer indexing)
                hidden_states = output.hidden_states[1:-1]

                # Next-token targets for the cross-entropy
                input_ids = batch["input_ids"]
                targets = input_ids[:, 1:]  # (batch_size, seq_len - 1)

                for layer_idx, hidden_state in enumerate(hidden_states):
                    lens_logits = lens(hidden_state, layer_idx)

                    kl_sum, kl_tokens = compute_kl_divergence(lens_logits, model_probs)
                    total_kl[layer_idx] += kl_sum
                    total_tokens_kl[layer_idx] += kl_tokens

                    ce_sum, entropy_sum, ce_tokens = compute_ce_and_entropy(lens_logits, targets)
                    total_ce[layer_idx] += ce_sum
                    total_entropy[layer_idx] += entropy_sum
                    total_tokens_ce[layer_idx] += ce_tokens

        final_metrics = self._aggregate_metrics(total_kl, total_ce, total_entropy, total_tokens_kl, total_tokens_ce, nats_to_bpb)

        save_metrics(final_metrics, self.model.name_or_path, self.lens_pretrained_path)

        return final_metrics

    def _aggregate_metrics(self, total_kl, total_ce, total_entropy, total_tokens_kl, total_tokens_ce, nats_to_bpb) -> Dict[str, List[float]]:
        """Turn the per-layer sums into averages per token, converted from nats to bits per byte."""
        final_metrics = {
            "Kullback-Leibler": [],
            "Cross-Entropy": [],
            "Entropy": []
        }

        layer_indices = sorted(total_tokens_kl.keys())
        if not layer_indices:
            logger.warning("No layers were evaluated.")
        else:
            # KL covers every position, CE and entropy one less per sequence, hence separate counts
            ce_layer_indices = sorted(total_tokens_ce.keys())

            for i in layer_indices:
                avg_kl = total_kl[i] / total_tokens_kl[i] if total_tokens_kl[i] > 0 else 0.0
                final_metrics["Kullback-Leibler"].append(avg_kl * nats_to_bpb)

            for i in ce_layer_indices:
                avg_ce = total_ce[i] / total_tokens_ce[i] if total_tokens_ce[i] > 0 else 0.0
                final_metrics["Cross-Entropy"].append(avg_ce * nats_to_bpb)

                avg_entropy = total_entropy[i] / total_tokens_ce[i] if total_tokens_ce[i] > 0 else 0.0
                final_metrics["Entropy"].append(avg_entropy * nats_to_bpb)

        return final_metrics
