import os
import json
import logging
from dataclasses import dataclass
from tqdm import tqdm
from typing import Optional, Union, Tuple, List, Dict, Literal
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from transformers import set_seed, PreTrainedModel, PreTrainedTokenizer

from src.lens import *
from src import BenchmarkDataset, TransformersModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_surprisal(lens_logits: torch.Tensor, final_probs: torch.Tensor, eval_mask: torch.Tensor) -> Tuple[float, int]:
    """
    How surprised the model's final layer is by the token the lens predicts at an intermediate layer.

    For the lens' top token t_i = argmax(lens_logits), the surprisal is -log P_final(t_i), in nats.
    It is low when the intermediate layer already points to a token the model would also produce.

    Args:
        lens_logits: (batch, seq_len, vocab) logits decoded by the lens.
        final_probs: (batch, seq_len, vocab) output probabilities of the model.
        eval_mask: (batch, seq_len) boolean mask of the positions to count.

    Returns:
        The summed surprisal over the masked positions, and the number of those positions.
    """
    t_i = lens_logits.argmax(dim=-1)
    prob_of_t_i_at_final = final_probs.gather(-1, t_i.unsqueeze(-1)).squeeze(-1)
    # Clamped so that a token with zero probability gives a large but finite value instead of inf
    surprisal = -torch.log(prob_of_t_i_at_final.clamp(min=1e-10))

    num_tokens = eval_mask.sum().item()
    surprisal_sum = surprisal[eval_mask].sum().item() if num_tokens > 0 else 0.0

    return surprisal_sum, num_tokens


def compute_precision_recall(lens_logits: torch.Tensor, final_logits: torch.Tensor, eval_mask: torch.Tensor) -> Tuple[Dict[str, float], int]:
    """
    Agreement between the lens' top predictions and the model's, at every masked position.

    Precision@k: the lens' top-1 token is among the model's top-k.
    Recall@k: the model's top-1 token is among the lens' top-k.
    Precision@1 and Recall@1 both reduce to "same top-1 token", so they coincide.

    Returns:
        The number of hits for each metric over the masked positions, and the number of those positions.
    """
    t_i = lens_logits.argmax(dim=-1)
    t_final = final_logits.argmax(dim=-1)

    top_1_final = final_logits.topk(1, dim=-1).indices
    top_5_final = final_logits.topk(5, dim=-1).indices
    top_1_lens = lens_logits.topk(1, dim=-1).indices
    top_5_lens = lens_logits.topk(5, dim=-1).indices

    precision_1 = (t_i.unsqueeze(-1) == top_1_final).any(dim=-1)
    precision_5 = (t_i.unsqueeze(-1) == top_5_final).any(dim=-1)
    recall_1 = (t_final.unsqueeze(-1) == top_1_lens).any(dim=-1)
    recall_5 = (t_final.unsqueeze(-1) == top_5_lens).any(dim=-1)

    num_tokens = eval_mask.sum().item()

    sums = {
        "Precision@1": precision_1[eval_mask].sum().item() if num_tokens > 0 else 0.0,
        "Precision@5": precision_5[eval_mask].sum().item() if num_tokens > 0 else 0.0,
        "Recall@1": recall_1[eval_mask].sum().item() if num_tokens > 0 else 0.0,
        "Recall@5": recall_5[eval_mask].sum().item() if num_tokens > 0 else 0.0,
    }

    return sums, num_tokens


def save_metrics(metrics: Dict[str, Dict[str, float]], model_name_or_path: str, dataset_name: str, lens_pretrained_path: Optional[str]):
    """
    Write the metrics of one benchmark to outputs/lsc/<model>/<dataset>/<file>.json.

    The file is named after the last two parts of the lens path, e.g. "TunedLens-AdamW-100_steps.json",
    or "LogitLens.json" when there is no trained lens.
    """
    model_name = model_name_or_path.split('/')[-1].replace(".", "_")
    output_dir = Path(os.path.join("outputs", "lsc", model_name, dataset_name))
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
class LSCEvaluator:
    """
    Measures, layer by layer, how close the lens' predictions are to the model's final ones on benchmark answers.

    Each benchmark item is fed as a full chat (prompt + reference answer, see BenchmarkDataset), and
    only the positions of the answer are scored. Per layer, it reports the surprisal of the lens'
    top token under the final distribution, and Precision@k / Recall@k between the two top-k sets.
    This indicates from which layer on the model has already "decided" its output.

    Attributes:
        model: The model the lens was trained on.
        dataset_names: Benchmarks to evaluate on (see BenchmarkDataset for the supported names).
        lens_type: Which lens to evaluate.
        batch_size: Items per forward pass.
        seed: Random seed.
        lens_pretrained_path: Path of the lens checkpoint relative to checkpoints/<model>/,
            e.g. "TunedLens-AdamW/100_steps". Required for every lens except the LogitLens.
    """
    model: TransformersModel
    dataset_names: List[str]
    lens_type: Literal["LogitLens", "TunedLens", "AnchorLens", "LoRALens"]
    batch_size: int
    seed: int
    lens_pretrained_path: Optional[str] = None

    def get_lens(self, model: PreTrainedModel) -> Union[LogitLens, TunedLens, AnchorLens, LoRALens]:
        """Build the LogitLens, or load the trained lens from checkpoints/<model>/<lens_pretrained_path>."""
        if self.lens_type == "LogitLens":
            # So that save_metrics names the output file "LogitLens.json"
            self.lens_pretrained_path = None
            logger.info(f"Using LogitLens.")
            return LogitLens.from_model(model)

        if self.lens_type in ["TunedLens", "AnchorLens", "LoRALens"]:
            if self.lens_pretrained_path is None:
                raise ValueError(f"lens_pretrained_path is required for {self.lens_type}")

            model_str = self.model.name_or_path.split('/')[-1].replace(".", "_")
            self.lens_pretrained_path = os.path.join("checkpoints", model_str, self.lens_pretrained_path)

            lens_class = {"TunedLens": TunedLens, "AnchorLens": AnchorLens, "LoRALens": LoRALens}[self.lens_type]
            return lens_class.from_model_and_pretrained(model, self.lens_pretrained_path)

        raise ValueError(f"Unknown lens type: '{self.lens_type}'")

    def setup(self) -> Tuple[Union[LogitLens, TunedLens, AnchorLens, LoRALens], PreTrainedModel, PreTrainedTokenizer]:
        """Load the model, its tokenizer and the lens. Benchmarks are loaded one at a time in `evaluate`."""
        logger.info(f"Setting seed {self.seed}...")
        set_seed(self.seed)

        model, tokenizer = self.model.load()
        lens = self.get_lens(model)

        logger.info("Setup completed.")

        return lens, model, tokenizer

    def evaluate(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Evaluate on every benchmark in `dataset_names`, saving each one's results as soon as it is done.

        Returns:
            {dataset: {metric: {layer index (as a string): value}}}.
        """
        lens, model, tokenizer = self.setup()

        lens.eval()
        model.eval()

        all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

        logger.info("Starting evaluation.")

        for dataset_name in self.dataset_names:
            logger.info(f"=== Starting Evaluation for Dataset: {dataset_name} ===")

            benchmark_ds = BenchmarkDataset(dataset_name)
            dataloader = benchmark_ds.get_dataloader(
                tokenizer,
                batch_size=self.batch_size,
                model_name=self.model.name_or_path
            )

            # Sums and token counts per layer, divided only at the end
            total_surprisal = defaultdict(float)
            total_precision_recall = defaultdict(lambda: defaultdict(float))
            total_active_tokens = defaultdict(int)

            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
                    input_ids = batch["input_ids"].to(model.device)
                    attention_mask = batch["attention_mask"].to(model.device)
                    completion_mask = batch["completion_mask"].to(model.device)

                    # Sequences are left-padded, so positions must start at 0 on the first real token,
                    # not on the first padding token (a plain forward call would count from the padding)
                    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

                    # Position t predicts token t + 1, so the prediction of the first answer token is
                    # made on the last prompt token. Shifting the mask left by one adds that position.
                    shifted_mask = torch.cat([
                        completion_mask[:, 1:],
                        torch.zeros_like(completion_mask[:, :1])
                    ], dim=1)
                    eval_mask = completion_mask | shifted_mask

                    output = model(
                        input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        output_hidden_states=True,
                        use_cache=False
                    )

                    final_logits = output.logits
                    final_probs = final_logits.float().softmax(dim=-1)

                    # Drop the embeddings and the last block's output (see BaseLens for the layer indexing)
                    hidden_states = output.hidden_states[1:-1]

                    for layer_idx, hidden_state in enumerate(hidden_states):
                        lens_logits = lens(hidden_state, layer_idx)

                        surprisal_sum, num_tokens = compute_surprisal(lens_logits, final_probs, eval_mask)
                        total_surprisal[layer_idx] += surprisal_sum

                        pr_sums, _ = compute_precision_recall(lens_logits, final_logits, eval_mask)
                        for metric_name, val_sum in pr_sums.items():
                            total_precision_recall[layer_idx][metric_name] += val_sum

                        total_active_tokens[layer_idx] += num_tokens

            dataset_metrics = self._aggregate_metrics(total_surprisal, total_precision_recall, total_active_tokens)

            save_metrics(dataset_metrics, self.model.name_or_path, dataset_name, self.lens_pretrained_path)

            all_results[dataset_name] = dataset_metrics

            # Explicitly delete dataloader and dataset wrapper to free RAM for the next dataset
            del benchmark_ds
            del dataloader

        return all_results

    def _aggregate_metrics(self, total_surprisal, total_precision_recall, total_active_tokens) -> Dict[str, Dict[str, float]]:
        """
        Turn the per-layer sums into averages over the scored positions.

        Surprisal stays in nats; Precision and Recall become fractions in [0, 1]. Layer indices
        are stored as strings, since that is what they become in JSON anyway.
        """
        final_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)

        layer_indices = sorted(total_active_tokens.keys())
        if not layer_indices:
            logger.warning("No layers were evaluated.")
        else:
            for layer_idx in layer_indices:
                count = total_active_tokens[layer_idx]

                avg_surprisal = total_surprisal[layer_idx] / count if count > 0 else 0.0
                final_metrics["Surprisal"][str(layer_idx)] = avg_surprisal

                for metric_name, val_sum in total_precision_recall[layer_idx].items():
                    avg_val = val_sum / count if count > 0 else 0.0
                    final_metrics[metric_name][str(layer_idx)] = avg_val

        return dict(final_metrics)
