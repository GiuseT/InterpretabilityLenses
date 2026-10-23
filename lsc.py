"""
Compare a lens' per-layer predictions with the model's final ones on benchmark answers (see LSCEvaluator).

It reuses the eval configs, so the same file drives both scripts:

    python lsc.py --config reproduce/config/eval/Qwen3-32B/TunedLens-AdamW-100_steps.json

All eight benchmarks supported by BenchmarkDataset are evaluated, one after the other. Results
go to outputs/lsc/<model>/<benchmark>/. Run it from the repository root, where the checkpoints/ folder is.
"""
import argparse
import json
import logging
from pathlib import Path

from src import TransformersModel, LSCEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Compare a lens' per-layer predictions with the model's final ones on benchmark answers.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (JSON).")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    logger.info(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        config = json.load(f)

    try:
        model_config = config.pop("model")
    except KeyError as e:
        raise ValueError("Configuration file must contain a 'model' section.") from e

    # Drop the eval-only keys, if present: the benchmarks replace the text corpus
    config.pop("dataset", None)
    config.pop("num_eval_tokens", None)

    logger.info("Initializing Model...")
    try:
        model = TransformersModel(**model_config)
    except TypeError as e:
        logger.error(f"Error initializing Model: {e}")
        raise ValueError("Invalid configuration parameters for Model.") from e

    logger.info("Initializing LSCEvaluator...")
    try:
        evaluator = LSCEvaluator(
            model=model,
            dataset_names=["MMLU-Pro", "GPQA-Diamond", "MATH-500", "AIME-2024", "AIME-2025", "HumanEval", "MBPP", "IFEval"],
            **config
        )
    except TypeError as e:
        logger.error(f"Error initializing LSCEvaluator: {e}")
        raise ValueError(
            "Invalid configuration parameters for LSCEvaluator. "
            "Ensure your JSON contains 'lens_type', 'batch_size' and 'seed' (plus 'lens_pretrained_path' for trained lenses)."
        ) from e

    logger.info("Starting LSC evaluation...")
    evaluator.evaluate()
    logger.info("Evaluation finished.")

if __name__ == "__main__":
    main()
