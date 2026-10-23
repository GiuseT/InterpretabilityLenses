"""
Evaluate a lens on held-out text: per-layer KL, cross-entropy and entropy (see Evaluator).

    python eval.py --config reproduce/config/eval/Qwen3-32B/TunedLens-AdamW-100_steps.json

It needs a "model" and a "dataset" section; every other key is passed to Evaluator. Results go to
outputs/eval/<model>/. Run it from the repository root, where the checkpoints/ folder is.
"""
import argparse
import json
import logging
from pathlib import Path

from src import TransformersModel, Dataset, Evaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a lens model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (JSON).")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    logger.info(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        config = json.load(f)

    # The two sections become their own objects; what is left in `config` goes to Evaluator as is
    try:
        model_config = config.pop("model")
        dataset_config = config.pop("dataset")
    except KeyError as e:
        raise ValueError(f"Configuration file must contain 'model' and 'dataset' sections.") from e

    # A TypeError here means an unknown or missing key in one of the sections
    logger.info("Initializing components...")
    try:
        model = TransformersModel(**model_config)
        dataset = Dataset(**dataset_config)
    except TypeError as e:
        logger.error(f"Error initializing components: {e}")
        raise ValueError("Invalid configuration parameters for components.") from e

    logger.info("Initializing Evaluator...")
    try:
        evaluator = Evaluator(
            model=model,
            data=dataset,
            **config
        )
    except TypeError as e:
        logger.error(f"Error initializing Evaluator: {e}")
        raise ValueError("Invalid configuration parameters for Evaluator.") from e

    logger.info("Starting evaluation...")
    evaluator.evaluate()
    logger.info("Evaluation finished.")

if __name__ == "__main__":
    main()
