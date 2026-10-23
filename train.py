"""
Train a lens on a model.

    python train.py --config reproduce/config/train/Qwen3-32B/TunedLens-AdamW.json

The short form relative to reproduce/config/train/ (--config Qwen3-32B/TunedLens-AdamW.json) also works.
The config needs a "model", a "dataset" and an "optimizer" section (see TransformersModel, Dataset and
Optimizer); every other key is passed to Trainer. Run it from the repository root: the dataset,
the checkpoints and the snapshots are all read and written relative to it.
"""
import argparse
import json
import logging
import os

from pathlib import Path

from src import TransformersModel, Optimizer, Dataset, Trainer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train a lens model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (JSON).")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        # Older commands give the path relative to reproduce/config/train/
        config_path = Path(os.path.join("reproduce", "config", "train", args.config))
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    logger.info(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        config = json.load(f)

    # The three sections become their own objects; what is left in `config` goes to Trainer as is
    try:
        model_config = config.pop("model")
        dataset_config = config.pop("dataset")
        optimizer_config = config.pop("optimizer")
    except KeyError as e:
        raise ValueError(f"Configuration file must contain 'model', 'dataset', and 'optimizer' sections.") from e

    # A TypeError here means an unknown or missing key in one of the sections
    logger.info("Initializing components...")
    try:
        model = TransformersModel(**model_config)
        dataset = Dataset(**dataset_config)
        optimizer = Optimizer(**optimizer_config)
    except TypeError as e:
        logger.error(f"Error initializing components: {e}")
        raise ValueError("Invalid configuration parameters for components.") from e

    logger.info("Initializing Trainer...")
    try:
        trainer = Trainer(
            model=model,
            data=dataset,
            opt=optimizer,
            **config
        )
    except TypeError as e:
        logger.error(f"Error initializing Trainer: {e}")
        raise ValueError("Invalid configuration parameters for Trainer.") from e

    logger.info("Starting training...")
    trainer.execute()
    logger.info("Training finished.")

if __name__ == "__main__":
    main()
