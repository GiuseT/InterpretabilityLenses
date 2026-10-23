import logging
import os
import math

from typing import Dict, List, Tuple
from dataclasses import dataclass
from multiprocessing import cpu_count
from datasets import Dataset as HfDataset
from transformers import PreTrainedTokenizerBase


logger = logging.getLogger(__name__)


def chunk_and_tokenize(
    data: HfDataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    num_proc: int = min(cpu_count() // 2, 8),
    text_key: str = "text",
    max_seq_len: int = 2048,
    return_final_batch: bool = False,
    load_from_cache_file: bool = True,
) -> tuple[HfDataset, float]:
    """
    Tokenize a text dataset into fixed-length chunks, GPT pre-training style.

    Documents are joined with the EOS token as separator (the stream also starts with one) and the
    token stream is cut into chunks of exactly `max_seq_len` tokens, so there is no padding: long
    documents span several chunks, and short ones share a chunk with their neighbors.

    Args:
        data: Dataset with a text column.
        tokenizer: Tokenizer of the model under study.
        num_proc: Worker processes for `datasets.map`.
        text_key: Name of the text column.
        max_seq_len: Chunk length in tokens. Capped by `tokenizer.model_max_length`.
        return_final_batch: Keep the last, shorter chunk of each group of 4096 documents
            instead of dropping it.
        load_from_cache_file: Reuse the `datasets` cache if this exact tokenization was done before.

    Returns:
        The tokenized dataset (a single "input_ids" column, in torch format), and the factor
        that converts a loss in nats per token into bits per byte (see `Dataset.load`).
    """

    def _tokenize_fn(x: Dict[str, List]):
        chunk_size = min(tokenizer.model_max_length, max_seq_len)
        # The empty string at the front makes the joined text start with an EOS token
        joined_text = tokenizer.eos_token.join([""] + x[text_key])
        output = tokenizer(
            joined_text,
            max_length=chunk_size,
            return_attention_mask=False,
            return_overflowing_tokens=True,
            truncation=True,
        )

        # Fast tokenizers already return one row per chunk. Slow ones return the first chunk
        # plus everything past it as a flat list, which we have to cut up ourselves.
        if overflow := output.pop("overflowing_tokens", None):
            # Slow Tokenizers return unnested lists of ints
            assert isinstance(output["input_ids"][0], int)

            chunks = [output["input_ids"]] + [
                overflow[i * chunk_size : (i + 1) * chunk_size]
                for i in range(math.ceil(len(overflow) / chunk_size))
            ]
            output = {"input_ids": chunks}

        # Counted before dropping the last chunk: the ratio below only needs tokens and bytes of the same text
        total_tokens = sum(len(ids) for ids in output["input_ids"])
        total_bytes = len(joined_text.encode("utf-8"))

        if not return_final_batch:
            # The last chunk is almost always shorter than chunk_size, and we don't want padding
            output = {k: v[:-1] for k, v in output.items()}

        output_batch_size = len(output["input_ids"])

        if output_batch_size == 0:
            raise ValueError(
                "Not enough data to create a single batch complete batch."
                " Either allow the final batch to be returned, or supply more data."
            )

        # map() needs one value per output row, so the totals of this group are spread across
        # its rows. Only their sums are used afterwards.
        div, rem = divmod(total_tokens, output_batch_size)
        output["length"] = [div] * output_batch_size
        output["length"][-1] += rem

        div, rem = divmod(total_bytes, output_batch_size)
        output["bytes"] = [div] * output_batch_size
        output["bytes"][-1] += rem

        return output

    data = data.map(
        _tokenize_fn,
        # Each batch drops its last chunk, so larger batches waste fewer tokens
        batched=True,
        batch_size=4096,
        num_proc=num_proc,
        remove_columns=data.column_names,
        load_from_cache_file=load_from_cache_file,
    )
    # nats/token * tokens/byte = nats/byte, and dividing by ln(2) turns nats into bits
    nats_to_bpb = (sum(data["length"]) / sum(data["bytes"])) / math.log(2)

    data = data.with_format("torch", columns=["input_ids"]).select_columns(["input_ids"])

    return data, nats_to_bpb


@dataclass
class Dataset:
    """
    The text corpus used to train and evaluate the lenses.

    Attributes:
        name: File name inside reproduce/dataset/, in JSON Lines format (one {"text": ...} object per line).
        text_column: Name of the field holding the text.
        max_seq_len: Length of each tokenized chunk.
        dataset_shuffle: Shuffle the documents before chunking.
        dataset_shuffle_seed: Seed for that shuffle.
    """

    name: str
    text_column: str = "text"
    max_seq_len: int = 2048
    dataset_shuffle: bool = True
    dataset_shuffle_seed: int = 42

    def load(self, tokenizer: PreTrainedTokenizerBase) -> Tuple[HfDataset, float]:
        """
        Read, shuffle and tokenize the corpus with `chunk_and_tokenize`.

        Returns:
            The tokenized chunks and the nats-to-bits-per-byte factor. Multiplying a loss in nats
            per token by this factor gives bits per byte, which, unlike a per-token loss, can be
            compared across models with different tokenizers (Gao et al., 2020, "The Pile", sec. 3.1).
        """
        logger.info(f"Loading dataset '{self.name}'...")

        dataset = HfDataset.from_json(os.path.join("reproduce", "dataset", self.name))

        logger.info(f"Dataset has {len(dataset):,} samples and columns: {dataset.column_names}.")

        if self.dataset_shuffle:
            logger.info(f"Shuffling dataset with seed {self.dataset_shuffle_seed}...")
            dataset = dataset.shuffle(self.dataset_shuffle_seed)

        dataset, nats_to_bpb = chunk_and_tokenize(
            dataset,
            tokenizer,
            text_key=self.text_column,
            max_seq_len=self.max_seq_len
        )

        return dataset, nats_to_bpb
