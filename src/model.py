import logging

from typing import Optional, Literal, Tuple, Dict
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Precision = Literal["bfloat16", "float32", "float16", "auto"]
AttentionImplementation = Literal["eager", "sdpa", "flash_attention_2", "flash_attention_3", "flash_attention_4"]


@dataclass
class TransformersModel:
    """
    Settings for loading the model under study, together with its tokenizer, from the Hugging Face Hub.

    The model is loaded frozen and in eval mode: only the lenses are trained, never the model.

    Attributes:
        name_or_path: Hub id or local path of the model.
        precision: Dtype of the model weights.
        attn_implementation: Attention backend passed to `from_pretrained`.
        tensor_parallelism: Shard the model across GPUs with transformers' automatic TP plan.
        device_map: Passed to `from_pretrained` as is (e.g. "auto"). If None, the model is loaded on CPU.
        max_memory: Memory budget per GPU index, e.g. {0: "20GiB", 1: "40GiB"}. Only useful with `device_map`.
    """
    name_or_path: str
    precision: Precision = "bfloat16"
    attn_implementation: AttentionImplementation = "flash_attention_2"
    tensor_parallelism: bool = False
    device_map: Optional[str] = None
    max_memory: Optional[Dict[int, str]] = None

    def load_tokenizer(self) -> PreTrainedTokenizerBase:
        """Load the tokenizer, falling back to the EOS token for padding if the tokenizer has no pad token."""
        logger.info(f"Loading tokenizer for '{self.name_or_path}'...")
        tokenizer = AutoTokenizer.from_pretrained(self.name_or_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            logger.info("Tokenizer does not have a pad_token_id. Default it to eos_token_id.")
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load the model (frozen, in eval mode) and its tokenizer."""
        logger.info(f"Loading pretrained weights for '{self.name_or_path}'...")

        kwargs = dict(
            pretrained_model_name_or_path=self.name_or_path,
            low_cpu_mem_usage=True,
            dtype=self.precision,
            attn_implementation=self.attn_implementation,
            trust_remote_code=True
        )

        if self.device_map is not None:
            kwargs["device_map"] = self.device_map

        if self.max_memory is not None:
            # Keys arrive as strings when the config comes from JSON/CLI, but transformers wants GPU indices as ints
            kwargs["max_memory"] = {int(k): v for k, v in self.max_memory.items()}

        if self.tensor_parallelism:
            kwargs["tp_plan"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(**kwargs).eval()
        model.requires_grad_(False)

        return model, self.load_tokenizer()
