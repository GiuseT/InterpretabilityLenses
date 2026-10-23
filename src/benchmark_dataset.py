import os
import json
import random
import zlib
import torch

from typing import List, Dict, Any
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset

# --- Prompt Templates ---
# Each template turns a benchmark item into the user message of a chat. Note the double braces in
# MATH_TEMPLATE: they produce a literal \boxed{} after str.format.
MC_TEMPLATE = """Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}"""

MATH_TEMPLATE = """Solve the following math problem step by step. Put your answer inside \boxed{{}}.

{question}

Remember to put your answer inside \boxed{{}}."""

HE_TEMPLATE = """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

{question}"""

# The 3-shot prompt from the original MBPP paper (Austin et al., 2021), with its [BEGIN]/[DONE] delimiters
MBPP_TEMPLATE = """You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:

assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)
assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)
assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)
[BEGIN]
def similar_elements(test_tup1, test_tup2):
  res = tuple(set(test_tup1) & set(test_tup2))
  return (res)
[DONE]
You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:

assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
[BEGIN]
import math
def is_not_prime(n):
    result = False
    for i in range(2,int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = True
    return result
[DONE]
You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:

assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]
[BEGIN]
import heapq as hq
def heap_queue_largest(nums,n):
  largest_nums = hq.nlargest(n, nums)
  return largest_nums
[DONE]
You are an expert Python programmer, and here is your task: {question} Your code should pass these tests:

{tests}
[BEGIN]"""

DIRECT_TEMPLATE = """{question}"""

# System message prepended for models whose name contains the key. Other models get no system message.
MODEL2SYSPROMPT = {
    "Qwen2.5": {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    "OLMo": {"role": "system",
             "content": "You are OLMo 2, a helpful and harmless AI Assistant built by the Allen Institute for AI."}
}


class BenchmarkDataset:
    """
    Benchmark items turned into full chat transcripts (prompt + reference answer), for evaluating
    lenses on the answer tokens.

    No text is generated: the reference completion (the dataset's solution or chain of thought)
    is placed in the assistant turn, and the model reads the whole transcript in one forward pass.
    Each batch also comes with a mask that marks which tokens belong to the completion.

    Supported names: "MMLU-Pro", "GPQA-Diamond", "MATH-500", "AIME-2024", "AIME-2025", "HumanEval",
    "MBPP" and "IFEval". For IFEval there is no reference answer, so the completions are
    Llama-3.1-405B generations.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset = self._load_and_prepare()

    def _load_and_prepare(self):
        """Download the benchmark and reduce each item to a "formatted_prompt" and a "formatted_completion"."""
        name = self.dataset_name

        if name == "MMLU-Pro":
            # The validation split is the one that ships with chain-of-thought answers
            ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
            return ds.map(self._format_mmlu_pro, remove_columns=ds.column_names)

        elif name == "GPQA-Diamond":
            ds = load_dataset("Idavidrein/gpqa", name="gpqa_diamond", split="train")
            return ds.map(self._format_gpqa, remove_columns=ds.column_names)

        elif name == "MATH-500":
            ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
            return ds.map(lambda x: {
                "formatted_prompt": MATH_TEMPLATE.format(question=x["problem"]),
                "formatted_completion": x["solution"]
            }, remove_columns=ds.column_names)

        elif name == "AIME-2024":
            ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
            return ds.map(lambda x: {
                "formatted_prompt": MATH_TEMPLATE.format(question=x["Problem"]),
                "formatted_completion": x["Solution"]
            }, remove_columns=ds.column_names)

        elif name == "AIME-2025":
            ds = load_dataset("Sunny8781/AIME2025_w_solution", split="test")
            return ds.map(lambda x: {
                "formatted_prompt": MATH_TEMPLATE.format(question=x["problem"]),
                "formatted_completion": x["solution"]
            }, remove_columns=ds.column_names)

        elif name == "HumanEval":
            ds = load_dataset("openai/openai_humaneval", split="test")
            return ds.map(lambda x: {
                "formatted_prompt": HE_TEMPLATE.format(question=x["prompt"]),
                "formatted_completion": x["canonical_solution"]
            }, remove_columns=ds.column_names)

        elif name == "MBPP":
            ds = load_dataset("google-research-datasets/mbpp", name="sanitized", split="test")
            return ds.map(self._format_mbpp, remove_columns=ds.column_names)

        elif name == "IFEval":
            # IFEval has no gold answers: these are responses generated by Llama-3.1-405B
            ds = load_dataset("harpreetsahota/llama3_1-405B-on-IFEval", split="train")
            return ds.map(lambda x: {
                "formatted_prompt": DIRECT_TEMPLATE.format(question=x["prompt"]),
                "formatted_completion": x["generated_text"]
            }, remove_columns=ds.column_names)

        else:
            raise ValueError(f"Dataset {self.dataset_name} is not supported.")

    @staticmethod
    def _format_mmlu_pro(row: Dict[str, Any]) -> Dict[str, str]:
        """Lettered options (A, B, ...) in the prompt; the CoT followed by "ANSWER: <letter>" as completion."""
        options = row["options"]
        # chr(65 + i) is the i-th uppercase letter
        letters = ", ".join([chr(65 + i) for i in range(len(options))])
        choices_str = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])

        target_letter = chr(65 + row["answer_index"])
        cot = row.get("cot_content", "")
        completion = f"{cot}\nANSWER: {target_letter}".strip()

        return {
            "formatted_prompt": MC_TEMPLATE.format(letters=letters, question=row["question"], choices=choices_str),
            "formatted_completion": completion
        }

    @staticmethod
    def _format_gpqa(row: Dict[str, Any]) -> Dict[str, str]:
        """
        Same format as MMLU-Pro, with the expert explanation as CoT.

        GPQA stores the correct answer in its own column, so without shuffling it would always be
        option A. The options are shuffled with a seed derived from the question.
        """
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"]
        ]
        # crc32 rather than hash(): Python randomizes string hashes per process, so the order would change between runs
        rng = random.Random(zlib.crc32(row["Question"].encode("utf-8")))
        rng.shuffle(choices)

        letters = ", ".join([chr(65 + i) for i in range(len(choices))])
        choices_str = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(choices)])

        correct_idx = choices.index(row["Correct Answer"])
        target_letter = chr(65 + correct_idx)
        cot = row.get("Explanation", "")
        completion = f"{cot}\nANSWER: {target_letter}".strip()

        return {
            "formatted_prompt": MC_TEMPLATE.format(letters=letters, question=row["Question"], choices=choices_str),
            "formatted_completion": completion
        }

    @staticmethod
    def _format_mbpp(row: Dict[str, Any]) -> Dict[str, str]:
        """The task and its asserts go into the 3-shot prompt; the reference code is the completion."""
        tests_str = "\n".join(row["test_list"])
        return {
            "formatted_prompt": MBPP_TEMPLATE.format(question=row["prompt"], tests=tests_str),
            "formatted_completion": row["code"]
        }

    def get_dataloader(self, tokenizer, model_name: str, batch_size: int = 1) -> DataLoader:
        """
        Format every item with the model's chat template, tokenize it, and batch it.

        The completion mask is found by tokenizing the chat twice, once up to the assistant header
        and once in full: tokens past the length of the first are the completion. This assumes
        the first tokenization is a prefix of the second, which holds for usual chat templates.

        Note that the tokenizer is modified in place (left padding, and EOS as pad token if it had none).

        Args:
            tokenizer: Tokenizer of the model under study, with a chat template.
            model_name: Name or path of the model. Used to pick the system prompt and the
                template options (reasoning effort for gpt-oss, thinking disabled otherwise).
            batch_size: Items per batch. Shorter sequences are left-padded to the longest one.

        Returns:
            A DataLoader yielding dicts with three (batch, seq_len) tensors: "input_ids",
            "attention_mask" (1 on real tokens, 0 on padding) and "completion_mask" (True on
            completion tokens, False on padding and prompt tokens).
        """
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        sys_prompt = next((MODEL2SYSPROMPT[k] for k in MODEL2SYSPROMPT if k in model_name), None)

        def map_chat_template(batch):
            input_ids_list = []
            prompt_lengths = []

            for prompt, completion in zip(batch["formatted_prompt"], batch["formatted_completion"]):
                prompt_messages = [sys_prompt, {"role": "user", "content": prompt}] if sys_prompt else [
                    {"role": "user", "content": prompt}]

                full_messages = prompt_messages + [{"role": "assistant", "content": completion}]

                # Options for reasoning models' templates: gpt-oss writes the effort into its system message,
                # and enable_thinking=False (e.g. Qwen3) frames the answer as one without a thinking trace
                kwargs = {"tokenize": False}
                if "gpt-oss" in model_name:
                    kwargs["reasoning_effort"] = "low"
                else:
                    kwargs["enable_thinking"] = False

                try:
                    prompt_text = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, **kwargs)
                    full_text = tokenizer.apply_chat_template(full_messages, add_generation_prompt=False, **kwargs)
                except Exception:
                    # Fallback if specific kwargs are not supported by the tokenizer
                    prompt_text = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True,
                                                                tokenize=False)
                    full_text = tokenizer.apply_chat_template(full_messages, add_generation_prompt=False,
                                                              tokenize=False)

                # The chat template already contains the special tokens (BOS etc.), so don't add them twice
                prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

                input_ids_list.append(full_ids)
                prompt_lengths.append(len(prompt_ids))

            return {"input_ids": input_ids_list, "prompt_length": prompt_lengths}

        tokenized_ds = self.dataset.map(
            map_chat_template,
            batched=True,
            remove_columns=["formatted_prompt", "formatted_completion"],
            desc="Applying Chat Template & Tokenizing"
        )

        def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
            prompt_lengths = [item["prompt_length"] for item in batch]

            max_len = max(len(seq) for seq in input_ids)

            padded_input_ids = []
            attention_masks = []
            completion_masks = []

            for seq, p_len in zip(input_ids, prompt_lengths):
                pad_len = max_len - len(seq)

                if pad_len > 0:
                    pad_tensor = torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)
                    padded_seq = torch.cat([pad_tensor, seq])
                    # Mask: 0 for padding + 0 for prompt + 1 for completion
                    mask = torch.cat([
                        torch.zeros(pad_len + p_len, dtype=torch.bool),
                        torch.ones(len(seq) - p_len, dtype=torch.bool)
                    ])
                else:
                    padded_seq = seq
                    mask = torch.cat([
                        torch.zeros(p_len, dtype=torch.bool),
                        torch.ones(len(seq) - p_len, dtype=torch.bool)
                    ])

                padded_input_ids.append(padded_seq)
                attention_masks.append(torch.cat([
                    torch.zeros(pad_len, dtype=torch.long),
                    torch.ones(len(seq), dtype=torch.long)
                ]))
                completion_masks.append(mask)

            return {
                "input_ids": torch.stack(padded_input_ids),
                "attention_mask": torch.stack(attention_masks),
                "completion_mask": torch.stack(completion_masks)
            }

        return DataLoader(
            tokenized_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(4, batch_size),
            collate_fn=collate_fn
        )
