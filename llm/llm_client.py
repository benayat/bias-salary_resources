import logging
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from tqdm.auto import tqdm
from vllm import SamplingParams, LLM, TokensPrompt
from transformers import AutoTokenizer


@dataclass
class LLMResourceConfig:
    gpu_memory_utilization: float
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    block_size: int
    tensor_parallel_size: int
    dtype: str
    trust_remote_code: bool
    disable_log_stats: bool
    max_parallel_loading_workers: Optional[int] = None
    enable_prefix_caching: bool = True
    enforce_eager: bool = False
    use_transformers: bool = False
    enable_chunked_prefill: bool = False
    # attention_backend: Optional[str] = "flashinfer"

    def scale_for_model_size(self, model_size_b: float):
        """Scale config parameters for a given model size (in billions) to fit within VRAM, based on a 3B baseline."""
        if model_size_b <= 0:
            raise ValueError("Model size must be positive.")
        scale_factor = 1 / model_size_b
        print(f"scale_factor: {scale_factor} for model size {model_size_b}B")
        self.gpu_memory_utilization = 0.9
        # self.gpu_memory_utilization = min(0.9, max(0.9 * scale_factor, 0.7))
        self.max_num_seqs = int(128 * scale_factor)
        self.max_num_batched_tokens = int(65536 * scale_factor)

    def to_vllm_config(self) -> Dict[str, Any]:
        """Convert to a configuration dictionary for vLLM."""
        return {
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "block_size": self.block_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "disable_log_stats": self.disable_log_stats,
            "max_parallel_loading_workers": self.max_parallel_loading_workers,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enforce_eager": self.enforce_eager,
            "model_impl": "transformers" if self.use_transformers else "vllm",
            # "attention_backend": self.attention_backend,
            "enable_chunked_prefill": self.enable_chunked_prefill,
        }


@dataclass
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2
    n: int = 1
    seed: int = 12345


class LLMClient:
    """
    Thin wrapper around vLLM that:
      * Pre-tokenizes chat-style prompts using HF tokenizer + chat_template.
      * Uses llm.generate(prompts=<token_ids or text>) as the only generation path.
    """

    def __init__(
            self,
            model_name: str,
            config: LLMResourceConfig,
    ):
        self.model_name = model_name
        self.config = config
        self.disable_thinking = ("Qwen3" in model_name or "deepseek" in model_name) and not "Instruct" in model_name
        self.is_deepseek = "deepseek" in model_name.lower()

        # vLLM engine
        self.llm = LLM(self.model_name, **config.to_vllm_config())

        # HF tokenizer for chat templates + tokenization
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=self.config.trust_remote_code,
        )

        logging.info(
            f"LLMClient initialized for model={self.model_name} ")

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _post_process_output(self, text: str) -> str:
        """
        Post-process output text. For deepseek models, extract only content after </think> tag.

        Args:
            text: Raw output text from the model

        Returns:
            Processed text (content after </think> for deepseek, unchanged for others)
        """
        if not self.is_deepseek:
            return text

        # For deepseek models, extract content after </think> tag
        nthink_end = "</think>\n\n"
        if nthink_end in text:
            # Find the position after </think> and return everything after it
            idx = text.find(nthink_end) + len(nthink_end)
            return text[idx:].strip()

        # If no </think> tag found, return as-is
        return text

    def _messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert a chat `messages` list into a single text using the chat template,
        without tokenizing.

        This matches the "batch_text" pipeline:
          - tokenize=False
          - add_generation_prompt=True
          - chat_template_kwargs={"enable_thinking": False}
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False if self.disable_thinking else True
        )

    def _batch_tokenize_messages(self, prompts: List[Dict[str, Any]]) -> list[TokensPrompt]:
        """
        Pre-tokenize a batch of prompts with HF fast tokenizer.

        Each prompt is expected to be:
          { "messages": [...], "metadata": {...} }

        Steps:
          1) messages -> text via chat template (tokenize=False)
          2) batch tokenization with add_special_tokens=False
        """
        # 1) Build chat-formatted texts
        messages_list = [p["messages"] for p in prompts]
        texts = [
            self._messages_to_text(msgs)
            for msgs in tqdm(messages_list, desc="Building chat texts")
        ]

        # 2) Batched tokenization (HF fast tokenizer does internal parallelism)
        enc = self.tokenizer(
            texts,
            padding=False,
            truncation=False,
            add_special_tokens=False,  # chat_template already added special tokens
            return_attention_mask=False,
        )
        return [
            TokensPrompt(prompt_token_ids=ids)
            for ids in enc["input_ids"]
        ]

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def run_batch(
            self,
            prompts: List[Dict[str, Any]],
            sampling_params: SamplingConfig,
            output_field: str = "output",
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of prompts and return results with metadata.

        Each item in `prompts` should look like:
          {
            "messages": [...],      # HF-style chat messages
            "metadata": {...},      # optional
          }

          * Pre-tokenize via chat_template(tokenize=False) + tokenizer(batch).
          * Call llm.generate(prompts=<List[List[int]]>, ...).
        """
        params = SamplingParams(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            max_tokens=sampling_params.max_tokens,
            n=sampling_params.n,
            seed=sampling_params.seed,
        )
        try:
            # Default / fast path: pre-tokenize to token IDs and feed directly
            tokenized_prompts = self._batch_tokenize_messages(prompts)
            outputs = self.llm.generate(
                prompts=tokenized_prompts,
                sampling_params=params,
                use_tqdm=True,
            )

            return [
                {
                    **prompts[i].get("metadata", {}),
                    output_field: self._post_process_output(outputs[i].outputs[0].text.strip()),
                }
                for i in range(len(outputs))
            ]
        except Exception as e:
            logging.exception("Error in run_batch")
            return [
                {
                    **prompts[i].get("metadata", {}),
                    output_field: f"[ERROR] {str(e)}",
                }
                for i in range(len(prompts))
            ]

    def delete_client(self):
        """Delete the LLM client and clear CUDA cache."""
        logging.info("Deleting LLM client and clearing CUDA cache")
        del self.llm
        torch.cuda.empty_cache()
        logging.info("LLM client deleted and CUDA cache cleared")

    def reset_client_to_another_model(self, model_name: str):
        """Reset the LLM client to use a different model, using the same config."""
        logging.info(f"Resetting llm model to {model_name}")
        self.model_name = model_name
        self.disable_thinking = ("Qwen3" in model_name or "deepseek" in model_name) and not "Instruct" in model_name
        self.is_deepseek = "deepseek" in model_name.lower()
        self.delete_client()

        # Recreate vLLM engine
        self.llm = LLM(self.model_name, **self.config.to_vllm_config())

        # Recreate tokenizer for the new model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            trust_remote_code=self.config.trust_remote_code,
        )

        logging.info(f"LLM client reset to model: {self.model_name}")
