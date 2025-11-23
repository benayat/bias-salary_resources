import logging
import multiprocessing
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union

import torch
from vllm import SamplingParams, LLM


def pre_tokenize_prompts(prompts, tokenizer, pool_size):
    """Pre-tokenize prompts using multiprocessing."""
    with multiprocessing.Pool(processes=pool_size) as pool:
        tokenized = pool.map(tokenizer, prompts)
    return tokenized


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
    # attention_backend: Optional[str] = "flashinfer"

    def scale_for_model_size(self, model_size_b: float):
        """Scale config parameters for a given model size (in billions) to fit within VRAM, based on a 3B baseline."""
        if model_size_b <= 0:
            raise ValueError("Model size must be positive.")
        scale_factor = 3 / model_size_b
        print(f"scale_factor: {scale_factor} for model size {model_size_b}B")
        self.gpu_memory_utilization = 0.9
        # self.gpu_memory_utilization = min(0.9, max(0.9 * scale_factor,0.7))
        self.max_num_seqs = int(128 * scale_factor)
        self.max_num_batched_tokens = int(65536 * scale_factor)

    def to_vllm_config(self) -> Dict[str, Any]:
        """Convert to a configuration dictionary for vLLM."""
        return {
            'gpu_memory_utilization': self.gpu_memory_utilization,
            'max_model_len': self.max_model_len,
            'max_num_seqs': self.max_num_seqs,
            'max_num_batched_tokens': self.max_num_batched_tokens,
            'block_size': self.block_size,
            'tensor_parallel_size': self.tensor_parallel_size,
            'dtype': self.dtype,
            'trust_remote_code': self.trust_remote_code,
            'disable_log_stats': self.disable_log_stats,
            'max_parallel_loading_workers': self.max_parallel_loading_workers,
            'enable_prefix_caching': self.enable_prefix_caching,
            'enforce_eager': self.enforce_eager,
            'model_impl': 'transformers' if self.use_transformers else 'vllm',
            # 'attention_backend': self.attention_backend
        }


@dataclass
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2
    # batch_size: int = 100


class LLMClient:
    def __init__(self, model_name: str, config: LLMResourceConfig, disable_pre_tokenization=False, pre_tokenization_pool_size=None):
        self.model_name = model_name
        self.llm = LLM(self.model_name, **config.to_vllm_config())
        self.config = config
        self.disable_pre_tokenization = disable_pre_tokenization
        self.pre_tokenization_pool_size = pre_tokenization_pool_size or max(1, int(0.75 * os.cpu_count()))
        logging.info(f"Pre-tokenization {'disabled' if self.disable_pre_tokenization else 'enabled'}.")
        if not self.disable_pre_tokenization:
            logging.info(f"Pre-tokenization pool size: {self.pre_tokenization_pool_size}")
        # Initialize tokenizer
        # Adjust tokenizer initialization to use vLLM-compatible tokenization
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name).encode

    def run_batch_simple(self, prompts: List[str], sampling_params: SamplingConfig) -> Union[
        list[dict[str, str]], list[tuple[str, Any]]]:
        """Generate outputs for a batch of prompts using llm.generate and return a list of {prompt: output}."""
        params = SamplingParams(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            max_tokens=sampling_params.max_tokens,
        )
        try:
            outputs = self.llm.generate(prompts, sampling_params=params, use_tqdm=True)
            return [
                (prompts[i], output.outputs[0].text.strip())
                for i, output in enumerate(outputs)
            ]
        except Exception as e:
            return [
                {prompts[i]: f"[ERROR] {str(e)}"}
                for i in range(len(prompts))
            ]

    def run_batch(self, prompts: List[Dict], sampling_params: SamplingConfig, output_field: str = "output") -> List[
        Dict]:
        """Process a batch of prompts and return results with metadata"""
        params = SamplingParams(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            max_tokens=sampling_params.max_tokens,
        )
        try:
            # Tokenize prompts and apply chat template
            tokenized_messages = [self.tokenizer.apply_chat_template(prompt["messages"], tokenize=True, add_generation_prompt=True, chat_template_kwargs={"enable_thinking": False}, return_tensors="pt") for prompt in prompts]
            outputs = self.llm.generate(
                prompts=tokenized_messages,
                sampling_params=params,
                use_tqdm=True
            )
            return [
                {
                    **prompts[i].get("metadata", {}),
                    output_field: output.outputs[0].text.strip()
                }
                for i, output in enumerate(outputs)
            ]
        except Exception as e:
            return [
                {
                    **prompts[i].get("metadata", {}),
                    output_field: f"[ERROR] {str(e)}"
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
        self.delete_client()
        self.llm = LLM(self.model_name, **self.config.to_vllm_config())
        logging.info(f"LLM client reset to model: {self.model_name}")