import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from openai import OpenAI


@dataclass
class OpenAIConfig:
    api_key: str
    base_url: str = "https://api.openai.com/v1"


@dataclass
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2
    seed: int = 12345


class LLMClient:
    """
    LLM client using OpenAI-compatible API.
    Maintains the same interface as the vLLM-based client.
    """

    def __init__(
        self,
        model_name: str,
        config: OpenAIConfig,
    ):
        self.model_name = model_name
        self.config = config

        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

        logging.info(
            f"OpenAI LLMClient initialized for model={self.model_name} ")

    def run_batch(
        self,
        prompts: List[Dict[str, Any]],
        sampling_params: SamplingConfig,
        output_field: str = "output",
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of prompts and return results with metadata.
        Since OpenAI API doesn't support true batching, process sequentially.
        """
        results = []
        for prompt in prompts:
            try:
                # Build API call parameters
                messages = prompt["messages"]
                api_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": sampling_params.temperature,
                    # "max_tokens": sampling_params.max_tokens,
                    "seed": sampling_params.seed,
                    # "n": sampling_params.n,
                }

                # OpenAI uses max_completion_tokens, other providers use max_tokens
                if "api.openai.com" in self.config.base_url:
                    api_params["max_completion_tokens"] = sampling_params.max_tokens
                    api_params["reasoning_effort"] = "none"
                else:
                    api_params["max_tokens"] = sampling_params.max_tokens

                if "grok" in self.model_name:
                    api_params["extra_body"] = {"reasoning": {"enabled": False}}

                # Add seed if provided (DeepSeek and Google don't support it, so skip)
                if sampling_params.seed is not None and not ("deepseek.com" in self.config.base_url or "google" in self.config.base_url):
                    api_params["seed"] = sampling_params.seed

                # DeepSeek-specific: disable thinking mode
                if "deepseek.com" in self.config.base_url:
                    api_params["extra_body"] = {"thinking": {"type": "disabled"}}

                if "gpt-oss" in self.model_name:
                    api_params["extra_body"] = {"reasoning_effort": "low"}

                if "anthropic.com" not in self.config.base_url:
                    api_params["top_p"] = sampling_params.top_p

                response = self.client.chat.completions.create(**api_params)
                # print(response)
                output = response.choices[0].message.content.strip()
                results.append({
                    **prompt.get("metadata", {}),
                    output_field: output,
                })
            except Exception as e:
                logging.exception("Error in run_batch for prompt")
                results.append({
                    **prompt.get("metadata", {}),
                    output_field: f"[ERROR] {str(e)}",
                })
        return results

    def delete_client(self):
        """Delete the OpenAI client."""
        logging.info("Deleting OpenAI client")
        self.client = None
        logging.info("OpenAI client deleted")

    def reset_client_to_another_model(self, model_name: str):
        """Reset the client to use a different model."""
        logging.info(f"Resetting OpenAI client to model {model_name}")
        self.model_name = model_name
        # Recreate client if needed, but since model is in create, it's fine
        logging.info(f"OpenAI client reset to model: {self.model_name}")
