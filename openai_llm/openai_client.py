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
                messages = prompt["messages"]
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                    max_tokens=sampling_params.max_tokens,
                )
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
