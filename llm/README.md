# LLM Client Package

A modular, extensible package for working with Large Language Models using different backends.

## Package Structure

```
llm/
├── __init__.py                 # Public API exports
├── config.py                   # Configuration classes
├── base_client.py             # Abstract base class defining the interface
├── vllm_client.py             # vLLM implementation (high-performance)
├── pytorch_client.py          # PyTorch/Transformers implementation
└── llm_client.py              # Backwards compatibility module
```

## Components

### Configuration Classes (`config.py`)

- **`LLMResourceConfig`**: Configuration for GPU/resource allocation
  - `gpu_memory_utilization`: Fraction of GPU memory to use
  - `max_model_len`: Maximum sequence length
  - `max_num_seqs`: Maximum number of sequences
  - `batch_size`: Batch size for processing
  - `dtype`: Data type (e.g., "float16", "bfloat16")
  - And more...

- **`SamplingConfig`**: Configuration for text generation
  - `temperature`: Sampling temperature (0 = greedy)
  - `top_p`: Nucleus sampling parameter
  - `max_tokens`: Maximum tokens to generate
  - `batch_size`: Batch size for processing

### Base Class (`base_client.py`)

- **`BaseLLMClient`**: Abstract base class defining the common interface
  - `run_batch_simple()`: Simple batch generation
  - `run_batch()`: Batch generation with metadata
  - `delete_client()`: Cleanup and memory management
  - `reset_client_to_another_model()`: Model switching

### Client Implementations

#### vLLM Client (`vllm_client.py`)

- **`VLLMClient`**: High-performance inference using vLLM
  - Optimized for throughput and latency
  - Supports paged attention and continuous batching
  - Best for production deployments

#### PyTorch Client (`pytorch_client.py`)

- **`PyTorchLLMClient`**: Standard implementation using Hugging Face Transformers
  - Uses native PyTorch for inference
  - Automatic device detection (CUDA/CPU)
  - Supports chat templates
  - Good for development and testing

## Usage Examples

### Basic Usage (vLLM)

```python
from llm import VLLMClient, LLMResourceConfig, SamplingConfig

# Configure resources
config = LLMResourceConfig(
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    max_num_seqs=256,
    max_num_batched_tokens=8192,
    block_size=16,
    tensor_parallel_size=1,
    dtype="float16"
)

# Initialize client
client = VLLMClient("meta-llama/Llama-3-8B", config)

# Configure sampling
sampling = SamplingConfig(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# Generate responses
prompts = ["Hello, how are you?", "What is Python?"]
results = client.run_batch_simple(prompts, sampling)

for prompt, response in results:
    print(f"Q: {prompt}\nA: {response}\n")
```

### Using PyTorch Client

```python
from llm import PyTorchLLMClient, SamplingConfig

# Initialize PyTorch client (simpler config)
client = PyTorchLLMClient("meta-llama/Llama-3-8B")

# Same interface!
sampling = SamplingConfig(temperature=0.7, max_tokens=512)
results = client.run_batch_simple(prompts, sampling)
```

### Batch Processing with Metadata

```python
# Prepare prompts with metadata
prompts = [
    {
        "messages": [
            {"role": "user", "content": "What is AI?"}
        ],
        "metadata": {"id": 1, "category": "tech"}
    },
    {
        "messages": [
            {"role": "user", "content": "Explain quantum computing"}
        ],
        "metadata": {"id": 2, "category": "science"}
    }
]

# Process batch
results = client.run_batch(prompts, sampling, output_field="response")

for result in results:
    print(f"ID: {result['id']}")
    print(f"Category: {result['category']}")
    print(f"Response: {result['response']}\n")
```

### Polymorphic Usage

```python
from llm import BaseLLMClient, VLLMClient, PyTorchLLMClient

def process_with_any_client(client: BaseLLMClient, prompts: list):
    """Works with any client implementation!"""
    sampling = SamplingConfig(temperature=0.0, max_tokens=256)
    return client.run_batch_simple(prompts, sampling)

# Use with either client
vllm_results = process_with_any_client(VLLMClient("model", config), prompts)
pytorch_results = process_with_any_client(PyTorchLLMClient("model"), prompts)
```

### Model Switching

```python
# Switch to a different model
client.reset_client_to_another_model("meta-llama/Llama-3-70B")

# Clean up resources
client.delete_client()
```

## Backwards Compatibility

Existing code using `LLMClient` will continue to work without changes:

```python
# Old code still works!
from llm.llm_client import LLMClient, SamplingConfig, LLMResourceConfig

client = LLMClient("model", config)  # Uses VLLMClient internally
```

## Adding New Clients

To add a new client implementation:

1. Create a new file in the `llm/` directory (e.g., `openai_client.py`)
2. Inherit from `BaseLLMClient`
3. Implement all abstract methods
4. Export from `__init__.py`

```python
from llm.base_client import BaseLLMClient

class OpenAIClient(BaseLLMClient):
    def run_batch_simple(self, prompts, sampling_params):
        # Implementation here
        pass
    
    # ... implement other methods
```

## Benefits

✅ **Separation of Concerns**: Each component has a single responsibility
✅ **Extensibility**: Easy to add new client implementations
✅ **Type Safety**: Clear interfaces with type hints
✅ **Backwards Compatible**: Existing code continues to work
✅ **Testability**: Each module can be tested independently
✅ **Documentation**: Clear module boundaries and purposes

