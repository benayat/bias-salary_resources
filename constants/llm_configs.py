from llm.llm_client import LLMResourceConfig, SamplingConfig

# Configuration for home setup with RTX 3090 (single GPU, 24GB VRAM)
HOME_CONFIG = LLMResourceConfig(
    gpu_memory_utilization=0.9,  # Use 90% of GPU memory for efficiency
    max_model_len=1024,  # Limited to ~250 expected tokens (input + output) for efficiency
    max_num_seqs=128,  # Moderate concurrency
    max_num_batched_tokens=131072,
    block_size=16,  # Standard KV cache block size
    tensor_parallel_size=1,  # Single GPU
    dtype="auto",  # Automatic data type selection
    trust_remote_code=True,  # Allow custom models
    disable_log_stats=True,  # Disable verbose logging
    max_parallel_loading_workers=4,  # Parallel loading for faster startup
    enable_prefix_caching=True,  # Enable prefix caching
    enforce_eager=False,  # Use default execution mode
    use_transformers=False,  # Use vLLM backend
)

# Configuration for university HPC with H200/B200 GPUs (multi-GPU cluster)
HPC_CONFIG = LLMResourceConfig(
    gpu_memory_utilization=0.95,  # Higher utilization for cluster efficiency
    max_model_len=1024,  # Limited to ~250 expected tokens (input + output) for efficiency
    max_num_seqs=256,  # Higher concurrency for batch processing
    max_num_batched_tokens=262144,  # Larger batch size for throughput
    block_size=32,  # Larger block size for better performance
    tensor_parallel_size=4,
    dtype="bfloat16",
    trust_remote_code=True,  # Allow custom models
    disable_log_stats=True,  # Disable verbose logging
    max_parallel_loading_workers=8,  # More workers for faster loading
    enable_prefix_caching=True,  # Enable prefix caching
    enforce_eager=False,  # Use default execution mode
    use_transformers=False,  # Use vLLM backend
)

# Default sampling configuration for both setups
DEFAULT_SAMPLING_CONFIG = SamplingConfig(
    temperature=0.0,  # Deterministic for estimation tasks
    top_p=1.0,  # No nucleus sampling
    max_tokens=1024,  # Sufficient for compensation estimates
    batch_size=100,  # Batch size for processing
)
