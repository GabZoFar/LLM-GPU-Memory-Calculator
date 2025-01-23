from typing import Dict, Union
from huggingface_hub import get_safetensors_metadata
import argparse
import sys

# Example: 
# python get_gpu_memory.py Qwen/Qwen2.5-7B-Instruct

# Dictionary mapping dtype strings to their byte sizes
bytes_per_dtype: Dict[str, float] = {
    "int4": 0.5,
    "int8": 1,
    "float8": 1,
    "float16": 2,
    "bfloat16": 2,  # Added bfloat16 as it's common
    "float32": 4,
}


def calculate_gpu_memory(parameter_count: int, bytes_per_param: float) -> float:
    """
    Calculates the GPU memory required for serving a Large Language Model (LLM).
    Uses binary calculation (1024³) for more accurate GiB measurement.

    The calculation follows these steps:
    1. Calculate total bytes needed: parameter_count * bytes_per_parameter
    2. Convert to GiB using binary prefix (1024³ bytes per GiB)
    3. Add 18% overhead for CUDA kernels, gradients, and other runtime requirements

    Args:
        parameter_count: Total number of parameters in the model
        bytes_per_param: Number of bytes per parameter based on dtype (e.g., 2 for float16)

    Returns:
        float: Estimated GPU memory required in GiB (binary gigabytes)

    Examples:
        >>> calculate_gpu_memory(7000000000, bytes_per_dtype["float16"])  # 7B model in float16
        15.91
        >>> calculate_gpu_memory(13000000000, bytes_per_dtype["int8"])  # 13B model in int8
        14.76
    """
    # Calculate total bytes needed
    total_bytes = parameter_count * bytes_per_param
    
    # Convert to GiB (1024³ bytes per GiB) and add 18% overhead
    memory_gib = (total_bytes / (1024**3)) * 1.18
    
    return round(memory_gib, 2)


def get_model_size(model_id: str, dtype: str = "float16") -> Union[float, None]:
    """
    Get the estimated GPU memory requirement for a Hugging Face model.

    This function:
    1. Validates the requested dtype
    2. Fetches the model's parameter count from its safetensors metadata
    3. Calculates the required GPU memory based on parameter count and dtype

    Args:
        model_id: Hugging Face model ID (e.g., "meta-llama/Llama-2-7b-hf")
        dtype: Data type for model loading ("float16", "int8", etc.)

    Returns:
        float or None: Estimated GPU memory in GiB, or None if estimation fails

    Examples:
        >>> get_model_size("facebook/opt-350m")
        0.82
        >>> get_model_size("meta-llama/Llama-2-7b-hf", dtype="int8")
        7.95

    Raises:
        ValueError: If the dtype is not supported
    """
    try:
        if dtype not in bytes_per_dtype:
            raise ValueError(
                f"Unsupported dtype: {dtype}. Supported types: {list(bytes_per_dtype.keys())}"
            )

        metadata = get_safetensors_metadata(model_id)
        if not metadata or not metadata.parameter_count:
            raise ValueError(f"Could not fetch metadata for model: {model_id}")

        parameter_count = list(metadata.parameter_count.values())[0]
        return calculate_gpu_memory(int(parameter_count), bytes_per_dtype[dtype])

    except Exception as e:
        print(f"Error estimating model size: {str(e)}", file=sys.stderr)
        return None


def main():
    """
    Command-line interface for GPU memory estimation.

    Usage:
        python get_memory_size.py meta-llama/Llama-2-7b-hf --dtype float16
        python get_memory_size.py facebook/opt-350m --dtype int8

    The script will print the estimated GPU memory requirement in GiB for the specified
    model and data type. If the estimation fails, it will print an error message to stderr.
    """
    parser = argparse.ArgumentParser(
        description="Estimate GPU memory requirements for Hugging Face models"
    )
    parser.add_argument(
        "model_id", 
        help="Hugging Face model ID (e.g., meta-llama/Llama-2-7b-hf)"
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=bytes_per_dtype.keys(),
        help="Data type for model loading (affects memory usage)"
    )

    args = parser.parse_args()
    size = get_model_size(args.model_id, args.dtype)

    if size is not None:
        print(
            f"Estimated GPU memory requirement for {args.model_id}: {size:.2f} GiB ({args.dtype})"
        )
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
