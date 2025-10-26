import logging
from typing import Optional
import torch


def get_device(device_preference: str) -> torch.device:
    """
    Get PyTorch device with auto-detection and fallback.

    Supports three modes:
    1. "auto" - Automatically detect and use best available device (CUDA > CPU)
    2. "cuda"/"gpu" - Request GPU with fallback to CPU if unavailable
    3. "cpu" - Force CPU-only execution

    Args:
        device_preference: User's device preference ("auto", "cpu", "cuda", or "gpu")

    Returns:
        torch.device: Selected device with fallback

    Examples:
        >>> device = get_device("auto")  # Auto-detect: returns cuda:0 if available
        >>> device = get_device("cuda")  # Returns cuda:0 if available, else cpu
        >>> device = get_device("cpu")   # Always returns cpu
    """
    # Normalize device preference
    device_preference = device_preference.lower().strip()

    # AUTO MODE: Auto-detection of best device
    if device_preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            cuda_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logging.info(
                f"[AUTO-DETECT] Using CUDA GPU: {cuda_name} "
                f"({gpu_memory_gb:.1f} GB VRAM available)"
            )
            return device
        else:
            logging.info(
                "[AUTO-DETECT] No CUDA GPU detected. Using CPU. "
                "For faster training, ensure CUDA is installed and GPU is available."
            )
            return torch.device("cpu")

    # CPU MODE: User override to force CPU
    if device_preference == "cpu":
        logging.info("Using device: CPU (user requested)")
        return torch.device("cpu")

    # GPU MODE: Request GPU with fallback
    if device_preference in ("cuda", "gpu"):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            cuda_name = torch.cuda.get_device_name(0)
            logging.info(f"Using device: CUDA (GPU: {cuda_name})")
            return device
        else:
            logging.warning(
                "CUDA/GPU requested but not available. Falling back to CPU. "
                "To use GPU, ensure CUDA is properly installed and a compatible GPU is present."
            )
            return torch.device("cpu")

    # INVALID DEVICE: Default to auto-detection
    logging.warning(
        f"Invalid device preference '{device_preference}'. Using auto-detection. "
        f"Valid options: 'auto', 'cpu', 'cuda', 'gpu'"
    )
    # Recursively call with "auto" for fallback
    return get_device("auto")


def calculate_optimal_gpus_per_client(
    num_clients: int, gpu_memory_gb: Optional[float] = None
) -> float:
    """
    Calculate optimal GPU allocation per client for Flower simulations.

    Uses conservative memory allocation to prevent OOM errors while maximizing
    parallelism. Based on Flower best practices for simulation resource management.

    Strategy:
    - For small GPUs (<6GB): More conservative allocation
    - For medium GPUs (6-12GB): Balanced allocation
    - For large GPUs (>12GB): More aggressive parallelism

    Args:
        num_clients: Total number of clients in simulation
        gpu_memory_gb: GPU memory in GB (auto-detected if None)

    Returns:
        float: Recommended GPU fraction per client (for client_resources)

    Examples:
        >>> calculate_optimal_gpus_per_client(5, gpu_memory_gb=8.0)
        0.25  # Allows 4 clients in parallel, safe for 8GB GPU
    """
    if gpu_memory_gb is None and torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    if gpu_memory_gb is None:
        # No GPU available, return 0
        return 0.0

    # Conservative allocation based on GPU memory
    # These values prevent OOM while allowing good parallelism
    if gpu_memory_gb < 6:
        # Small GPU: 2 clients max
        max_parallel = 2
    elif gpu_memory_gb < 12:
        # Medium GPU: 3-4 clients
        max_parallel = 4
    elif gpu_memory_gb < 24:
        # Large GPU: 6-8 clients
        max_parallel = 6
    else:
        # Very large GPU: 8-10 clients
        max_parallel = 8

    # Don't over-allocate if we have fewer clients than max parallel
    max_parallel = min(max_parallel, num_clients)

    # Return fraction: 1/max_parallel
    # E.g., for 4 parallel clients: 1/4 = 0.25
    if max_parallel > 0:
        return 1.0 / max_parallel
    else:
        return 0.0


def get_device_name(device: torch.device) -> str:
    """
    Get human-readable device name.

    Args:
        device: PyTorch device object

    Returns:
        str: Human-readable device name

    Examples:
        >>> get_device_name(torch.device("cpu"))
        'CPU'
        >>> get_device_name(torch.device("cuda:0"))
        'CUDA (NVIDIA GeForce RTX 3090)'
    """
    if device.type == "cpu":
        return "CPU"
    elif device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(device.index or 0)
            return f"CUDA ({gpu_name})"
        except Exception:
            return "CUDA"
    else:
        return str(device)
