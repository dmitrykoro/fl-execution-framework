import logging

import torch


def get_device(device_preference: str) -> torch.device:
    """
    Get PyTorch device with automatic fallback to CPU if CUDA is not available.

    Args:
        device_preference: User's device preference ("cpu", "cuda", or "gpu")

    Returns:
        torch.device: Selected device, with fallback to CPU if CUDA unavailable

    Examples:
        >>> device = get_device("cuda")  # Returns cuda:0 if available, else cpu
        >>> device = get_device("cpu")   # Always returns cpu
    """
    # Normalize device preference
    device_preference = device_preference.lower().strip()

    # If user explicitly requested CPU, honor that
    if device_preference == "cpu":
        logging.info("Using device: CPU (as requested)")
        return torch.device("cpu")

    # For "cuda" or "gpu" requests, check availability
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

    # Invalid device specification
    logging.warning(
        f"Invalid device preference '{device_preference}'. Defaulting to CPU. "
        f"Valid options: 'cpu', 'cuda', 'gpu'"
    )
    return torch.device("cpu")


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
