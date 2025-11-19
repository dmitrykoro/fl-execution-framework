import logging
import torch

"""Device utilities for PyTorch."""


def get_device(device_preference: str) -> torch.device:
    """
    Get PyTorch device with GPU detection.

    Supports two modes:
    1. "cpu" - Force CPU-only execution
    2. "gpu" - GPU detection (CPU fallback)

    Args:
        device_preference: "cpu" or "gpu"

    Returns:
        torch.device: Selected device
    """
    # Normalize device preference
    device_preference = device_preference.lower().strip()

    # CPU MODE: Force CPU
    if device_preference == "cpu":
        logging.info("Using device: CPU")
        return torch.device("cpu")

    # GPU MODE: Detection
    if device_preference == "gpu":
        # NVIDIA CUDA (Only CUDA GPU acceleration currently supported)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            cuda_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logging.info(
                f"Using CUDA GPU: {cuda_name} ({gpu_memory_gb:.1f} GB VRAM available)"
            )
            return device

        # Apple Silicon MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logging.warning(
                "Detected Apple Silicon GPU (MPS), but framework only supports NVIDIA CUDA GPUs. "
                "Falling back to CPU. For GPU acceleration, use a machine with NVIDIA GPU."
            )
            return torch.device("cpu")

        # AMD ROCm
        if (
            hasattr(torch, "version")
            and hasattr(torch.version, "hip")
            and torch.version.hip is not None
        ):
            logging.warning(
                "Detected AMD GPU (ROCm), but framework only supports NVIDIA CUDA GPUs. "
                "Falling back to CPU. For GPU acceleration, use a machine with NVIDIA GPU."
            )
            return torch.device("cpu")

        # No GPU detected
        logging.info(
            "GPU requested but no compatible GPU detected. Using CPU. "
            "For GPU acceleration, ensure NVIDIA GPU and CUDA drivers are installed."
        )
        return torch.device("cpu")

    # Default to CPU
    logging.warning(
        f"Invalid device preference '{device_preference}'. Valid options: 'cpu' or 'gpu'. "
        "Defaulting to CPU."
    )
    return torch.device("cpu")


def get_device_name(device: torch.device) -> str:
    """
    Get human-readable device name.

    Args:
        device: PyTorch device object

    Returns:
        str: Human-readable device name
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
