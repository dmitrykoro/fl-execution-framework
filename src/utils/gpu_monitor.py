"""GPU memory monitoring utilities for federated learning experiments."""

import logging
import torch
from typing import Dict, Union


class GPUMemoryMonitor:
    """Monitor and log GPU memory usage during experiments."""

    def __init__(self, device: Union[str, torch.device], log_level: int = logging.INFO):
        """Initialize GPU memory monitor.

        Args:
            device: PyTorch device to monitor (string or torch.device)
            log_level: Logging level for memory reports
        """
        # Convert string to torch.device if needed
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.is_cuda = self.device.type == "cuda"
        self.log_level = log_level
        self.initial_allocated = 0
        self.initial_reserved = 0

        if self.is_cuda:
            self.initial_allocated = torch.cuda.memory_allocated(device)
            self.initial_reserved = torch.cuda.memory_reserved(device)

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics.

        Returns:
            Dictionary with memory stats in GB:
                - allocated: Currently allocated memory
                - reserved: Memory reserved by caching allocator
                - free: Difference between reserved and allocated
                - max_allocated: Peak allocated memory
        """
        if not self.is_cuda:
            return {}

        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1e9

        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "free_gb": round(reserved - allocated, 2),
            "max_allocated_gb": round(max_allocated, 2),
        }

    def log_memory_usage(self, context: str = ""):
        """Log current GPU memory usage.

        Args:
            context: Contextual message (e.g., "After round 10")
        """
        if not self.is_cuda:
            return

        stats = self.get_memory_stats()
        message = (
            f"GPU Memory {context}: "
            f"{stats['allocated_gb']:.2f}GB allocated, "
            f"{stats['reserved_gb']:.2f}GB reserved, "
            f"{stats['free_gb']:.2f}GB free (peak: {stats['max_allocated_gb']:.2f}GB)"
        )
        logging.log(self.log_level, message)

    def check_memory_threshold(self, threshold_percent: float = 90.0) -> bool:
        """Check if GPU memory usage exceeds threshold.

        Args:
            threshold_percent: Warning threshold (0-100)

        Returns:
            True if memory exceeds threshold
        """
        if not self.is_cuda:
            return False

        total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        usage_percent = (allocated / total_memory) * 100

        if usage_percent >= threshold_percent:
            logging.warning(
                f"GPU memory usage HIGH: {usage_percent:.1f}% "
                f"({allocated:.2f}GB / {total_memory:.2f}GB)"
            )
            return True

        return False

    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)


def log_gpu_memory(device: Union[str, torch.device], context: str = ""):
    """Convenience function for quick memory logging.

    Args:
        device: PyTorch device (string or torch.device)
        context: Contextual message
    """
    device_obj = torch.device(device) if isinstance(device, str) else device
    if device_obj.type == "cuda":
        monitor = GPUMemoryMonitor(device_obj)
        monitor.log_memory_usage(context)
