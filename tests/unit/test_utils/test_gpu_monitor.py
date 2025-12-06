"""Unit tests for GPU Memory Monitor."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.utils.gpu_monitor import GPUMemoryMonitor, log_gpu_memory


@pytest.fixture
def mock_cuda():
    """Mocks torch.cuda functions."""
    with patch("src.utils.gpu_monitor.torch.cuda") as mock:
        mock.get_device_properties.return_value.total_memory = 10 * 1e9
        mock.memory_allocated.return_value = 2 * 1e9
        mock.memory_reserved.return_value = 4 * 1e9
        mock.max_memory_allocated.return_value = 5 * 1e9

        yield mock


@pytest.fixture
def mock_logging():
    """Mocks the logging module."""
    with patch("src.utils.gpu_monitor.logging") as mock:
        yield mock


class TestGPUMemoryMonitor:
    """Tests for GPUMemoryMonitor."""

    def test_init_cpu(self):
        """Tests initialization on CPU."""
        monitor = GPUMemoryMonitor(device="cpu")
        assert not monitor.is_cuda
        assert monitor.initial_allocated == 0

    def test_init_cuda(self, mock_cuda):
        """Tests initialization on CUDA."""
        device = MagicMock()
        device.type = "cuda"

        monitor = GPUMemoryMonitor(device=device)
        assert monitor.is_cuda
        assert monitor.initial_allocated == 2 * 1e9

    def test_get_memory_stats_cpu(self):
        """Tests memory statistics on CPU."""
        monitor = GPUMemoryMonitor(device="cpu")
        stats = monitor.get_memory_stats()
        assert stats == {}

    def test_get_memory_stats_cuda(self, mock_cuda):
        """Tests memory statistics on CUDA."""
        device = MagicMock()
        device.type = "cuda"
        monitor = GPUMemoryMonitor(device=device)

        stats = monitor.get_memory_stats()

        assert stats["allocated_gb"] == 2.0
        assert stats["reserved_gb"] == 4.0
        assert stats["free_gb"] == 2.0
        assert stats["max_allocated_gb"] == 5.0

    def test_log_memory_usage(self, mock_cuda, mock_logging):
        """Tests memory usage logging."""
        device = MagicMock()
        device.type = "cuda"
        monitor = GPUMemoryMonitor(device=device)

        monitor.log_memory_usage(context="TestCtx")

        mock_logging.log.assert_called_once()
        args = mock_logging.log.call_args[0]
        assert args[0] == logging.INFO
        assert "TestCtx" in args[1]
        assert "2.00GB allocated" in args[1]

    def test_check_memory_threshold_below(self, mock_cuda, mock_logging):
        """Tests memory threshold check within limits."""
        device = MagicMock()
        device.type = "cuda"
        monitor = GPUMemoryMonitor(device=device)

        result = monitor.check_memory_threshold(threshold_percent=90.0)

        assert not result
        mock_logging.warning.assert_not_called()

    def test_check_memory_threshold_exceeded(self, mock_cuda, mock_logging):
        """Tests memory threshold check exceeding limits."""
        device = MagicMock()
        device.type = "cuda"
        monitor = GPUMemoryMonitor(device=device)

        mock_cuda.memory_allocated.return_value = 9.5 * 1e9

        result = monitor.check_memory_threshold(threshold_percent=90.0)

        assert result
        mock_logging.warning.assert_called_once()
        assert "95.0%" in mock_logging.warning.call_args[0][0]

    def test_reset_peak_stats(self, mock_cuda):
        """Tests resetting peak memory statistics."""
        device = MagicMock()
        device.type = "cuda"
        monitor = GPUMemoryMonitor(device=device)

        monitor.reset_peak_stats()
        mock_cuda.reset_peak_memory_stats.assert_called_with(device)


def test_log_gpu_memory_helper(mock_cuda, mock_logging):
    """Tests the log_gpu_memory helper function."""
    with patch("src.utils.gpu_monitor.torch.device") as mock_dev_cls:
        device_mock = MagicMock()
        device_mock.type = "cuda"
        mock_dev_cls.return_value = device_mock

        log_gpu_memory("cuda:0", "HelperTest")

        mock_logging.log.assert_called_once()
        assert "HelperTest" in mock_logging.log.call_args[0][1]
