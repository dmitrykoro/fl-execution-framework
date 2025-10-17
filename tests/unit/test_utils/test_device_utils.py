import unittest
from unittest.mock import Mock, patch

import torch

from src.utils.device_utils import (
    calculate_optimal_gpus_per_client,
    get_device,
    get_device_name,
)


class TestDeviceUtils(unittest.TestCase):
    """Test device selection and fallback functionality"""

    def test_get_device_cpu_request(self):
        """CPU should always be returned when explicitly requested"""
        device = get_device("cpu")
        self.assertEqual(device.type, "cpu")
        self.assertIsInstance(device, torch.device)

    def test_get_device_cpu_uppercase(self):
        """Should handle uppercase device names"""
        device = get_device("CPU")
        self.assertEqual(device.type, "cpu")

    def test_get_device_cpu_with_whitespace(self):
        """Should handle device names with whitespace"""
        device = get_device("  cpu  ")
        self.assertEqual(device.type, "cpu")

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA Test GPU")
    def test_get_device_cuda_available(self, mock_get_name, mock_is_available):
        """CUDA should be returned when available and requested"""
        device = get_device("cuda")
        self.assertEqual(device.type, "cuda")
        self.assertIsInstance(device, torch.device)
        mock_is_available.assert_called_once()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA Test GPU")
    def test_get_device_gpu_available(self, mock_get_name, mock_is_available):
        """GPU alias should work same as CUDA"""
        device = get_device("gpu")
        self.assertEqual(device.type, "cuda")
        mock_is_available.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_device_cuda_fallback(self, mock_is_available):
        """Should fallback to CPU when CUDA requested but unavailable"""
        device = get_device("cuda")
        self.assertEqual(device.type, "cpu")
        mock_is_available.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_device_gpu_fallback(self, mock_is_available):
        """Should fallback to CPU when GPU requested but unavailable"""
        device = get_device("gpu")
        self.assertEqual(device.type, "cpu")
        mock_is_available.assert_called_once()

    def test_get_device_invalid_preference(self):
        """Invalid device preference should use auto-detection"""
        device = get_device("quantum")
        # Should auto-detect (either cuda or cpu depending on availability)
        self.assertIn(device.type, ["cpu", "cuda"])

    def test_get_device_empty_string(self):
        """Empty string should use auto-detection"""
        device = get_device("")
        # Should auto-detect (either cuda or cpu depending on availability)
        self.assertIn(device.type, ["cpu", "cuda"])

    def test_get_device_name_cpu(self):
        """Should return CPU for CPU device"""
        device = torch.device("cpu")
        name = get_device_name(device)
        self.assertEqual(name, "CPU")

    @patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 3090")
    def test_get_device_name_cuda(self, mock_get_name):
        """Should return GPU name for CUDA device"""
        device = torch.device("cuda:0")
        name = get_device_name(device)
        self.assertIn("CUDA", name)
        self.assertIn("NVIDIA GeForce RTX 3090", name)

    @patch("torch.cuda.get_device_name", side_effect=Exception("GPU error"))
    def test_get_device_name_cuda_error(self, mock_get_name):
        """Should handle errors gracefully when getting GPU name"""
        device = torch.device("cuda:0")
        name = get_device_name(device)
        self.assertEqual(name, "CUDA")

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA Test GPU")
    @patch("torch.cuda.get_device_properties")
    def test_get_device_auto_with_gpu(
        self, mock_props, mock_get_name, mock_is_available
    ):
        """Auto mode should select GPU when CUDA is available"""
        mock_props.return_value = Mock(total_memory=8 * 1024**3)
        device = get_device("auto")
        self.assertEqual(device.type, "cuda")
        mock_is_available.assert_called()

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_device_auto_without_gpu(self, mock_is_available):
        """Auto mode should fallback to CPU when no GPU available"""
        device = get_device("auto")
        self.assertEqual(device.type, "cpu")
        mock_is_available.assert_called()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA Test GPU")
    def test_explicit_cpu_overrides_gpu(self, mock_get_name, mock_is_available):
        """CPU should be used even when GPU is available if explicitly requested"""
        device = get_device("cpu")
        self.assertEqual(device.type, "cpu")


class TestGPUAllocation(unittest.TestCase):
    """Test optimal GPU allocation calculation for Flower simulations"""

    @patch("torch.cuda.is_available", return_value=False)
    def test_no_gpu_returns_zero(self, mock_cuda):
        """Should return 0 when no GPU is available"""
        result = calculate_optimal_gpus_per_client(num_clients=5)
        self.assertEqual(result, 0.0)

    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.is_available", return_value=True)
    def test_small_gpu_conservative_allocation(self, mock_is_available, mock_props):
        """Small GPU (<6GB) should use conservative allocation"""
        mock_props.return_value = Mock(total_memory=4 * 1024**3)
        result = calculate_optimal_gpus_per_client(num_clients=10)
        self.assertEqual(result, 0.5)

    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.is_available", return_value=True)
    def test_medium_gpu_balanced_allocation(self, mock_is_available, mock_props):
        """Medium GPU (6-12GB) should use balanced allocation"""
        mock_props.return_value = Mock(total_memory=8 * 1024**3)
        result = calculate_optimal_gpus_per_client(num_clients=10)
        self.assertEqual(result, 0.25)

    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.is_available", return_value=True)
    def test_large_gpu_aggressive_allocation(self, mock_is_available, mock_props):
        """Large GPU (12-24GB) should use more aggressive allocation"""
        mock_props.return_value = Mock(total_memory=16 * 1024**3)
        result = calculate_optimal_gpus_per_client(num_clients=20)
        self.assertAlmostEqual(result, 1.0 / 6, places=5)

    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.is_available", return_value=True)
    def test_very_large_gpu_max_allocation(self, mock_is_available, mock_props):
        """Very large GPU (>24GB) should allow maximum parallelism"""
        mock_props.return_value = Mock(total_memory=32 * 1024**3)
        result = calculate_optimal_gpus_per_client(num_clients=20)
        self.assertEqual(result, 0.125)

    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.is_available", return_value=True)
    def test_allocation_respects_client_count(self, mock_is_available, mock_props):
        """GPU allocation should not exceed actual client count"""
        mock_props.return_value = Mock(total_memory=8 * 1024**3)
        result = calculate_optimal_gpus_per_client(num_clients=2)
        self.assertEqual(result, 0.5)

    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.is_available", return_value=True)
    def test_single_client_gets_full_gpu(self, mock_is_available, mock_props):
        """Single client should get full GPU allocation"""
        mock_props.return_value = Mock(total_memory=8 * 1024**3)
        result = calculate_optimal_gpus_per_client(num_clients=1)
        self.assertEqual(result, 1.0)

    def test_explicit_gpu_memory_parameter(self):
        """Explicit GPU memory parameter should override auto-detection"""
        result = calculate_optimal_gpus_per_client(num_clients=10, gpu_memory_gb=8.0)
        self.assertEqual(result, 0.25)

    @patch("torch.cuda.is_available", return_value=True)
    def test_zero_clients_edge_case(self, mock_is_available):
        """Zero clients should return 0 allocation"""
        result = calculate_optimal_gpus_per_client(num_clients=0)
        self.assertEqual(result, 0.0)

    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.is_available", return_value=True)
    def test_allocation_consistency(self, mock_is_available, mock_props):
        """GPU allocation should be consistent across multiple calls"""
        mock_props.return_value = Mock(total_memory=8 * 1024**3)
        result1 = calculate_optimal_gpus_per_client(num_clients=5)
        result2 = calculate_optimal_gpus_per_client(num_clients=5)
        self.assertEqual(result1, result2)


if __name__ == "__main__":
    unittest.main()
