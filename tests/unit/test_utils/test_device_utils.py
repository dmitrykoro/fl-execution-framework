import unittest
from unittest.mock import patch

import torch

from src.utils.device_utils import get_device, get_device_name


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
        """Invalid device preference should default to CPU"""
        device = get_device("quantum")
        self.assertEqual(device.type, "cpu")

    def test_get_device_empty_string(self):
        """Empty string should default to CPU"""
        device = get_device("")
        self.assertEqual(device.type, "cpu")

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


if __name__ == "__main__":
    unittest.main()
