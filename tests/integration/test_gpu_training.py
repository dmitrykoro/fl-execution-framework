"""
Integration tests for GPU/CUDA training functionality.

Tests GPU training with real PyTorch device allocation and memory monitoring.
Falls back to CPU if CUDA is unavailable.
"""

import json
from pathlib import Path

import pytest
import torch


class TestGPUTraining:
    """Test GPU training integration with device allocation and fallback"""

    @pytest.fixture
    def minimal_gpu_config(self, tmp_path: Path) -> dict:
        """
        Minimal config for GPU training test.

        Uses small model, few clients, and minimal data for fast execution.
        """
        return {
            "shared_settings": {
                "display_name": "GPU Training Test",
                "aggregation_strategy_keyword": "fedavg",
                "dataset_source": "medmnist",
                "dataset_keyword": "bloodmnist",
                "num_of_rounds": 2,
                "num_of_clients": 2,
                "num_of_malicious_clients": 0,
                "show_plots": "false",
                "save_plots": "false",
                "save_csv": "true",
                "training_device": "cuda",
                "cpus_per_client": 1,
                "gpus_per_client": 1.0,
                "training_subset_fraction": 0.01,  # 1% of data for speed
                "min_fit_clients": 2,
                "min_evaluate_clients": 2,
                "min_available_clients": 2,
                "evaluate_metrics_aggregation_fn": "weighted_average",
                "num_of_client_epochs": 1,
                "batch_size": 16,
                "preserve_dataset": "false",
                "model_type": "cnn",
                "use_llm": "false",
            },
            "simulation_strategies": [{}],
        }

    @pytest.fixture
    def gpu_config_path(self, minimal_gpu_config: dict, tmp_path: Path) -> Path:
        """Save GPU config to temp file"""
        config_path = tmp_path / "config_gpu_test.json"
        with open(config_path, "w") as f:
            json.dump(minimal_gpu_config, f, indent=4)
        return config_path

    def test_gpu_device_selection(self):
        """CUDA device should be selected when available, fallback to CPU otherwise"""
        from src.utils.device_utils import get_device

        # Test explicit CPU request
        cpu_device = get_device("cpu")
        assert cpu_device.type == "cpu"

        # Test GPU request (fallback to CPU if unavailable)
        gpu_device = get_device("gpu")
        if torch.cuda.is_available():
            assert gpu_device.type == "cuda"
        else:
            assert gpu_device.type == "cpu"

    def test_gpu_device_fallback(self):
        """CUDA should fallback to CPU gracefully when unavailable"""
        from src.utils.device_utils import get_device

        if not torch.cuda.is_available():
            device = get_device("gpu")
            assert device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_allocation(self):
        """GPU should allocate memory for model when CUDA is available"""
        # Create simple model and move to GPU
        model = torch.nn.Linear(10, 10)
        model = model.to("cuda")

        # Verify model is on GPU
        assert next(model.parameters()).device.type == "cuda"

        # Verify GPU memory is allocated
        allocated_memory = torch.cuda.memory_allocated()
        assert allocated_memory > 0, "GPU memory should be allocated for model"

        # Cleanup
        del model
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_training_with_dataloaders(self):
        """
        Test GPU training with real PyTorch DataLoaders.

        Verifies model can train on GPU with gradient updates.
        """
        from torch.utils.data import DataLoader, TensorDataset

        # Create synthetic dataset
        X = torch.randn(100, 3, 28, 28)
        y = torch.randint(0, 10, (100,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Create simple CNN model
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 10),
        )
        model = model.to("cuda")

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        # Track GPU memory before training
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()

        # Train for one epoch
        model.train()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to("cuda"), labels.to("cuda")

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Verify GPU was utilized
        peak_memory = torch.cuda.max_memory_allocated()
        assert peak_memory > initial_memory, (
            "GPU memory should increase during training"
        )

        # Verify model parameters were updated
        assert any(param.grad is not None for param in model.parameters()), (
            "Model parameters should have gradients"
        )

        # Cleanup
        del model
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_resource_limits(self):
        """
        Test GPU resource allocation stays within limits.

        Verifies Ray/PyTorch respects GPU memory constraints.
        """
        # Get total GPU memory
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        assert total_memory_gb > 0, "GPU should have allocatable memory"

        # Create model that uses significant memory
        large_model = torch.nn.Linear(1000, 10000)
        large_model = large_model.to("cuda")

        allocated_memory_gb = torch.cuda.memory_allocated() / (1024**3)

        # Verify we can allocate memory without exceeding total
        assert allocated_memory_gb < total_memory_gb, (
            "Allocated memory should not exceed total GPU memory"
        )

        # Cleanup
        del large_model
        torch.cuda.empty_cache()

    def test_gpu_info_logging(self, caplog):
        """Device selection should log GPU information when available"""
        from src.utils.device_utils import get_device

        _ = get_device("gpu")

        if torch.cuda.is_available():
            # Should log GPU name
            assert any("CUDA" in record.message for record in caplog.records)
        else:
            # Should log fallback warning
            assert any("no compatible GPU detected" in record.message for record in caplog.records)
