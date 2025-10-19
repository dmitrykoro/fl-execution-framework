"""
Unit tests for DynamicCNN model.

Tests dynamic CNN initialization, forward passes, parameter extraction,
and adaptability to different dataset characteristics.
"""

import torch
import torch.nn as nn

from src.network_models.dynamic_cnn import DynamicCNN
from tests.common import pytest


class TestDynamicCNNInitialization:
    """Test DynamicCNN model initialization."""

    def test_initialization_default_params(self) -> None:
        """Test DynamicCNN initialization with default parameters."""
        model = DynamicCNN(num_classes=10)

        assert isinstance(model, nn.Module)
        assert model.num_classes == 10
        assert model.input_channels == 1
        assert model.input_height == 28
        assert model.input_width == 28

    def test_initialization_custom_params(self) -> None:
        """Test DynamicCNN initialization with custom parameters."""
        model = DynamicCNN(
            num_classes=5,
            input_channels=3,
            input_height=64,
            input_width=64,
        )

        assert model.num_classes == 5
        assert model.input_channels == 3
        assert model.input_height == 64
        assert model.input_width == 64

    def test_initialization_rgb_images(self) -> None:
        """Test DynamicCNN initialization for RGB images."""
        model = DynamicCNN(num_classes=10, input_channels=3)

        assert model.input_channels == 3
        assert isinstance(model.conv1, nn.Conv2d)
        assert model.conv1.in_channels == 3

    def test_initialization_grayscale_images(self) -> None:
        """Test DynamicCNN initialization for grayscale images."""
        model = DynamicCNN(num_classes=10, input_channels=1)

        assert model.input_channels == 1
        assert isinstance(model.conv1, nn.Conv2d)
        assert model.conv1.in_channels == 1

    def test_layers_exist(self) -> None:
        """Test that all expected layers are created."""
        model = DynamicCNN(num_classes=10)

        assert hasattr(model, "conv1")
        assert hasattr(model, "conv2")
        assert hasattr(model, "pool")
        assert hasattr(model, "fc1")
        assert hasattr(model, "fc2")
        assert hasattr(model, "dropout")

    def test_conv_layer_configurations(self) -> None:
        """Test convolutional layer configurations."""
        model = DynamicCNN(num_classes=10, input_channels=3)

        # Conv1: input_channels -> 32
        assert model.conv1.in_channels == 3
        assert model.conv1.out_channels == 32
        assert model.conv1.kernel_size == (5, 5)
        assert model.conv1.padding == (2, 2)

        # Conv2: 32 -> 64
        assert model.conv2.in_channels == 32
        assert model.conv2.out_channels == 64
        assert model.conv2.kernel_size == (5, 5)
        assert model.conv2.padding == (2, 2)

    def test_pooling_layer_configuration(self) -> None:
        """Test pooling layer configuration."""
        model = DynamicCNN(num_classes=10)

        assert isinstance(model.pool, nn.MaxPool2d)
        assert model.pool.kernel_size == 2
        assert model.pool.stride == 2

    def test_fc_layer_output_size(self) -> None:
        """Test FC2 output size matches num_classes."""
        model = DynamicCNN(num_classes=15)

        assert model.fc2.out_features == 15

    def test_dropout_rate(self) -> None:
        """Test dropout layer configuration."""
        model = DynamicCNN(num_classes=10)

        assert isinstance(model.dropout, nn.Dropout)
        assert model.dropout.p == 0.5


class TestDynamicCNNForwardPass:
    """Test DynamicCNN forward pass."""

    @pytest.mark.parametrize(
        "num_classes,input_channels,height,width",
        [
            (10, 1, 28, 28),  # MNIST-like
            (10, 3, 32, 32),  # CIFAR-like
            (2, 1, 64, 64),  # Binary classification, large image
            (5, 3, 224, 224),  # ImageNet-like
        ],
    )
    def test_forward_pass_output_shape(
        self,
        num_classes: int,
        input_channels: int,
        height: int,
        width: int,
    ) -> None:
        """Test forward pass produces correct output shape."""
        model = DynamicCNN(
            num_classes=num_classes,
            input_channels=input_channels,
            input_height=height,
            input_width=width,
        )
        model.eval()

        batch_size = 4
        x = torch.randn(batch_size, input_channels, height, width)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, num_classes)

    def test_forward_pass_single_sample(self) -> None:
        """Test forward pass with single sample (batch_size=1)."""
        model = DynamicCNN(num_classes=10)
        model.eval()

        x = torch.randn(1, 1, 28, 28)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 10)

    def test_forward_pass_large_batch(self) -> None:
        """Test forward pass with large batch size."""
        model = DynamicCNN(num_classes=10)
        model.eval()

        batch_size = 128
        x = torch.randn(batch_size, 1, 28, 28)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 10)

    def test_forward_pass_no_nans(self) -> None:
        """Test forward pass produces no NaN values."""
        model = DynamicCNN(num_classes=10)
        model.eval()

        x = torch.randn(4, 1, 28, 28)

        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output).any()

    def test_forward_pass_not_all_zeros(self) -> None:
        """Test forward pass produces non-zero outputs."""
        model = DynamicCNN(num_classes=10)
        model.eval()

        x = torch.randn(4, 1, 28, 28)

        with torch.no_grad():
            output = model(x)

        assert not torch.allclose(output, torch.zeros_like(output))

    def test_forward_pass_training_mode(self) -> None:
        """Test forward pass in training mode."""
        model = DynamicCNN(num_classes=10)
        model.train()

        x = torch.randn(4, 1, 28, 28)
        output = model(x)

        assert output.shape == (4, 10)
        assert output.requires_grad

    def test_forward_pass_eval_mode(self) -> None:
        """Test forward pass in evaluation mode."""
        model = DynamicCNN(num_classes=10)
        model.eval()

        x = torch.randn(4, 1, 28, 28)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 10)
        assert not output.requires_grad


class TestDynamicCNNAdaptability:
    """Test DynamicCNN adaptability to different input dimensions."""

    @pytest.mark.parametrize(
        "height,width,expected_fc1_in",
        [
            (28, 28, 64 * 7 * 7),  # 28/4 = 7
            (32, 32, 64 * 8 * 8),  # 32/4 = 8
            (64, 64, 64 * 16 * 16),  # 64/4 = 16
            (224, 224, 64 * 56 * 56),  # 224/4 = 56
        ],
    )
    def test_fc1_adapts_to_input_size(
        self, height: int, width: int, expected_fc1_in: int
    ) -> None:
        """Test FC1 layer adapts to different input dimensions."""
        model = DynamicCNN(
            num_classes=10,
            input_channels=1,
            input_height=height,
            input_width=width,
        )

        assert model.fc1.in_features == expected_fc1_in
        assert model.fc1.out_features == 512

    def test_handles_square_images(self) -> None:
        """Test model handles square images."""
        model = DynamicCNN(num_classes=10, input_height=32, input_width=32)
        x = torch.randn(2, 1, 32, 32)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 10)

    def test_handles_non_square_images(self) -> None:
        """Test model handles non-square images."""
        model = DynamicCNN(num_classes=10, input_height=28, input_width=36)
        x = torch.randn(2, 1, 28, 36)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 10)

    @pytest.mark.parametrize("num_classes", [2, 5, 10, 100, 1000])
    def test_handles_different_num_classes(self, num_classes: int) -> None:
        """Test model adapts to different number of classes."""
        model = DynamicCNN(num_classes=num_classes)
        x = torch.randn(2, 1, 28, 28)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, num_classes)


class TestDynamicCNNParameters:
    """Test DynamicCNN parameter management."""

    def test_has_trainable_parameters(self) -> None:
        """Test model has trainable parameters."""
        model = DynamicCNN(num_classes=10)
        params = list(model.parameters())

        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_parameter_count(self) -> None:
        """Test model has reasonable parameter count."""
        model = DynamicCNN(num_classes=10)
        total_params = sum(p.numel() for p in model.parameters())

        # Should have parameters but not excessive
        assert 10000 < total_params < 10_000_000

    def test_state_dict_extraction(self) -> None:
        """Test state dict can be extracted."""
        model = DynamicCNN(num_classes=10)
        state_dict = model.state_dict()

        assert len(state_dict) > 0
        assert "conv1.weight" in state_dict
        assert "conv2.weight" in state_dict
        assert "fc1.weight" in state_dict
        assert "fc2.weight" in state_dict

    def test_state_dict_loading(self) -> None:
        """Test state dict can be loaded."""
        model1 = DynamicCNN(num_classes=10)
        model2 = DynamicCNN(num_classes=10)

        state_dict = model1.state_dict()
        model2.load_state_dict(state_dict)

        # Verify parameters match
        for (n1, p1), (n2, p2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)

    def test_zero_grad(self) -> None:
        """Test gradients can be zeroed."""
        model = DynamicCNN(num_classes=10)
        x = torch.randn(2, 1, 28, 28)
        target = torch.randint(0, 10, (2,))

        # Forward + backward
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()

        # Verify gradients exist
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

        # Zero gradients
        model.zero_grad()

        # Verify gradients are zeroed
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is None or torch.allclose(
                    p.grad, torch.zeros_like(p.grad)
                )


class TestDynamicCNNGradients:
    """Test DynamicCNN gradient computation."""

    def test_gradients_computed(self) -> None:
        """Test gradients are computed during backprop."""
        model = DynamicCNN(num_classes=10)
        model.train()

        x = torch.randn(4, 1, 28, 28)
        target = torch.randint(0, 10, (4,))

        output = model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()

        # Check all parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()

    def test_gradients_non_zero(self) -> None:
        """Test gradients are non-zero after backprop."""
        model = DynamicCNN(num_classes=10)
        model.train()

        x = torch.randn(4, 1, 28, 28)
        target = torch.randint(0, 10, (4,))

        output = model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()

        # At least some gradients should be non-zero
        has_nonzero_grad = False
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                if not torch.allclose(param.grad, torch.zeros_like(param.grad)):
                    has_nonzero_grad = True
                    break

        assert has_nonzero_grad


class TestDynamicCNNTrainingEvalModes:
    """Test DynamicCNN training and evaluation modes."""

    def test_train_mode_sets_training_flag(self) -> None:
        """Test train() sets training flag."""
        model = DynamicCNN(num_classes=10)
        model.train()

        assert model.training

    def test_eval_mode_sets_eval_flag(self) -> None:
        """Test eval() sets evaluation flag."""
        model = DynamicCNN(num_classes=10)
        model.eval()

        assert not model.training

    def test_dropout_active_in_training(self) -> None:
        """Test dropout is active in training mode."""
        model = DynamicCNN(num_classes=10)
        model.train()

        x = torch.randn(10, 1, 28, 28)

        # Run multiple forward passes
        outputs = []
        for _ in range(5):
            output = model(x)
            outputs.append(output.clone())

        # Outputs should differ due to dropout
        all_same = True
        for i in range(1, len(outputs)):
            if not torch.allclose(outputs[0], outputs[i]):
                all_same = False
                break

        assert not all_same

    def test_dropout_inactive_in_eval(self) -> None:
        """Test dropout is inactive in evaluation mode."""
        model = DynamicCNN(num_classes=10)
        model.eval()

        x = torch.randn(10, 1, 28, 28)

        # Run multiple forward passes
        outputs = []
        with torch.no_grad():
            for _ in range(5):
                output = model(x)
                outputs.append(output.clone())

        # Outputs should be identical in eval mode
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i])


class TestDynamicCNNDeviceCompatibility:
    """Test DynamicCNN device compatibility."""

    def test_cpu_device(self) -> None:
        """Test model works on CPU."""
        model = DynamicCNN(num_classes=10).to("cpu")
        x = torch.randn(2, 1, 28, 28).to("cpu")

        with torch.no_grad():
            output = model(x)

        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self) -> None:
        """Test model works on CUDA."""
        model = DynamicCNN(num_classes=10).to("cuda")
        x = torch.randn(2, 1, 28, 28).to("cuda")

        with torch.no_grad():
            output = model(x)

        assert output.device.type == "cuda"


class TestDynamicCNNRepr:
    """Test DynamicCNN string representation."""

    def test_repr_contains_num_classes(self) -> None:
        """Test __repr__ contains number of classes."""
        model = DynamicCNN(num_classes=10)
        repr_str = repr(model)

        assert "classes=10" in repr_str

    def test_repr_contains_input_shape(self) -> None:
        """Test __repr__ contains input shape."""
        model = DynamicCNN(
            num_classes=10,
            input_channels=3,
            input_height=64,
            input_width=64,
        )
        repr_str = repr(model)

        assert "input_shape=(3, 64, 64)" in repr_str

    def test_repr_format(self) -> None:
        """Test __repr__ has correct format."""
        model = DynamicCNN(
            num_classes=5, input_channels=1, input_height=28, input_width=28
        )
        repr_str = repr(model)

        assert repr_str == "DynamicCNN(classes=5, input_shape=(1, 28, 28))"


class TestDynamicCNNIntegration:
    """Integration tests for DynamicCNN."""

    def test_training_loop_simulation(self) -> None:
        """Test simulated training loop."""
        model = DynamicCNN(num_classes=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        model.train()

        # Simulate training steps
        for _ in range(3):
            x = torch.randn(4, 1, 28, 28)
            target = torch.randint(0, 10, (4,))

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            assert loss.item() >= 0
            assert not torch.isnan(loss)

    def test_evaluation_simulation(self) -> None:
        """Test simulated evaluation."""
        model = DynamicCNN(num_classes=10)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for _ in range(3):
                x = torch.randn(4, 1, 28, 28)
                target = torch.randint(0, 10, (4,))

                output = model(x)
                loss = criterion(output, target)

                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = total_loss / 3
        accuracy = correct / total

        assert avg_loss >= 0
        assert 0 <= accuracy <= 1

    def test_model_saving_loading(self) -> None:
        """Test model can be saved and loaded via state dict."""
        model1 = DynamicCNN(num_classes=10)
        model2 = DynamicCNN(num_classes=10)

        # Save state
        state_dict = model1.state_dict()

        # Load state
        model2.load_state_dict(state_dict)

        # Verify predictions match
        x = torch.randn(2, 1, 28, 28)
        model1.eval()
        model2.eval()

        with torch.no_grad():
            output1 = model1(x)
            output2 = model2(x)

        assert torch.allclose(output1, output2)

    def test_different_architectures_independent(self) -> None:
        """Test models with different configs are independent."""
        model1 = DynamicCNN(num_classes=10, input_height=28, input_width=28)
        model2 = DynamicCNN(num_classes=5, input_height=32, input_width=32)

        # Different architectures
        assert model1.fc2.out_features != model2.fc2.out_features
        assert model1.fc1.in_features != model2.fc1.in_features

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
    def test_handles_various_batch_sizes(self, batch_size: int) -> None:
        """Test model handles various batch sizes."""
        model = DynamicCNN(num_classes=10)
        model.eval()

        x = torch.randn(batch_size, 1, 28, 28)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 10)
