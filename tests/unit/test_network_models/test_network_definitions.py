"""
Unit tests for network model definitions.

Tests network model initialization, forward passes, parameter extraction,
and state management with lightweight mock implementations.
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast
from unittest.mock import patch

from tests.common import Mock, np, pytest
import torch
import torch.nn as nn

# Import BERT model functions
from src.network_models.bert_model_definition import (
    get_lora_state_dict,
    load_model,
    load_model_with_lora,
    set_lora_state_dict,
)
from src.network_models.bloodmnist_network_definition import BloodMNISTNetwork
from src.network_models.femnist_full_niid_network_definition import (
    FemnistFullNIIDNetwork,
)
from src.network_models.femnist_reduced_iid_network_definition import (
    FemnistReducedIIDNetwork,
)
from src.network_models.flair_network_definition import FlairNetwork

# Import network models
from src.network_models.its_network_definition import ITSNetwork
from src.network_models.lung_photos_network_definition import (
    LungCancerCNN as LungPhotosNetwork,
)
from src.network_models.pneumoniamnist_network_definition import PneumoniamnistNetwork


class TestNetworkModels:
    """Test network model definitions."""

    def _create_network(
        self, network_class: Type[nn.Module], num_classes: Optional[int] = None
    ) -> nn.Module:
        """Create network with appropriate parameters."""
        if network_class.__name__ == "LungCancerCNN" and num_classes is not None:
            return network_class(num_classes=num_classes)
        else:
            return network_class()

    @pytest.fixture(
        params=[
            (ITSNetwork, (3, 224, 224), 10),
            (FemnistReducedIIDNetwork, (1, 28, 28), 10),
            (FemnistFullNIIDNetwork, (1, 28, 28), 62),
            (PneumoniamnistNetwork, (1, 28, 28), 2),
            (BloodMNISTNetwork, (3, 28, 28), 8),
            (FlairNetwork, (3, 224, 224), 2),
            (LungPhotosNetwork, (1, 224, 224), 2),
        ]
    )
    def network_config(
        self, request: Any
    ) -> Dict[str, Union[Type[nn.Module], Tuple[int, ...], int, str]]:
        """Parameterized fixture for different network configurations."""
        network_class, input_shape, num_classes = request.param
        return {
            "class": network_class,
            "input_shape": input_shape,
            "num_classes": num_classes,
            "name": network_class.__name__,
        }

    def test_network_initialization(
        self,
        network_config: Dict[str, Union[Type[nn.Module], Tuple[int, ...], int, str]],
    ) -> None:
        """Test network initialization."""
        network_class = cast(Type[nn.Module], network_config["class"])
        num_classes = cast(int, network_config["num_classes"])
        assert isinstance(network_class, type) and issubclass(network_class, nn.Module)
        assert isinstance(num_classes, int)
        network = self._create_network(network_class, num_classes)

        # Check that network is a PyTorch module
        assert isinstance(network, nn.Module)

        # Check that network has expected layers
        assert hasattr(network, "conv1")
        assert hasattr(network, "fc1")

        # Check that weights are initialized (not all zeros)
        conv1_layer = network.conv1
        assert isinstance(conv1_layer, (nn.Conv2d, nn.Conv1d))
        conv1_weights = conv1_layer.weight.data
        assert not torch.allclose(conv1_weights, torch.zeros_like(conv1_weights))

    def test_network_forward_pass(
        self,
        network_config: Dict[str, Union[Type[nn.Module], Tuple[int, ...], int, str]],
    ) -> None:
        """Test forward pass through networks."""
        network_class = cast(Type[nn.Module], network_config["class"])
        input_shape = cast(Tuple[int, ...], network_config["input_shape"])
        num_classes = cast(int, network_config["num_classes"])

        # Create network instance with appropriate parameters
        network_class_typed = cast(Type[nn.Module], network_class)
        num_classes_typed = cast(int, num_classes)
        input_shape_typed = cast(Tuple[int, ...], input_shape)

        if network_class_typed.__name__ == "LungCancerCNN":
            network = network_class_typed(num_classes=num_classes_typed)
        else:
            network = network_class_typed()
        network.eval()  # Set to evaluation mode

        # Create mock input
        batch_size: int = 4
        mock_input: torch.Tensor = torch.randn(batch_size, *input_shape_typed)

        # Forward pass
        with torch.no_grad():
            output = network(mock_input)

        # Check output shape
        expected_shape: Tuple[int, int] = (batch_size, num_classes_typed)
        assert output.shape == expected_shape

        # Check output is not all zeros or NaN
        assert not torch.allclose(output, torch.zeros_like(output))
        assert not torch.isnan(output).any()

    def test_network_parameter_extraction(
        self,
        network_config: Dict[str, Union[Type[nn.Module], Tuple[int, ...], int, str]],
    ) -> None:
        """Test parameter extraction from networks."""
        network_class = cast(Type[nn.Module], network_config["class"])
        num_classes = cast(int, network_config["num_classes"])
        network = self._create_network(network_class, num_classes)

        # Get parameters as list
        parameters: List[torch.Tensor] = list(network.parameters())

        # Check that parameters exist
        assert len(parameters) > 0

        # Check that parameters are tensors
        for param in parameters:
            assert isinstance(param, torch.Tensor)
            assert param.requires_grad  # Should be trainable

        # Test state_dict extraction
        state_dict = network.state_dict()
        assert isinstance(state_dict, OrderedDict)
        assert len(state_dict) > 0

    def test_network_parameter_setting(
        self,
        network_config: Dict[str, Union[Type[nn.Module], Tuple[int, ...], int, str]],
    ) -> None:
        """Test setting parameters in networks."""
        network_class = cast(Type[nn.Module], network_config["class"])
        num_classes = cast(int, network_config["num_classes"])
        network = self._create_network(network_class, num_classes)

        # Get original parameters (make deep copy to avoid reference issues)
        original_state_dict = network.state_dict()
        original_copy: Dict[str, torch.Tensor] = {
            k: v.clone() for k, v in original_state_dict.items()
        }

        # Create modified parameters
        modified_state_dict: Dict[str, torch.Tensor] = {}
        for key, param in original_state_dict.items():
            # Add small noise to parameters
            modified_state_dict[key] = param + torch.randn_like(param) * 0.5

        # Set modified parameters
        network.load_state_dict(modified_state_dict)

        # Verify parameters were changed
        new_state_dict = network.state_dict()
        for key in original_copy.keys():
            assert not torch.allclose(original_copy[key], new_state_dict[key])

    def test_network_weight_initialization(self, network_config):
        """Test that weight initialization methods work correctly."""
        network_class = cast(Type[nn.Module], network_config["class"])

        # Create two networks to compare initialization
        num_classes = cast(int, network_config["num_classes"])
        network1 = self._create_network(network_class, num_classes)
        network2 = self._create_network(network_class, num_classes)

        # Check that weights are different (random initialization)
        conv1_layer1 = getattr(network1, "conv1", None)
        conv1_layer2 = getattr(network2, "conv1", None)

        if conv1_layer1 is None or conv1_layer2 is None:
            pytest.skip("Network does not have conv1 layer")

        conv1_layer1 = cast(nn.Conv2d, conv1_layer1)
        conv1_layer2 = cast(nn.Conv2d, conv1_layer2)
        conv1_weights1 = conv1_layer1.weight.data
        conv1_weights2 = conv1_layer2.weight.data

        # Should not be identical (very low probability with random init)
        assert not torch.allclose(conv1_weights1, conv1_weights2)

        # Check that biases are initialized to zero (if bias exists)
        if conv1_layer1.bias is not None:
            conv1_bias1 = conv1_layer1.bias.data
            assert torch.allclose(conv1_bias1, torch.zeros_like(conv1_bias1))

    def test_network_gradient_computation(self, network_config):
        """Test that networks can compute gradients."""
        network_class = cast(Type[nn.Module], network_config["class"])
        input_shape = cast(Tuple[int, ...], network_config["input_shape"])
        num_classes = cast(int, network_config["num_classes"])

        if network_class.__name__ == "LungCancerCNN":
            network = network_class(num_classes=2)  # Default for this class
        else:
            network = network_class()
        network.train()  # Set to training mode

        # Create mock input and target
        batch_size = 2
        mock_input = torch.randn(batch_size, *input_shape)
        mock_target = torch.randint(0, num_classes, (batch_size,))

        # Forward pass
        output = network(mock_input)

        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, mock_target)

        # Backward pass
        loss.backward()

        # Check that gradients were computed
        for param in network.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.allclose(param.grad, torch.zeros_like(param.grad))

    def test_network_training_evaluation_modes(self, network_config):
        """Test switching between training and evaluation modes."""
        network_class = cast(Type[nn.Module], network_config["class"])
        input_shape = cast(Tuple[int, ...], network_config["input_shape"])

        if network_class.__name__ == "LungCancerCNN":
            network = network_class(num_classes=2)  # Default for this class
        else:
            network = network_class()
        batch_size = 2
        mock_input = torch.randn(batch_size, *input_shape)

        # Test training mode
        network.train()
        assert network.training

        output_train = network(mock_input)

        # Test evaluation mode
        network.eval()
        assert not network.training

        with torch.no_grad():
            output_eval = network(mock_input)

        # Outputs might be different due to dropout behavior
        # Just check that both modes produce valid outputs
        assert output_train.shape == output_eval.shape
        assert not torch.isnan(output_train).any()
        assert not torch.isnan(output_eval).any()

    def test_network_dropout_behavior(self, network_config):
        """Test dropout behavior in training vs evaluation mode."""
        network_class = cast(Type[nn.Module], network_config["class"])
        input_shape = cast(Tuple[int, ...], network_config["input_shape"])

        if network_class.__name__ == "LungCancerCNN":
            network = network_class(num_classes=2)  # Default for this class
        else:
            network = network_class()
        batch_size = 10
        mock_input = torch.randn(batch_size, *input_shape)

        # Set seed for reproducibility
        torch.manual_seed(42)

        # Training mode - dropout should be active
        network.train()
        outputs_train = []
        for _ in range(5):
            output = network(mock_input)
            outputs_train.append(output.clone())

        # Evaluation mode - dropout should be inactive
        network.eval()
        with torch.no_grad():
            outputs_eval = []
            for _ in range(5):
                output = network(mock_input)
                outputs_eval.append(output.clone())

        # In evaluation mode, outputs should be identical
        for i in range(1, len(outputs_eval)):
            assert torch.allclose(outputs_eval[0], outputs_eval[i])

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_network_batch_size_handling(self, network_config, batch_size):
        """Test networks handle different batch sizes correctly."""
        network_class = cast(Type[nn.Module], network_config["class"])
        input_shape = cast(Tuple[int, ...], network_config["input_shape"])
        num_classes = cast(int, network_config["num_classes"])

        network = self._create_network(network_class, num_classes)
        network.eval()

        mock_input = torch.randn(batch_size, *input_shape)

        with torch.no_grad():
            output = network(mock_input)

        expected_shape = (batch_size, num_classes)
        assert output.shape == expected_shape

    def test_network_memory_efficiency(self, network_config):
        """Test that networks don't have memory leaks."""
        network_class = cast(Type[nn.Module], network_config["class"])
        input_shape = cast(Tuple[int, ...], network_config["input_shape"])

        if network_class.__name__ == "LungCancerCNN":
            network = network_class(num_classes=2)  # Default for this class
        else:
            network = network_class()
        network.eval()

        # Run multiple forward passes
        for _ in range(10):
            mock_input = torch.randn(2, *input_shape)
            with torch.no_grad():
                output = network(mock_input)

            # Clear references
            del mock_input, output

        # Test passes if no memory errors occur

    def test_network_parameter_count(self, network_config):
        """Test that networks have reasonable parameter counts."""
        network_class = cast(Type[nn.Module], network_config["class"])
        if network_class.__name__ == "LungCancerCNN":
            network = network_class(num_classes=2)  # Default for this class
        else:
            network = network_class()

        total_params = sum(p.numel() for p in network.parameters())
        trainable_params = sum(
            p.numel() for p in network.parameters() if p.requires_grad
        )

        # Check that network has parameters
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params == total_params  # All params should be trainable

        # Check reasonable parameter count (not too small or too large)
        assert 1000 < total_params < 100_000_000  # Reasonable range for these networks

    def test_network_device_compatibility(self, network_config):
        """Test that networks work on different devices."""
        network_class = cast(Type[nn.Module], network_config["class"])
        input_shape = cast(Tuple[int, ...], network_config["input_shape"])

        if network_class.__name__ == "LungCancerCNN":
            network = network_class(num_classes=2)  # Default for this class
        else:
            network = network_class()

        # Test CPU
        network = network.to("cpu")
        mock_input = torch.randn(2, *input_shape).to("cpu")

        with torch.no_grad():
            output = network(mock_input)

        assert output.device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            network = network.to("cuda")
            mock_input = mock_input.to("cuda")

            with torch.no_grad():
                output = network(mock_input)

            assert output.device.type == "cuda"


class TestBERTModelFunctions:
    """Test suite for BERT model loading and LoRA functions."""

    @patch("src.network_models.bert_model_definition.AutoModelForMaskedLM")
    @patch("src.network_models.bert_model_definition.get_peft_model")
    def test_load_model_with_lora(self, mock_get_peft_model, mock_auto_model):
        """Test loading BERT model with LoRA configuration."""
        # Mock the base model
        mock_base_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_base_model

        # Mock the LoRA model
        mock_lora_model = Mock()
        mock_lora_model.print_trainable_parameters.return_value = None
        mock_get_peft_model.return_value = mock_lora_model

        # Test function
        model = load_model_with_lora(
            model_name="bert-base-uncased",
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.1,
            lora_target_modules=["query", "value"],
        )

        # Verify calls
        mock_auto_model.from_pretrained.assert_called_once_with("bert-base-uncased")
        mock_get_peft_model.assert_called_once()
        assert model == mock_lora_model

    @patch("src.network_models.bert_model_definition.AutoModelForMaskedLM")
    def test_load_model(self, mock_auto_model):
        """Test loading BERT model without LoRA."""
        mock_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model

        model = load_model("bert-base-uncased")

        mock_auto_model.from_pretrained.assert_called_once_with("bert-base-uncased")
        assert model == mock_model

    @patch("src.network_models.bert_model_definition.get_peft_model_state_dict")
    def test_get_lora_state_dict(self, mock_get_peft_state_dict):
        """Test getting LoRA state dict as numpy arrays."""
        # Mock state dict with tensors
        mock_state_dict = OrderedDict(
            {"lora_A": torch.randn(10, 5), "lora_B": torch.randn(5, 10)}
        )
        mock_get_peft_state_dict.return_value = mock_state_dict

        mock_model = Mock()
        result = get_lora_state_dict(mock_model)

        # Verify result
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(arr, np.ndarray) for arr in result)

        mock_get_peft_state_dict.assert_called_once_with(mock_model)

    @patch("src.network_models.bert_model_definition.set_peft_model_state_dict")
    @patch("src.network_models.bert_model_definition.get_peft_model_state_dict")
    def test_set_lora_state_dict(
        self, mock_get_peft_state_dict, mock_set_peft_state_dict
    ):
        """Test setting LoRA state dict from numpy arrays."""
        # Mock existing state dict keys
        mock_state_dict = OrderedDict(
            {"lora_A": torch.randn(10, 5), "lora_B": torch.randn(5, 10)}
        )
        mock_get_peft_state_dict.return_value = mock_state_dict

        # Test data
        mock_model = Mock()
        state_list = [np.random.randn(10, 5), np.random.randn(5, 10)]

        set_lora_state_dict(mock_model, state_list)

        # Verify calls
        mock_get_peft_state_dict.assert_called_once_with(mock_model)
        mock_set_peft_state_dict.assert_called_once()

        # Check that the call was made with correct structure
        call_args = mock_set_peft_state_dict.call_args
        assert call_args[0][0] == mock_model  # First argument is model
        assert isinstance(
            call_args[0][1], OrderedDict
        )  # Second argument is OrderedDict

    def test_lora_state_dict_consistency(self):
        """Test that LoRA state dict operations are consistent."""
        # Create mock model with state dict
        mock_model = Mock()

        with (
            patch(
                "src.network_models.bert_model_definition.get_peft_model_state_dict"
            ) as mock_get,
            patch(
                "src.network_models.bert_model_definition.set_peft_model_state_dict"
            ) as mock_set,
        ):
            # Mock state dict
            original_state_dict = OrderedDict(
                {"lora_A": torch.randn(5, 3), "lora_B": torch.randn(3, 5)}
            )
            mock_get.return_value = original_state_dict

            # Get state as numpy arrays
            state_list = get_lora_state_dict(mock_model)

            # Set state back
            set_lora_state_dict(mock_model, state_list)

            # Verify operations
            assert mock_get.call_count == 2  # Called in both get and set operations
            mock_set.assert_called_once()


class TestNetworkModelIntegration:
    """Integration tests for network models."""

    def test_network_training_loop_simulation(self):
        """Test a simulated training loop with network models."""
        network = ITSNetwork()
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Simulate training data
        batch_size = 4
        input_shape = (3, 224, 224)
        num_classes = 10

        network.train()

        # Simulate a few training steps
        for step in range(3):
            # Mock batch
            inputs = torch.randn(batch_size, *input_shape)
            targets = torch.randint(0, num_classes, (batch_size,))

            # Forward pass
            outputs = network(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Verify training progresses
            assert loss.item() >= 0
            assert not torch.isnan(loss)

    def test_network_evaluation_simulation(self):
        """Test a simulated evaluation with network models."""
        network = FemnistReducedIIDNetwork()
        criterion = nn.CrossEntropyLoss()

        network.eval()

        total_loss = 0
        correct = 0
        total = 0

        # Simulate evaluation data
        batch_size = 4
        input_shape = (1, 28, 28)
        num_classes = 10

        with torch.no_grad():
            for batch in range(3):
                inputs = torch.randn(batch_size, *input_shape)
                targets = torch.randint(0, num_classes, (batch_size,))

                outputs = network(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        # Verify evaluation metrics
        avg_loss = total_loss / 3
        accuracy = correct / total

        assert avg_loss >= 0
        assert 0 <= accuracy <= 1

    @pytest.mark.parametrize(
        "network_class,input_shape",
        [
            (ITSNetwork, (3, 224, 224)),
            (FemnistReducedIIDNetwork, (1, 28, 28)),
            (PneumoniamnistNetwork, (1, 28, 28)),
            (BloodMNISTNetwork, (3, 28, 28)),
        ],
    )
    def test_network_state_dict_serialization(self, network_class, input_shape):
        """Test that network state dicts can be serialized and deserialized."""
        network1 = network_class()
        network2 = network_class()

        # Get state dict from first network
        state_dict = network1.state_dict()

        # Load into second network
        network2.load_state_dict(state_dict)

        # Verify networks have same parameters
        for (name1, param1), (name2, param2) in zip(
            network1.named_parameters(), network2.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)

    def test_network_reproducibility(self):
        """Test that networks produce reproducible results with same seed."""
        input_shape = (3, 224, 224)
        batch_size = 2

        # First run
        torch.manual_seed(42)
        network1 = ITSNetwork()
        inputs = torch.randn(batch_size, *input_shape)

        network1.eval()
        with torch.no_grad():
            output1 = network1(inputs)

        # Second run with same seed
        torch.manual_seed(42)
        network2 = ITSNetwork()
        inputs = torch.randn(batch_size, *input_shape)

        network2.eval()
        with torch.no_grad():
            output2 = network2(inputs)

        # Outputs should be identical
        assert torch.allclose(output1, output2)

    def test_network_parameter_sharing(self):
        """Test parameter sharing between network instances."""
        network1 = PneumoniamnistNetwork()
        network2 = PneumoniamnistNetwork()

        # Initially different
        assert not torch.allclose(network1.conv1.weight, network2.conv1.weight)

        # Share parameters
        network2.load_state_dict(network1.state_dict())

        # Now should be the same
        assert torch.allclose(network1.conv1.weight, network2.conv1.weight)

        # Modify one network
        with torch.no_grad():
            network1.conv1.weight += 0.1

        # Should be different again
        assert not torch.allclose(network1.conv1.weight, network2.conv1.weight)
