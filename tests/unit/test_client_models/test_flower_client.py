"""
Unit tests for FlowerClient class.

Tests client parameter handling, model operations, training and evaluation
with mocked PyTorch operations and data.
"""

from collections import OrderedDict
from typing import Any, Dict
from unittest.mock import patch

import torch

from src.client_models.flower_client import FlowerClient
from tests.common import Mock, np, pytest
from tests.fixtures.sample_models import MockCNNNetwork

NDArray = np.ndarray
Config = Dict[str, Any]
Metrics = Dict[str, Any]


class TestFlowerClient:
    """Test FlowerClient class."""

    @pytest.fixture
    def mock_network(self):
        """Create mock network."""
        return MockCNNNetwork(num_classes=10, input_channels=3)

    @pytest.fixture
    def mock_trainloader(self):
        """Create a mock training data loader."""
        # Create mock data
        mock_data = [
            (torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))) for _ in range(5)
        ]

        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter(mock_data))
        mock_loader.__len__ = Mock(return_value=5)
        mock_loader.dataset = Mock()
        mock_loader.dataset.__len__ = Mock(return_value=20)  # 5 batches * 4 samples

        return mock_loader

    @pytest.fixture
    def mock_valloader(self):
        """Create a mock validation data loader."""
        # Create mock data
        mock_data = [
            (torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))) for _ in range(3)
        ]

        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter(mock_data))
        mock_loader.__len__ = Mock(return_value=3)
        mock_loader.dataset = Mock()
        mock_loader.dataset.__len__ = Mock(return_value=12)  # 3 batches * 4 samples

        return mock_loader

    @pytest.fixture
    def mock_transformer_trainloader(self):
        """Create a mock transformer training data loader."""
        mock_data = []
        for _ in range(3):
            batch = {
                "input_ids": torch.randint(0, 1000, (2, 128)),
                "attention_mask": torch.ones(2, 128),
                "labels": torch.randint(0, 10, (2, 128)),
            }
            mock_data.append(batch)

        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter(mock_data))
        mock_loader.__len__ = Mock(return_value=3)
        mock_loader.dataset = Mock()
        mock_loader.dataset.__len__ = Mock(return_value=6)

        return mock_loader

    @pytest.fixture
    def mock_transformer_valloader(self):
        """Create a mock transformer validation data loader."""
        mock_data = []
        for _ in range(2):
            batch = {
                "input_ids": torch.randint(0, 1000, (2, 128)),
                "attention_mask": torch.ones(2, 128),
                "labels": torch.randint(0, 10, (2, 128)),
            }
            mock_data.append(batch)

        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter(mock_data))
        mock_loader.__len__ = Mock(return_value=2)
        mock_loader.dataset = Mock()
        mock_loader.dataset.__len__ = Mock(return_value=4)

        return mock_loader

    @pytest.fixture
    def flower_client_cnn(self, mock_network, mock_trainloader, mock_valloader):
        """Create FlowerClient instance for CNN testing."""
        return FlowerClient(
            client_id=1,
            net=mock_network,
            trainloader=mock_trainloader,
            valloader=mock_valloader,
            training_device="cpu",
            num_of_client_epochs=2,
            model_type="cnn",
            use_lora=False,
            num_malicious_clients=0,
        )

    @pytest.fixture
    def flower_client_transformer(
        self, mock_transformer_trainloader, mock_transformer_valloader
    ):
        """Create FlowerClient instance for transformer testing."""
        mock_net = Mock()
        mock_net.parameters.return_value = [torch.randn(10, 5), torch.randn(10)]

        return FlowerClient(
            client_id=1,
            net=mock_net,
            trainloader=mock_transformer_trainloader,
            valloader=mock_transformer_valloader,
            training_device="cpu",
            num_of_client_epochs=2,
            model_type="transformer",
            use_lora=False,
            num_malicious_clients=0,
        )

    def test_init(self, mock_network, mock_trainloader, mock_valloader):
        """Test initialization."""
        client = FlowerClient(
            client_id=5,
            net=mock_network,
            trainloader=mock_trainloader,
            valloader=mock_valloader,
            training_device="cuda",
            num_of_client_epochs=3,
            model_type="cnn",
            use_lora=True,
            num_malicious_clients=2,
        )

        assert client.client_id == 5
        assert client.net == mock_network
        assert client.trainloader == mock_trainloader
        assert client.valloader == mock_valloader
        assert client.training_device == "cuda"
        assert client.model_type == "cnn"
        assert client.num_of_client_epochs == 3
        assert client.use_lora
        assert client.num_malicious_clients == 2

    def test_get_parameters_without_lora(self, flower_client_cnn):
        """Test get_parameters method without LoRA."""
        parameters = flower_client_cnn.get_parameters(config={})

        # Should return list of numpy arrays
        assert isinstance(parameters, list)
        assert all(isinstance(param, np.ndarray) for param in parameters)

        # Should match network parameters
        expected_params = [
            param.cpu().detach().numpy() for param in flower_client_cnn.net.parameters()
        ]
        assert len(parameters) == len(expected_params)

    @patch("src.client_models.flower_client.get_peft_model_state_dict")
    def test_get_parameters_with_lora(self, mock_get_peft, flower_client_cnn):
        """Test get_parameters method with LoRA."""
        flower_client_cnn.use_lora = True

        # Mock LoRA state dict
        mock_state_dict = OrderedDict(
            {"lora_A": torch.randn(10, 5), "lora_B": torch.randn(5, 10)}
        )
        mock_get_peft.return_value = mock_state_dict

        parameters = flower_client_cnn.get_parameters(config={})

        assert isinstance(parameters, list)
        assert len(parameters) == 2
        mock_get_peft.assert_called_once_with(flower_client_cnn.net)

    def test_set_parameters_without_lora(self, flower_client_cnn):
        """Test set_parameters method without LoRA."""
        # Get original parameters
        original_params = flower_client_cnn.get_parameters(config={})

        # Create new parameters with different values
        new_params = [param + 0.1 for param in original_params]

        # Set new parameters
        flower_client_cnn.set_parameters(flower_client_cnn.net, new_params)

        # Verify parameters were updated
        updated_params = flower_client_cnn.get_parameters(config={})
        for orig, new, updated in zip(original_params, new_params, updated_params):
            np.testing.assert_allclose(updated, new, rtol=1e-5)

    @patch("src.client_models.flower_client.set_peft_model_state_dict")
    @patch("src.client_models.flower_client.get_peft_model_state_dict")
    def test_set_parameters_with_lora(
        self, mock_get_peft, mock_set_peft, flower_client_cnn
    ):
        """Test set_parameters method with LoRA."""
        flower_client_cnn.use_lora = True

        # Mock LoRA state dict keys
        mock_get_peft.return_value = OrderedDict(
            {"lora_A": torch.randn(10, 5), "lora_B": torch.randn(5, 10)}
        )

        # Create new parameters
        new_params = [np.random.randn(10, 5), np.random.randn(5, 10)]

        flower_client_cnn.set_parameters(flower_client_cnn.net, new_params)

        mock_get_peft.assert_called_once_with(flower_client_cnn.net)
        mock_set_peft.assert_called_once()

    def test_train_cnn_model(self, flower_client_cnn):
        """Test training method for CNN model."""
        with patch("torch.optim.Adam") as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer

            # Mock the training loop
            flower_client_cnn.train(
                net=flower_client_cnn.net,
                trainloader=flower_client_cnn.trainloader,
                epochs=1,
                verbose=False,
            )

            # Verify optimizer was created and used
            mock_optimizer_class.assert_called_once()
            # Check that the optimizer was called with the network parameters
            call_args = mock_optimizer_class.call_args[0]
            assert len(list(call_args[0])) == len(
                list(flower_client_cnn.net.parameters())
            )
            assert mock_optimizer.zero_grad.call_count >= 1
            assert mock_optimizer.step.call_count >= 1

    def test_train_transformer_model(self, flower_client_transformer):
        """Test training method for transformer model."""
        # Mock transformer outputs
        mock_outputs = Mock()
        mock_outputs.loss = torch.tensor(0.5, requires_grad=True)
        mock_outputs.logits = torch.randn(2, 128, 10)

        flower_client_transformer.net.return_value = mock_outputs

        with patch("torch.optim.AdamW") as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer

            flower_client_transformer.train(
                net=flower_client_transformer.net,
                trainloader=flower_client_transformer.trainloader,
                epochs=1,
                verbose=False,
            )

            # Verify optimizer was created and used
            mock_optimizer_class.assert_called_once()
            assert mock_optimizer.zero_grad.call_count >= 1
            assert mock_optimizer.step.call_count >= 1

    def test_train_with_fedprox(self, flower_client_transformer):
        """Test training with FedProx regularization."""
        # Setup for FedProx
        flower_client_transformer.use_lora = True
        flower_client_transformer.client_id = 5  # >= num_malicious_clients

        # Mock transformer outputs
        mock_outputs = Mock()
        mock_outputs.loss = torch.tensor(0.5, requires_grad=True)
        mock_outputs.logits = torch.randn(2, 128, 10)

        flower_client_transformer.net.return_value = mock_outputs

        # Mock global parameters
        global_params = [torch.randn(10, 5), torch.randn(10)]

        with (
            patch("torch.optim.AdamW") as mock_optimizer_class,
            patch.object(
                flower_client_transformer, "get_parameters"
            ) as mock_get_params,
        ):
            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer
            mock_get_params.return_value = [param.numpy() for param in global_params]

            flower_client_transformer.train(
                net=flower_client_transformer.net,
                trainloader=flower_client_transformer.trainloader,
                epochs=1,
                verbose=False,
                global_params=global_params,
                mu=0.01,
            )

            # Verify training completed
            assert mock_optimizer.step.call_count >= 1

    def test_train_unsupported_model_type(self, flower_client_cnn):
        """Test training with unsupported model type raises error."""
        flower_client_cnn.model_type = "unsupported"

        with pytest.raises(ValueError, match="Unsupported model type: unsupported"):
            flower_client_cnn.train(
                net=flower_client_cnn.net,
                trainloader=flower_client_cnn.trainloader,
                epochs=1,
            )

    def test_test_cnn_model(self, flower_client_cnn):
        """Test evaluation method for CNN model."""
        loss, accuracy = flower_client_cnn.test(
            net=flower_client_cnn.net, testloader=flower_client_cnn.valloader
        )

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1

    def test_test_transformer_model(self, flower_client_transformer):
        """Test evaluation method for transformer model."""
        # Mock transformer outputs
        mock_outputs = Mock()
        mock_outputs.loss = torch.tensor(0.3)
        mock_outputs.logits = torch.randn(2, 128, 10)

        flower_client_transformer.net.return_value = mock_outputs

        loss, accuracy = flower_client_transformer.test(
            net=flower_client_transformer.net,
            testloader=flower_client_transformer.valloader,
        )

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1

    def test_test_unsupported_model_type(self, flower_client_cnn):
        """Test evaluation with unsupported model type raises error."""
        flower_client_cnn.model_type = "unsupported"

        with pytest.raises(ValueError, match="Unsupported model type: unsupported"):
            flower_client_cnn.test(
                net=flower_client_cnn.net, testloader=flower_client_cnn.valloader
            )

    def test_fit_method(self, flower_client_cnn):
        """Test fit method integrates training and parameter handling."""
        # Get initial parameters
        initial_params = flower_client_cnn.get_parameters(config={})

        # Create new parameters
        new_params = [param + 0.1 for param in initial_params]

        with patch.object(flower_client_cnn, "train") as mock_train:
            mock_train.return_value = (0.5, 0.8)
            result_params, dataset_size, metrics = flower_client_cnn.fit(
                new_params, config={}
            )

            # Verify training was called
            mock_train.assert_called_once()

            # Verify return values
            assert isinstance(result_params, list)
            assert isinstance(dataset_size, int)
            assert isinstance(metrics, dict)
            assert dataset_size == len(flower_client_cnn.trainloader.dataset)

    def test_fit_method_transformer_with_lora(self, flower_client_transformer):
        """Test fit method for transformer with LoRA."""
        flower_client_transformer.use_lora = True
        flower_client_transformer.client_id = 5  # >= num_malicious_clients

        # Mock parameters
        mock_params = [np.random.randn(10, 5), np.random.randn(10)]

        with (
            patch.object(
                flower_client_transformer, "get_parameters"
            ) as mock_get_params,
            patch.object(
                flower_client_transformer, "set_parameters"
            ) as mock_set_params,
            patch.object(flower_client_transformer, "train") as mock_train,
        ):
            mock_get_params.return_value = mock_params
            mock_train.return_value = (0.3, 0.9)

            result_params, dataset_size, metrics = flower_client_transformer.fit(
                mock_params, config={}
            )

            # Verify methods were called
            mock_set_params.assert_called_once_with(
                flower_client_transformer.net, mock_params
            )
            mock_train.assert_called_once()

            # Verify return values
            assert isinstance(result_params, list)
            assert isinstance(dataset_size, int)
            assert isinstance(metrics, dict)

    def test_fit_unsupported_model_type_in_gradient_calculation(
        self, flower_client_cnn
    ):
        """Test fit method handles unsupported model type in gradient calculation."""
        flower_client_cnn.model_type = "unsupported"
        initial_params = flower_client_cnn.get_parameters(config={})

        with pytest.raises(ValueError, match="Unsupported model type: unsupported"):
            flower_client_cnn.fit(initial_params, config={})

    def test_evaluate_method(self, flower_client_cnn):
        """Test evaluate method."""
        # Get initial parameters
        initial_params = flower_client_cnn.get_parameters(config={})

        with patch.object(flower_client_cnn, "test") as mock_test:
            mock_test.return_value = (0.5, 0.85)

            loss, dataset_size, metrics = flower_client_cnn.evaluate(
                initial_params, config={}
            )

            # Verify test was called
            mock_test.assert_called_once_with(
                flower_client_cnn.net, flower_client_cnn.valloader
            )

            # Verify return values
            assert isinstance(loss, float)
            assert isinstance(dataset_size, int)
            assert isinstance(metrics, dict)
            assert "accuracy" in metrics
            assert dataset_size == len(flower_client_cnn.valloader.dataset)

    @pytest.mark.parametrize(
        "model_type,expected_criterion",
        [
            ("cnn", "CrossEntropyLoss"),
            ("transformer", None),  # No criterion for transformer
        ],
    )
    def test_training_uses_correct_loss_function(
        self,
        model_type,
        expected_criterion,
        mock_network,
        mock_trainloader,
        mock_valloader,
    ):
        """Test that training uses correct loss function for different model types."""
        if model_type == "transformer":
            # Use mock network for transformer
            mock_net = Mock()
            mock_outputs = Mock()
            mock_outputs.loss = torch.tensor(0.5, requires_grad=True)
            mock_outputs.logits = torch.randn(2, 128, 10)
            mock_net.return_value = mock_outputs
            # Create mock transformer trainloader
            mock_data = []
            for _ in range(3):
                batch = {
                    "input_ids": torch.randint(0, 1000, (2, 128)),
                    "attention_mask": torch.ones(2, 128),
                    "labels": torch.randint(0, 10, (2, 128)),
                }
                mock_data.append(batch)

            trainloader = Mock()
            trainloader.__iter__ = Mock(return_value=iter(mock_data))
            trainloader.__len__ = Mock(return_value=3)
            trainloader.dataset = Mock()
            trainloader.dataset.__len__ = Mock(return_value=6)
        else:
            mock_net = mock_network
            trainloader = mock_trainloader

        client = FlowerClient(
            client_id=1,
            net=mock_net,
            trainloader=trainloader,
            valloader=mock_valloader,
            training_device="cpu",
            num_of_client_epochs=1,
            model_type=model_type,
        )

        if model_type == "cnn":
            with patch("torch.nn.CrossEntropyLoss") as mock_criterion:
                mock_criterion.return_value = Mock(
                    return_value=torch.tensor(0.5, requires_grad=True)
                )
                with patch("torch.optim.Adam"):
                    client.train(mock_net, trainloader, epochs=1)
                mock_criterion.assert_called()
        else:
            with patch("torch.optim.AdamW"):
                client.train(mock_net, trainloader, epochs=1)

    def test_malicious_client_behavior(self, flower_client_transformer):
        """Test that malicious clients don't use FedProx regularization."""
        flower_client_transformer.use_lora = True
        flower_client_transformer.client_id = 0  # < num_malicious_clients
        flower_client_transformer.num_malicious_clients = 2

        # Mock transformer outputs
        mock_outputs = Mock()
        mock_outputs.loss = torch.tensor(0.5, requires_grad=True)
        mock_outputs.logits = torch.randn(2, 128, 10)

        flower_client_transformer.net.return_value = mock_outputs

        # Mock global parameters
        global_params = [torch.randn(10, 5), torch.randn(10)]

        with patch("torch.optim.AdamW") as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer

            flower_client_transformer.train(
                net=flower_client_transformer.net,
                trainloader=flower_client_transformer.trainloader,
                epochs=1,
                verbose=False,
                global_params=global_params,
                mu=0.01,
            )

            # Training should complete without FedProx regularization
            assert mock_optimizer.step.call_count >= 1

    def test_verbose_training_output(self, flower_client_cnn, capsys):
        """Test verbose training produces output."""
        with patch("torch.optim.Adam"):
            flower_client_cnn.train(
                net=flower_client_cnn.net,
                trainloader=flower_client_cnn.trainloader,
                epochs=1,
                verbose=True,
            )

        captured = capsys.readouterr()
        assert "Epoch 1:" in captured.out
        assert "train loss" in captured.out
        assert "accuracy" in captured.out

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_handling(self, mock_network, mock_trainloader, mock_valloader):
        """Test that client handles different devices correctly."""
        client = FlowerClient(
            client_id=1,
            net=mock_network,
            trainloader=mock_trainloader,
            valloader=mock_valloader,
            training_device="cuda",
            num_of_client_epochs=1,
            model_type="cnn",
        )

        assert client.training_device == "cuda"

        # Test that device is used in training (mocked)
        with patch("torch.optim.Adam"):
            client.train(mock_network, mock_trainloader, epochs=1)

    def test_parameter_consistency(self, flower_client_cnn):
        """Test that parameters remain consistent through get/set operations."""
        # Get initial parameters
        initial_params = flower_client_cnn.get_parameters(config={})

        # Set the same parameters back
        flower_client_cnn.set_parameters(flower_client_cnn.net, initial_params)

        # Get parameters again
        final_params = flower_client_cnn.get_parameters(config={})

        # Should be the same (within floating point precision)
        for initial, final in zip(initial_params, final_params):
            np.testing.assert_allclose(initial, final, rtol=1e-5)

    def test_empty_dataloader_handling(self, flower_client_cnn):
        """Test handling of empty dataloaders."""
        # Create empty dataloader
        empty_loader = Mock()
        empty_loader.__iter__ = Mock(return_value=iter([]))
        empty_loader.__len__ = Mock(return_value=0)
        empty_loader.dataset = Mock()
        empty_loader.dataset.__len__ = Mock(return_value=0)

        # Training with empty loader should not crash
        with patch("torch.optim.Adam"):
            flower_client_cnn.train(flower_client_cnn.net, empty_loader, epochs=1)

        # Evaluation with empty loader should return reasonable values
        loss, accuracy = flower_client_cnn.test(flower_client_cnn.net, empty_loader)
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)

    def test_verbose_training_output_transformer(
        self, flower_client_transformer, capsys
    ):
        """Test verbose training produces output for transformer model."""
        # Mock transformer outputs
        mock_outputs = Mock()
        mock_outputs.loss = torch.tensor(0.5, requires_grad=True)
        mock_outputs.logits = torch.randn(2, 128, 10)

        flower_client_transformer.net.return_value = mock_outputs

        with patch("torch.optim.AdamW") as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer

            flower_client_transformer.train(
                net=flower_client_transformer.net,
                trainloader=flower_client_transformer.trainloader,
                epochs=1,
                verbose=True,
            )

        captured = capsys.readouterr()
        assert "Epoch 1:" in captured.out
        assert "train loss" in captured.out
        assert "accuracy" in captured.out
