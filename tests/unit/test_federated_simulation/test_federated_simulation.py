"""Unit tests for FederatedSimulation.

Focuses on high-level orchestration: property assignment, strategy
creation, and simulation run invocation.
"""

from unittest.mock import Mock, patch

import pytest
from src.data_models.simulation_strategy_config import StrategyConfig
from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.dataset_handlers.dataset_handler import DatasetHandler
from src.federated_simulation import FederatedSimulation


class TestFederatedSimulation:
    """Tests for FederatedSimulation high-level behavior."""

    @pytest.fixture
    def mock_strategy_config(self):
        """Return a StrategyConfig populated with reasonable test values."""
        return StrategyConfig(
            aggregation_strategy_keyword="trust",
            dataset_keyword="its",
            model_type="cnn",
            use_llm="false",
            num_of_rounds=3,
            num_of_clients=5,
            num_of_malicious_clients=1,
            training_subset_fraction=0.8,
            training_device="cpu",
            batch_size=32,
            num_of_client_epochs=2,
            min_fit_clients=5,
            min_evaluate_clients=5,
            min_available_clients=5,
            # Trust-specific parameters
            trust_threshold=0.7,
            beta_value=0.5,
            begin_removing_from_round=2,
            num_of_clusters=1,
        )

    @pytest.fixture
    def mock_output_directory(self, tmp_path):
        """Return a temporary directory path for test outputs."""
        output_dir = tmp_path / "test_output"
        output_dir.mkdir()
        return str(output_dir)

    @pytest.fixture
    def mock_dataset_handler(self):
        """Return a Mock implementing DatasetHandler interface used in tests."""
        handler = Mock(spec=DatasetHandler)
        handler.poisoned_client_ids = set()
        return handler

    @pytest.fixture
    def federated_simulation(
        self,
        mock_strategy_config,
        mock_output_directory,
        mock_dataset_handler,
    ):
        """Return a FederatedSimulation instance with property assignment patched.

        Patches _assign_all_properties to avoid heavy setup during construction.
        """
        with patch.object(FederatedSimulation, "_assign_all_properties"):
            return FederatedSimulation(
                mock_strategy_config, mock_output_directory, mock_dataset_handler
            )

    def test_initialization_with_strategy_config(
        self, mock_strategy_config, mock_dataset_handler, mock_output_directory
    ):
        """Test FederatedSimulation initialization with strategy config."""
        with patch.object(FederatedSimulation, "_assign_all_properties"):
            simulation = FederatedSimulation(
                mock_strategy_config, mock_output_directory, mock_dataset_handler
            )

            assert simulation.strategy_config == mock_strategy_config
            assert isinstance(simulation.strategy_history, SimulationStrategyHistory)
            assert simulation.strategy_history.strategy_config == mock_strategy_config

    def test_initialization_creates_strategy_history(self, federated_simulation):
        """Test initialization creates strategy history."""
        assert hasattr(federated_simulation, "strategy_history")
        assert federated_simulation.strategy_history is not None
        assert isinstance(
            federated_simulation.strategy_history, SimulationStrategyHistory
        )

    def _setup_mock_loaders_and_model(self, federated_simulation):
        mock_trainloader = Mock()
        mock_valloader = Mock()
        federated_simulation._trainloaders = [
            mock_trainloader
        ] * federated_simulation.strategy_config.num_of_clients
        federated_simulation._valloaders = [
            mock_valloader
        ] * federated_simulation.strategy_config.num_of_clients

        mock_model = Mock()
        mock_state = {
            "layer1.weight": Mock(
                cpu=Mock(return_value=Mock(numpy=Mock(return_value=[1.0])))
            ),
            "layer1.bias": Mock(
                cpu=Mock(return_value=Mock(numpy=Mock(return_value=[0.1])))
            ),
        }
        mock_model.state_dict.return_value = mock_state
        mock_model.to = Mock(return_value=mock_model)
        federated_simulation._network_model = mock_model

    def test_get_model_params(self):
        mock_model = Mock()
        mock_state = {
            "layer1.weight": Mock(
                cpu=Mock(return_value=Mock(numpy=Mock(return_value=[1.0])))
            ),
            "layer1.bias": Mock(
                cpu=Mock(return_value=Mock(numpy=Mock(return_value=[0.1])))
            ),
        }
        mock_model.state_dict.return_value = mock_state

        result = FederatedSimulation._get_model_params(mock_model)

        assert isinstance(result, list)
        assert len(result) == 2

    @patch("src.federated_simulation.flwr.simulation.start_simulation")
    def test_run_simulation_calls_start_simulation(
        self, mock_start_simulation, federated_simulation
    ):
        mock_start_simulation.return_value = Mock()
        federated_simulation._aggregation_strategy = Mock()
        federated_simulation.run_simulation()
        mock_start_simulation.assert_called_once()

    def test_assign_all_properties_calls_sub_methods(self, federated_simulation):
        """Test _assign_all_properties calls dataset, network, and strategy assignment."""
        with patch.object(
            federated_simulation, "_assign_dataset_loaders_and_network_model"
        ) as mock_assign_dataset, patch.object(
            federated_simulation, "_assign_aggregation_strategy"
        ) as mock_assign_strategy:
            federated_simulation._assign_all_properties()
            mock_assign_dataset.assert_called_once()
            mock_assign_strategy.assert_called_once()

    def test_assign_dataset_loaders_and_network_model_its_dataset(
        self, federated_simulation
    ):
        """Test assignment for ITS dataset."""
        federated_simulation.strategy_config.dataset_keyword = "its"
        with patch(
            "src.federated_simulation.ImageDatasetLoader"
        ) as mock_loader_class, patch(
            "src.federated_simulation.ITSNetwork"
        ) as mock_network_class:
            mock_loader, mock_network = Mock(), Mock()
            mock_trainloaders, mock_valloaders = [Mock()], [Mock()]
            mock_loader.load_datasets.return_value = (
                mock_trainloaders,
                mock_valloaders,
            )
            mock_loader_class.return_value = mock_loader
            mock_network_class.return_value = mock_network

            federated_simulation._assign_dataset_loaders_and_network_model()

            assert federated_simulation._network_model == mock_network
            assert federated_simulation._trainloaders == mock_trainloaders
            assert federated_simulation._valloaders == mock_valloaders
            mock_loader.load_datasets.assert_called_once()

    @patch("src.federated_simulation.ImageDatasetLoader")
    @patch("src.federated_simulation.FemnistReducedIIDNetwork")
    def test_assign_dataset_loaders_and_network_model_femnist_iid_dataset(
        self,
        mock_network_class,
        mock_loader_class,
        mock_strategy_config,
        mock_output_directory,
        mock_dataset_handler,
    ):
        """Test assignment for FEMNIST IID dataset."""
        mock_strategy_config.dataset_keyword = "femnist_iid"
        mock_loader, mock_network = Mock(), Mock()
        mock_loader.load_datasets.return_value = ([Mock()], [Mock()])

        # Mock network model with proper state_dict
        mock_state_dict = {"layer1.weight": Mock(), "layer1.bias": Mock()}
        for param in mock_state_dict.values():
            import numpy as np

            param.cpu.return_value.numpy.return_value = np.array([1.0, 2.0])
        mock_network.state_dict.return_value = mock_state_dict

        mock_loader_class.return_value = mock_loader
        mock_network_class.return_value = mock_network

        simulation = FederatedSimulation(
            mock_strategy_config, mock_output_directory, mock_dataset_handler
        )

        assert simulation._network_model == mock_network

    @patch("src.federated_simulation.ImageDatasetLoader")
    @patch("src.federated_simulation.FemnistFullNIIDNetwork")
    def test_assign_dataset_loaders_and_network_model_femnist_niid_dataset(
        self,
        mock_network_class,
        mock_loader_class,
        mock_strategy_config,
        mock_output_directory,
        mock_dataset_handler,
    ):
        """Test assignment for FEMNIST NIID dataset."""
        mock_strategy_config.dataset_keyword = "femnist_niid"
        mock_loader, mock_network = Mock(), Mock()
        mock_loader.load_datasets.return_value = ([Mock()], [Mock()])

        # Mock network model with proper state_dict
        mock_state_dict = {"layer1.weight": Mock(), "layer1.bias": Mock()}
        for param in mock_state_dict.values():
            import numpy as np

            param.cpu.return_value.numpy.return_value = np.array([1.0, 2.0])
        mock_network.state_dict.return_value = mock_state_dict

        mock_loader_class.return_value = mock_loader
        mock_network_class.return_value = mock_network

        simulation = FederatedSimulation(
            mock_strategy_config, mock_output_directory, mock_dataset_handler
        )

        assert simulation._network_model == mock_network

    @patch("src.federated_simulation.MedQuADDatasetLoader")
    @patch("src.federated_simulation.load_model")
    def test_assign_dataset_loaders_and_network_model_medquad_dataset(
        self,
        mock_load_model,
        mock_loader_class,
        mock_strategy_config,
        mock_output_directory,
        mock_dataset_handler,
    ):
        """Test assignment for MedQuAD dataset."""
        mock_strategy_config.dataset_keyword = "medquad"
        mock_strategy_config.model_type = "transformer"
        mock_strategy_config.use_llm = "true"
        mock_strategy_config.llm_model = "bert-base-uncased"
        mock_strategy_config.llm_chunk_size = 512
        mock_strategy_config.mlm_probability = 0.15
        mock_strategy_config.llm_finetuning = "full"
        mock_loader = Mock()
        mock_loader.load_datasets.return_value = ([Mock()], [Mock()])
        mock_loader_class.return_value = mock_loader

        # Mock the model returned by load_model with proper state_dict
        mock_model = Mock()
        mock_state_dict = {"layer1.weight": Mock(), "layer1.bias": Mock()}
        for param in mock_state_dict.values():
            import numpy as np

            param.cpu.return_value.numpy.return_value = np.array([1.0, 2.0])
        mock_model.state_dict.return_value = mock_state_dict
        mock_load_model.return_value = mock_model

        FederatedSimulation(
            mock_strategy_config, mock_output_directory, mock_dataset_handler
        )

        mock_loader_class.assert_called_once()
        mock_load_model.assert_called_once()

    @patch("src.federated_simulation.TrustBasedRemovalStrategy")
    def test_assign_aggregation_strategy_trust_strategy(
        self,
        mock_strategy_class,
        mock_strategy_config,
        mock_output_directory,
        mock_dataset_handler,
    ):
        """Test assignment for trust-based aggregation strategy."""
        mock_strategy_config.aggregation_strategy_keyword = "trust"

        # Mock the network model that would be assigned
        mock_network = Mock()
        mock_state_dict = {"layer1.weight": Mock(), "layer1.bias": Mock()}
        for param in mock_state_dict.values():
            import numpy as np

            param.cpu.return_value.numpy.return_value = np.array([1.0, 2.0])
        mock_network.state_dict.return_value = mock_state_dict

        with patch.object(
            FederatedSimulation, "_assign_dataset_loaders_and_network_model"
        ):
            with patch.object(
                FederatedSimulation, "_get_model_params"
            ) as mock_get_params:
                import numpy as np

                mock_get_params.return_value = [np.array([1.0, 2.0])]
                FederatedSimulation(
                    mock_strategy_config, mock_output_directory, mock_dataset_handler
                )
                mock_strategy_class.assert_called_once()

    def test_client_fn_creates_flower_client(self, federated_simulation):
        self._setup_mock_loaders_and_model(federated_simulation)

        with patch("src.federated_simulation.FlowerClient") as mock_flower_client:
            # Mock FlowerClient instance and its to_client method
            from flwr.client import Client

            mock_flower_instance = Mock()
            mock_flower_instance.to_client.return_value = Client()
            mock_flower_client.return_value = mock_flower_instance

            result = federated_simulation.client_fn("0")
            assert mock_flower_client.called
            assert isinstance(result, Client)

    def test_get_model_params_extracts_parameters(self):
        """Test _get_model_params extracts model parameters correctly."""
        mock_model = Mock()
        mock_model.state_dict.return_value = {
            "layer1.weight": Mock(),
            "layer1.bias": Mock(),
        }
        result = FederatedSimulation._get_model_params(mock_model)
        assert isinstance(result, list)
        mock_model.state_dict.assert_called_once()

    def test_get_model_params_handles_lora_model(self):
        """Test _get_model_params handles LORA models correctly."""
        with patch("src.federated_simulation.isinstance") as mock_isinstance:
            mock_isinstance.return_value = (
                True  # Make isinstance(model, PeftModel) return True
            )
            mock_model = Mock()
            with patch(
                "src.federated_simulation.get_peft_model_state_dict"
            ) as mock_get_peft:
                mock_state_dict = {"lora_layer.weight": Mock()}
                # Configure mock parameter to have proper cpu().numpy() chain
                import numpy as np

                mock_state_dict[
                    "lora_layer.weight"
                ].cpu.return_value.numpy.return_value = np.array([1.0, 2.0])
                mock_get_peft.return_value = mock_state_dict
                result = FederatedSimulation._get_model_params(mock_model)
                mock_get_peft.assert_called_once_with(mock_model)
                assert isinstance(result, list)

    @patch("src.federated_simulation.flwr.simulation.start_simulation")
    def test_run_simulation_passes_correct_parameters(
        self,
        mock_start_simulation,
        federated_simulation,
    ):
        """Test run_simulation passes correct parameters to Flower simulation."""
        mock_start_simulation.return_value = Mock()
        federated_simulation._assign_all_properties()
        federated_simulation.run_simulation()
        args, kwargs = mock_start_simulation.call_args
        assert "strategy" in kwargs
        assert kwargs["strategy"] == federated_simulation._aggregation_strategy
