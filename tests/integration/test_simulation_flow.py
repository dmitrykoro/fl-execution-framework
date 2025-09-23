"""
Integration tests for FederatedSimulation class.

Tests simulation initialization, component assignment, and execution workflows
with mocked dependencies.
"""

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import numpy as np
import pytest
from src.data_models.simulation_strategy_config import StrategyConfig
from src.federated_simulation import FederatedSimulation

from tests.fixtures.mock_datasets import MockDatasetHandler
from tests.fixtures.sample_models import MockCNNNetwork, MockNetwork

NDArray = np.ndarray


def _get_base_strategy_config() -> Dict[str, Any]:
    """Return base strategy configuration for testing."""
    return {
        "aggregation_strategy_keyword": "trust",
        "dataset_keyword": "its",
        "num_of_rounds": 3,
        "num_of_clients": 5,
        "num_of_malicious_clients": 1,
        "trust_threshold": 0.7,
        "beta_value": 0.5,
        "begin_removing_from_round": 1,
        "remove_clients": True,
        "min_fit_clients": 3,
        "min_evaluate_clients": 3,
        "min_available_clients": 5,
        "evaluate_metrics_aggregation_fn": "weighted_average",
        "training_device": "cpu",
        "cpus_per_client": 1,
        "gpus_per_client": 0.0,
        "batch_size": 32,
        "num_of_client_epochs": 1,
        "training_subset_fraction": 1.0,
        "model_type": "cnn",
        "use_llm": False,
        "llm_finetuning": None,
    }


class TestFederatedSimulationInitialization:
    """Test FederatedSimulation initialization."""

    @pytest.fixture
    def mock_dataset_handler(self) -> MockDatasetHandler:
        """Create mock dataset handler."""
        handler = MockDatasetHandler(dataset_type="its")
        handler.setup_dataset(num_clients=5)
        return handler

    @pytest.fixture
    def temp_dataset_dir(self, tmp_path: Path) -> str:
        """Create temporary dataset directory."""
        dataset_dir = tmp_path / "datasets" / "its"
        dataset_dir.mkdir(parents=True)
        return str(dataset_dir)

    def test_simulation_initialization_with_trust_strategy(
        self, temp_dataset_dir: str, mock_dataset_handler: MockDatasetHandler
    ) -> None:
        """Test initialization with trust strategy."""
        # Arrange
        base_config = _get_base_strategy_config()
        strategy_config = StrategyConfig.from_dict(base_config)

        # Act
        with (
            patch("src.federated_simulation.ImageDatasetLoader") as mock_loader,
            patch("src.federated_simulation.ITSNetwork") as mock_network,
        ):
            # Mock dataset loader
            mock_loader_instance = Mock()
            mock_loader_instance.load_datasets.return_value = (
                [Mock() for _ in range(5)],  # trainloaders
                [Mock() for _ in range(5)],  # valloaders
            )
            mock_loader.return_value = mock_loader_instance

            # Mock network
            mock_network_instance = MockNetwork()
            mock_network.return_value = mock_network_instance

            simulation = FederatedSimulation(
                strategy_config=strategy_config,
                dataset_dir=temp_dataset_dir,
                dataset_handler=mock_dataset_handler,
            )

        # Assert
        assert simulation.strategy_config == strategy_config
        assert simulation._dataset_dir == temp_dataset_dir
        assert simulation.dataset_handler == mock_dataset_handler
        assert simulation._network_model is not None
        assert simulation._aggregation_strategy is not None
        assert simulation._trainloaders is not None
        assert simulation._valloaders is not None
        assert len(simulation._trainloaders) == 5
        assert len(simulation._valloaders) == 5

    def test_simulation_initialization_with_pid_strategy(
        self, temp_dataset_dir: str, mock_dataset_handler: MockDatasetHandler
    ) -> None:
        """Test initialization with PID strategy."""
        # Arrange
        pid_config = _get_base_strategy_config()
        pid_config.update(
            {
                "aggregation_strategy_keyword": "pid",
                "Kp": 1.0,
                "Ki": 0.1,
                "Kd": 0.01,
                "num_std_dev": 2.0,
            }
        )
        strategy_config = StrategyConfig.from_dict(pid_config)

        # Act
        with (
            patch("src.federated_simulation.ImageDatasetLoader") as mock_loader,
            patch("src.federated_simulation.ITSNetwork") as mock_network,
        ):
            mock_loader_instance = Mock()
            mock_loader_instance.load_datasets.return_value = (
                [Mock() for _ in range(5)],
                [Mock() for _ in range(5)],
            )
            mock_loader.return_value = mock_loader_instance

            mock_network_instance = MockNetwork()
            mock_network.return_value = mock_network_instance

            simulation = FederatedSimulation(
                strategy_config=strategy_config,
                dataset_dir=temp_dataset_dir,
                dataset_handler=mock_dataset_handler,
            )

        # Assert
        assert simulation.strategy_config.aggregation_strategy_keyword == "pid"
        assert simulation._aggregation_strategy is not None
        # Verify PID-specific attributes are set
        assert hasattr(simulation._aggregation_strategy, "ki")
        assert hasattr(simulation._aggregation_strategy, "kp")
        assert hasattr(simulation._aggregation_strategy, "kd")

    def test_simulation_initialization_with_krum_strategy(
        self, temp_dataset_dir: str, mock_dataset_handler: MockDatasetHandler
    ) -> None:
        """Test initialization with Krum strategy."""
        # Arrange
        krum_config = _get_base_strategy_config()
        krum_config.update(
            {"aggregation_strategy_keyword": "krum", "num_krum_selections": 3}
        )
        strategy_config = StrategyConfig.from_dict(krum_config)

        # Act
        with (
            patch("src.federated_simulation.ImageDatasetLoader") as mock_loader,
            patch("src.federated_simulation.ITSNetwork") as mock_network,
        ):
            mock_loader_instance = Mock()
            mock_loader_instance.load_datasets.return_value = (
                [Mock() for _ in range(5)],
                [Mock() for _ in range(5)],
            )
            mock_loader.return_value = mock_loader_instance

            mock_network_instance = MockNetwork()
            mock_network.return_value = mock_network_instance

            simulation = FederatedSimulation(
                strategy_config=strategy_config,
                dataset_dir=temp_dataset_dir,
                dataset_handler=mock_dataset_handler,
            )

        # Assert
        assert simulation.strategy_config.aggregation_strategy_keyword == "krum"
        assert simulation._aggregation_strategy is not None
        assert hasattr(simulation._aggregation_strategy, "num_krum_selections")

    @pytest.mark.parametrize(
        "dataset_type,expected_network",
        [
            ("its", "ITSNetwork"),
            ("femnist_iid", "FemnistReducedIIDNetwork"),
            ("femnist_niid", "FemnistFullNIIDNetwork"),
            ("flair", "FlairNetwork"),
            ("pneumoniamnist", "PneumoniamnistNetwork"),
            ("bloodmnist", "BloodmnistNetwork"),
            ("lung_photos", "LungCancerCNN"),
        ],
    )
    def test_dataset_and_network_assignment(
        self,
        temp_dataset_dir: str,
        mock_dataset_handler: MockDatasetHandler,
        dataset_type: str,
        expected_network: str,
    ) -> None:
        """Test dataset loader and network assignment for different dataset types."""
        # Arrange
        config = _get_base_strategy_config()
        config["dataset_keyword"] = dataset_type
        strategy_config = StrategyConfig.from_dict(config)

        # Act
        with (
            patch("src.federated_simulation.ImageDatasetLoader") as mock_loader,
            patch(f"src.federated_simulation.{expected_network}") as mock_network,
        ):
            mock_loader_instance = Mock()
            mock_loader_instance.load_datasets.return_value = (
                [Mock() for _ in range(5)],
                [Mock() for _ in range(5)],
            )
            mock_loader.return_value = mock_loader_instance

            mock_network_instance = MockCNNNetwork()
            mock_network.return_value = mock_network_instance

            simulation = FederatedSimulation(
                strategy_config=strategy_config,
                dataset_dir=temp_dataset_dir,
                dataset_handler=mock_dataset_handler,
            )

        # Assert
        mock_loader.assert_called_once()
        mock_network.assert_called_once()
        assert simulation._network_model is not None

    def test_simulation_initialization_with_invalid_strategy(
        self, temp_dataset_dir: str, mock_dataset_handler: MockDatasetHandler
    ) -> None:
        """Test FederatedSimulation initialization with invalid strategy raises error."""
        # Arrange
        invalid_config = _get_base_strategy_config()
        invalid_config["aggregation_strategy_keyword"] = "invalid_strategy"
        strategy_config = StrategyConfig.from_dict(invalid_config)

        # Act & Assert
        with (
            patch("src.federated_simulation.ImageDatasetLoader") as mock_loader,
            patch("src.federated_simulation.ITSNetwork") as mock_network,
        ):
            mock_loader_instance = Mock()
            mock_loader_instance.load_datasets.return_value = (
                [Mock() for _ in range(5)],
                [Mock() for _ in range(5)],
            )
            mock_loader.return_value = mock_loader_instance
            mock_network.return_value = MockNetwork()

            with pytest.raises(
                NotImplementedError,
                match="The strategy invalid_strategy not implemented",
            ):
                FederatedSimulation(
                    strategy_config=strategy_config,
                    dataset_dir=temp_dataset_dir,
                    dataset_handler=mock_dataset_handler,
                )

    def test_simulation_strategy_history_initialization(
        self, temp_dataset_dir: str, mock_dataset_handler: MockDatasetHandler
    ) -> None:
        """Test that SimulationStrategyHistory is properly initialized."""
        # Arrange
        base_config = _get_base_strategy_config()
        strategy_config = StrategyConfig.from_dict(base_config)

        # Act
        with (
            patch("src.federated_simulation.ImageDatasetLoader") as mock_loader,
            patch("src.federated_simulation.ITSNetwork") as mock_network,
        ):
            mock_loader_instance = Mock()
            mock_loader_instance.load_datasets.return_value = (
                [Mock() for _ in range(5)],
                [Mock() for _ in range(5)],
            )
            mock_loader.return_value = mock_loader_instance
            mock_network.return_value = MockNetwork()

            simulation = FederatedSimulation(
                strategy_config=strategy_config,
                dataset_dir=temp_dataset_dir,
                dataset_handler=mock_dataset_handler,
            )

        # Assert
        assert simulation.strategy_history is not None
        assert simulation.strategy_history.strategy_config == strategy_config
        assert simulation.strategy_history.dataset_handler == mock_dataset_handler
        assert simulation.strategy_history.rounds_history is not None


class TestFederatedSimulationExecution:
    """Test FederatedSimulation execution."""

    @pytest.fixture
    def mock_simulation(self) -> FederatedSimulation:
        """Create a mock simulation for testing execution."""
        base_config = _get_base_strategy_config()
        strategy_config = StrategyConfig.from_dict(base_config)
        mock_dataset_handler = MockDatasetHandler(dataset_type="its")
        mock_dataset_handler.setup_dataset(num_clients=5)

        with (
            patch("src.federated_simulation.ImageDatasetLoader") as mock_loader,
            patch("src.federated_simulation.ITSNetwork") as mock_network,
        ):
            mock_loader_instance = Mock()
            mock_loader_instance.load_datasets.return_value = (
                [Mock() for _ in range(5)],
                [Mock() for _ in range(5)],
            )
            mock_loader.return_value = mock_loader_instance
            mock_network.return_value = MockNetwork()

            simulation = FederatedSimulation(
                strategy_config=strategy_config,
                dataset_dir="/tmp/test",
                dataset_handler=mock_dataset_handler,
            )

        return simulation

    @patch("src.federated_simulation.flwr.simulation.start_simulation")
    def test_run_simulation_calls_flower_with_correct_parameters(
        self, mock_start_simulation: Mock, mock_simulation: FederatedSimulation
    ) -> None:
        """Test that run_simulation calls Flower with correct parameters."""
        # Act
        mock_simulation.run_simulation()

        # Assert
        mock_start_simulation.assert_called_once()
        call_args = mock_start_simulation.call_args

        # Verify call arguments
        assert "client_fn" in call_args.kwargs
        assert call_args.kwargs["num_clients"] == 5
        assert call_args.kwargs["config"].num_rounds == 3
        assert call_args.kwargs["strategy"] == mock_simulation._aggregation_strategy
        assert call_args.kwargs["client_resources"]["num_cpus"] == 1
        assert call_args.kwargs["client_resources"]["num_gpus"] == 0.0

    def test_client_fn_creates_flower_client(
        self, mock_simulation: FederatedSimulation
    ) -> None:
        """Test that client_fn creates FlowerClient with correct parameters."""
        # Arrange
        client_id = "0"

        # Mock FlowerClient
        with patch("src.federated_simulation.FlowerClient") as mock_flower_client:
            mock_client_instance = Mock()
            mock_client_instance.to_client.return_value = Mock()
            mock_flower_client.return_value = mock_client_instance

            # Act
            mock_simulation.client_fn(client_id)

            # Assert
            mock_flower_client.assert_called_once()
            call_args = mock_flower_client.call_args

            # Verify FlowerClient initialization parameters
            assert call_args.kwargs["client_id"] == 0
            assert call_args.kwargs["net"] is not None
            assert call_args.kwargs["trainloader"] is not None
            assert call_args.kwargs["valloader"] is not None
            assert call_args.kwargs["training_device"] == "cpu"
            assert call_args.kwargs["num_of_client_epochs"] == 1
            assert call_args.kwargs["num_malicious_clients"] == 1

            mock_client_instance.to_client.assert_called_once()

    def test_client_fn_with_different_client_ids(
        self, mock_simulation: FederatedSimulation
    ) -> None:
        """Test client_fn with different client IDs uses correct data loaders."""
        # Mock FlowerClient
        with patch("src.federated_simulation.FlowerClient") as mock_flower_client:
            mock_client_instance = Mock()
            mock_client_instance.to_client.return_value = Mock()
            mock_flower_client.return_value = mock_client_instance

            # Test different client IDs
            for client_id in ["0", "1", "2", "3", "4"]:
                # Act
                mock_simulation.client_fn(client_id)

                # Assert
                call_args = mock_flower_client.call_args
                assert call_args.kwargs["client_id"] == int(client_id)
                if mock_simulation._trainloaders is not None:
                    assert (
                        call_args.kwargs["trainloader"]
                        == mock_simulation._trainloaders[int(client_id)]
                    )
                if mock_simulation._valloaders is not None:
                    assert (
                        call_args.kwargs["valloader"]
                        == mock_simulation._valloaders[int(client_id)]
                    )

    def test_get_model_params_with_regular_model(self) -> None:
        """Test _get_model_params with regular PyTorch model."""
        # Arrange
        model = MockNetwork(num_classes=10, input_size=100)

        # Act
        params: List[NDArray] = FederatedSimulation._get_model_params(model)

        # Assert
        assert isinstance(params, list)
        assert len(params) > 0
        assert all(isinstance(param, np.ndarray) for param in params)

        # Verify parameter shapes match model
        model_params = list(model.parameters())
        assert len(params) == len(model_params)
        for param, model_param in zip(params, model_params):
            assert param.shape == model_param.shape

    @patch("src.federated_simulation.flwr.simulation.start_simulation")
    def test_simulation_execution_with_mocked_flower_components(
        self, mock_start_simulation: Mock, mock_simulation: FederatedSimulation
    ) -> None:
        """Test complete simulation execution with mocked Flower components."""
        # Arrange
        mock_start_simulation.return_value = None  # Simulate successful completion

        # Act
        mock_simulation.run_simulation()

        # Assert
        mock_start_simulation.assert_called_once()

        # Verify that the simulation can create clients
        with patch("src.federated_simulation.FlowerClient") as mock_flower_client:
            mock_client_instance = Mock()
            mock_client_instance.to_client.return_value = Mock()
            mock_flower_client.return_value = mock_client_instance

            # Test client creation
            for i in range(mock_simulation.strategy_config.num_of_clients):
                client = mock_simulation.client_fn(str(i))
                assert client is not None

    def test_simulation_component_assignment_consistency(self) -> None:
        """Test that component assignment is consistent across initialization."""
        # Arrange
        base_config = _get_base_strategy_config()
        strategy_config = StrategyConfig.from_dict(base_config)
        mock_dataset_handler = MockDatasetHandler(dataset_type="its")
        mock_dataset_handler.setup_dataset(num_clients=5)

        # Act
        with (
            patch("src.federated_simulation.ImageDatasetLoader") as mock_loader,
            patch("src.federated_simulation.ITSNetwork") as mock_network,
        ):
            mock_loader_instance = Mock()
            mock_loader_instance.load_datasets.return_value = (
                [Mock() for _ in range(5)],
                [Mock() for _ in range(5)],
            )
            mock_loader.return_value = mock_loader_instance
            mock_network.return_value = MockNetwork()

            simulation = FederatedSimulation(
                strategy_config=strategy_config,
                dataset_dir="/tmp/test",
                dataset_handler=mock_dataset_handler,
            )

        # Assert component consistency
        if simulation._trainloaders is not None:
            assert simulation.strategy_config.num_of_clients == len(
                simulation._trainloaders
            )
        if simulation._valloaders is not None:
            assert simulation.strategy_config.num_of_clients == len(
                simulation._valloaders
            )
        assert simulation._aggregation_strategy is not None
        assert simulation._network_model is not None

        # Verify strategy history is properly linked
        assert simulation.strategy_history.strategy_config == strategy_config
        assert simulation.strategy_history.dataset_handler == mock_dataset_handler


class TestFederatedSimulationErrorHandling:
    """Test error handling in FederatedSimulation."""

    def test_initialization_with_invalid_dataset_keyword(self) -> None:
        """Test initialization with invalid dataset keyword."""
        # Arrange
        invalid_config = _get_base_strategy_config()
        invalid_config["dataset_keyword"] = "invalid_dataset"
        strategy_config = StrategyConfig.from_dict(invalid_config)
        mock_dataset_handler = MockDatasetHandler()

        # Act & Assert
        # The code logs an error and calls sys.exit(-1) for invalid dataset keywords
        with pytest.raises(SystemExit) as exc_info:
            FederatedSimulation(
                strategy_config=strategy_config,
                dataset_dir="/tmp/test",
                dataset_handler=mock_dataset_handler,
            )
        assert exc_info.value.code == -1

    @patch("src.federated_simulation.flwr.simulation.start_simulation")
    def test_simulation_handles_flower_exceptions(
        self, mock_start_simulation: Mock
    ) -> None:
        """Test that simulation properly handles Flower simulation exceptions."""
        # Arrange
        base_config = _get_base_strategy_config()
        strategy_config = StrategyConfig.from_dict(base_config)
        mock_dataset_handler = MockDatasetHandler(dataset_type="its")
        mock_dataset_handler.setup_dataset(num_clients=5)

        with (
            patch("src.federated_simulation.ImageDatasetLoader") as mock_loader,
            patch("src.federated_simulation.ITSNetwork") as mock_network,
        ):
            mock_loader_instance = Mock()
            mock_loader_instance.load_datasets.return_value = (
                [Mock() for _ in range(5)],
                [Mock() for _ in range(5)],
            )
            mock_loader.return_value = mock_loader_instance
            mock_network.return_value = MockNetwork()

            simulation = FederatedSimulation(
                strategy_config=strategy_config,
                dataset_dir="/tmp/test",
                dataset_handler=mock_dataset_handler,
            )

        # Mock Flower to raise an exception
        mock_start_simulation.side_effect = RuntimeError("Flower simulation failed")

        # Act & Assert
        with pytest.raises(RuntimeError, match="Flower simulation failed"):
            simulation.run_simulation()

    def test_client_fn_with_invalid_client_id(self) -> None:
        """Test client_fn with invalid client ID."""
        # Arrange
        base_config = _get_base_strategy_config()
        strategy_config = StrategyConfig.from_dict(base_config)
        mock_dataset_handler = MockDatasetHandler(dataset_type="its")
        mock_dataset_handler.setup_dataset(num_clients=5)

        with (
            patch("src.federated_simulation.ImageDatasetLoader") as mock_loader,
            patch("src.federated_simulation.ITSNetwork") as mock_network,
        ):
            mock_loader_instance = Mock()
            mock_loader_instance.load_datasets.return_value = (
                [Mock() for _ in range(5)],
                [Mock() for _ in range(5)],
            )
            mock_loader.return_value = mock_loader_instance
            mock_network.return_value = MockNetwork()

            simulation = FederatedSimulation(
                strategy_config=strategy_config,
                dataset_dir="/tmp/test",
                dataset_handler=mock_dataset_handler,
            )

        # Act & Assert - Test with client ID beyond available range
        with pytest.raises(IndexError):
            simulation.client_fn("10")  # Only 5 clients available (0-4)
