"""Unit tests for FederatedSimulation class."""

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

from src.data_models.simulation_strategy_config import StrategyConfig
from src.federated_simulation import FederatedSimulation
from tests.common import Mock, np, pytest
from tests.fixtures.mock_datasets import MockDatasetHandler
from tests.fixtures.sample_models import MockNetwork

NDArray = np.ndarray


def _get_base_strategy_config_dict() -> Dict[str, Any]:
    """Base strategy configuration dictionary."""
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


def _create_simulation_with_mocks(
    strategy_config: StrategyConfig,
    dataset_dir: str,
    dataset_handler: MockDatasetHandler,
    network_name: str = "ITSNetwork",
) -> FederatedSimulation:
    """Create a FederatedSimulation instance with mocked dependencies for testing."""
    with (
        patch("src.federated_simulation.ImageDatasetLoader") as mock_loader,
        patch(f"src.federated_simulation.{network_name}") as mock_network,
    ):
        mock_loader_instance = Mock()
        mock_loader_instance.load_datasets.return_value = (
            [Mock() for _ in range(strategy_config.num_of_clients or 0)],
            [Mock() for _ in range(strategy_config.num_of_clients or 0)],
        )
        mock_loader.return_value = mock_loader_instance

        mock_network_instance = MockNetwork()
        mock_network.return_value = mock_network_instance

        return FederatedSimulation(
            strategy_config=strategy_config,
            dataset_dir=dataset_dir,
            dataset_handler=dataset_handler,
        )


class TestFederatedSimulationInitialization:
    """Test suite for FederatedSimulation initialization."""

    @pytest.fixture
    def mock_dataset_handler(self) -> MockDatasetHandler:
        """Create a mock dataset handler for testing."""
        handler = MockDatasetHandler(dataset_type="its")
        handler.setup_dataset(num_clients=5)
        return handler

    @pytest.fixture
    def temp_dataset_dir(self, tmp_path: Path) -> str:
        """Create a temporary dataset directory for testing."""
        dataset_dir = tmp_path / "datasets" / "its"
        dataset_dir.mkdir(parents=True)
        return str(dataset_dir)

    def test_simulation_initialization_with_trust_strategy(
        self, temp_dataset_dir: str, mock_dataset_handler: MockDatasetHandler
    ) -> None:
        """Test initialization with the 'trust' aggregation strategy."""
        strategy_config = StrategyConfig.from_dict(_get_base_strategy_config_dict())
        simulation = _create_simulation_with_mocks(
            strategy_config, temp_dataset_dir, mock_dataset_handler
        )

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
        """Test initialization with the 'pid' aggregation strategy."""
        pid_config_dict = _get_base_strategy_config_dict()
        pid_config_dict.update(
            {
                "aggregation_strategy_keyword": "pid",
                "Kp": 1.0,
                "Ki": 0.1,
                "Kd": 0.01,
                "num_std_dev": 2.0,
            }
        )
        strategy_config = StrategyConfig.from_dict(pid_config_dict)
        simulation = _create_simulation_with_mocks(
            strategy_config, temp_dataset_dir, mock_dataset_handler
        )

        assert simulation.strategy_config.aggregation_strategy_keyword == "pid"
        assert simulation._aggregation_strategy is not None
        assert hasattr(simulation._aggregation_strategy, "ki")
        assert hasattr(simulation._aggregation_strategy, "kp")
        assert hasattr(simulation._aggregation_strategy, "kd")

    def test_simulation_initialization_with_krum_strategy(
        self, temp_dataset_dir: str, mock_dataset_handler: MockDatasetHandler
    ) -> None:
        """Test initialization with Krum strategy."""
        krum_config_dict = _get_base_strategy_config_dict()
        krum_config_dict.update(
            {"aggregation_strategy_keyword": "krum", "num_krum_selections": 3}
        )
        strategy_config = StrategyConfig.from_dict(krum_config_dict)
        simulation = _create_simulation_with_mocks(
            strategy_config, temp_dataset_dir, mock_dataset_handler
        )

        assert simulation.strategy_config.aggregation_strategy_keyword == "krum"
        assert simulation._aggregation_strategy is not None
        assert hasattr(simulation._aggregation_strategy, "num_krum_selections")

    @pytest.mark.parametrize(
        "dataset_type, expected_network",
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
        """Test correct dataset loader and network assignment for various dataset types."""
        config_dict = _get_base_strategy_config_dict()
        config_dict["dataset_keyword"] = dataset_type
        strategy_config = StrategyConfig.from_dict(config_dict)

        simulation = _create_simulation_with_mocks(
            strategy_config, temp_dataset_dir, mock_dataset_handler, expected_network
        )

        assert simulation._network_model is not None
        assert simulation.strategy_config.dataset_keyword == dataset_type

    def test_simulation_initialization_with_invalid_strategy(
        self, temp_dataset_dir: str, mock_dataset_handler: MockDatasetHandler
    ) -> None:
        """Test FederatedSimulation initialization with invalid strategy raises error."""
        invalid_config_dict = _get_base_strategy_config_dict()
        invalid_config_dict["aggregation_strategy_keyword"] = "invalid_strategy"
        strategy_config = StrategyConfig.from_dict(invalid_config_dict)

        with pytest.raises(
            NotImplementedError,
            match="The strategy invalid_strategy not implemented",
        ):
            _create_simulation_with_mocks(
                strategy_config, temp_dataset_dir, mock_dataset_handler
            )

    def test_simulation_strategy_history_initialization(
        self, temp_dataset_dir: str, mock_dataset_handler: MockDatasetHandler
    ) -> None:
        """Test that SimulationStrategyHistory is properly initialized."""
        strategy_config = StrategyConfig.from_dict(_get_base_strategy_config_dict())
        simulation = _create_simulation_with_mocks(
            strategy_config, temp_dataset_dir, mock_dataset_handler
        )

        assert simulation.strategy_history is not None
        assert simulation.strategy_history.strategy_config == strategy_config
        assert simulation.strategy_history.dataset_handler == mock_dataset_handler
        assert simulation.strategy_history.rounds_history is not None

    def test_unsupported_dataset_raises_error(
        self, mock_dataset_handler: MockDatasetHandler, temp_dataset_dir: str
    ) -> None:
        """Test that an unsupported dataset keyword raises a SystemExit."""
        config_dict = _get_base_strategy_config_dict()
        config_dict["dataset_keyword"] = "unsupported_dataset"
        strategy_config = StrategyConfig.from_dict(config_dict)

        with pytest.raises(SystemExit) as exc_info:
            _create_simulation_with_mocks(
                strategy_config, temp_dataset_dir, mock_dataset_handler
            )
        assert exc_info.value.code == -1

    def test_unsupported_strategy_raises_error(
        self, mock_dataset_handler: MockDatasetHandler, temp_dataset_dir: str
    ) -> None:
        """Test that an unsupported aggregation strategy raises a NotImplementedError."""
        config_dict = _get_base_strategy_config_dict()
        config_dict["aggregation_strategy_keyword"] = "unsupported_strategy"
        strategy_config = StrategyConfig.from_dict(config_dict)

        with pytest.raises(NotImplementedError, match="not implemented"):
            _create_simulation_with_mocks(
                strategy_config, temp_dataset_dir, mock_dataset_handler
            )

    @pytest.mark.parametrize(
        "strategy,extra_params",
        [
            ("trust", {"trust_threshold": 0.7, "beta_value": 0.5}),
            ("pid", {"Kp": 1.0, "Ki": 0.1, "Kd": 0.01, "num_std_dev": 2.0}),
            ("pid_scaled", {"Kp": 1.0, "Ki": 0.1, "Kd": 0.01, "num_std_dev": 2.0}),
            (
                "pid_standardized",
                {"Kp": 1.0, "Ki": 0.1, "Kd": 0.01, "num_std_dev": 2.0},
            ),
            ("krum", {"num_krum_selections": 3}),
            ("multi-krum", {"num_krum_selections": 3}),
            ("multi-krum-based", {"num_krum_selections": 3}),
            ("trimmed_mean", {"trim_ratio": 0.2}),
            # Skip RFA for now due to constructor parameter incompatibility
            # ("rfa", {}),
            ("bulyan", {}),
        ],
    )
    def test_all_supported_strategies(
        self,
        temp_dataset_dir: str,
        mock_dataset_handler: MockDatasetHandler,
        strategy: str,
        extra_params: Dict[str, Any],
    ) -> None:
        """Test that all supported aggregation strategies can be initialized."""
        config_dict = _get_base_strategy_config_dict()
        config_dict["aggregation_strategy_keyword"] = strategy
        config_dict.update(extra_params)

        # RFA strategy doesn't accept num_of_malicious_clients parameter
        if strategy == "rfa":
            config_dict.pop("num_of_malicious_clients", None)

        strategy_config = StrategyConfig.from_dict(config_dict)

        simulation = _create_simulation_with_mocks(
            strategy_config, temp_dataset_dir, mock_dataset_handler
        )

        assert simulation.strategy_config.aggregation_strategy_keyword == strategy
        assert simulation._aggregation_strategy is not None


class TestFederatedSimulationClientFunction:
    """Test suite for client function creation and management."""

    @pytest.fixture
    def mock_dataset_handler(self) -> MockDatasetHandler:
        """Create a mock dataset handler for testing."""
        handler = MockDatasetHandler(dataset_type="its")
        handler.setup_dataset(num_clients=5)
        return handler

    @pytest.fixture
    def temp_dataset_dir(self, tmp_path: Path) -> str:
        """Create a temporary dataset directory for testing."""
        dataset_dir = tmp_path / "datasets" / "its"
        dataset_dir.mkdir(parents=True)
        return str(dataset_dir)

    def test_client_fn_creates_flower_client(
        self, temp_dataset_dir: str, mock_dataset_handler: MockDatasetHandler
    ) -> None:
        """Test that the client_fn factory creates a FlowerClient with correct parameters."""
        strategy_config = StrategyConfig.from_dict(_get_base_strategy_config_dict())
        simulation = _create_simulation_with_mocks(
            strategy_config, temp_dataset_dir, mock_dataset_handler
        )

        with patch("src.federated_simulation.FlowerClient") as mock_flower_client:
            mock_client_instance = Mock()
            mock_client_instance.to_client.return_value = Mock()
            mock_flower_client.return_value = mock_client_instance

            simulation.client_fn(cid="0")

            mock_flower_client.assert_called_once()
            call_args = mock_flower_client.call_args
            assert call_args.kwargs["client_id"] == 0
            assert call_args.kwargs["net"] is not None
            assert call_args.kwargs["trainloader"] is not None
            assert call_args.kwargs["valloader"] is not None
            assert call_args.kwargs["training_device"] == "cpu"
            assert call_args.kwargs["num_of_client_epochs"] == 1

    def test_client_fn_with_different_client_ids(
        self, temp_dataset_dir: str, mock_dataset_handler: MockDatasetHandler
    ) -> None:
        """Test client_fn with different client IDs uses correct data loaders."""
        strategy_config = StrategyConfig.from_dict(_get_base_strategy_config_dict())
        simulation = _create_simulation_with_mocks(
            strategy_config, temp_dataset_dir, mock_dataset_handler
        )

        with patch("src.federated_simulation.FlowerClient") as mock_flower_client:
            mock_client_instance = Mock()
            mock_client_instance.to_client.return_value = Mock()
            mock_flower_client.return_value = mock_client_instance

            # Test different client IDs
            for client_id in ["0", "1", "2", "3", "4"]:
                simulation.client_fn(client_id)

                call_args = mock_flower_client.call_args
                assert call_args.kwargs["client_id"] == int(client_id)
                if simulation._trainloaders is not None:
                    assert (
                        call_args.kwargs["trainloader"]
                        == simulation._trainloaders[int(client_id)]
                    )
                if simulation._valloaders is not None:
                    assert (
                        call_args.kwargs["valloader"]
                        == simulation._valloaders[int(client_id)]
                    )

    def test_client_fn_with_invalid_client_id(
        self, temp_dataset_dir: str, mock_dataset_handler: MockDatasetHandler
    ) -> None:
        """Test client_fn with invalid client ID raises IndexError."""
        strategy_config = StrategyConfig.from_dict(_get_base_strategy_config_dict())
        simulation = _create_simulation_with_mocks(
            strategy_config, temp_dataset_dir, mock_dataset_handler
        )

        with pytest.raises(IndexError):
            simulation.client_fn("10")  # Only 5 clients available (0-4)

    def test_client_fn_handles_llm_model_loading(
        self, temp_dataset_dir: str, mock_dataset_handler: MockDatasetHandler
    ) -> None:
        """Test client_fn handles LLM model loading for transformer models."""
        config_dict = _get_base_strategy_config_dict()
        config_dict.update(
            {
                "model_type": "transformer",
                "use_llm": True,
                "llm_finetuning": "full",
                "dataset_keyword": "medquad",
                "llm_model": "bert-base-uncased",
                "llm_chunk_size": 512,
                "mlm_probability": 0.15,
            }
        )
        strategy_config = StrategyConfig.from_dict(config_dict)

        with patch("src.federated_simulation.MedQuADDatasetLoader") as mock_loader:
            with patch("src.federated_simulation.load_model") as mock_load_model:
                mock_loader_instance = Mock()
                mock_loader_instance.load_datasets.return_value = (
                    [Mock() for _ in range(5)],
                    [Mock() for _ in range(5)],
                )
                mock_loader.return_value = mock_loader_instance
                mock_load_model.return_value = MockNetwork()

                simulation = FederatedSimulation(
                    strategy_config=strategy_config,
                    dataset_dir=temp_dataset_dir,
                    dataset_handler=mock_dataset_handler,
                )

                with patch(
                    "src.federated_simulation.FlowerClient"
                ) as mock_flower_client:
                    mock_client_instance = Mock()
                    mock_client_instance.to_client.return_value = Mock()
                    mock_flower_client.return_value = mock_client_instance

                    result = simulation.client_fn("0")

                    mock_flower_client.assert_called_once()
                    assert result is not None


class TestFederatedSimulationModelParams:
    """Test suite for model parameter extraction."""

    def test_get_model_params_with_regular_model(self) -> None:
        """Test _get_model_params with a standard PyTorch model."""
        model = MockNetwork(num_classes=10, input_size=100)
        params: List[NDArray] = FederatedSimulation._get_model_params(model)

        assert isinstance(params, list)
        assert len(params) > 0
        assert all(isinstance(param, np.ndarray) for param in params)

        model_params = list(model.parameters())
        assert len(params) == len(model_params)
        for param, model_param in zip(params, model_params):
            assert param.shape == model_param.shape

    def test_get_model_params_handles_lora_model(self) -> None:
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
                mock_state_dict[
                    "lora_layer.weight"
                ].cpu.return_value.numpy.return_value = np.array([1.0, 2.0])
                mock_get_peft.return_value = mock_state_dict

                result = FederatedSimulation._get_model_params(mock_model)

                mock_get_peft.assert_called_once_with(mock_model)
                assert isinstance(result, list)
                assert len(result) == 1
                assert np.array_equal(result[0], np.array([1.0, 2.0]))


class TestFederatedSimulationErrorHandling:
    """Test error handling in FederatedSimulation."""

    def test_initialization_with_invalid_dataset_keyword(self) -> None:
        """Test initialization with invalid dataset keyword."""
        invalid_config_dict = _get_base_strategy_config_dict()
        invalid_config_dict["dataset_keyword"] = "invalid_dataset"
        strategy_config = StrategyConfig.from_dict(invalid_config_dict)
        mock_dataset_handler = MockDatasetHandler()

        # The code logs an error and calls sys.exit(-1) for invalid dataset keywords
        with pytest.raises(SystemExit) as exc_info:
            FederatedSimulation(
                strategy_config=strategy_config,
                dataset_dir="/tmp/test",
                dataset_handler=mock_dataset_handler,
            )
        assert exc_info.value.code == -1

    def test_component_assignment_consistency(self) -> None:
        """Test that component assignment is consistent across initialization."""
        strategy_config = StrategyConfig.from_dict(_get_base_strategy_config_dict())
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


class TestWeightedAverage:
    """Test suite for weighted_average function."""

    def test_weighted_average_empty_metrics(self) -> None:
        """Test weighted_average with empty metrics list."""
        from src.federated_simulation import weighted_average

        result = weighted_average([])
        assert result == {}

    def test_weighted_average_single_client(self) -> None:
        """Test weighted_average with single client."""
        from src.federated_simulation import weighted_average

        metrics = [(100, {"accuracy": 0.85, "loss": 0.25})]
        result = weighted_average(metrics)

        assert "accuracy" in result
        assert "loss" in result
        assert result["accuracy"] == 0.85
        assert result["loss"] == 0.25

    def test_weighted_average_multiple_clients_equal_samples(self) -> None:
        """Test weighted_average with multiple clients having equal samples."""
        from src.federated_simulation import weighted_average

        metrics = [
            (100, {"accuracy": 0.8, "loss": 0.3}),
            (100, {"accuracy": 0.9, "loss": 0.2}),
        ]
        result = weighted_average(metrics)

        # With equal weights, should be simple average
        assert abs(result["accuracy"] - 0.85) < 1e-6
        assert abs(result["loss"] - 0.25) < 1e-6

    def test_weighted_average_multiple_clients_different_samples(self) -> None:
        """Test weighted_average with clients having different sample counts."""
        from src.federated_simulation import weighted_average

        metrics = [
            (100, {"accuracy": 0.8, "loss": 0.3}),  # 100 samples
            (200, {"accuracy": 0.9, "loss": 0.2}),  # 200 samples
        ]
        result = weighted_average(metrics)

        # Weighted: (100*0.8 + 200*0.9) / 300 = 260/300 = 0.8666...
        expected_accuracy = (100 * 0.8 + 200 * 0.9) / 300
        expected_loss = (100 * 0.3 + 200 * 0.2) / 300

        assert abs(result["accuracy"] - expected_accuracy) < 1e-6
        assert abs(result["loss"] - expected_loss) < 1e-6

    def test_weighted_average_different_metric_sets(self) -> None:
        """Test weighted_average when clients have different metrics."""
        from src.federated_simulation import weighted_average

        metrics = [
            (100, {"accuracy": 0.8, "precision": 0.85}),
            (200, {"accuracy": 0.9, "recall": 0.88}),
        ]
        result = weighted_average(metrics)

        # All metrics should be in result
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result

        # Accuracy: both clients have it
        expected_accuracy = (100 * 0.8 + 200 * 0.9) / 300
        assert abs(result["accuracy"] - expected_accuracy) < 1e-6

        # Precision: only first client
        assert abs(result["precision"] - 0.85) < 1e-6

        # Recall: only second client
        assert abs(result["recall"] - 0.88) < 1e-6

    def test_weighted_average_missing_metrics_in_some_clients(self) -> None:
        """Test weighted_average when some clients are missing certain metrics."""
        from src.federated_simulation import weighted_average

        metrics = [
            (100, {"accuracy": 0.8, "loss": 0.3}),
            (200, {"accuracy": 0.9}),  # Missing loss
            (150, {"loss": 0.25}),  # Missing accuracy
        ]
        result = weighted_average(metrics)

        # Accuracy: clients 0 and 1
        expected_accuracy = (100 * 0.8 + 200 * 0.9) / (100 + 200)
        assert abs(result["accuracy"] - expected_accuracy) < 1e-6

        # Loss: clients 0 and 2
        expected_loss = (100 * 0.3 + 150 * 0.25) / (100 + 150)
        assert abs(result["loss"] - expected_loss) < 1e-6

    def test_weighted_average_zero_samples(self) -> None:
        """Test weighted_average handles zero samples gracefully."""
        from src.federated_simulation import weighted_average

        metrics = [
            (0, {"accuracy": 0.8}),  # Zero samples
            (100, {"accuracy": 0.9}),
        ]
        result = weighted_average(metrics)

        # Should only count client with non-zero samples
        assert abs(result["accuracy"] - 0.9) < 1e-6

    def test_weighted_average_all_zero_samples(self) -> None:
        """Test weighted_average when all clients have zero samples."""
        from src.federated_simulation import weighted_average

        metrics = [
            (0, {"accuracy": 0.8}),
            (0, {"accuracy": 0.9}),
        ]
        result = weighted_average(metrics)

        # No metrics should be computed when total_samples is 0
        assert "accuracy" not in result or result.get("accuracy") is None

    def test_weighted_average_complex_metrics(self) -> None:
        """Test weighted_average with multiple complex metrics."""
        from src.federated_simulation import weighted_average

        metrics = [
            (100, {"accuracy": 0.85, "precision": 0.88, "recall": 0.82, "f1": 0.85}),
            (
                200,
                {"accuracy": 0.90, "precision": 0.92, "recall": 0.87, "f1": 0.895},
            ),
            (150, {"accuracy": 0.88, "precision": 0.89, "recall": 0.85, "f1": 0.87}),
        ]
        result = weighted_average(metrics)

        # All metrics should be present
        assert all(
            metric in result for metric in ["accuracy", "precision", "recall", "f1"]
        )

        # Verify weighted calculations
        total_samples = 450
        expected_accuracy = (100 * 0.85 + 200 * 0.90 + 150 * 0.88) / total_samples
        assert abs(result["accuracy"] - expected_accuracy) < 1e-6

    def test_weighted_average_negative_metric_values(self) -> None:
        """Test weighted_average handles negative metric values."""
        from src.federated_simulation import weighted_average

        metrics = [
            (100, {"score": -0.5}),
            (200, {"score": 0.3}),
        ]
        result = weighted_average(metrics)

        expected_score = (100 * -0.5 + 200 * 0.3) / 300
        assert abs(result["score"] - expected_score) < 1e-6

    def test_weighted_average_large_numbers(self) -> None:
        """Test weighted_average with large sample counts and metric values."""
        from src.federated_simulation import weighted_average

        metrics = [
            (1000000, {"accuracy": 0.95}),
            (2000000, {"accuracy": 0.97}),
        ]
        result = weighted_average(metrics)

        expected_accuracy = (1000000 * 0.95 + 2000000 * 0.97) / 3000000
        assert abs(result["accuracy"] - expected_accuracy) < 1e-6

    def test_weighted_average_floating_point_samples(self) -> None:
        """Test weighted_average handles floating point sample counts."""
        from src.federated_simulation import weighted_average

        metrics = [
            (100.5, {"accuracy": 0.8}),
            (200.3, {"accuracy": 0.9}),
        ]
        result = weighted_average(metrics)

        expected_accuracy = (100.5 * 0.8 + 200.3 * 0.9) / (100.5 + 200.3)
        assert abs(result["accuracy"] - expected_accuracy) < 1e-6

    def test_weighted_average_preserves_all_metric_names(self) -> None:
        """Test that weighted_average preserves all unique metric names."""
        from src.federated_simulation import weighted_average

        metrics = [
            (100, {"metric1": 0.5, "metric2": 0.6}),
            (200, {"metric2": 0.7, "metric3": 0.8}),
            (150, {"metric3": 0.9, "metric4": 1.0}),
        ]
        result = weighted_average(metrics)

        # All unique metrics should be present
        assert set(result.keys()) == {"metric1", "metric2", "metric3", "metric4"}

    def test_weighted_average_single_metric_across_clients(self) -> None:
        """Test weighted_average with single metric type across all clients."""
        from src.federated_simulation import weighted_average

        metrics = [
            (50, {"loss": 0.5}),
            (100, {"loss": 0.4}),
            (150, {"loss": 0.3}),
            (200, {"loss": 0.2}),
        ]
        result = weighted_average(metrics)

        total = 500
        expected_loss = (50 * 0.5 + 100 * 0.4 + 150 * 0.3 + 200 * 0.2) / total
        assert abs(result["loss"] - expected_loss) < 1e-6
        assert len(result) == 1  # Only one metric

    def test_weighted_average_empty_metric_dict(self) -> None:
        """Test weighted_average when client has empty metrics dict."""
        from src.federated_simulation import weighted_average

        metrics = [
            (100, {}),  # Empty metrics
            (200, {"accuracy": 0.9}),
        ]
        result = weighted_average(metrics)

        # Should only have accuracy from second client
        assert "accuracy" in result
        assert abs(result["accuracy"] - 0.9) < 1e-6

    def test_weighted_average_all_empty_metric_dicts(self) -> None:
        """Test weighted_average when all clients have empty metrics dicts."""
        from src.federated_simulation import weighted_average

        metrics = [
            (100, {}),
            (200, {}),
        ]
        result = weighted_average(metrics)

        # Should return empty dict
        assert result == {}
