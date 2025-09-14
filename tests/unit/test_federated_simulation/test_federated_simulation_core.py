"""
Core unit tests for FederatedSimulation class.

Tests the core federated learning simulation orchestration functionality.
"""

from unittest.mock import Mock, patch

import pytest
from src.data_models.simulation_strategy_config import StrategyConfig
from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.dataset_handlers.dataset_handler import DatasetHandler
from src.federated_simulation import FederatedSimulation


class TestFederatedSimulationCore:
    """Test suite for FederatedSimulation core functionality."""

    @pytest.fixture
    def mock_strategy_config(self):
        """Create mock strategy configuration for testing."""
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
    def mock_dataset_handler(self):
        """Create mock dataset handler for testing."""
        handler = Mock(spec=DatasetHandler)
        handler.poisoned_client_ids = set()
        return handler

    def create_simulation(
        self, strategy_config, tmp_path, mock_dataset_handler, patch_assign=True
    ):
        """Helper method to create FederatedSimulation instances with mocked dependencies."""
        dataset_dir = tmp_path / "datasets"
        dataset_dir.mkdir(exist_ok=True)

        if patch_assign:
            with patch.object(FederatedSimulation, "_assign_all_properties"):
                return FederatedSimulation(
                    strategy_config, str(dataset_dir), mock_dataset_handler
                )
        else:
            return FederatedSimulation(
                strategy_config, str(dataset_dir), mock_dataset_handler
            )

    def test_initialization_with_strategy_config(
        self, mock_strategy_config, mock_dataset_handler, tmp_path
    ):
        """Test FederatedSimulation initialization with strategy config."""
        simulation = self.create_simulation(
            mock_strategy_config, tmp_path, mock_dataset_handler
        )

        assert simulation.strategy_config == mock_strategy_config
        assert isinstance(simulation.strategy_history, SimulationStrategyHistory)
        assert simulation.strategy_history.strategy_config == mock_strategy_config
        assert simulation.dataset_handler == mock_dataset_handler

    def test_initialization_sets_dataset_directory(
        self, mock_strategy_config, mock_dataset_handler, tmp_path
    ):
        """Test initialization sets dataset directory correctly."""
        dataset_dir = tmp_path / "test_datasets"
        dataset_dir.mkdir()

        with patch.object(FederatedSimulation, "_assign_all_properties"):
            simulation = FederatedSimulation(
                mock_strategy_config, str(dataset_dir), mock_dataset_handler
            )

        assert simulation._dataset_dir == str(dataset_dir)

    @patch("src.federated_simulation.flwr.simulation.start_simulation")
    def test_run_simulation_calls_start_simulation(
        self,
        mock_start_simulation,
        mock_strategy_config,
        mock_dataset_handler,
        tmp_path,
    ):
        """Test run_simulation calls Flower's start_simulation."""
        mock_start_simulation.return_value = Mock()
        simulation = self.create_simulation(
            mock_strategy_config, tmp_path, mock_dataset_handler
        )

        # Mock required attributes
        simulation.aggregation_strategy = Mock()

        simulation.run_simulation()

        mock_start_simulation.assert_called_once()
        call_args = mock_start_simulation.call_args
        assert "client_fn" in call_args[1]
        assert "num_clients" in call_args[1]
        assert "config" in call_args[1]
        assert "strategy" in call_args[1]

    def test_assign_all_properties_calls_sub_methods(
        self, mock_strategy_config, mock_dataset_handler, tmp_path
    ):
        """Test _assign_all_properties calls dataset and strategy assignment methods."""
        simulation = self.create_simulation(
            mock_strategy_config, tmp_path, mock_dataset_handler
        )

        with patch.object(
            simulation, "_assign_dataset_loaders_and_network_model"
        ) as mock_assign_dataset:
            with patch.object(
                simulation, "_assign_aggregation_strategy"
            ) as mock_assign_strategy:
                simulation._assign_all_properties()

                mock_assign_dataset.assert_called_once()
                mock_assign_strategy.assert_called_once()

    def test_assign_dataset_loaders_its_dataset(
        self, mock_strategy_config, mock_dataset_handler, tmp_path
    ):
        """Test dataset loader assignment for ITS dataset."""
        mock_strategy_config.dataset_keyword = "its"
        simulation = self.create_simulation(
            mock_strategy_config, tmp_path, mock_dataset_handler
        )

        with patch("src.federated_simulation.ImageDatasetLoader") as mock_loader_class:
            with patch("src.federated_simulation.ITSNetwork") as mock_network_class:
                mock_loader = Mock()
                mock_network = Mock()
                mock_loader_class.return_value = mock_loader
                mock_network_class.return_value = mock_network

                simulation._assign_dataset_loaders_and_network_model()

                assert simulation.dataset_loader == mock_loader
                assert simulation.network_model == mock_network

    def test_assign_dataset_loaders_femnist_iid_dataset(
        self, mock_strategy_config, mock_dataset_handler, tmp_path
    ):
        """Test dataset loader assignment for FEMNIST IID dataset."""
        mock_strategy_config.dataset_keyword = "femnist_iid"
        simulation = self.create_simulation(
            mock_strategy_config, tmp_path, mock_dataset_handler
        )

        with patch("src.federated_simulation.ImageDatasetLoader") as mock_loader_class:
            with patch(
                "src.federated_simulation.FemnistReducedIIDNetwork"
            ) as mock_network_class:
                mock_loader = Mock()
                mock_network = Mock()
                mock_loader_class.return_value = mock_loader
                mock_network_class.return_value = mock_network

                simulation._assign_dataset_loaders_and_network_model()

                assert simulation.dataset_loader == mock_loader
                assert simulation.network_model == mock_network

    def test_assign_dataset_loaders_medquad_dataset(
        self, mock_strategy_config, mock_dataset_handler, tmp_path
    ):
        """Test dataset loader assignment for MedQuAD dataset with transformer model."""
        mock_strategy_config.dataset_keyword = "medquad"
        mock_strategy_config.model_type = "transformer"
        mock_strategy_config.use_llm = "true"
        simulation = self.create_simulation(
            mock_strategy_config, tmp_path, mock_dataset_handler
        )

        with patch(
            "src.federated_simulation.MedQuADDatasetLoader"
        ) as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader

            simulation._assign_dataset_loaders_and_network_model()

            assert simulation.dataset_loader == mock_loader

    def test_assign_aggregation_strategy_trust(
        self, mock_strategy_config, mock_dataset_handler, tmp_path
    ):
        """Test aggregation strategy assignment for trust strategy."""
        mock_strategy_config.aggregation_strategy_keyword = "trust"
        simulation = self.create_simulation(
            mock_strategy_config, tmp_path, mock_dataset_handler
        )

        with patch(
            "src.federated_simulation.TrustBasedRemovalStrategy"
        ) as mock_strategy_class:
            mock_strategy = Mock()
            mock_strategy_class.return_value = mock_strategy

            simulation._assign_aggregation_strategy()

            assert simulation.aggregation_strategy == mock_strategy

    def test_assign_aggregation_strategy_pid(
        self, mock_strategy_config, mock_dataset_handler, tmp_path
    ):
        """Test aggregation strategy assignment for PID strategy."""
        mock_strategy_config.aggregation_strategy_keyword = "pid"
        mock_strategy_config.Kp = 1.0
        mock_strategy_config.Ki = 0.1
        mock_strategy_config.Kd = 0.01
        mock_strategy_config.num_std_dev = 2.0
        simulation = self.create_simulation(
            mock_strategy_config, tmp_path, mock_dataset_handler
        )

        with patch(
            "src.federated_simulation.PIDBasedRemovalStrategy"
        ) as mock_strategy_class:
            mock_strategy = Mock()
            mock_strategy_class.return_value = mock_strategy

            simulation._assign_aggregation_strategy()

            assert simulation.aggregation_strategy == mock_strategy

    def test_assign_aggregation_strategy_multi_krum(
        self, mock_strategy_config, mock_dataset_handler, tmp_path
    ):
        """Test aggregation strategy assignment for Multi-Krum strategy."""
        mock_strategy_config.aggregation_strategy_keyword = "multi-krum"
        mock_strategy_config.num_krum_selections = 3
        simulation = self.create_simulation(
            mock_strategy_config, tmp_path, mock_dataset_handler
        )

        with patch("src.federated_simulation.MultiKrumStrategy") as mock_strategy_class:
            mock_strategy = Mock()
            mock_strategy_class.return_value = mock_strategy

            simulation._assign_aggregation_strategy()

            assert simulation.aggregation_strategy == mock_strategy

    def test_client_fn_creates_flower_client(
        self, mock_strategy_config, mock_dataset_handler, tmp_path
    ):
        """Test client_fn creates FlowerClient instances."""
        simulation = self.create_simulation(
            mock_strategy_config, tmp_path, mock_dataset_handler
        )

        # Mock required components
        simulation.dataset_loader = Mock()
        simulation.network_model = Mock()

        with patch("src.federated_simulation.FlowerClient") as mock_flower_client:
            mock_client = Mock()
            mock_flower_client.return_value = mock_client

            result = simulation.client_fn("0")

            mock_flower_client.assert_called_once()
            assert result == mock_client

    def test_client_fn_handles_llm_model_loading(
        self, mock_strategy_config, mock_dataset_handler, tmp_path
    ):
        """Test client_fn handles LLM model loading for transformer models."""
        mock_strategy_config.model_type = "transformer"
        mock_strategy_config.use_llm = "true"
        mock_strategy_config.llm_finetuning = "full"
        simulation = self.create_simulation(
            mock_strategy_config, tmp_path, mock_dataset_handler
        )

        simulation.dataset_loader = Mock()

        with patch("src.federated_simulation.load_model") as mock_load_model:
            with patch("src.federated_simulation.FlowerClient") as mock_flower_client:
                mock_model = Mock()
                mock_load_model.return_value = mock_model
                mock_client = Mock()
                mock_flower_client.return_value = mock_client

                simulation.client_fn("0")

                mock_load_model.assert_called_once()
                mock_flower_client.assert_called_once()

    def test_client_fn_handles_lora_model_loading(
        self, mock_strategy_config, mock_dataset_handler, tmp_path
    ):
        """Test client_fn handles LORA model loading for transformer models."""
        mock_strategy_config.model_type = "transformer"
        mock_strategy_config.use_llm = "true"
        mock_strategy_config.llm_finetuning = "lora"
        simulation = self.create_simulation(
            mock_strategy_config, tmp_path, mock_dataset_handler
        )

        simulation.dataset_loader = Mock()

        with patch("src.federated_simulation.load_model_with_lora") as mock_load_lora:
            with patch("src.federated_simulation.FlowerClient") as mock_flower_client:
                mock_model = Mock()
                mock_load_lora.return_value = mock_model
                mock_client = Mock()
                mock_flower_client.return_value = mock_client

                simulation.client_fn("0")

                mock_load_lora.assert_called_once()

    def test_get_model_params_extracts_parameters(self):
        """Test _get_model_params extracts model parameters correctly."""
        mock_model = Mock()
        mock_model.state_dict.return_value = {
            "layer1.weight": Mock(),
            "layer1.bias": Mock(),
        }

        with patch("src.federated_simulation.ndarrays_to_parameters") as mock_ndarrays:
            mock_parameters = Mock()
            mock_ndarrays.return_value = mock_parameters

            result = FederatedSimulation._get_model_params(mock_model)

            mock_model.state_dict.assert_called_once()
            mock_ndarrays.assert_called_once()
            assert result == mock_parameters

    def test_get_model_params_handles_peft_model(self):
        """Test _get_model_params handles PEFT (LoRA) models correctly."""
        mock_model = Mock()
        mock_model.__class__.__name__ = "PeftModel"

        with patch(
            "src.federated_simulation.get_peft_model_state_dict"
        ) as mock_get_peft:
            with patch(
                "src.federated_simulation.ndarrays_to_parameters"
            ) as mock_ndarrays:
                mock_state_dict = {"lora_layer.weight": Mock()}
                mock_get_peft.return_value = mock_state_dict
                mock_parameters = Mock()
                mock_ndarrays.return_value = mock_parameters

                result = FederatedSimulation._get_model_params(mock_model)

                mock_get_peft.assert_called_once_with(mock_model)
                mock_ndarrays.assert_called_once()
                assert result == mock_parameters

    def test_unsupported_dataset_raises_error(
        self, mock_strategy_config, mock_dataset_handler, tmp_path
    ):
        """Test that unsupported dataset raises ValueError."""
        mock_strategy_config.dataset_keyword = "unsupported_dataset"
        simulation = self.create_simulation(
            mock_strategy_config, tmp_path, mock_dataset_handler
        )

        with pytest.raises(ValueError) as exc_info:
            simulation._assign_dataset_loaders_and_network_model()

        assert "Unsupported dataset" in str(exc_info.value)

    def test_unsupported_strategy_raises_error(
        self, mock_strategy_config, mock_dataset_handler, tmp_path
    ):
        """Test that unsupported aggregation strategy raises ValueError."""
        mock_strategy_config.aggregation_strategy_keyword = "unsupported_strategy"
        simulation = self.create_simulation(
            mock_strategy_config, tmp_path, mock_dataset_handler
        )

        with pytest.raises(ValueError) as exc_info:
            simulation._assign_aggregation_strategy()

        assert "Unsupported aggregation strategy" in str(exc_info.value)

    def test_simulation_with_all_supported_datasets(
        self, mock_dataset_handler, tmp_path
    ):
        """Test simulation initialization works with all supported dataset types."""
        dataset_keywords = [
            "its",
            "femnist_iid",
            "femnist_niid",
            "flair",
            "pneumoniamnist",
            "bloodmnist",
            "lung_photos",
            "medquad",
        ]

        for dataset in dataset_keywords:
            config = StrategyConfig(
                aggregation_strategy_keyword="trust",
                dataset_keyword=dataset,
                model_type="transformer" if dataset == "medquad" else "cnn",
                use_llm="true" if dataset == "medquad" else "false",
                num_of_rounds=2,
                num_of_clients=3,
                trust_threshold=0.7,
                beta_value=0.5,
                begin_removing_from_round=2,
                num_of_clusters=1,
            )

            # Should not raise exception during initialization
            simulation = self.create_simulation(config, tmp_path, mock_dataset_handler)
            assert simulation.strategy_config.dataset_keyword == dataset

    def test_simulation_with_all_supported_strategies(
        self, mock_dataset_handler, tmp_path
    ):
        """Test simulation initialization works with all supported aggregation strategies."""
        strategies = [
            "trust",
            "pid",
            "pid_scaled",
            "pid_standardized",
            "krum",
            "multi-krum",
            "multi-krum-based",
            "trimmed_mean",
            "rfa",
            "bulyan",
        ]

        for strategy in strategies:
            config = StrategyConfig(
                aggregation_strategy_keyword=strategy,
                dataset_keyword="its",
                model_type="cnn",
                use_llm="false",
                num_of_rounds=2,
                num_of_clients=3,
                # Add strategy-specific parameters
                trust_threshold=0.7 if strategy == "trust" else None,
                beta_value=0.5 if strategy == "trust" else None,
                begin_removing_from_round=2 if strategy == "trust" else None,
                num_of_clusters=1 if strategy == "trust" else None,
                Kp=1.0 if "pid" in strategy else None,
                Ki=0.1 if "pid" in strategy else None,
                Kd=0.01 if "pid" in strategy else None,
                num_std_dev=2.0 if "pid" in strategy else None,
                num_krum_selections=3 if "krum" in strategy else None,
                trim_ratio=0.2 if strategy == "trimmed_mean" else None,
            )

            # Should not raise exception during initialization
            simulation = self.create_simulation(config, tmp_path, mock_dataset_handler)
            assert simulation.strategy_config.aggregation_strategy_keyword == strategy
