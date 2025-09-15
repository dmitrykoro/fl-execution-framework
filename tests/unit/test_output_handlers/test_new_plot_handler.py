from unittest.mock import Mock, patch

import matplotlib
import pytest
from src.data_models.client_info import ClientInfo
from src.data_models.round_info import RoundsInfo
from src.data_models.simulation_strategy_config import StrategyConfig
from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.federated_simulation import FederatedSimulation
from src.output_handlers.directory_handler import DirectoryHandler
from src.output_handlers.new_plot_handler import (
    _generate_multi_string_strategy_label,
    _generate_single_string_strategy_label,
    bar_width,
    plot_size,
    show_plots_within_strategy,
)

matplotlib.use("Agg")  # Use non-interactive backend


class TestPlotHandler:
    """Test suite for new_plot_handler plotting functionality"""

    @pytest.fixture
    def mock_strategy_config(self):
        """Create a mock strategy configuration for testing"""
        return StrategyConfig(
            aggregation_strategy_keyword="trust",
            dataset_keyword="its",
            remove_clients=True,
            begin_removing_from_round=2,
            num_of_clients=10,
            num_of_malicious_clients=2,
            num_of_client_epochs=3,
            batch_size=32,
            show_plots=True,
            save_plots=False,
        )

    @pytest.fixture
    def mock_client_info_list(self):
        """Create mock client info list with metrics for testing"""
        clients = []
        for i in range(3):
            client = ClientInfo(client_id=i, num_of_rounds=3)
            client.loss_history = [0.8 - i * 0.1, 0.6 - i * 0.1, 0.4 - i * 0.1]
            client.accuracy_history = [0.6 + i * 0.1, 0.7 + i * 0.1, 0.8 + i * 0.1]
            clients.append(client)
        return clients

    @pytest.fixture
    def mock_simulation_strategy(self, mock_strategy_config, mock_client_info_list):
        """Create mock federated simulation for testing"""

        simulation = Mock(spec=FederatedSimulation)
        simulation.strategy_config = mock_strategy_config

        # Mock strategy history
        strategy_history = Mock(spec=SimulationStrategyHistory)
        strategy_history.get_all_clients.return_value = mock_client_info_list

        # Add the missing rounds_history attribute
        mock_rounds_history = Mock(spec=RoundsInfo)
        mock_rounds_history.removal_threshold_history = []
        strategy_history.rounds_history = mock_rounds_history

        simulation.strategy_history = strategy_history

        return simulation

    @pytest.fixture
    def mock_directory_handler(self):
        """Create mock directory handler for testing"""
        handler = Mock(spec=DirectoryHandler)
        handler.dirname = "/tmp/test_output"
        return handler

    def test_generate_single_string_strategy_label(self, mock_strategy_config):
        """Test _generate_single_string_strategy_label creates correct label"""
        label = _generate_single_string_strategy_label(mock_strategy_config)

        assert "strategy: trust" in label
        assert "dataset: its" in label
        assert "remove: True" in label
        assert "remove_from: 2" in label
        assert "total clients: 10" in label
        assert "bad_clients: 2" in label
        assert "client_epochs: 3" in label
        assert "batch_size: 32" in label

    def test_generate_single_string_strategy_label_no_removal(self):
        """Test label generation when client removal is disabled"""
        config = StrategyConfig(
            aggregation_strategy_keyword="fedavg",
            dataset_keyword="femnist",
            remove_clients=False,
            num_of_clients=5,
            num_of_malicious_clients=0,
            num_of_client_epochs=1,
            batch_size=16,
        )

        label = _generate_single_string_strategy_label(config)

        assert "remove: False" in label
        assert "remove_from: n/a" in label

    def test_generate_multi_string_strategy_label(self, mock_strategy_config):
        """Test _generate_multi_string_strategy_label replaces commas with newlines"""
        multi_label = _generate_multi_string_strategy_label(mock_strategy_config)
        single_label = _generate_single_string_strategy_label(mock_strategy_config)

        # Should be same content but with newlines instead of commas
        assert multi_label == single_label.replace(", ", "\n")
        assert "\n" in multi_label
        assert ", " not in multi_label

    def test_show_plots_within_strategy_returns_early_when_no_plots_enabled(
        self, mock_simulation_strategy, mock_directory_handler
    ):
        """Test show_plots_within_strategy returns early when plots are disabled"""
        # Disable both plot options
        mock_simulation_strategy.strategy_config.show_plots = False
        mock_simulation_strategy.strategy_config.save_plots = False

        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            show_plots_within_strategy(mock_simulation_strategy, mock_directory_handler)

            # Should not call matplotlib functions
            mock_subplots.assert_not_called()

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_show_plots_within_strategy_creates_plots_when_enabled(
        self,
        mock_savefig,
        mock_show,
        mock_figure,
        mock_simulation_strategy,
        mock_directory_handler,
    ):
        """Test show_plots_within_strategy creates plots when enabled"""
        # Enable plot showing
        mock_simulation_strategy.strategy_config.show_plots = True
        mock_simulation_strategy.strategy_config.save_plots = False

        show_plots_within_strategy(mock_simulation_strategy, mock_directory_handler)

        # Should create plots
        mock_figure.assert_called()
        # Should show plots
        mock_show.assert_called()

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_show_plots_within_strategy_saves_plots_when_enabled(
        self,
        mock_savefig,
        mock_show,
        mock_figure,
        mock_simulation_strategy,
        mock_directory_handler,
    ):
        """Test show_plots_within_strategy saves plots when save is enabled"""
        # Enable plot saving
        mock_simulation_strategy.strategy_config.save_plots = True
        mock_simulation_strategy.strategy_config.show_plots = False

        show_plots_within_strategy(mock_simulation_strategy, mock_directory_handler)

        # Should create plots
        mock_figure.assert_called()
        # Should save plots
        mock_savefig.assert_called()
        # Should not show plots
        mock_show.assert_not_called()

    @patch("matplotlib.pyplot.subplots")
    def test_show_plots_within_strategy_handles_empty_client_list(
        self, mock_subplots, mock_simulation_strategy, mock_directory_handler
    ):
        """Test show_plots_within_strategy handles empty client list gracefully"""
        # Mock empty client list
        mock_simulation_strategy.strategy_history.get_all_clients.return_value = []

        # Mock matplotlib components
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)

        # Should not raise exception with empty client list
        show_plots_within_strategy(mock_simulation_strategy, mock_directory_handler)

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.plot")
    def test_show_plots_within_strategy_uses_client_metrics(
        self, mock_plot, mock_figure, mock_simulation_strategy, mock_directory_handler
    ):
        """Test show_plots_within_strategy uses client loss and accuracy metrics"""
        show_plots_within_strategy(mock_simulation_strategy, mock_directory_handler)

        # Should call plot method, verifying that the metric-processing loop is entered
        mock_plot.assert_called()

    def test_plot_size_constant(self):
        """Test plot_size constant is correctly defined"""

        assert plot_size == (11, 7)
        assert len(plot_size) == 2
        assert all(isinstance(dim, int) for dim in plot_size)

    def test_bar_width_constant(self):
        """Test bar_width constant is correctly defined"""

        assert bar_width == 0.2
        assert isinstance(bar_width, float)

    @patch("matplotlib.pyplot.subplots")
    def test_show_plots_with_both_options_enabled(
        self, mock_subplots, mock_simulation_strategy, mock_directory_handler
    ):
        """Test show_plots_within_strategy when both show and save are enabled"""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)

        # Enable both options
        mock_simulation_strategy.strategy_config.show_plots = True
        mock_simulation_strategy.strategy_config.save_plots = True

        with patch("matplotlib.pyplot.show") as mock_show:
            with patch("matplotlib.pyplot.savefig") as mock_savefig:
                show_plots_within_strategy(
                    mock_simulation_strategy, mock_directory_handler
                )

                # Should both show and save
                mock_show.assert_called()
                mock_savefig.assert_called()

    def test_strategy_label_handles_none_values(self):
        """Test strategy label generation handles None values gracefully"""
        config = StrategyConfig(
            aggregation_strategy_keyword="fedavg",
            dataset_keyword=None,
            remove_clients=None,
            num_of_clients=None,
            num_of_malicious_clients=None,
            num_of_client_epochs=None,
            batch_size=None,
        )

        # Should not raise exception with None values
        label = _generate_single_string_strategy_label(config)
        assert "strategy: fedavg" in label
        assert "None" in label  # None values should be converted to string
