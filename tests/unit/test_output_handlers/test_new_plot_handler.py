from unittest.mock import patch

import matplotlib

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
    export_plot_data_json,
    plot_size,
    show_inter_strategy_plots,
    show_plots_within_strategy,
)
from tests.common import Mock, np, pytest

matplotlib.use("Agg")


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
        mock_rounds_history.plottable_metrics = []  # Add plottable_metrics for export_plot_data_json
        mock_rounds_history.get_metric_by_name = Mock(return_value=None)
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

    @pytest.fixture
    def mock_multiple_strategies(self, mock_strategy_config):
        """Create multiple mock simulation strategies for inter-strategy testing"""
        strategies = []
        for i in range(2):
            simulation = Mock(spec=FederatedSimulation)
            config = StrategyConfig(
                aggregation_strategy_keyword=f"strategy_{i}",
                dataset_keyword="test_dataset",
                remove_clients=i % 2 == 0,
                num_of_clients=5 + i,
                num_of_malicious_clients=i,
                num_of_client_epochs=2 + i,
                batch_size=16 + i * 8,
                show_plots=True,
                save_plots=False,
            )
            simulation.strategy_config = config

            # Mock strategy history with rounds_history
            strategy_history = Mock(spec=SimulationStrategyHistory)

            # Mock client info with rounds
            client_info = Mock(spec=ClientInfo)
            client_info.rounds = [1, 2, 3]
            strategy_history.get_all_clients.return_value = [client_info]

            # Mock rounds_history with metrics
            rounds_history = Mock(spec=RoundsInfo)
            rounds_history.plottable_metrics = ["accuracy", "loss"]
            rounds_history.barable_metrics = ["num_clients"]
            rounds_history.get_metric_by_name.side_effect = (
                lambda metric: [0.7 + i * 0.1, 0.8 + i * 0.1, 0.9 + i * 0.1]
                if metric in ["accuracy", "loss", "num_clients"]
                else []
            )

            strategy_history.rounds_history = rounds_history
            simulation.strategy_history = strategy_history
            strategies.append(simulation)

        return strategies

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_show_inter_strategy_plots_line_plots(
        self,
        mock_savefig,
        mock_show,
        mock_figure,
        mock_multiple_strategies,
        mock_directory_handler,
    ):
        """Test show_inter_strategy_plots creates line plots for plottable metrics"""
        show_inter_strategy_plots(mock_multiple_strategies, mock_directory_handler)

        # Should create figure for each plottable metric
        assert mock_figure.call_count >= 2  # accuracy and loss
        mock_show.assert_called()

    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.bar")
    def test_show_inter_strategy_plots_bar_plots(
        self,
        mock_bar,
        mock_figure,
        mock_show,
        mock_tight_layout,
        mock_multiple_strategies,
        mock_directory_handler,
    ):
        """Test show_inter_strategy_plots creates bar plots for barable metrics"""
        show_inter_strategy_plots(mock_multiple_strategies, mock_directory_handler)

        mock_bar.assert_called()

    def test_show_inter_strategy_plots_returns_early_when_plots_disabled(
        self, mock_multiple_strategies, mock_directory_handler
    ):
        """Test show_inter_strategy_plots returns early when plots are disabled"""
        # Disable both plot options for first strategy
        mock_multiple_strategies[0].strategy_config.show_plots = False
        mock_multiple_strategies[0].strategy_config.save_plots = False

        with patch("matplotlib.pyplot.figure") as mock_figure:
            show_inter_strategy_plots(mock_multiple_strategies, mock_directory_handler)
            mock_figure.assert_not_called()

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.savefig")
    def test_show_inter_strategy_plots_saves_when_enabled(
        self,
        mock_savefig,
        mock_figure,
        mock_multiple_strategies,
        mock_directory_handler,
    ):
        """Test show_inter_strategy_plots saves plots when save_plots is enabled"""
        # Enable saving for first strategy
        mock_multiple_strategies[0].strategy_config.save_plots = True
        mock_multiple_strategies[0].strategy_config.show_plots = False

        show_inter_strategy_plots(mock_multiple_strategies, mock_directory_handler)

        mock_savefig.assert_called()

    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.plot")
    def test_show_inter_strategy_plots_handles_empty_metrics(
        self,
        mock_plot,
        mock_figure,
        mock_show,
        mock_tight_layout,
        mock_multiple_strategies,
        mock_directory_handler,
    ):
        """Test show_inter_strategy_plots handles strategies with empty metrics"""
        mock_multiple_strategies[
            0
        ].strategy_history.rounds_history.get_metric_by_name.return_value = []

        show_inter_strategy_plots(mock_multiple_strategies, mock_directory_handler)

        mock_figure.assert_called()

    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.legend")
    def test_show_inter_strategy_plots_legend_handling(
        self,
        mock_legend,
        mock_figure,
        mock_show,
        mock_tight_layout,
        mock_multiple_strategies,
        mock_directory_handler,
    ):
        """Test show_inter_strategy_plots handles legend display conditionally"""
        with patch("matplotlib.pyplot.gca") as mock_gca:
            mock_ax = Mock()
            mock_ax.get_legend_handles_labels.return_value = (["handle1"], ["label1"])
            mock_gca.return_value = mock_ax

            show_inter_strategy_plots(mock_multiple_strategies, mock_directory_handler)

            mock_legend.assert_called()

    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_show_inter_strategy_plots_no_legend_when_empty(
        self,
        mock_figure,
        mock_show,
        mock_tight_layout,
        mock_multiple_strategies,
        mock_directory_handler,
    ):
        """Test show_inter_strategy_plots skips legend when no handles/labels"""
        with patch("matplotlib.pyplot.gca") as mock_gca:
            with patch("matplotlib.pyplot.legend") as mock_legend:
                mock_ax = Mock()
                mock_ax.get_legend_handles_labels.return_value = ([], [])
                mock_gca.return_value = mock_ax

                show_inter_strategy_plots(
                    mock_multiple_strategies, mock_directory_handler
                )

                mock_legend.assert_not_called()

    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.plot")
    def test_show_plots_within_strategy_with_removal_threshold(
        self,
        mock_plot,
        mock_figure,
        mock_show,
        mock_tight_layout,
        mock_simulation_strategy,
        mock_directory_handler,
    ):
        """Test show_plots_within_strategy handles removal threshold plotting"""
        mock_simulation_strategy.strategy_history.rounds_history.removal_threshold_history = [
            0.5,
            0.6,
            0.7,
        ]

        mock_client = mock_simulation_strategy.strategy_history.get_all_clients()[0]
        mock_client.plottable_metrics = ["removal_criterion_history"]
        mock_client.get_metric_by_name = Mock(return_value=[0.4, 0.5, 0.8])
        mock_client.rounds = [1, 2, 3]
        mock_client.aggregation_participation_history = [1, 1, 0]

        show_plots_within_strategy(mock_simulation_strategy, mock_directory_handler)

        assert mock_plot.call_count >= 2

    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.plot")
    def test_show_plots_within_strategy_no_removal_threshold(
        self,
        mock_plot,
        mock_figure,
        mock_show,
        mock_tight_layout,
        mock_simulation_strategy,
        mock_directory_handler,
    ):
        """Test show_plots_within_strategy when no removal threshold exists"""
        mock_simulation_strategy.strategy_history.rounds_history.removal_threshold_history = []

        mock_client = mock_simulation_strategy.strategy_history.get_all_clients()[0]
        mock_client.plottable_metrics = ["removal_criterion_history"]
        mock_client.get_metric_by_name = Mock(return_value=[0.4, 0.5, 0.8])
        mock_client.rounds = [1, 2, 3]
        mock_client.aggregation_participation_history = [1, 1, 0]

        show_plots_within_strategy(mock_simulation_strategy, mock_directory_handler)

        mock_plot.assert_called()

    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.plot")
    def test_show_plots_within_strategy_mismatched_dimensions(
        self,
        mock_plot,
        mock_figure,
        mock_show,
        mock_tight_layout,
        mock_simulation_strategy,
        mock_directory_handler,
    ):
        """Test show_plots_within_strategy handles mismatched data dimensions"""
        mock_client = mock_simulation_strategy.strategy_history.get_all_clients()[0]

        mock_client.rounds = [1, 2, 3, 4, 5]
        mock_client.accuracy_history = [0.4, 0.5, 0.8]
        mock_client.aggregation_participation_history = [1, 1, 0]
        mock_client.plottable_metrics = ["accuracy_history"]

        show_plots_within_strategy(mock_simulation_strategy, mock_directory_handler)

        mock_plot.assert_called()

    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.plot")
    def test_show_plots_within_strategy_malicious_client_labeling(
        self,
        mock_plot,
        mock_figure,
        mock_show,
        mock_tight_layout,
        mock_simulation_strategy,
        mock_directory_handler,
    ):
        """Test show_plots_within_strategy labels malicious clients correctly"""
        mock_client = mock_simulation_strategy.strategy_history.get_all_clients()[0]
        mock_client.is_malicious = True
        mock_client.client_id = 5
        mock_client.plottable_metrics = ["accuracy_history"]
        mock_client.accuracy_history = [0.4, 0.5, 0.8]
        mock_client.rounds = [1, 2, 3]
        mock_client.aggregation_participation_history = [1, 1, 0]

        with patch("matplotlib.pyplot.legend"):
            show_plots_within_strategy(mock_simulation_strategy, mock_directory_handler)

        call_args = [call[1] for call in mock_plot.call_args_list if "label" in call[1]]
        malicious_labels = [
            args["label"]
            for args in call_args
            if "client_5 (malicious)" in args["label"]
        ]
        assert len(malicious_labels) > 0

    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.plot")
    def test_show_plots_within_strategy_excluded_values_plotting(
        self,
        mock_plot,
        mock_figure,
        mock_show,
        mock_tight_layout,
        mock_simulation_strategy,
        mock_directory_handler,
    ):
        """Test show_plots_within_strategy plots excluded values with X markers"""
        mock_client = mock_simulation_strategy.strategy_history.get_all_clients()[0]
        mock_client.plottable_metrics = ["accuracy_history"]
        mock_client.accuracy_history = [0.4, 0.5, 0.8]
        mock_client.rounds = [1, 2, 3]
        mock_client.aggregation_participation_history = [
            1,
            0,
            1,
        ]

        show_plots_within_strategy(mock_simulation_strategy, mock_directory_handler)

        x_marker_calls = [
            call
            for call in mock_plot.call_args_list
            if len(call[0]) >= 3 and "kx" in call[0]
        ]
        assert len(x_marker_calls) > 0

    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.show")
    def test_show_plots_within_strategy_directory_handler_usage(
        self,
        mock_show,
        mock_tight_layout,
        mock_simulation_strategy,
        mock_directory_handler,
    ):
        """Test show_plots_within_strategy uses directory handler for save path"""
        mock_simulation_strategy.strategy_config.save_plots = True
        mock_simulation_strategy.strategy_config.show_plots = False
        mock_directory_handler.new_plots_dirname = "/test/plots"

        with patch("matplotlib.pyplot.savefig") as mock_savefig:
            with patch("matplotlib.pyplot.figure"):
                show_plots_within_strategy(
                    mock_simulation_strategy, mock_directory_handler
                )

        save_calls = [call[0][0] for call in mock_savefig.call_args_list]
        assert any("/test/plots/" in path for path in save_calls)

    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_show_inter_strategy_plots_bar_chart_positioning(
        self,
        mock_figure,
        mock_show,
        mock_tight_layout,
        mock_multiple_strategies,
        mock_directory_handler,
    ):
        """Test show_inter_strategy_plots positions bar charts correctly"""
        with patch("matplotlib.pyplot.bar") as mock_bar:
            with patch("numpy.arange") as mock_arange:
                mock_arange.return_value = np.array([0, 1, 2])

                show_inter_strategy_plots(
                    mock_multiple_strategies, mock_directory_handler
                )

                bar_calls = mock_bar.call_args_list
                if bar_calls:
                    x_positions = [call[0][0] for call in bar_calls]
                    assert len(x_positions) > 0

    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.gca")
    @patch("matplotlib.pyplot.figure")
    def test_show_plots_within_strategy_axis_configuration(
        self,
        mock_figure,
        mock_gca,
        mock_show,
        mock_tight_layout,
        mock_simulation_strategy,
        mock_directory_handler,
    ):
        """Test show_plots_within_strategy configures axes correctly"""
        mock_ax = Mock()
        mock_gca.return_value = mock_ax

        show_plots_within_strategy(mock_simulation_strategy, mock_directory_handler)

        mock_ax.xaxis.set_major_locator.assert_called()

    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.gca")
    @patch("matplotlib.pyplot.figure")
    def test_show_inter_strategy_plots_axis_configuration(
        self,
        mock_figure,
        mock_gca,
        mock_show,
        mock_tight_layout,
        mock_multiple_strategies,
        mock_directory_handler,
    ):
        """Test show_inter_strategy_plots configures axes correctly for bar charts"""
        mock_ax = Mock()
        mock_ax.get_legend_handles_labels.return_value = ([], [])
        mock_gca.return_value = mock_ax

        show_inter_strategy_plots(mock_multiple_strategies, mock_directory_handler)

        mock_ax.set_xticks.assert_called()
        mock_ax.set_xticklabels.assert_called()

    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.show")
    @patch("math.ceil")
    @patch("matplotlib.pyplot.legend")
    @patch("matplotlib.pyplot.figure")
    def test_show_plots_within_strategy_legend_columns(
        self,
        mock_figure,
        mock_legend,
        mock_ceil,
        mock_show,
        mock_tight_layout,
        mock_simulation_strategy,
        mock_directory_handler,
    ):
        """Test show_plots_within_strategy calculates legend columns correctly"""
        mock_ceil.return_value = 3

        show_plots_within_strategy(mock_simulation_strategy, mock_directory_handler)

        mock_ceil.assert_called()
        legend_calls = [
            call for call in mock_legend.call_args_list if "ncol" in call[1]
        ]
        assert len(legend_calls) > 0

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.figure")
    def test_show_plots_within_strategy_layout_adjustment(
        self,
        mock_figure,
        mock_tight_layout,
        mock_show,
        mock_simulation_strategy,
        mock_directory_handler,
    ):
        """Test show_plots_within_strategy calls tight_layout"""
        show_plots_within_strategy(mock_simulation_strategy, mock_directory_handler)

        mock_tight_layout.assert_called()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.figure")
    def test_show_inter_strategy_plots_layout_adjustment(
        self,
        mock_figure,
        mock_tight_layout,
        mock_show,
        mock_multiple_strategies,
        mock_directory_handler,
    ):
        """Test show_inter_strategy_plots calls tight_layout"""
        show_inter_strategy_plots(mock_multiple_strategies, mock_directory_handler)

        mock_tight_layout.assert_called()

    def test_plot_configuration_constants_access(self):
        """Test that plot configuration constants are accessible and have expected types"""
        assert isinstance(plot_size, tuple)
        assert len(plot_size) == 2
        assert isinstance(bar_width, (int, float))
        assert bar_width > 0

    # --- Tests for export_plot_data_json ---

    def test_export_plot_data_json_returns_early_when_save_disabled(
        self, mock_simulation_strategy, mock_directory_handler, tmp_path
    ):
        """Test export_plot_data_json returns early when save_plots is disabled"""
        mock_simulation_strategy.strategy_config.save_plots = False
        mock_directory_handler.new_plots_dirname = str(tmp_path)

        export_plot_data_json(mock_simulation_strategy, mock_directory_handler)

        # Should not create any JSON files
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 0

    def test_export_plot_data_json_creates_file_when_enabled(
        self, mock_simulation_strategy, mock_directory_handler, tmp_path
    ):
        """Test export_plot_data_json creates JSON file when save_plots is enabled"""
        mock_simulation_strategy.strategy_config.save_plots = True
        mock_simulation_strategy.strategy_config.strategy_number = 0
        mock_directory_handler.new_plots_dirname = str(tmp_path)

        export_plot_data_json(mock_simulation_strategy, mock_directory_handler)

        # Should create plot_data_0.json
        json_file = tmp_path / "plot_data_0.json"
        assert json_file.exists()

    def test_export_plot_data_json_valid_structure(
        self, mock_simulation_strategy, mock_directory_handler, tmp_path
    ):
        """Test export_plot_data_json creates valid JSON structure"""
        import json

        mock_simulation_strategy.strategy_config.save_plots = True
        mock_simulation_strategy.strategy_config.strategy_number = 0
        mock_directory_handler.new_plots_dirname = str(tmp_path)

        export_plot_data_json(mock_simulation_strategy, mock_directory_handler)

        json_file = tmp_path / "plot_data_0.json"
        with open(json_file, "r") as f:
            data = json.load(f)

        # Verify top-level structure
        assert "per_client_metrics" in data
        assert "round_metrics" in data
        assert isinstance(data["per_client_metrics"], list)
        assert isinstance(data["round_metrics"], dict)

    def test_export_plot_data_json_includes_client_data(
        self, mock_simulation_strategy, mock_directory_handler, tmp_path
    ):
        """Test export_plot_data_json includes per-client data"""
        import json

        mock_simulation_strategy.strategy_config.save_plots = True
        mock_simulation_strategy.strategy_config.strategy_number = 0
        mock_directory_handler.new_plots_dirname = str(tmp_path)

        # Setup client data
        clients = mock_simulation_strategy.strategy_history.get_all_clients()
        assert len(clients) == 3

        export_plot_data_json(mock_simulation_strategy, mock_directory_handler)

        json_file = tmp_path / "plot_data_0.json"
        with open(json_file, "r") as f:
            data = json.load(f)

        # Should have data for all 3 clients
        assert len(data["per_client_metrics"]) == 3

        # Verify client data structure
        client_data = data["per_client_metrics"][0]
        assert "client_id" in client_data
        assert "is_malicious" in client_data
        assert "rounds" in client_data
        assert "aggregation_participation" in client_data
        assert "metrics" in client_data

    def test_export_plot_data_json_client_metrics(
        self, mock_simulation_strategy, mock_directory_handler, tmp_path
    ):
        """Test export_plot_data_json includes client metrics correctly"""
        import json

        mock_simulation_strategy.strategy_config.save_plots = True
        mock_simulation_strategy.strategy_config.strategy_number = 0
        mock_directory_handler.new_plots_dirname = str(tmp_path)

        export_plot_data_json(mock_simulation_strategy, mock_directory_handler)

        json_file = tmp_path / "plot_data_0.json"
        with open(json_file, "r") as f:
            data = json.load(f)

        client_data = data["per_client_metrics"][0]
        assert "loss_history" in client_data["metrics"]
        assert "accuracy_history" in client_data["metrics"]

        # Verify values match mock data
        assert client_data["metrics"]["loss_history"] == [0.8, 0.6, 0.4]
        assert client_data["metrics"]["accuracy_history"] == [0.6, 0.7, 0.8]

    def test_export_plot_data_json_includes_rounds(
        self, mock_simulation_strategy, mock_directory_handler, tmp_path
    ):
        """Test export_plot_data_json includes rounds array"""
        import json

        mock_simulation_strategy.strategy_config.save_plots = True
        mock_simulation_strategy.strategy_config.strategy_number = 0
        mock_directory_handler.new_plots_dirname = str(tmp_path)

        export_plot_data_json(mock_simulation_strategy, mock_directory_handler)

        json_file = tmp_path / "plot_data_0.json"
        with open(json_file, "r") as f:
            data = json.load(f)

        assert "rounds" in data
        assert isinstance(data["rounds"], list)

    def test_export_plot_data_json_includes_round_metrics(
        self, mock_simulation_strategy, mock_directory_handler, tmp_path
    ):
        """Test export_plot_data_json includes round-level metrics"""
        import json

        mock_simulation_strategy.strategy_config.save_plots = True
        mock_simulation_strategy.strategy_config.strategy_number = 0
        mock_directory_handler.new_plots_dirname = str(tmp_path)

        # Setup round metrics
        rounds_history = mock_simulation_strategy.strategy_history.rounds_history
        rounds_history.plottable_metrics = ["aggregated_loss", "average_accuracy"]
        rounds_history.get_metric_by_name = Mock(
            side_effect=lambda metric: [0.5, 0.4, 0.3]
            if metric in ["aggregated_loss", "average_accuracy"]
            else None
        )

        export_plot_data_json(mock_simulation_strategy, mock_directory_handler)

        json_file = tmp_path / "plot_data_0.json"
        with open(json_file, "r") as f:
            data = json.load(f)

        assert "aggregated_loss" in data["round_metrics"]
        assert "average_accuracy" in data["round_metrics"]
        assert data["round_metrics"]["aggregated_loss"] == [0.5, 0.4, 0.3]

    def test_export_plot_data_json_includes_removal_threshold(
        self, mock_simulation_strategy, mock_directory_handler, tmp_path
    ):
        """Test export_plot_data_json includes removal threshold when available"""
        import json

        mock_simulation_strategy.strategy_config.save_plots = True
        mock_simulation_strategy.strategy_config.strategy_number = 0
        mock_directory_handler.new_plots_dirname = str(tmp_path)

        # Add removal threshold history
        mock_simulation_strategy.strategy_history.rounds_history.removal_threshold_history = [
            0.5,
            0.6,
            0.7,
        ]

        export_plot_data_json(mock_simulation_strategy, mock_directory_handler)

        json_file = tmp_path / "plot_data_0.json"
        with open(json_file, "r") as f:
            data = json.load(f)

        assert "removal_threshold_history" in data
        assert data["removal_threshold_history"] == [0.5, 0.6, 0.7]

    def test_export_plot_data_json_no_removal_threshold(
        self, mock_simulation_strategy, mock_directory_handler, tmp_path
    ):
        """Test export_plot_data_json handles missing removal threshold"""
        import json

        mock_simulation_strategy.strategy_config.save_plots = True
        mock_simulation_strategy.strategy_config.strategy_number = 0
        mock_directory_handler.new_plots_dirname = str(tmp_path)

        # No removal threshold
        mock_simulation_strategy.strategy_history.rounds_history.removal_threshold_history = []

        export_plot_data_json(mock_simulation_strategy, mock_directory_handler)

        json_file = tmp_path / "plot_data_0.json"
        with open(json_file, "r") as f:
            data = json.load(f)

        assert "removal_threshold_history" not in data

    def test_export_plot_data_json_strategy_number_in_filename(
        self, mock_simulation_strategy, mock_directory_handler, tmp_path
    ):
        """Test export_plot_data_json uses strategy_number in filename"""
        mock_simulation_strategy.strategy_config.save_plots = True
        mock_simulation_strategy.strategy_config.strategy_number = 42
        mock_directory_handler.new_plots_dirname = str(tmp_path)

        export_plot_data_json(mock_simulation_strategy, mock_directory_handler)

        json_file = tmp_path / "plot_data_42.json"
        assert json_file.exists()

    def test_export_plot_data_json_empty_client_list(
        self, mock_simulation_strategy, mock_directory_handler, tmp_path
    ):
        """Test export_plot_data_json handles empty client list"""
        import json

        mock_simulation_strategy.strategy_config.save_plots = True
        mock_simulation_strategy.strategy_config.strategy_number = 0
        mock_directory_handler.new_plots_dirname = str(tmp_path)

        # Empty client list
        mock_simulation_strategy.strategy_history.get_all_clients.return_value = []

        export_plot_data_json(mock_simulation_strategy, mock_directory_handler)

        json_file = tmp_path / "plot_data_0.json"
        with open(json_file, "r") as f:
            data = json.load(f)

        assert data["per_client_metrics"] == []
        assert data["rounds"] == []

    def test_export_plot_data_json_skips_none_metrics(
        self, mock_simulation_strategy, mock_directory_handler, tmp_path
    ):
        """Test export_plot_data_json skips None round metrics"""
        import json

        mock_simulation_strategy.strategy_config.save_plots = True
        mock_simulation_strategy.strategy_config.strategy_number = 0
        mock_directory_handler.new_plots_dirname = str(tmp_path)

        # Setup round metrics with some None values
        rounds_history = mock_simulation_strategy.strategy_history.rounds_history
        rounds_history.plottable_metrics = ["metric1", "metric2"]
        rounds_history.get_metric_by_name = Mock(
            side_effect=lambda metric: [0.5, 0.4] if metric == "metric1" else None
        )

        export_plot_data_json(mock_simulation_strategy, mock_directory_handler)

        json_file = tmp_path / "plot_data_0.json"
        with open(json_file, "r") as f:
            data = json.load(f)

        # Should only include metric1, not metric2
        assert "metric1" in data["round_metrics"]
        assert "metric2" not in data["round_metrics"]

    def test_export_plot_data_json_malicious_client_flag(
        self, mock_simulation_strategy, mock_directory_handler, tmp_path
    ):
        """Test export_plot_data_json correctly exports is_malicious flag"""
        import json

        mock_simulation_strategy.strategy_config.save_plots = True
        mock_simulation_strategy.strategy_config.strategy_number = 0
        mock_directory_handler.new_plots_dirname = str(tmp_path)

        # Set first client as malicious
        clients = mock_simulation_strategy.strategy_history.get_all_clients()
        clients[0].is_malicious = True
        clients[1].is_malicious = False

        export_plot_data_json(mock_simulation_strategy, mock_directory_handler)

        json_file = tmp_path / "plot_data_0.json"
        with open(json_file, "r") as f:
            data = json.load(f)

        assert data["per_client_metrics"][0]["is_malicious"] is True
        assert data["per_client_metrics"][1]["is_malicious"] is False

    def test_export_plot_data_json_aggregation_participation(
        self, mock_simulation_strategy, mock_directory_handler, tmp_path
    ):
        """Test export_plot_data_json includes aggregation participation history"""
        import json

        mock_simulation_strategy.strategy_config.save_plots = True
        mock_simulation_strategy.strategy_config.strategy_number = 0
        mock_directory_handler.new_plots_dirname = str(tmp_path)

        # Set aggregation participation
        clients = mock_simulation_strategy.strategy_history.get_all_clients()
        clients[0].aggregation_participation_history = [1, 1, 0]

        export_plot_data_json(mock_simulation_strategy, mock_directory_handler)

        json_file = tmp_path / "plot_data_0.json"
        with open(json_file, "r") as f:
            data = json.load(f)

        assert data["per_client_metrics"][0]["aggregation_participation"] == [1, 1, 0]

    def test_export_plot_data_json_json_formatting(
        self, mock_simulation_strategy, mock_directory_handler, tmp_path
    ):
        """Test export_plot_data_json creates properly formatted JSON with indentation"""
        mock_simulation_strategy.strategy_config.save_plots = True
        mock_simulation_strategy.strategy_config.strategy_number = 0
        mock_directory_handler.new_plots_dirname = str(tmp_path)

        export_plot_data_json(mock_simulation_strategy, mock_directory_handler)

        json_file = tmp_path / "plot_data_0.json"
        content = json_file.read_text()

        # Should have indentation (indent=2 in json.dump)
        assert "  " in content
        # Should be valid JSON
        import json

        json.loads(content)

    def test_export_plot_data_json_multiple_strategies(
        self, mock_simulation_strategy, mock_directory_handler, tmp_path
    ):
        """Test export_plot_data_json creates separate files for different strategies"""
        mock_directory_handler.new_plots_dirname = str(tmp_path)

        # Export for strategy 0
        mock_simulation_strategy.strategy_config.save_plots = True
        mock_simulation_strategy.strategy_config.strategy_number = 0
        export_plot_data_json(mock_simulation_strategy, mock_directory_handler)

        # Export for strategy 1
        mock_simulation_strategy.strategy_config.strategy_number = 1
        export_plot_data_json(mock_simulation_strategy, mock_directory_handler)

        # Should create two separate files
        assert (tmp_path / "plot_data_0.json").exists()
        assert (tmp_path / "plot_data_1.json").exists()

    def test_export_plot_data_json_client_id_types(
        self, mock_simulation_strategy, mock_directory_handler, tmp_path
    ):
        """Test export_plot_data_json handles different client_id types"""
        import json

        mock_simulation_strategy.strategy_config.save_plots = True
        mock_simulation_strategy.strategy_config.strategy_number = 0
        mock_directory_handler.new_plots_dirname = str(tmp_path)

        # Set various client ID types
        clients = mock_simulation_strategy.strategy_history.get_all_clients()
        clients[0].client_id = 0
        clients[1].client_id = 1
        clients[2].client_id = 2

        export_plot_data_json(mock_simulation_strategy, mock_directory_handler)

        json_file = tmp_path / "plot_data_0.json"
        with open(json_file, "r") as f:
            data = json.load(f)

        # All client_ids should be preserved
        client_ids = [c["client_id"] for c in data["per_client_metrics"]]
        assert client_ids == [0, 1, 2]
