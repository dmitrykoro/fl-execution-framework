import argparse
import json
import logging
import os
import torch

# Suppress joblib CPU count warnings
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)

from src.config_loaders.config_loader import ConfigLoader

from src.output_handlers.directory_handler import DirectoryHandler
from src.output_handlers import new_plot_handler

from src.federated_simulation import FederatedSimulation

from src.data_models.simulation_strategy_config import StrategyConfig

from src.dataset_handlers.dataset_handler import DatasetHandler


def _serialize_config_for_logging(config_dict: dict) -> str:
    """Serialize config dict to JSON, converting torch.device to string."""
    serializable_dict = config_dict.copy()
    if isinstance(serializable_dict.get("training_device"), torch.device):
        serializable_dict["training_device"] = str(serializable_dict["training_device"])
    return json.dumps(serializable_dict, indent=4)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FL Execution Framework - Run federated learning simulations"
    )

    parser.add_argument(
        "config_filename",
        type=str,
        nargs="?",
        default="example_strategy_config.json",
        help="Config filename (default: example_strategy_config.json)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging verbosity (default: INFO)"
    )

    return parser.parse_args()


class SimulationRunner:
    def __init__(
            self,
            config_filename: str,
            log_level: str = "INFO"
    ) -> None:

        # Configure logging only if not already configured
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=getattr(logging, log_level),
                format="%(levelname)s: %(message)s"
            )

        self._config_loader = ConfigLoader(
            usecase_config_path=f"config/simulation_strategies/{config_filename}",
            dataset_config_path=f"config/dataset_keyword_to_dataset_dir.json"
        )
        self._simulation_strategy_config_dicts = self._config_loader.get_usecase_config_list()
        self._dataset_config_list = self._config_loader.get_dataset_config_list()
        self._directory_handler = DirectoryHandler()

    def run(self):
        """Run simulations according to the specified usecase config"""

        executed_simulation_strategies = []

        for strategy_config_dict, strategy_number in zip(
                self._simulation_strategy_config_dicts,
                range(len(self._simulation_strategy_config_dicts))
        ):
            logging.info(
                "\n" + "-" * 50 + f"Executing new strategy" + "-" * 50 + "\n" +
                "Strategy config:\n" +
                _serialize_config_for_logging(strategy_config_dict)
            )

            strategy_config = StrategyConfig.from_dict(strategy_config_dict)
            setattr(strategy_config, "strategy_number", strategy_number)

            self._directory_handler.assign_dataset_dir(strategy_number)

            dataset_handler = DatasetHandler(
                strategy_config=strategy_config,
                directory_handler=self._directory_handler,
                dataset_config_list=self._dataset_config_list
            )
            dataset_handler.setup_dataset()

            simulation_strategy = FederatedSimulation(
                strategy_config=strategy_config,
                dataset_dir=self._directory_handler.dataset_dir,
                dataset_handler=dataset_handler,
                directory_handler=self._directory_handler,
            )
            simulation_strategy.run_simulation()

            executed_simulation_strategies.append(simulation_strategy)

            # generate per-client plots
            new_plot_handler.show_plots_within_strategy(simulation_strategy, self._directory_handler)

            simulation_strategy.strategy_history.calculate_additional_rounds_data()
            self._directory_handler.save_csv_and_config(simulation_strategy.strategy_history)

            dataset_handler.teardown_dataset()

        # after all strategies are executed, show comparison averaging plots
        new_plot_handler.show_inter_strategy_plots(executed_simulation_strategies, self._directory_handler)


if __name__ == "__main__":
    """Run simulation with config file and optional log level."""
    args = parse_arguments()

    simulation_runner = SimulationRunner(args.config_filename, args.log_level)
    simulation_runner.run()