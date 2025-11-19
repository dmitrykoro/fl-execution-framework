import argparse
import json
import logging
import os
import time
import torch
import gc
import ray

# Suppress joblib CPU count warnings
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)

# Set HuggingFace cache directory to avoid API rate limits
os.environ["HF_HOME"] = "./cache/huggingface"

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
        help="Config filename (default: example_strategy_config.json)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging verbosity (default: INFO)",
    )

    return parser.parse_args()


class SimulationRunner:
    def __init__(self, config_filename: str, log_level: str = "INFO") -> None:
        # Configure logging only if not already configured
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=getattr(logging, log_level), format="%(levelname)s: %(message)s"
            )

        # Prevent Flower's logs from duplicating
        flwr_logger = logging.getLogger("flwr")
        flwr_logger.propagate = False

        self._config_loader = ConfigLoader(
            usecase_config_path=f"config/simulation_strategies/{config_filename}",
            dataset_config_path="config/dataset_keyword_to_dataset_dir.json",
        )
        self._simulation_strategy_config_dicts = (
            self._config_loader.get_usecase_config_list()
        )
        self._dataset_config_list = self._config_loader.get_dataset_config_list()
        self._directory_handler = DirectoryHandler()

    def run(self):
        """Run simulations according to the specified usecase config"""

        # Extend Ray worker timeout to ensure reliability in multi-strategy runs
        os.environ["RAY_worker_register_timeout_seconds"] = "60"
        logging.debug(
            "Set RAY_worker_register_timeout_seconds=60 for multi-strategy reliability"
        )

        executed_simulation_strategies = []

        for strategy_config_dict, strategy_number in zip(
            self._simulation_strategy_config_dicts,
            range(len(self._simulation_strategy_config_dicts)),
        ):
            dataset_handler = None
            simulation_strategy = None

            try:
                logging.info(
                    "\n"
                    + "-" * 50
                    + "Executing new strategy"
                    + "-" * 50
                    + "\n"
                    + "Strategy config:\n"
                    + _serialize_config_for_logging(strategy_config_dict)
                )

                strategy_config = StrategyConfig.from_dict(strategy_config_dict)
                setattr(strategy_config, "strategy_number", strategy_number)

                self._directory_handler.assign_dataset_dir(strategy_number)

                dataset_handler = DatasetHandler(
                    strategy_config=strategy_config,
                    directory_handler=self._directory_handler,
                    dataset_config_list=self._dataset_config_list,
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
                new_plot_handler.show_plots_within_strategy(
                    simulation_strategy, self._directory_handler
                )

                simulation_strategy.strategy_history.calculate_additional_rounds_data()
                self._directory_handler.save_csv_and_config(
                    simulation_strategy.strategy_history
                )

            finally:
                # Cleanup resources even if simulation crashes
                if dataset_handler is not None:
                    dataset_handler.teardown_dataset()

                # Ensure Ray is initialized before shutting it down to prevent errors in multi-strategy runs
                if ray.is_initialized():
                    logging.debug("Shutting down Ray before cleanup...")
                    ray.shutdown()
                    time.sleep(3.0)
                    logging.debug("Ray shutdown complete after 3s cleanup delay")

                logging.debug(f"Cleaning up resources after strategy {strategy_number}")

                # Clear GPU memory if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logging.debug("GPU cache cleared")

                # Keep only the strategy_history for plotting
                if simulation_strategy is not None:
                    simulation_strategy._network_model = None
                    simulation_strategy._trainloaders = None
                    simulation_strategy._valloaders = None
                    simulation_strategy._dataset_loader = None

                # Force garbage collection to free memory
                gc.collect()
                logging.debug("Garbage collection completed")

        # after all strategies are executed, show comparison averaging plots
        new_plot_handler.show_inter_strategy_plots(
            executed_simulation_strategies, self._directory_handler
        )


if __name__ == "__main__":
    """Run simulation with config file and optional log level."""
    args = parse_arguments()

    simulation_runner = SimulationRunner(args.config_filename, args.log_level)
    simulation_runner.run()
