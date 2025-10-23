import json
import logging
import sys

from src.config_loaders.config_loader import ConfigLoader

from src.output_handlers.directory_handler import DirectoryHandler
from src.output_handlers import new_plot_handler

from src.federated_simulation import FederatedSimulation

from src.data_models.simulation_strategy_config import StrategyConfig

from src.dataset_handlers.dataset_handler import DatasetHandler


class SimulationRunner:
    def __init__(
            self,
            config_filename: str
    ) -> None:

        # Configure logging only if not already configured
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.INFO)

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
                json.dumps(strategy_config_dict, indent=4)
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
    """Run simulation with config file in command line args or default example."""
    if len(sys.argv) > 1:
        config_filename = sys.argv[1]
    else:
        config_filename = "example_strategy_config.json"

    simulation_runner = SimulationRunner(config_filename)
    simulation_runner.run()