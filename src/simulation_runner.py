import json
import logging

from config_loaders.config_loader import ConfigLoader

from output_handlers import plot_handler
from output_handlers.directory_handler import DirectoryHandler

from federated_simulation import FederatedSimulation

from data_models.simulation_strategy_config import StrategyConfig


class SimulationRunner:

    def __init__(
            self,
            config_filename: str
    ) -> None:

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

        for strategy_config_dict, i in zip(  # yes, it's bad
                self._simulation_strategy_config_dicts,
                range(len(self._simulation_strategy_config_dicts))
        ):
            logging.info(
                "\n" + "-" * 50 + f"Executing new strategy" + "-" * 50 + "\n" +
                "Strategy config:\n" +
                json.dumps(strategy_config_dict, indent=4)
            )

            strategy_config = StrategyConfig.from_dict(strategy_config_dict)
            setattr(strategy_config, "strategy_number", i)

            simulation_strategy = FederatedSimulation(strategy_config, self._dataset_config_list)
            simulation_strategy.run_simulation()

            executed_simulation_strategies.append(simulation_strategy)

            # after the execution of the strategy, show plots per client
            plot_handler.show_plots_within_strategy(simulation_strategy, self._directory_handler)
            self._directory_handler.save_all(simulation_strategy)

        # after all strategies are executed, show comparison averaging plots
        plot_handler.show_comparing_plots_among_strategies(executed_simulation_strategies, self._directory_handler)


"""Put the filename of the json strategy from config/simulation_strategies here"""
simulation_runner = SimulationRunner("femnist_iid.json")
simulation_runner.run()
