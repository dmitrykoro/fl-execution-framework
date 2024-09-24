import json
import logging
import os
import sys

import flwr
from flwr.client import Client, ClientApp, NumPyClient

from config_loaders.config_loader import ConfigLoader
from dataset_loaders.image_dataset_loader import ImageDatasetLoader
from dataset_loaders.image_transformers.its_image_transformer import its_image_transformer
from dataset_loaders.image_transformers.femnist_image_transformer import femnist_image_transformer
from network_models.its_network_definition import ITSNetwork
from network_models.femnist_network_definition import FemnistNetwork
from client_models.flower_client import FlowerClient
from simulation_strategies.trust_based_removal_srategy import TrustBasedRemovalStrategy

from output_handlers.plot_handler import PlotHandler
from output_handlers.directory_handler import DirectoryHandler
from utils.additional_data_calculator import AdditionalDataCalculator


class SimulationRunner:

    def __init__(
            self,
            config_filename: str
    ) -> None:
        self.valloaders = None
        self.trainloaders = None
        self.network = None
        self.training_device = None
        self.num_of_client_epochs = None

        self.csv_loss_history_filenames = []

        logging.basicConfig(level=logging.INFO)

        self.config_loader = ConfigLoader(
            usecase_config_path=f"config/simulation_strategies/{config_filename}",
            dataset_config_path=f"config/dataset_keyword_to_dataset_dir.json"
        )
        self.simulation_strategies = self.config_loader.get_usecase_config_list()
        self.datasets_folder = os.path.join("datasets")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.join(script_dir))

        self.additional_data_calculator = AdditionalDataCalculator()
        self.plot_handler = None
        self.directory_handler = DirectoryHandler()

        self.strategy_history = []

    def run(self):
        """Run simulations according to the specified usecase config"""

        for strategy in self.simulation_strategies:
            logging.info(
                "\n" + "-" * 50 + f"Executing new strategy" + "-" * 50 + "\n" +
                "Strategy config:\n" +
                json.dumps(strategy, indent=4)
            )

            (
                dataset_keyword,
                show_plots,
                save_plots,
                save_csv,
                training_device,
                cpus_per_client,
                gpus_per_client,
                num_of_rounds,
                remove_clients,
                begin_removing_from_round,
                num_of_clusters,
                trust_threshold,
                reputation_threshold,
                beta_value,
                num_of_client_epochs,
                num_of_clients,
                batch_size,
                training_subset_fraction,
                min_fit_clients,
                min_evaluate_clients,
                min_available_clients,
                evaluate_metrics_aggregation_fn
            ) = self._parse_strategy(strategy)
            logging.info(f"Successfully parsed config for strategy: {dataset_keyword}")

            self.plot_handler = PlotHandler(
                show_plots=show_plots,
                save_plots=save_plots,
                num_of_rounds=num_of_rounds,
                directory_handler=self.directory_handler
            )

            if dataset_keyword == "its":
                dataset_loader = ImageDatasetLoader(
                    transformer=its_image_transformer,
                    dataset_dir=self.config_loader.get_dataset_folder_name(dataset_keyword),
                    num_of_clients=num_of_clients,
                    batch_size=batch_size,
                    training_subset_fraction=training_subset_fraction
                )
                self.network = ITSNetwork()

            elif dataset_keyword in ("femnist_niid", "femnist_iid"):
                dataset_loader = ImageDatasetLoader(
                    transformer=femnist_image_transformer,
                    dataset_dir=self.config_loader.get_dataset_folder_name(dataset_keyword),
                    num_of_clients=num_of_clients,
                    batch_size=batch_size,
                    training_subset_fraction=training_subset_fraction
                )
                self.network = FemnistNetwork()
            else:
                logging.error(
                    f"Non-existing dataset_keyword: {dataset_keyword}. Please check {self.config_loader.dataset_config_path}"
                )
                sys.exit(-1)

            self.num_of_client_epochs = num_of_client_epochs

            self.trainloaders, self.valloaders = dataset_loader.load_datasets()

            strategy = TrustBasedRemovalStrategy(
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                remove_clients=remove_clients,
                beta_value=beta_value,
                trust_threshold=trust_threshold,
                begin_removing_from_round=begin_removing_from_round
            )

            flwr.simulation.start_simulation(
                client_fn=self.client_fn,
                num_clients=num_of_clients,
                config=flwr.server.ServerConfig(num_rounds=num_of_rounds),
                strategy=strategy,
                client_resources={"num_cpus": cpus_per_client, "num_gpus": gpus_per_client},
            )

            # calculate_distances_relation(accuracy_trust_reputation_distance_data)

            full_strategy_data = self.additional_data_calculator.calculate_data(strategy.rounds_history)

            strategy_id = (
                f'{dataset_keyword}, '
                f'remove: {remove_clients}, '
                f'clients: {num_of_clients}, '
                f'rounds: {num_of_rounds}, '
                f'remove_from: {begin_removing_from_round if remove_clients else "n/a"}'
                f', min_fit_clients: {min_fit_clients}'
                f', min_available_clients: {min_available_clients}'
            )
            self.plot_handler.show_plots_per_strategy(full_strategy_data, strategy_id)

            # to compare data among strategies
            self.strategy_history.append(
                {
                    'strategy_id': strategy_id,
                    'rounds_history': {
                        f'{round_number}': round_data['round_info']
                        for round_number, round_data in full_strategy_data.items()
                    }
                }
            )

        self.plot_handler.show_plots_among_strategies(self.strategy_history)

    @staticmethod
    def _parse_strategy(strategy: dict) -> tuple:
        """Parse the strategy from the provided dict"""
        try:
            return (
                strategy["dataset_keyword"],
                True if strategy["show_plots"] == "true" else False,
                True if strategy["save_plots"] == "true" else False,
                True if strategy["save_csv"] == "true" else False,
                strategy["training_device"],
                strategy["cpus_per_client"],
                strategy["gpus_per_client"],
                strategy["num_of_rounds"],
                True if strategy["remove_clients"] == "true" else False,
                strategy["begin_removing_from_round"],
                strategy["num_of_clusters"],
                strategy["trust_threshold"],
                strategy["reputation_threshold"],
                strategy["beta_value"],
                strategy["num_of_client_epochs"],
                strategy["num_of_clients"],
                strategy["batch_size"],
                strategy["training_subset_fraction"],
                strategy["min_fit_clients"],
                strategy["min_evaluate_clients"],
                strategy["min_available_clients"],
                strategy["evaluate_metrics_aggregation_fn"]
            )

        except Exception as e:
            logging.error(f"Error while parsing the strategy. Error: {e}")

    def client_fn(self, cid: str) -> Client:
        """Create a Flower client."""

        net = self.network.to(self.training_device)

        trainloader = self.trainloaders[int(cid)]
        valloader = self.valloaders[int(cid)]

        return FlowerClient(net, trainloader, valloader, self.training_device, self.num_of_client_epochs).to_client()


"""Possible options: femnist_iid.json for testing on FEMNIST in IID manner, its.json for testing on ITS"""
simulation_runner = SimulationRunner("femnist_iid.json")
simulation_runner.run()
