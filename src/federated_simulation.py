import logging
import sys
import os

import flwr
from flwr.client import Client, ClientApp, NumPyClient


from dataset_loaders.image_dataset_loader import ImageDatasetLoader
from dataset_loaders.image_transformers.its_image_transformer import its_image_transformer
from dataset_loaders.image_transformers.femnist_image_transformer import femnist_image_transformer
from dataset_loaders.image_transformers.flair_image_transformer import flair_image_transformer
from dataset_loaders.image_transformers.pneumoniamnist_image_transformer import pneumoniamnist_image_transformer
from dataset_loaders.image_transformers.bloodmnist_image_transformer import bloodmnist_image_transformer
from dataset_loaders.image_transformers.lung_photos_image_transformer import lung_cancer_image_transformer

from network_models.its_network_definition import ITSNetwork
from network_models.femnist_reduced_iid_network_definition import FemnistReducedIIDNetwork
from network_models.femnist_full_niid_network_definition import FemnistFullNIIDNetwork
from network_models.flair_network_definition import FlairNetwork
from network_models.pneumoniamnist_network_definition import PneumoniamnistNetwork
from network_models.bloodmnist_network_definition import BloodmnistNetwork
from network_models.lung_photos_network_definition import LungCancerCNN


from client_models.flower_client import FlowerClient

from simulation_strategies.trust_based_removal_srategy import TrustBasedRemovalStrategy
from simulation_strategies.pid_based_removal_strategy import PIDBasedRemovalStrategy
from simulation_strategies.krum_based_removal_strategy import KrumBasedRemovalStrategy
from simulation_strategies.multi_krum_based_removal_strategy import MultiKrumBasedRemovalStrategy
from simulation_strategies.trimmed_mean_based_removal_strategy import TrimmedMeanBasedRemovalStrategy
from simulation_strategies.mutli_krum_strategy import MultiKrumStrategy
from simulation_strategies.rfa_based_removal_strategy import RFABasedRemovalStrategy
from simulation_strategies.bulyan_strategy import BulyanStrategy

from data_models.simulation_strategy_config import StrategyConfig
from data_models.simulation_strategy_history import SimulationStrategyHistory
from data_models.round_info import RoundsInfo

from dataset_handlers.dataset_handler import DatasetHandler


class FederatedSimulation:
    def __init__(
            self,
            strategy_config: StrategyConfig,
            dataset_dir: os.path,
            dataset_handler: DatasetHandler
    ):
        self.strategy_config = strategy_config
        self.rounds_history = None

        self.dataset_handler = dataset_handler

        self.strategy_history = SimulationStrategyHistory(
            strategy_config=self.strategy_config,
            dataset_handler=self.dataset_handler,
            rounds_history=RoundsInfo(simulation_strategy_config=self.strategy_config)
        )

        self._dataset_dir = dataset_dir

        self._network_model = None
        self._aggregation_strategy = None

        self._trainloaders = None
        self._valloaders = None

        self._assign_all_properties()

    def run_simulation(self) -> None:
        """Start federated simulation"""

        flwr.simulation.start_simulation(
            client_fn=self.client_fn,
            num_clients=self.strategy_config.num_of_clients,
            config=flwr.server.ServerConfig(num_rounds=self.strategy_config.num_of_rounds),
            strategy=self._aggregation_strategy,
            client_resources={
                "num_cpus": self.strategy_config.cpus_per_client,
                "num_gpus": self.strategy_config.gpus_per_client
            },
        )

    def _assign_all_properties(self) -> None:
        """Assign simulation properties based on strategy_dict"""

        self._assign_dataset_loaders_and_network_model()
        self._assign_aggregation_strategy()

    def _assign_dataset_loaders_and_network_model(self) -> None:
        """Assign dataset loader and the corresponding network model"""

        dataset_keyword = self.strategy_config.dataset_keyword
        num_of_clients = self.strategy_config.num_of_clients
        batch_size = self.strategy_config.batch_size
        training_subset_fraction = self.strategy_config.training_subset_fraction

        if dataset_keyword == "its":
            dataset_loader = ImageDatasetLoader(
                transformer=its_image_transformer,
                dataset_dir=self._dataset_dir,
                num_of_clients=num_of_clients,
                batch_size=batch_size,
                training_subset_fraction=training_subset_fraction
            )
            self._network_model = ITSNetwork()

        elif dataset_keyword == "femnist_iid":
            dataset_loader = ImageDatasetLoader(
                transformer=femnist_image_transformer,
                dataset_dir=self._dataset_dir,
                num_of_clients=num_of_clients,
                batch_size=batch_size,
                training_subset_fraction=training_subset_fraction
            )
            self._network_model = FemnistReducedIIDNetwork()

        elif dataset_keyword == "femnist_niid":
            dataset_loader = ImageDatasetLoader(
                transformer=femnist_image_transformer,
                dataset_dir=self._dataset_dir,
                num_of_clients=num_of_clients,
                batch_size=batch_size,
                training_subset_fraction=training_subset_fraction
            )
            self._network_model = FemnistFullNIIDNetwork()

        elif dataset_keyword == "flair":
            dataset_loader = ImageDatasetLoader(
                transformer=flair_image_transformer,
                dataset_dir=self._dataset_dir,
                num_of_clients=num_of_clients,
                batch_size=batch_size,
                training_subset_fraction=training_subset_fraction
            )
            self._network_model = FlairNetwork()

        elif dataset_keyword == "pneumoniamnist":
            dataset_loader = ImageDatasetLoader(
                transformer=pneumoniamnist_image_transformer,
                dataset_dir=self._dataset_dir,
                num_of_clients=num_of_clients,
                batch_size=batch_size,
                training_subset_fraction=training_subset_fraction
            )
            self._network_model = PneumoniamnistNetwork()
        elif dataset_keyword == "bloodmnist":
            dataset_loader = ImageDatasetLoader(
                transformer=bloodmnist_image_transformer,
                dataset_dir=self._dataset_dir,
                num_of_clients=num_of_clients,
                batch_size=batch_size,
                training_subset_fraction=training_subset_fraction
            )
            self._network_model = BloodmnistNetwork()
        elif dataset_keyword == "lung_photos":
            dataset_loader = ImageDatasetLoader(
                transformer=lung_cancer_image_transformer,
                dataset_dir=self._dataset_dir,
                num_of_clients=num_of_clients,
                batch_size=batch_size,
                training_subset_fraction=training_subset_fraction
            )
            self._network_model = LungCancerCNN()
        else:
            logging.error(
                f"You are parsing a strategy for dataset: {dataset_keyword}. "
                f"Check that you assign a correct dataset_loader at the code above."
            )
            sys.exit(-1)

        self._trainloaders, self._valloaders = dataset_loader.load_datasets()

    def _assign_aggregation_strategy(self) -> None:
        """Assign aggregation strategy"""

        aggregation_strategy_keyword = self.strategy_config.aggregation_strategy_keyword

        if aggregation_strategy_keyword == "trust":
            self._aggregation_strategy = TrustBasedRemovalStrategy(
                min_fit_clients=self.strategy_config.min_fit_clients,
                min_evaluate_clients=self.strategy_config.min_evaluate_clients,
                min_available_clients=self.strategy_config.min_available_clients,
                evaluate_metrics_aggregation_fn=self.strategy_config.evaluate_metrics_aggregation_fn,
                remove_clients=self.strategy_config.remove_clients,
                beta_value=self.strategy_config.beta_value,
                trust_threshold=self.strategy_config.trust_threshold,
                strategy_history=self.strategy_history,
                begin_removing_from_round=self.strategy_config.begin_removing_from_round
            )
        elif aggregation_strategy_keyword in ("pid", "pid_scaled", "pid_standardized"):
            self._aggregation_strategy = PIDBasedRemovalStrategy(
                min_fit_clients=self.strategy_config.min_fit_clients,
                min_evaluate_clients=self.strategy_config.min_evaluate_clients,
                min_available_clients=self.strategy_config.min_available_clients,
                evaluate_metrics_aggregation_fn=self.strategy_config.evaluate_metrics_aggregation_fn,
                remove_clients=self.strategy_config.remove_clients,
                begin_removing_from_round=self.strategy_config.begin_removing_from_round,
                ki=self.strategy_config.Ki,
                kp=self.strategy_config.Kp,
                kd=self.strategy_config.Kd,
                num_std_dev=self.strategy_config.num_std_dev,
                strategy_history=self.strategy_history,
                network_model=self._network_model,
                aggregation_strategy_keyword=aggregation_strategy_keyword
            )
        elif aggregation_strategy_keyword == "krum":
            self._aggregation_strategy = KrumBasedRemovalStrategy(
               min_fit_clients=self.strategy_config.min_fit_clients,
               min_evaluate_clients=self.strategy_config.min_evaluate_clients,
               min_available_clients=self.strategy_config.min_available_clients,
               evaluate_metrics_aggregation_fn=self.strategy_config.evaluate_metrics_aggregation_fn,
               remove_clients=self.strategy_config.remove_clients,
               begin_removing_from_round=self.strategy_config.begin_removing_from_round,
               num_malicious_clients=self.strategy_config.num_of_malicious_clients,
               strategy_history=self.strategy_history,
               num_krum_selections=self.strategy_config.num_krum_selections  # Use to simulate different Attack strategies
            )
        elif aggregation_strategy_keyword == "multi-krum-based":
            self._aggregation_strategy = MultiKrumBasedRemovalStrategy(
                min_fit_clients=self.strategy_config.min_fit_clients,
                min_evaluate_clients=self.strategy_config.min_evaluate_clients,
                min_available_clients=self.strategy_config.min_available_clients,
                evaluate_metrics_aggregation_fn=self.strategy_config.evaluate_metrics_aggregation_fn,
                remove_clients=self.strategy_config.remove_clients,
                begin_removing_from_round=self.strategy_config.begin_removing_from_round,
                num_of_malicious_clients=self.strategy_config.num_of_malicious_clients,
                strategy_history=self.strategy_history,
                num_krum_selections=self.strategy_config.num_krum_selections
            )
        elif aggregation_strategy_keyword == "multi-krum":
            self._aggregation_strategy = MultiKrumStrategy(
                min_fit_clients=self.strategy_config.min_fit_clients,
                min_evaluate_clients=self.strategy_config.min_evaluate_clients,
                min_available_clients=self.strategy_config.min_available_clients,
                evaluate_metrics_aggregation_fn=self.strategy_config.evaluate_metrics_aggregation_fn,
                remove_clients=self.strategy_config.remove_clients,
                begin_removing_from_round=self.strategy_config.begin_removing_from_round,
                num_of_malicious_clients=self.strategy_config.num_of_malicious_clients,
                strategy_history=self.strategy_history,
                num_krum_selections=self.strategy_config.num_krum_selections
            )
        elif aggregation_strategy_keyword == "trimmed_mean":
            self._aggregation_strategy = TrimmedMeanBasedRemovalStrategy(
                min_fit_clients=self.strategy_config.min_fit_clients,
                min_evaluate_clients=self.strategy_config.min_evaluate_clients,
                min_available_clients=self.strategy_config.min_available_clients,
                evaluate_metrics_aggregation_fn=self.strategy_config.evaluate_metrics_aggregation_fn,
                remove_clients=self.strategy_config.remove_clients,
                begin_removing_from_round=self.strategy_config.begin_removing_from_round,
                strategy_history=self.strategy_history,
                trim_ratio=self.strategy_config.trim_ratio
            )

        elif aggregation_strategy_keyword == "rfa":
            self._aggregation_strategy = RFABasedRemovalStrategy(
                min_fit_clients=self.strategy_config.min_fit_clients,
                min_evaluate_clients=self.strategy_config.min_evaluate_clients,
                min_available_clients=self.strategy_config.min_available_clients,
                evaluate_metrics_aggregation_fn=self.strategy_config.evaluate_metrics_aggregation_fn,
                remove_clients=self.strategy_config.remove_clients,
                begin_removing_from_round=self.strategy_config.begin_removing_from_round,
                strategy_history=self.strategy_history,
                num_of_malicious_clients=self.strategy_config.num_of_malicious_clients
            )
        
        elif aggregation_strategy_keyword == "bulyan":
            self._aggregation_strategy = BulyanStrategy(
                min_fit_clients=self.strategy_config.min_fit_clients,
                min_evaluate_clients=self.strategy_config.min_evaluate_clients,
                min_available_clients=self.strategy_config.min_available_clients,
                remove_clients=self.strategy_config.remove_clients,
                begin_removing_from_round=self.strategy_config.begin_removing_from_round,
                strategy_history=self.strategy_history,
                network_model=self._network_model,
                aggregation_strategy_keyword=aggregation_strategy_keyword,
                # fix to the number of assumed malicious clients
                assumed_num_malicious=self.strategy_config.num_of_malicious_clients
            )

        else:
            raise NotImplementedError(f"The strategy {aggregation_strategy_keyword} not implemented!")

    def client_fn(self, cid: str) -> Client:
        """Create a Flower client."""

        net = self._network_model.to(self.strategy_config.training_device)

        trainloader = self._trainloaders[int(cid)]
        valloader = self._valloaders[int(cid)]

        return FlowerClient(
            net,
            trainloader,
            valloader,
            self.strategy_config.training_device,
            self.strategy_config.num_of_client_epochs
        ).to_client()
