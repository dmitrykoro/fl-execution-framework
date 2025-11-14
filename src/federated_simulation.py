import logging
import sys
import os

import flwr

from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy.aggregate import weighted_loss_avg
from typing import List, Tuple

from peft import PeftModel, get_peft_model_state_dict


from src.dataset_loaders.image_dataset_loader import ImageDatasetLoader
from src.dataset_loaders.image_transformers.its_image_transformer import its_image_transformer
from src.dataset_loaders.image_transformers.femnist_image_transformer import femnist_image_transformer
from src.dataset_loaders.image_transformers.flair_image_transformer import flair_image_transformer
from src.dataset_loaders.image_transformers.lung_photos_image_transformer import lung_cancer_image_transformer
from src.dataset_loaders.medquad_dataset_loader import MedQuADDatasetLoader
from src.dataset_loaders.image_transformers.medmnist_2d_grayscale_image_transformer import medmnist_2d_grayscale_image_transformer
from src.dataset_loaders.image_transformers.medmnist_2d_rgb_image_transformer import medmnist_2d_rgb_image_transformer


from src.network_models.its_network_definition import ITSNetwork
from src.network_models.femnist_reduced_iid_network_definition import FemnistReducedIIDNetwork
from src.network_models.femnist_full_niid_network_definition import FemnistFullNIIDNetwork
from src.network_models.flair_network_definition import FlairNetwork
from src.network_models.lung_photos_network_definition import LungCancerCNN
from src.network_models.pneumoniamnist_network_definition import PneumoniamnistNetwork
from src.network_models.bloodmnist_network_definition import BloodMNISTNetwork
from src.network_models.breastmnist_network_definition import BreastMNISTNetwork
from src.network_models.pathmnist_network_definition import PathMNISTNetwork
from src.network_models.dermamnist_network_definition import DermaMNISTNetwork
from src.network_models.octmnist_network_definition import OctMNISTNetwork
from src.network_models.retinamnist_network_definition import RetinaMNISTNetwork
from src.network_models.tissuemnist_network_definition import TissueMNISTNetwork
from src.network_models.organamnist_network_definition import OrganAMNISTNetwork
from src.network_models.organcmnist_network_definition import OrganCMNISTNetwork
from src.network_models.organsmnist_network_definition import OrganSMNISTNetwork
from src.network_models.bert_model_definition import load_model, load_model_with_lora

from src.client_models.flower_client import FlowerClient

from src.simulation_strategies.trust_based_removal_strategy import TrustBasedRemovalStrategy
from src.simulation_strategies.pid_based_removal_strategy import PIDBasedRemovalStrategy
from src.simulation_strategies.krum_based_removal_strategy import KrumBasedRemovalStrategy
from src.simulation_strategies.multi_krum_based_removal_strategy import MultiKrumBasedRemovalStrategy
from src.simulation_strategies.trimmed_mean_based_removal_strategy import TrimmedMeanBasedRemovalStrategy
from src.simulation_strategies.mutli_krum_strategy import MultiKrumStrategy
from src.simulation_strategies.rfa_based_removal_strategy import RFABasedRemovalStrategy
from src.simulation_strategies.bulyan_strategy import BulyanStrategy

from src.data_models.simulation_strategy_config import StrategyConfig
from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.data_models.round_info import RoundsInfo

from src.dataset_handlers.dataset_handler import DatasetHandler


def weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
    """Compute weighted average of metrics from multiple clients."""
    if not metrics:
        return {}

    # Extract metric names from the first client
    metric_names = set()
    for _, client_metrics in metrics:
        metric_names.update(client_metrics.keys())

    # Calculate weighted average for each metric
    weighted_metrics = {}
    for metric_name in metric_names:
        total_samples = 0
        weighted_sum = 0.0

        for num_samples, client_metrics in metrics:
            if metric_name in client_metrics:
                weighted_sum += num_samples * client_metrics[metric_name]
                total_samples += num_samples

        if total_samples > 0:
            weighted_metrics[metric_name] = weighted_sum / total_samples

    return weighted_metrics


class FederatedSimulation:
    def __init__(
            self,
            strategy_config: StrategyConfig,
            dataset_dir: str,
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
        self._dataset_loader = None

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

        common_kwargs = dict(
            dataset_dir=self._dataset_dir,
            num_of_clients=num_of_clients,
            batch_size=batch_size,
            training_subset_fraction=training_subset_fraction
        )

        if dataset_keyword == "its":
            dataset_loader = ImageDatasetLoader(
                transformer=its_image_transformer,
                **common_kwargs
            )
            self._network_model = ITSNetwork()

        elif dataset_keyword == "femnist_iid":
            dataset_loader = ImageDatasetLoader(
                transformer=femnist_image_transformer,
                **common_kwargs
            )
            self._network_model = FemnistReducedIIDNetwork()

        elif dataset_keyword == "femnist_niid":
            dataset_loader = ImageDatasetLoader(
                transformer=femnist_image_transformer,
                **common_kwargs
            )
            self._network_model = FemnistFullNIIDNetwork()

        elif dataset_keyword == "flair":
            dataset_loader = ImageDatasetLoader(
                transformer=flair_image_transformer,
                **common_kwargs
            )
            self._network_model = FlairNetwork()

        elif dataset_keyword == "pneumoniamnist":
            dataset_loader = ImageDatasetLoader(
                transformer=medmnist_2d_grayscale_image_transformer,
                **common_kwargs
            )
            self._network_model = PneumoniamnistNetwork()

        elif dataset_keyword == "bloodmnist":
            dataset_loader = ImageDatasetLoader(
                transformer=medmnist_2d_rgb_image_transformer,
                **common_kwargs
            )
            self._network_model = BloodMNISTNetwork()

        elif dataset_keyword == "breastmnist":
            dataset_loader = ImageDatasetLoader(
                transformer=medmnist_2d_grayscale_image_transformer,
                **common_kwargs
            )
            self._network_model = BreastMNISTNetwork()

        elif dataset_keyword == "pathmnist":
            dataset_loader = ImageDatasetLoader(
                transformer=medmnist_2d_rgb_image_transformer,
                **common_kwargs
            )
            self._network_model = PathMNISTNetwork()

        elif dataset_keyword == "dermamnist":
            dataset_loader = ImageDatasetLoader(
                transformer=medmnist_2d_rgb_image_transformer,
                **common_kwargs
            )
            self._network_model = DermaMNISTNetwork()

        elif dataset_keyword == "octmnist":
            dataset_loader = ImageDatasetLoader(
                transformer=medmnist_2d_grayscale_image_transformer,
                **common_kwargs
            )
            self._network_model = OctMNISTNetwork()

        elif dataset_keyword == "retinamnist":
            dataset_loader = ImageDatasetLoader(
                transformer=medmnist_2d_rgb_image_transformer,
                **common_kwargs
            )
            self._network_model = RetinaMNISTNetwork()

        elif dataset_keyword == "tissuemnist":
            dataset_loader = ImageDatasetLoader(
                transformer=medmnist_2d_grayscale_image_transformer,
                **common_kwargs
            )
            self._network_model = TissueMNISTNetwork()

        elif dataset_keyword == "organamnist":
            dataset_loader = ImageDatasetLoader(
                transformer=medmnist_2d_grayscale_image_transformer,
                **common_kwargs
            )
            self._network_model = OrganAMNISTNetwork()

        elif dataset_keyword == "organcmnist":
            dataset_loader = ImageDatasetLoader(
                transformer=medmnist_2d_grayscale_image_transformer,
                **common_kwargs
            )
            self._network_model = OrganCMNISTNetwork()

        elif dataset_keyword == "organsmnist":
            dataset_loader = ImageDatasetLoader(
                transformer=medmnist_2d_grayscale_image_transformer,
                **common_kwargs
            )
            self._network_model = OrganSMNISTNetwork()

        elif dataset_keyword == "lung_photos":
            dataset_loader = ImageDatasetLoader(
                transformer=lung_cancer_image_transformer,
                **common_kwargs
            )
            self._network_model = LungCancerCNN()

        elif dataset_keyword == "medquad":
            dataset_loader = MedQuADDatasetLoader(
                model_name=self.strategy_config.llm_model,
                chunk_size=self.strategy_config.llm_chunk_size,
                mlm_probability=self.strategy_config.mlm_probability,
                num_poisoned_clients=self.strategy_config.num_of_malicious_clients,
                **common_kwargs
            )
            if self.strategy_config.llm_finetuning == "lora":
                self._network_model = load_model_with_lora(
                    model_name=self.strategy_config.llm_model,
                    lora_rank=self.strategy_config.lora_rank,
                    lora_alpha=self.strategy_config.lora_alpha,
                    lora_dropout=self.strategy_config.lora_dropout,
                    lora_target_modules=["query", "value"],
                )
            else:
                self._network_model = load_model(
                    model_name=self.strategy_config.llm_model,
                )
        else:
            logging.error(
                f"You are parsing a strategy for dataset: {dataset_keyword}. "
                f"Check that you assign a correct dataset_loader at the code above."
            )
            sys.exit(-1)

        self._dataset_loader = dataset_loader
        self._trainloaders, self._valloaders = dataset_loader.load_datasets()

    def _assign_aggregation_strategy(self) -> None:
        """Assign aggregation strategy"""

        aggregation_strategy_keyword = self.strategy_config.aggregation_strategy_keyword

        common_kwargs = dict(
            initial_parameters=ndarrays_to_parameters(self._get_model_params(self._network_model)),
            min_fit_clients=self.strategy_config.min_fit_clients,
            min_evaluate_clients=self.strategy_config.min_evaluate_clients,
            min_available_clients=self.strategy_config.min_available_clients,
            evaluate_metrics_aggregation_fn=self.strategy_config.evaluate_metrics_aggregation_fn,
            fit_metrics_aggregation_fn=weighted_average,
            remove_clients=self.strategy_config.remove_clients,
            begin_removing_from_round=self.strategy_config.begin_removing_from_round,
            strategy_history=self.strategy_history,
        )

        if aggregation_strategy_keyword == "trust":
            self._aggregation_strategy = TrustBasedRemovalStrategy(
                beta_value=self.strategy_config.beta_value,
                trust_threshold=self.strategy_config.trust_threshold,
                **common_kwargs
            )
        elif aggregation_strategy_keyword in ("pid", "pid_scaled", "pid_standardized", "pid_standardized_score_based"):
            self._aggregation_strategy = PIDBasedRemovalStrategy(
                ki=self.strategy_config.Ki,
                kp=self.strategy_config.Kp,
                kd=self.strategy_config.Kd,
                num_std_dev=self.strategy_config.num_std_dev,
                network_model=self._network_model,
                aggregation_strategy_keyword=aggregation_strategy_keyword,
                use_lora=True if self.strategy_config.use_llm and self.strategy_config.llm_finetuning == "lora" else False,
                **common_kwargs
            )
        elif aggregation_strategy_keyword == "krum":
            self._aggregation_strategy = KrumBasedRemovalStrategy(
                num_malicious_clients=self.strategy_config.num_of_malicious_clients,
                num_krum_selections=self.strategy_config.num_krum_selections,
                **common_kwargs
            )
        elif aggregation_strategy_keyword == "multi-krum-based":
            self._aggregation_strategy = MultiKrumBasedRemovalStrategy(
                num_of_malicious_clients=self.strategy_config.num_of_malicious_clients,
                num_krum_selections=self.strategy_config.num_krum_selections,
                **common_kwargs
            )
        elif aggregation_strategy_keyword == "multi-krum":
            self._aggregation_strategy = MultiKrumStrategy(
                num_of_malicious_clients=self.strategy_config.num_of_malicious_clients,
                num_krum_selections=self.strategy_config.num_krum_selections,
                **common_kwargs
            )
        elif aggregation_strategy_keyword == "trimmed_mean":
            self._aggregation_strategy = TrimmedMeanBasedRemovalStrategy(
                trim_ratio=self.strategy_config.trim_ratio,
                **common_kwargs
            )

        elif aggregation_strategy_keyword == "rfa":
            self._aggregation_strategy = RFABasedRemovalStrategy(
                **common_kwargs
            )

        elif aggregation_strategy_keyword == "bulyan":
            self._aggregation_strategy = BulyanStrategy(
                num_krum_selections=self.strategy_config.num_krum_selections,
                **common_kwargs
            )

        else:
            raise NotImplementedError(f"The strategy {aggregation_strategy_keyword} not implemented!")

    def client_fn(self, cid: str) -> Client:
        """Create a Flower client."""

        net = self._network_model.to(self.strategy_config.training_device)

        use_lora = True if self.strategy_config.use_llm and self.strategy_config.llm_finetuning == "lora" else False

        trainloader = self._trainloaders[int(cid)]
        valloader = self._valloaders[int(cid)]

        return FlowerClient(
            client_id=int(cid),
            net=net,
            trainloader=trainloader,
            valloader=valloader,
            training_device=self.strategy_config.training_device,
            num_of_client_epochs=self.strategy_config.num_of_client_epochs,
            model_type=self.strategy_config.model_type,
            use_lora=use_lora,
            num_malicious_clients=self.strategy_config.num_of_malicious_clients,
        ).to_client()

    @staticmethod
    def _get_model_params(model):
        """
        Convert initial model params to suitable format.
        - For PEFT/LoRA models: return only LoRA adapter params
        - For regular models (CNN, etc.): return full state_dict
        """

        if isinstance(model, PeftModel):
            state_dict = get_peft_model_state_dict(model)
            return [val.cpu().numpy() for val in state_dict.values()]

        else:
            return [val.cpu().numpy() for _, val in model.state_dict().items()]
