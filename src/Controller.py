from typing import List, Tuple

import torch
from datasets.utils.logging import disable_progress_bar

import flwr as fl
from flwr.common import Metrics

from Network import Net
from FlowerClient import FlowerClient
from LoadDataset import LoadDataset
from TrustReputationStrategy import TrustPermanentRemovalStrategy
import Plots as plot_fn

from csv_writer import CSVWriter

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)
disable_progress_bar()

NUM_CLIENTS = 12
BATCH_SIZE = 4
ROOT_DIR = 'CLIENT_DATA'

# Loading dataset based on number of clients and batch size
data_loader = LoadDataset(ROOT_DIR, NUM_CLIENTS, BATCH_SIZE)
trainloaders, valloaders, testloaders = data_loader.load_datasets()

# data_loader.display_samples_from_loader(trainloaders[0], "Sample")


def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)

    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader).to_client()


# Specify the resources each of your clients need. By default, each
# client will be allocated 1x CPU and 0x GPUs
client_resources = {"num_cpus": 1, "num_gpus": 0.0}
if DEVICE.type == "cuda":
    # here we are asigning an entire GPU for each client.
    client_resources = {"num_cpus": 1, "num_gpus": 1.0}
    # Refer to our documentation for more details about Flower Simulations
    # and how to setup these `client_resources`.


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Create FedAvg strategy
# strategy = fl.server.strategy.FedAvg(
#     fraction_fit=1.0,
#     fraction_evaluate=1.0,
#     min_fit_clients=4,
#     min_evaluate_clients=4,
#     min_available_clients=4,
#     evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
# )


# RUN WITH CLIENTS REMOVAL

print('############# REMOVAL STRATEGY')
removal_strategy = TrustPermanentRemovalStrategy(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=4,  # Never sample less than 10 clients for training
    min_evaluate_clients=4,  # Never sample less than 5 clients for evaluation
    min_available_clients=4,  # Wait until all 10 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,
    remove_clients=True
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=removal_strategy,
    client_resources=client_resources,
)

accuracy_trust_reputation_data, loss_data = plot_fn.process_client_data(removal_strategy)

writer = CSVWriter(
    accuracy_trust_reputation_data=accuracy_trust_reputation_data,
    loss_data=loss_data,
    strategy_type='remove'
)
# writer.dump_to_file()
writer.write_to_csv()
writer.write_loss_to_csv()


# RUN WITHOUT CLIENTS REMOVAL

print('############# NO REMOVAL STRATEGY')
non_removal_strategy = TrustPermanentRemovalStrategy(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=4,  # Never sample less than 10 clients for training
    min_evaluate_clients=4,  # Never sample less than 5 clients for evaluation
    min_available_clients=4,  # Wait until all 10 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,
    remove_clients=False
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=non_removal_strategy,
    client_resources=client_resources,
)

accuracy_trust_reputation_data, loss_data = plot_fn.process_client_data(non_removal_strategy)

writer = CSVWriter(
    accuracy_trust_reputation_data=accuracy_trust_reputation_data,
    loss_data=loss_data,
    strategy_type='no_remove'
)
writer.write_to_csv()
writer.write_loss_to_csv()

'''plot_fn.plot_accuracy_history(data)
plot_fn.plot_accuracy_for_round(data, 1)
plot_fn.plot_accuracy_for_round(data, 5)
plot_fn.plot_accuracy_for_round(data, 10)
plot_fn.plot_trust_history(data)
plot_fn.plot_reputation_history(data)'''
