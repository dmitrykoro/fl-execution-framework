import logging

import flwr as fl
import numpy as np
import torch

from collections import OrderedDict
from typing import List

from src.network_models.bert_model_definition import get_peft_model_state_dict, set_peft_model_state_dict


class FlowerClient(fl.client.NumPyClient):
    def __init__(
            self,
            client_id,
            net,
            trainloader,
            valloader,
            training_device,
            num_of_client_epochs,
            model_type="cnn",
            use_lora=False,
            num_malicious_clients=0,
    ):
        self.client_id = client_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.training_device = training_device
        self.model_type = model_type
        self.num_of_client_epochs = num_of_client_epochs
        self.use_lora = use_lora
        self.num_malicious_clients = num_malicious_clients

    def set_parameters(self, net, parameters: List[np.ndarray]):
        if self.use_lora:
            params_dict = zip(get_peft_model_state_dict(net).keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            set_peft_model_state_dict(net, state_dict)
        else:
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.net.load_state_dict(state_dict, strict=False)

    def get_parameters(self, config):
        if self.use_lora:
            state_dict = get_peft_model_state_dict(self.net)
            return [val.cpu().numpy() for val in state_dict.values()]
        else:
            return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def train(self, net, trainloader, epochs: int, verbose=False, global_params=None, mu=0.01):
        """Train the network on the training set."""

        if self.model_type == "cnn":
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(net.parameters())
            net.train()

            for epoch in range(epochs):
                correct, total, epoch_loss = 0, 0, 0.0

                for images, labels in trainloader:
                    images, labels = images.to(self.training_device), labels.to(self.training_device)
                    optimizer.zero_grad()
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # Metrics
                    epoch_loss += loss
                    total += labels.size(0)
                    correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

                epoch_loss /= len(trainloader.dataset) if len(trainloader.dataset) > 0 else 1
                epoch_acc = correct / total if total > 0 else 0.0
                if verbose:
                    print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")

            return float(epoch_loss), float(epoch_acc)

        elif self.model_type == "transformer":
            optimizer = torch.optim.AdamW(net.parameters(), lr=5e-5)
            net.train()

            for epoch in range(epochs):
                total_loss = 0
                correct, total = 0, 0

                for batch in trainloader:
                    batch = {k: v.to(self.training_device) for k, v in batch.items()}
                    labels = batch["labels"]

                    outputs = net(**batch)
                    loss = outputs.loss

                    # Apply FedProx only if global_params are provided
                    if global_params is not None and self.client_id >= self.num_malicious_clients:
                        local_params = [torch.tensor(p, device=self.training_device) for p in self.get_parameters(config=None)]
                        prox_term = sum(torch.norm(lp - gp) ** 2 for lp, gp in zip(local_params, global_params))
                        loss = loss + (mu / 2) * prox_term

                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += loss.item()

                    if hasattr(outputs, "logits"):
                        preds = torch.argmax(outputs.logits, dim=-1)
                        mask = labels != -100
                        correct += (preds[mask] == labels[mask]).sum().item()
                        total += mask.sum().item()

                epoch_loss = total_loss / len(trainloader)
                epoch_acc = correct / total if total > 0 else 0

                if verbose:
                    print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")

            return float(epoch_loss), float(epoch_acc)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported types are 'cnn' and 'mlm'.")

    def test(self, net, testloader):
        """Evaluate the network on the entire test set."""

        if self.model_type == "cnn":
            criterion = torch.nn.CrossEntropyLoss()
            correct, total, loss = 0, 0, 0.0
            net.eval()

            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to(self.training_device), labels.to(self.training_device)
                    outputs = net(images)
                    loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            loss /= len(testloader.dataset) if len(testloader.dataset) > 0 else 1
            accuracy = correct / total if total > 0 else 0.0
            return loss, accuracy

        # add check for mlm as well
        elif self.model_type == "transformer":
            net.eval()
            total_loss = 0
            correct, total = 0, 0

            with torch.no_grad():
                for batch in testloader:
                    batch = {k: v.to(self.training_device) for k, v in batch.items()}
                    labels = batch["labels"]

                    outputs = net(**batch)
                    loss = outputs.loss.item()
                    total_loss += loss

                    if hasattr(outputs, "logits"):
                        preds = torch.argmax(outputs.logits, dim=-1)
                        mask = labels != -100
                        correct += (preds[mask] == labels[mask]).sum().item()
                        total += mask.sum().item()

            loss = total_loss / len(testloader)
            accuracy = correct / total if total > 0 else 0
            return loss, accuracy

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported types are 'cnn' and 'mlm'.")

    def fit(self, parameters, config):
        self.set_parameters(self.net, parameters)

        # Capture global LoRA parameters for FedProx
        global_params = None
        if self.model_type == "transformer" and self.use_lora and self.client_id >= self.num_malicious_clients:
            global_params = [torch.tensor(p, device=self.training_device) for p in self.get_parameters(config=None)]

        epoch_loss, epoch_acc = self.train(self.net, self.trainloader, epochs=self.num_of_client_epochs, global_params=global_params)

        return self.get_parameters(self.net), len(self.trainloader), {"loss": epoch_loss, "accuracy": epoch_acc}

    def evaluate(self, parameters, config):
        self.set_parameters(self.net, parameters)
        loss, accuracy = self.test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
