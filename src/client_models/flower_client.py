import flwr as fl
import numpy as np
import torch

from collections import OrderedDict
from typing import List


class FlowerClient(fl.client.NumPyClient):
    def __init__(
            self,
            net,
            trainloader,
            valloader,
            training_device,
            num_of_client_epochs,
            model_type="cnn"
    ):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.training_device = training_device
        self.model_type = model_type
        self.num_of_client_epochs = num_of_client_epochs

    # def set_parameters(self, net, state_list):
    #     """
    #     Load full model weights from list of numpy arrays.
    #     """
    #     keys = net.state_dict().keys()
    #     state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, state_list)})
    #     self.net.load_state_dict(state_dict, strict=True)

    def set_parameters(self, net, parameters: List[np.ndarray]):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=False)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def train(self, net, trainloader, epochs: int, verbose=False):
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
                epoch_loss /= len(trainloader.dataset)
                epoch_acc = correct / total
                if verbose:
                    print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")

        elif self.model_type == "transformer":
            optimizer = torch.optim.AdamW(net.parameters())
            net.train()

            for epoch in range(epochs):
                total_loss = 0
                correct, total = 0, 0

                for batch in trainloader:
                    batch = {k: v.to(self.training_device) for k, v in batch.items()}
                    labels = batch["labels"]

                    outputs = net(**batch)
                    loss = outputs.loss
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

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported types are 'cnn' and 'mlm'.")

    def test(self, net, testloader):
        """Evaluate the network on the entire test set."""

        if self.model_type == "transformer":
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
            loss /= len(testloader.dataset)
            accuracy = correct / total
            return loss, accuracy
        
        elif self.model_type == "mlm":
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
        self.train(self.net, self.trainloader, epochs=self.num_of_client_epochs)

        # calculate gradients
        optimizer = torch.optim.Adam(self.net.parameters())
        optimizer.zero_grad()
        criterion = torch.nn.CrossEntropyLoss()
        for images, labels in self.trainloader:
            images, labels = images.to(self.training_device), labels.to(self.training_device)
            outputs = self.net(images)
            loss = criterion(outputs, labels)
            loss.backward()

        return self.get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(self.net, parameters)
        loss, accuracy = self.test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
