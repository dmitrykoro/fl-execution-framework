import flwr as fl
import numpy as np
from collections import OrderedDict
from typing import List
import torch

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def set_parameters(self, net, parameters: List[np.ndarray]):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def train(self, net, trainloader, epochs: int, verbose=False):
        """Train the network on the training set."""
        #criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters())
        net.train()
        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for images, labels in trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
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
                print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

    def test(self, net, testloader):
        """Evaluate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        #criterion = torch.nn.BCELoss()
        correct, total, loss = 0, 0, 0.0
        net.eval()
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        loss /= len(testloader.dataset)
        accuracy = correct / total
        return loss, accuracy

    def fit(self, parameters, config):
        self.set_parameters(self.net, parameters)
        self.train(self.net, self.trainloader, epochs=1)
        # Calculate gradients
        optimizer = torch.optim.Adam(self.net.parameters())
        optimizer.zero_grad()
        criterion = torch.nn.CrossEntropyLoss()
        for images, labels in self.trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = self.net(images)
            loss = criterion(outputs, labels)
            loss.backward()
        parameters = self.get_parameters(self.net)
        
        return self.get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(self.net, parameters)
        loss, accuracy = self.test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
