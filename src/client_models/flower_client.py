import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import List

from src.network_models.bert_model_definition import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

# ---- Reproducibility helpers (you added these earlier) ----
from src.utils.seed import seed_everything, per_round_client_seed


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id,
        net,
        trainloader: DataLoader,
        valloader: DataLoader,
        training_device,
        num_of_client_epochs,
        model_type: str = "cnn",
        use_lora: bool = False,
        num_malicious_clients: int = 0,
    ):
        self.client_id = str(client_id)
        self.net = net
        self.trainloader = trainloader          # should already be deterministic from your loader patch
        self.valloader = valloader              # (shuffle=False)
        self.training_device = training_device
        self.model_type = model_type
        self.num_of_client_epochs = int(num_of_client_epochs)
        self.use_lora = use_lora
        self.num_malicious_clients = int(num_malicious_clients)

    # ---------- Parameter utilities ----------

    def set_parameters(self, net, parameters: List[np.ndarray]):
        if self.use_lora:
            params_dict = zip(get_peft_model_state_dict(net).keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            set_peft_model_state_dict(net, state_dict)
        else:
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.net.load_state_dict(state_dict, strict=False)

    def get_parameters(self, config):
        if self.use_lora:
            state_dict = get_peft_model_state_dict(self.net)
            return [val.detach().cpu().numpy() for val in state_dict.values()]
        else:
            return [val.detach().cpu().numpy() for _, val in self.net.state_dict().items()]

    # ---------- Training / Eval ----------

    def train(self, net, trainloader: DataLoader, epochs: int, verbose: bool = False, global_params=None, mu=0.01):
        """Train the network on the training set."""
        if self.model_type == "cnn":
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(net.parameters())
            net.train()

            for epoch in range(epochs):
                correct, total, epoch_loss = 0, 0, 0.0

                for images, labels in trainloader:
                    images, labels = images.to(self.training_device), labels.to(self.training_device)
                    optimizer.zero_grad(set_to_none=True)
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # Metrics
                    epoch_loss += float(loss.detach())
                    total += labels.size(0)
                    correct += (torch.max(outputs.detach(), 1)[1] == labels).sum().item()

                denom = len(trainloader.dataset) if hasattr(trainloader, "dataset") and len(trainloader.dataset) > 0 else 1
                epoch_loss /= denom
                epoch_acc = correct / total if total > 0 else 0.0
                if verbose:
                    print(f"Epoch {epoch + 1}: train loss {epoch_loss:.6f}, accuracy {epoch_acc:.4f}")

            return float(epoch_loss), float(epoch_acc)

        elif self.model_type == "transformer":
            optimizer = torch.optim.AdamW(net.parameters(), lr=5e-5)
            net.train()

            for epoch in range(epochs):
                total_loss = 0.0
                correct, total = 0, 0

                for batch in trainloader:
                    batch = {k: v.to(self.training_device) for k, v in batch.items()}
                    labels = batch["labels"]

                    outputs = net(**batch)
                    loss = outputs.loss

                    # FedProx (only for honest clients when global params available)
                    if (global_params is not None) and (int(self.client_id) >= self.num_malicious_clients):
                        local_params = [torch.tensor(p, device=self.training_device) for p in self.get_parameters(config=None)]
                        prox_term = sum(torch.norm(lp - gp) ** 2 for lp, gp in zip(local_params, global_params))
                        loss = loss + (mu / 2.0) * prox_term

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    total_loss += float(loss.detach())

                    if hasattr(outputs, "logits"):
                        preds = torch.argmax(outputs.logits.detach(), dim=-1)
                        mask = labels != -100
                        correct += (preds[mask] == labels[mask]).sum().item()
                        total += int(mask.sum().item())

                denom = len(trainloader) if len(trainloader) > 0 else 1
                epoch_loss = total_loss / denom
                epoch_acc = correct / total if total > 0 else 0.0

                if verbose:
                    print(f"Epoch {epoch + 1}: train loss {epoch_loss:.6f}, accuracy {epoch_acc:.4f}")

            return float(epoch_loss), float(epoch_acc)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported types are 'cnn' and 'transformer'.")

    def test(self, net, testloader: DataLoader):
        """Evaluate the network on the entire test set."""
        if self.model_type == "cnn":
            criterion = torch.nn.CrossEntropyLoss()
            correct, total, loss_sum = 0, 0, 0.0
            net.eval()

            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to(self.training_device), labels.to(self.training_device)
                    outputs = net(images)
                    loss_sum += float(criterion(outputs, labels).detach())
                    _, predicted = torch.max(outputs.detach(), 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            denom = len(testloader.dataset) if hasattr(testloader, "dataset") and len(testloader.dataset) > 0 else 1
            loss = loss_sum / denom
            accuracy = correct / total if total > 0 else 0.0
            return float(loss), float(accuracy)

        elif self.model_type == "transformer":
            net.eval()
            total_loss = 0.0
            correct, total = 0, 0

            with torch.no_grad():
                for batch in testloader:
                    batch = {k: v.to(self.training_device) for k, v in batch.items()}
                    labels = batch["labels"]

                    outputs = net(**batch)
                    total_loss += float(outputs.loss.detach())

                    if hasattr(outputs, "logits"):
                        preds = torch.argmax(outputs.logits.detach(), dim=-1)
                        mask = labels != -100
                        correct += (preds[mask] == labels[mask]).sum().item()
                        total += int(mask.sum().item())

            denom = len(testloader) if len(testloader) > 0 else 1
            loss = total_loss / denom
            accuracy = correct / total if total > 0 else 0.0
            return float(loss), float(accuracy)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported types are 'cnn' and 'transformer'.")

    # ---------- Flower hooks ----------

    def fit(self, parameters, config):
        # ---- Reproducibility: per-(client, round) seed ----
        # Expect "server_round" in config; take "seed" if the server injects it (fallback 1337).
        rnd = int(config.get("server_round", 0))
        global_seed = int(config.get("seed", 1337))
        round_seed = per_round_client_seed(global_seed=global_seed, cid=self.client_id, rnd=rnd)
        seed_everything(round_seed)

        # Load the incoming parameters
        self.set_parameters(self.net, parameters)

        # Capture global LoRA parameters for FedProx (only for honest clients)
        global_params = None
        if (self.model_type == "transformer") and self.use_lora and (int(self.client_id) >= self.num_malicious_clients):
            global_params = [torch.tensor(p, device=self.training_device) for p in self.get_parameters(config=None)]

        # Train with deterministic loader order (your loader already shuffles deterministically)
        epoch_loss, epoch_acc = self.train(
            self.net,
            self.trainloader,
            epochs=self.num_of_client_epochs,
            global_params=global_params,
        )

        # ---- Fixed offset reseed before gradient pass ----
        # Ensures this second pass consumes RNG in a known state across runs.
        seed_everything(round_seed + 1)

        # calculate gradients (kept for your pipeline; ensures identical order each run)
        optimizer = torch.optim.Adam(self.net.parameters())
        optimizer.zero_grad(set_to_none=True)

        if self.model_type == "cnn":
            criterion = torch.nn.CrossEntropyLoss()
            self.net.train()
            for images, labels in self.trainloader:
                images, labels = images.to(self.training_device), labels.to(self.training_device)
                outputs = self.net(images)
                loss = criterion(outputs, labels)
                loss.backward()

        elif self.model_type == "transformer":
            # No-op in your original code
            pass

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported types are 'cnn' and 'transformer'.")

        # NOTE: keeping your original semantics for num_examples returned (batches count),
        # to avoid changing aggregation behavior in your current experiments.
        return self.get_parameters(self.net), len(self.trainloader), {"loss": float(epoch_loss), "accuracy": float(epoch_acc)}

    def evaluate(self, parameters, config):
        # ---- Reproducibility: per-(client, round) seed ----
        rnd = int(config.get("server_round", 0))
        global_seed = int(config.get("seed", 1337))
        round_seed = per_round_client_seed(global_seed=global_seed, cid=self.client_id, rnd=rnd)
        seed_everything(round_seed)

        self.set_parameters(self.net, parameters)
        loss, accuracy = self.test(self.net, self.valloader)

        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
