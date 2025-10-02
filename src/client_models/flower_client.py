import logging

import flwr as fl
import numpy as np
import torch

from collections import OrderedDict
from typing import List

from network_models.bert_model_definition import get_peft_model_state_dict, set_peft_model_state_dict
from utils.legacy.ner_metrics_medmentions import StrictMentionAndDocEvaluator


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

    def _to_device_only_tensors(self, batch):
        return {k: (v.to(self.training_device) if torch.is_tensor(v) else v) for k, v in batch.items()}


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
                    batch = self._to_device_only_tensors(batch)
                    model_inputs = {
                        "input_ids": batch["input_ids"],
                        "attention_mask": batch.get("attention_mask"),
                        "labels": batch["labels"],
                    }
                    outputs = net(**model_inputs)
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
                        labels = model_inputs["labels"]
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
            total_loss = 0.0
            correct, total = 0, 0

            evaluator = None
            strict_enabled = None  # decide on first batch

            with torch.no_grad():
                for batch in testloader:
                    batch_dev = self._to_device_only_tensors(batch)

                    # Decide once whether we can run MedMentions strict metrics
                    if strict_enabled is None:
                        strict_enabled = ("doc_id" in batch) and ("word_length" in batch)
                        if strict_enabled:
                            evaluator = StrictMentionAndDocEvaluator(
                                id2label=self.net.config.id2label, label_only=True
                            )

                    model_inputs = {
                        "input_ids": batch_dev["input_ids"],
                        "attention_mask": batch_dev.get("attention_mask"),
                        "labels": batch_dev["labels"],
                    }

                    outputs = net(**model_inputs)
                    total_loss += outputs.loss.item()

                    if hasattr(outputs, "logits"):
                        preds = torch.argmax(outputs.logits, dim=-1)
                        labels = model_inputs["labels"]
                        mask = labels != -100
                        correct += (preds[mask] == labels[mask]).sum().item()
                        total   += mask.sum().item()

                        # Only update strict metrics if this is a MedMentions NER batch
                        if strict_enabled and evaluator is not None:
                            evaluator.update_batch(
                                outputs.logits.detach().cpu(),
                                labels.detach().cpu(),
                                batch["doc_id"],       # python list
                                batch["word_length"],  # python list
                            )

            loss = total_loss / max(len(testloader), 1)
            accuracy = (correct / total) if total > 0 else 0.0

            if strict_enabled and evaluator is not None:
                mm = evaluator.finalize()
                return loss, accuracy, mm
            else:
                # fall back to old two-tuple result for non-NER transformer tasks
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
        if self.model_type == "transformer":
            loss, accuracy,mm = self.test(self.net,self.valloader)
            metrics = {
                "accuracy": float(accuracy),
                "mention_precision": mm["mention_precision"],
                "mention_recall": mm["mention_recall"],
                "mention_f1": mm["mention_f1"],
                "document_precision": mm["document_precision"],
                "document_recall": mm["document_recall"],
                "document_f1": mm["document_f1"],
                # raw counts for micro-averaging on the server
                "tp_m": mm["tp_m"], "fp_m": mm["fp_m"], "fn_m": mm["fn_m"],
                "tp_d": mm["tp_d"], "fp_d": mm["fp_d"], "fn_d": mm["fn_d"],
            }
            return float(loss), len(self.valloader), metrics
        else:
            loss, accuracy = self.test(self.net, self.valloader)
            return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
