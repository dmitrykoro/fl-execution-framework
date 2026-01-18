import logging

import flwr as fl
import numpy as np
import torch

from collections import OrderedDict
from src.network_models.bert_model_definition import get_peft_model_state_dict, set_peft_model_state_dict
from src.attack_utils.poisoning import should_poison_this_round, apply_poisoning_attack
from src.attack_utils.attack_snapshots import save_attack_snapshot, save_visual_snapshot
from src.utils.legacy.ner_metrics_medmentions import StrictMentionAndDocEvaluator


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
            attacks_schedule=None,
            save_attack_snapshots=False,
            attack_snapshot_format="pickle_and_visual",
            snapshot_max_samples=5,
            output_dir=None,
            experiment_info=None,
            strategy_number=0,
            tokenizer=None,
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
        self.attacks_schedule = attacks_schedule
        self.save_attack_snapshots = save_attack_snapshots
        self.attack_snapshot_format = attack_snapshot_format
        self.snapshot_max_samples = snapshot_max_samples
        self.output_dir = output_dir
        self.experiment_info = experiment_info
        self.strategy_number = strategy_number
        self.tokenizer = tokenizer

    def _save_attack_snapshots(self, current_round, attack_configs, data_sample, labels_sample, original_data_sample=None, original_labels_sample=None):
        """Helper method to save attack snapshots for both CNN and transformer models.

        Args:
            current_round: Current training round
            attack_configs: List of attack configurations applied
            data_sample: Poisoned data (images for CNN, input_ids for transformer)
            labels_sample: Poisoned labels
            original_data_sample: Original unpoisoned data (images for CNN, input_ids for transformer)
            original_labels_sample: Original unpoisoned labels (for label flipping)
        """
        if not (self.save_attack_snapshots and self.output_dir):
            return

        # Save pickle/JSON snapshot
        save_attack_snapshot(
            client_id=self.client_id,
            round_num=current_round,
            attack_config=attack_configs,
            data_sample=data_sample,
            labels_sample=labels_sample,
            original_labels_sample=original_labels_sample,
            output_dir=self.output_dir,
            max_samples=self.snapshot_max_samples,
            save_format=self.attack_snapshot_format,
            experiment_info=self.experiment_info,
            strategy_number=self.strategy_number,
        )

        # Save visual snapshot (PNG for CNN, TXT for transformer)
        if self.attack_snapshot_format in ["visual", "pickle_and_visual"]:
            if self.model_type == "cnn" and original_data_sample is not None:
                save_visual_snapshot(
                    client_id=self.client_id,
                    round_num=current_round,
                    attack_config=attack_configs,
                    data_sample=data_sample.cpu().numpy(),
                    labels_sample=labels_sample.cpu().numpy(),
                    original_labels_sample=original_labels_sample.cpu().numpy() if original_labels_sample is not None else labels_sample.cpu().numpy(),
                    output_dir=self.output_dir,
                    experiment_info=self.experiment_info,
                    strategy_number=self.strategy_number,
                    original_data_sample=original_data_sample.cpu().numpy(),
                )
            elif self.model_type == "transformer" and original_data_sample is not None and self.tokenizer is not None:
                save_visual_snapshot(
                    client_id=self.client_id,
                    round_num=current_round,
                    attack_config=attack_configs,
                    data_sample=data_sample.cpu().numpy(),
                    labels_sample=labels_sample.cpu().numpy(),
                    original_labels_sample=original_labels_sample.cpu().numpy() if original_labels_sample is not None else labels_sample.cpu().numpy(),
                    output_dir=self.output_dir,
                    experiment_info=self.experiment_info,
                    strategy_number=self.strategy_number,
                    tokenizer=self.tokenizer,
                    original_data_sample=original_data_sample.cpu().numpy(),
                )

    def _to_device_only_tensors(self, batch):
        """Helper to move only tensor values to device, leaving non-tensors (like lists) as-is."""
        return {k: (v.to(self.training_device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def set_parameters(self, net, parameters: list[np.ndarray]):
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

    def train(self, net, trainloader, epochs: int, verbose=False, global_params=None, mu=0.01, config=None):
        """Train the network on the training set with optional dynamic poisoning."""
        current_round = config.get("server_round", 1) if config else 1

        if self.model_type == "cnn":
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(net.parameters())
            net.train()

            # Determine number of classes
            if hasattr(net, 'fc3'):
                num_classes = net.fc3.out_features
            elif hasattr(net, 'fc'):
                num_classes = net.fc.out_features
            else:
                # Default fallback
                num_classes = 10

            for epoch in range(epochs):
                correct, total, epoch_loss = 0, 0, 0.0

                for batch_idx, (images, labels) in enumerate(trainloader):
                    should_poison, attack_configs = should_poison_this_round(
                        current_round, self.client_id, self.attacks_schedule
                    )

                    if should_poison and attack_configs:
                        original_images = images.clone()
                        original_labels = labels.clone()

                        # Apply all attacks sequentially
                        for attack_config in attack_configs:
                            images, labels = apply_poisoning_attack(images, labels, attack_config, tokenizer=self.tokenizer, num_classes=num_classes)

                        # Save snapshot after all attacks applied
                        if epoch == 0 and batch_idx == 0:
                            self._save_attack_snapshots(
                                current_round=current_round,
                                attack_configs=attack_configs,
                                data_sample=images,
                                labels_sample=labels,
                                original_data_sample=original_images,
                                original_labels_sample=original_labels,
                            )

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

                    # Free intermediate tensors
                    del outputs, loss

                epoch_loss /= len(trainloader.dataset) if len(trainloader.dataset) > 0 else 1
                epoch_acc = correct / total if total > 0 else 0.0
                if verbose:
                    print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")

            return float(epoch_loss), float(epoch_acc)

        elif self.model_type == "transformer":
            optimizer = torch.optim.AdamW(net.parameters(), lr=5e-5)
            net.train()

            for epoch in range(epochs):
                logging.debug(f"[Client {self.client_id}] Starting epoch {epoch + 1}/{epochs}")
                total_loss = 0
                correct, total = 0, 0

                for batch_idx, batch in enumerate(trainloader):
                    should_poison, attack_configs = should_poison_this_round(
                        current_round, self.client_id, self.attacks_schedule
                    )
                    if should_poison and attack_configs:
                        # Capture original data before poisoning
                        original_input_ids = batch["input_ids"].clone()
                        original_labels = batch["labels"].clone()

                        # Save snapshot after all attacks applied
                        if epoch == 0 and batch_idx == 0:
                            self._save_attack_snapshots(
                                current_round=current_round,
                                attack_configs=attack_configs,
                                data_sample=batch["input_ids"],
                                labels_sample=batch["labels"],
                                original_data_sample=original_input_ids,
                                original_labels_sample=original_labels,
                            )

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

                    # Log progress every 10 batches
                    if (batch_idx + 1) % 10 == 0:
                        logging.debug(f"[Client {self.client_id}] Batch {batch_idx + 1}/{len(trainloader)} - Loss: {loss.item():.4f}")

                    # Free intermediate tensors
                    del outputs, loss, batch

                epoch_loss = total_loss / len(trainloader)
                epoch_acc = correct / total if total > 0 else 0
                logging.debug(f"[Client {self.client_id}] Epoch {epoch + 1} complete - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

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
        logging.debug(f"[Client {self.client_id}] Starting fit() - Setting parameters and beginning training")
        self.set_parameters(self.net, parameters)

        # Capture global LoRA parameters for FedProx
        global_params = None
        if self.model_type == "transformer" and self.use_lora and self.client_id >= self.num_malicious_clients:
            global_params = [torch.tensor(p, device=self.training_device) for p in self.get_parameters(config=None)]

        logging.debug(f"[Client {self.client_id}] Training for {self.num_of_client_epochs} epoch(s) with {len(self.trainloader)} batches")
        epoch_loss, epoch_acc = self.train(self.net, self.trainloader, epochs=self.num_of_client_epochs, global_params=global_params, config=config)
        logging.debug(f"[Client {self.client_id}] Training complete - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.get_parameters(self.net), len(self.trainloader), {"loss": epoch_loss, "accuracy": epoch_acc}

    def evaluate(self, parameters, config):
        self.set_parameters(self.net, parameters)

        if self.model_type == "transformer":
            out = self.test(self.net, self.valloader)

            # test() may return (loss, acc) or (loss, acc, mm)
            if isinstance(out, tuple) and len(out) == 3:
                loss, accuracy, mm = out
                metrics = {
                    "accuracy": float(accuracy),
                    "mention_precision": mm["mention_precision"],
                    "mention_recall":    mm["mention_recall"],
                    "mention_f1":        mm["mention_f1"],
                    "document_precision": mm["document_precision"],
                    "document_recall":    mm["document_recall"],
                    "document_f1":        mm["document_f1"],
                    # raw counts for micro-averaging on the server
                    "tp_m": mm["tp_m"], "fp_m": mm["fp_m"], "fn_m": mm["fn_m"],
                    "tp_d": mm["tp_d"], "fp_d": mm["fp_d"], "fn_d": mm["fn_d"],
                }
            else:
                loss, accuracy = out
                metrics = {"accuracy": float(accuracy)}

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return float(loss), len(self.valloader), metrics

        else:
            loss, accuracy = self.test(self.net, self.valloader)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
