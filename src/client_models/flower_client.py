import logging
import flwr as fl
import numpy as np
import torch

from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
from src.network_models.bert_model_definition import get_peft_model_state_dict, set_peft_model_state_dict
from src.dataset_handlers.dataset_handler import apply_poisoning_attack
from src.attack_utils.poisoning import (
    should_poison_this_round,
    apply_model_poisoning_attack,
    is_model_poisoning_attack,
)
from src.utils.legacy.ner_metrics_medmentions import StrictMentionAndDocEvaluator
from src.output_handlers import attack_snapshot_handler


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
            target_name="",
            sequence_len=12,
            strategy_history=None,
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
        self.target_name=target_name
        self.sequence_len=sequence_len
        self.strategy_history=strategy_history

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

        client_instance = self.strategy_history.get_client_by_id(self.client_id)
        attack_keywords_list = client_instance.get_current_attack_keywords(current_round)
        current_round_attack_configs = client_instance.get_current_attack_configs(current_round)

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
                    if current_round_attack_configs and epoch == 0 and batch_idx == 0:
                            original_images = images.clone()
                            original_labels = labels.clone()
                    for attack_type, attack_config in zip(attack_keywords_list, current_round_attack_configs):                        

                        images, labels = apply_poisoning_attack(
                            data=images,
                            labels=labels,
                            attack_type=attack_type,
                            attack_config=attack_config,
                            tokenizer=self.tokenizer,
                            num_classes=num_classes
                        )

                    # Save snapshot after all attacks applied
                    if current_round_attack_configs and epoch == 0 and batch_idx == 0:

                        attack_snapshot_handler.save_attack_snapshots(
                            attack_snapshot_format=self.attack_snapshot_format,
                            tokenizer=self.tokenizer,
                            model_type=self.model_type,
                            save_attack_snapshots=self.save_attack_snapshots,
                            client_id=self.client_id,
                            output_dir=self.output_dir,
                            max_samples=self.snapshot_max_samples,
                            experiment_info=self.experiment_info,
                            strategy_number=self.strategy_number,
                            current_round=current_round,
                            attack_configs=current_round_attack_configs,
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

                # calculate local client learning rate
                optimizer_lr = optimizer.param_groups[0]["lr"]
                num_steps = epochs * len(trainloader)
                eta_eff = optimizer_lr * num_steps

            return float(epoch_loss), float(epoch_acc), float(eta_eff)

        elif self.model_type == "transformer":
            optimizer = torch.optim.AdamW(net.parameters(), lr=5e-5)
            net.train()

            for epoch in range(epochs):
                logging.debug(f"[Client {self.client_id}] Starting epoch {epoch + 1}/{epochs}")
                total_loss = 0
                correct, total = 0, 0

                for batch_idx, batch in enumerate(trainloader):
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
                        local_params = [torch.tensor(p, device=self.training_device) for p in
                                        self.get_parameters(config=None)]
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
                        logging.debug(
                            f"[Client {self.client_id}] Batch {batch_idx + 1}/{len(trainloader)} - Loss: {loss.item():.4f}"
                        )

                    # Free intermediate tensors
                    del outputs, loss, batch

                epoch_loss = total_loss / len(trainloader)
                epoch_acc = correct / total if total > 0 else 0
                logging.debug(
                    f"[Client {self.client_id}] Epoch {epoch + 1} complete - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
                )

                if verbose:
                    print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")

            return float(epoch_loss), float(epoch_acc), 0.0
        
        elif self.model_type == "linear_regression":
            #TODO: Consider changing to something more robust, L1Loss for MAE or HuberLoss
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(net.parameters())
            net.train()

            # Collect snapshot data
            snapshot_features = []
            snapshot_targets = []
            snapshot_original_features = []
            snapshot_original_targets = []

            collected = 0

            for epoch in range(epochs):
                total, epoch_loss = 0, 0.0
                # For R^2 calcualtion
                ss_res, sum_y, sum_y_sq = 0.0, 0.0, 0.0

                # For denormalization
                y_mean = trainloader.dataset.y_mean.to(self.training_device)
                y_std = trainloader.dataset.y_std.to(self.training_device)

                for batch_idx, (features, targets) in enumerate(trainloader):
                    if current_round_attack_configs and epoch == 0  and collected < self.snapshot_max_samples:
                        original_features = features.clone()
                        original_targets = targets.clone()
                    # Apply all attacks sequentially
                    for attack_type, attack_config in zip(attack_keywords_list, current_round_attack_configs):
                        features, targets = apply_poisoning_attack(
                            data=features,
                            labels=targets,
                            attack_type=attack_type,
                            attack_config=attack_config,
                            dataset = trainloader.dataset,
                        )

                    # Collect snapshot data (only first epoch)
                    if current_round_attack_configs and epoch == 0 and collected < self.snapshot_max_samples:
                        remaining = self.snapshot_max_samples - collected
                        take = min(remaining, features.size(0))

                        snapshot_features.append(features[:take].clone())
                        snapshot_targets.append(targets[:take].clone())
                        snapshot_original_features.append(original_features[:take].clone())
                        snapshot_original_targets.append(original_targets[:take].clone())

                        collected += take

                    features = features.to(self.training_device)
                    targets = targets.to(self.training_device).float()
                    optimizer.zero_grad()
                    outputs = net(features)

                    # Ensure targets is the same shape as outputs
                    if outputs.shape != targets.shape:
                        targets = targets.view_as(outputs)

                    loss=criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    # Metrics
                    epoch_loss += loss.item() * features.size(0)

                    # denormalize for metrics
                    outputs_real = outputs * y_std + y_mean
                    targets_real = targets * y_std + y_mean
                    
                    ss_res += torch.sum((targets_real-outputs_real)**2).item()
                    sum_y += torch.sum(targets_real).item()
                    sum_y_sq += torch.sum(targets_real**2).item()

                    total += features.size(0)

                    # Free intermediate tensors
                    del outputs, loss
                
                epoch_loss /= total if total > 0 else 1
                y_mean = sum_y / total
                ss_tot = sum_y_sq - total * (y_mean**2)
                epoch_r2 = 1-(ss_res/ss_tot) if ss_tot > 0 else 0.0
                if verbose:
                    print(f"Epoch {epoch + 1}: train MSE {epoch_loss}")

                # calculate local client learning rate
                optimizer_lr = optimizer.param_groups[0]["lr"]
                num_steps = epochs * len(trainloader)
                eta_eff = optimizer_lr * num_steps

            # Save snapshot of attack
            if snapshot_features:
                attack_snapshot_handler.save_attack_snapshots(
                    attack_snapshot_format=self.attack_snapshot_format,
                    model_type=self.model_type,
                    save_attack_snapshots=self.save_attack_snapshots,
                    client_id=self.client_id,
                    output_dir=self.output_dir,
                    max_samples=self.snapshot_max_samples,
                    experiment_info=self.experiment_info,
                    strategy_number=self.strategy_number,
                    current_round=current_round,
                    attack_configs=current_round_attack_configs,
                    data_sample=torch.cat(snapshot_features),
                    labels_sample=torch.cat(snapshot_targets),
                    target_name=self.target_name,
                    original_data_sample=torch.cat(snapshot_original_features),
                    original_labels_sample=torch.cat(snapshot_original_targets),
                    dataset=trainloader.dataset,
                )            

            #TODO: Right now returning MSE, maybe use RMSE for more interprebility or MAE
            return float(epoch_loss), float(epoch_r2), float(eta_eff)

        elif self.model_type == "lstm":  
            criterion = torch.nn.SmoothL1Loss(reduction="sum")
            optimizer = torch.optim.Adam(net.parameters())
            net.train()

            # Collect snapshot data
            snapshot_features = []
            snapshot_targets = []
            snapshot_original_features = []
            snapshot_original_targets = []

            collected = 0

            for epoch in range(epochs):
                total, epoch_loss = 0, 0.0
                # For MARE calculation
                eps = 1e-8
                abs_rel_error_sum = 0.0

                # For denormalization
                y_mean = trainloader.dataset.y_mean.to(self.training_device)
                y_std = trainloader.dataset.y_std.to(self.training_device)

                for batch_idx, (features, targets) in enumerate(trainloader):
                    if current_round_attack_configs and epoch == 0  and collected < self.snapshot_max_samples:
                        original_features = features.clone()
                        original_targets = targets.clone()
                    # Apply all attacks sequentially
                    for attack_type, attack_config in zip(attack_keywords_list, current_round_attack_configs):
                        features, targets = apply_poisoning_attack(
                            data=features,
                            labels=targets,
                            attack_type=attack_type,
                            attack_config=attack_config,
                            dataset = trainloader.dataset,
                        )

                    # Collect snapshot data (only first epoch)
                    if current_round_attack_configs and epoch == 0 and collected < self.snapshot_max_samples:
                        remaining = self.snapshot_max_samples - collected
                        take = min(remaining, features.size(0))

                        snapshot_features.append(features[:take].clone())
                        snapshot_targets.append(targets[:take].clone())
                        snapshot_original_features.append(original_features[:take].clone())
                        snapshot_original_targets.append(original_targets[:take].clone())

                        collected += take

                    features = features.to(self.training_device)
                    targets = targets.to(self.training_device).float()
                    optimizer.zero_grad()
                    outputs = net(features)

                    # Ensure targets is the same shape as outputs
                    if outputs.shape != targets.shape:
                        targets = targets.view_as(outputs)

                    loss=criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    # Metrics
                    epoch_loss += loss.item()

                    # denormalize for metrics
                    outputs_real = outputs * y_std + y_mean
                    targets_real = targets * y_std + y_mean
                    
                    abs_rel_error_sum += torch.sum(
                        torch.abs(targets_real - outputs_real) / (torch.abs(targets_real) + eps)
                    ).item()

                    total += targets.numel()

                    # Free intermediate tensors
                    del outputs, loss
                
                # Normalize loss to per-element SmoothL1Loss
                epoch_loss /= total if total > 0 else 1
                epoch_mare = abs_rel_error_sum / total if total > 0 else 0.0
                if verbose:
                    print(f"Epoch {epoch + 1}: train SmoothL1Loss {epoch_loss}")

                # calculate local client learning rate
                optimizer_lr = optimizer.param_groups[0]["lr"]
                num_steps = epochs * len(trainloader)
                eta_eff = optimizer_lr * num_steps

            # Save snapshot of attack
            if snapshot_features:
                attack_snapshot_handler.save_attack_snapshots(
                    attack_snapshot_format=self.attack_snapshot_format,
                    model_type=self.model_type,
                    save_attack_snapshots=self.save_attack_snapshots,
                    client_id=self.client_id,
                    output_dir=self.output_dir,
                    max_samples=self.snapshot_max_samples,
                    experiment_info=self.experiment_info,
                    strategy_number=self.strategy_number,
                    current_round=current_round,
                    attack_configs=current_round_attack_configs,
                    data_sample=torch.cat(snapshot_features),
                    labels_sample=torch.cat(snapshot_targets),
                    target_name=self.target_name,
                    original_data_sample=torch.cat(snapshot_original_features),
                    original_labels_sample=torch.cat(snapshot_original_targets),
                    dataset=trainloader.dataset,
                )            

            return float(epoch_loss), float(epoch_mare), float(eta_eff)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported types are 'cnn', 'mlm' and 'linear_regression'.")

    def test(self, net, testloader, config=None):
        """Evaluate the network on the entire test set with optional dynamic poisoning."""
        current_round = config.get("server_round", 1) if config else 1

        client_instance = self.strategy_history.get_client_by_id(self.client_id)
        attack_keywords_list = client_instance.get_current_attack_keywords(current_round)
        current_round_attack_configs = client_instance.get_current_attack_configs(current_round)

        if self.model_type == "cnn":
            criterion = torch.nn.CrossEntropyLoss()
            correct, total, loss = 0, 0, 0.0
            net.eval()

            # Determine number of classes
            if hasattr(net, 'fc3'):
                num_classes = net.fc3.out_features
            elif hasattr(net, 'fc'):
                num_classes = net.fc.out_features
            else:
                # Default fallback
                num_classes = 10

            with torch.no_grad():
                for images, labels in testloader:

                    for attack_type, attack_config in zip(attack_keywords_list, current_round_attack_configs):
                        images, labels = apply_poisoning_attack(
                            data=images,
                            labels=labels,
                            attack_type=attack_type,
                            attack_config=attack_config,
                            tokenizer=self.tokenizer,
                            num_classes=num_classes
                        )

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
                        total += mask.sum().item()

                        # Only update strict metrics if this is a MedMentions NER batch
                        if strict_enabled and evaluator is not None:
                            evaluator.update_batch(
                                outputs.logits.detach().cpu(),
                                labels.detach().cpu(),
                                batch["doc_id"],  # python list
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
            
        elif self.model_type == "linear_regression":
            criterion = torch.nn.MSELoss(reduction="sum") # sum to compute SS_res directly
            net.eval()

            total, loss_sum, ss_res, ss_tot, y_true_sum, y_true_sq_sum = 0, 0.0, 0.0, 0.0, 0.0, 0.0

            # For denormalization
            y_mean = testloader.dataset.y_mean.to(self.training_device)
            y_std = testloader.dataset.y_std.to(self.training_device)

            with torch.no_grad():
                for features, targets in testloader:
                    # Apply all attacks sequentially
                    for attack_type, attack_config in zip(attack_keywords_list, current_round_attack_configs):
                        features, targets = apply_poisoning_attack(
                            data=features,
                            labels=targets,
                            attack_type=attack_type,
                            attack_config=attack_config,
                            dataset=testloader.dataset,
                        )

                    features, targets = features.to(self.training_device), targets.to(self.training_device)
                    outputs = net(features)

                    # denormalize for metrics
                    outputs_real = outputs * y_std + y_mean
                    targets_real = targets * y_std + y_mean

                    # Sum squared errors
                    batch_ss_res = torch.sum((targets_real-outputs_real)**2).item()
                    ss_res += batch_ss_res

                    # For SS_tot: sum of squares around mean
                    y_true_sum += torch.sum(targets_real).item()
                    y_true_sq_sum += torch.sum(targets_real**2).item()

                    # MSE loss for reporting
                    loss_sum += batch_ss_res
                    total += targets.size(0)
            
            # Normalize loss to per-sample MSE
            loss = loss_sum / total if total > 0 else 0.0

            # Compute total sum of squares
            mean_y = y_true_sum / total if total > 0 else 0.0
            ss_tot = y_true_sq_sum - total * (mean_y**2)
            r2 = 1-(ss_res/ss_tot) if ss_tot != 0 else 0.0

            return loss, ss_res, ss_tot, r2
        
        elif self.model_type == "lstm":
            criterion = torch.nn.SmoothL1Loss(reduction="sum")
            net.eval()

            #total, loss_sum, ss_res, ss_tot, y_true_sum, y_true_sq_sum = 0, 0.0, 0.0, 0.0, 0.0, 0.0
            total, smooth_l1_loss_sum, smooth_l1_loss_baseline_sum = 0, 0.0, 0.0
            eps = 1e-8
            abs_rel_error_sum, abs_rel_error_baseline_sum = 0.0, 0.0

            # For denormalization
            y_mean = testloader.dataset.y_mean.to(self.training_device)
            y_std = testloader.dataset.y_std.to(self.training_device)
            x_mean = testloader.dataset.x_mean.to(self.training_device)
            x_std = testloader.dataset.x_std.to(self.training_device)

            def _rolling_average_model(batch_sequences, window_size):
                """
                Simple moving avg model that serves as a baseline

                Note: uses last window
                """
                test_seqs = batch_sequences[:, -window_size:, :]    # (batch, window_size, 1)
                preds = test_seqs.mean(dim=1)                       # (batch, 1)
                return preds 

            with torch.no_grad():
                for features, targets in testloader:
                    # create baseline outputs using moving average before poisoning
                    features, targets = features.to(self.training_device), targets.to(self.training_device)
                    features_real = features * x_std + x_mean
                    targets_real = targets * y_std + y_mean

                    outputs_baseline = _rolling_average_model(features_real, self.sequence_len)
                    outputs_baseline = outputs_baseline.to(self.training_device)
                    outputs_baseline = outputs_baseline.view_as(targets_real)   # Ensure baseline_outputs is the same shape as targets

                    smooth_l1_loss_baseline_sum += criterion(outputs_baseline, targets_real).item()
                    abs_rel_error_baseline_sum += torch.sum(
                        torch.abs(targets_real - outputs_baseline) / torch.abs(targets_real + eps)
                    ).item()    

                    # Apply all attacks sequentially
                    for attack_type, attack_config in zip(attack_keywords_list, current_round_attack_configs):
                        features, targets = apply_poisoning_attack(
                            data=features,
                            labels=targets,
                            attack_type=attack_type,
                            attack_config=attack_config,
                            dataset=testloader.dataset,
                        )

                    outputs = net(features)

                    # denormalize for metrics
                    outputs_real = outputs * y_std + y_mean
                    targets_real = targets * y_std + y_mean

                    # SmoothL1Loss for reporting
                    smooth_l1_loss_sum += criterion(outputs_real, targets_real).item()

                    # Sum absolute relative errors for MARE
                    abs_rel_error_sum += torch.sum(
                        torch.abs(targets_real - outputs_real) / (torch.abs(targets_real) + eps)
                    ).item()

                    total += targets.numel()
            
            # Normalize loss to per-element SmoothL1Loss
            loss = smooth_l1_loss_sum / total if total > 0 else 0.0
            loss_baseline = smooth_l1_loss_baseline_sum / total if total > 0 else 0.0
            mare = abs_rel_error_sum / total if total > 0 else 0.0
            mare_baseline = abs_rel_error_baseline_sum / total if total > 0 else 0.0

            return loss, loss_baseline, mare, mare_baseline, total

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported types are 'cnn', 'mlm' and 'linear_regression'.")

    def fit(self, parameters, config):
        logging.debug(f"[Client {self.client_id}] Starting fit() - Setting parameters and beginning training")
        global_params_ndarrays = [p.copy() for p in parameters]
        self.set_parameters(self.net, parameters)

        # Capture global LoRA parameters for FedProx
        global_params = None
        if self.model_type == "transformer" and self.use_lora and self.client_id >= self.num_malicious_clients:
            global_params = [torch.tensor(p, device=self.training_device) for p in self.get_parameters(config=None)]

        logging.debug(
            f"[Client {self.client_id}] Training for {self.num_of_client_epochs} epoch(s) with {len(self.trainloader)} batches"
        )
        epoch_loss, epoch_acc, eta_eff = self.train(
            self.net, self.trainloader, epochs=self.num_of_client_epochs, global_params=global_params, config=config
        )
        logging.debug(
            f"[Client {self.client_id}] Training complete - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
        )

        if global_params is not None:
            del global_params

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # for Flower summary purposes
        if self.model_type == "linear_regression":
            metrics =  {"loss": epoch_loss, "r2": epoch_acc, "eta_eff": eta_eff}
        elif self.model_type == "lstm":
            metrics = {"loss": epoch_loss, "mare": epoch_acc, "eta_eff": eta_eff}
        else:
            metrics =  {"loss": epoch_loss, "accuracy": epoch_acc, "eta_eff": eta_eff}

        local_params = self.get_parameters(self.net)

        current_round = config.get("server_round", 1) if config else 1
        should_poison, attack_configs = should_poison_this_round(
            current_round, self.client_id, self.attacks_schedule
        )
        if should_poison and attack_configs:
            for attack_config in attack_configs:
                if is_model_poisoning_attack(attack_config.get("attack_type")):
                    local_params = apply_model_poisoning_attack(
                        local_params, global_params_ndarrays, metrics
                    )
        # For backwards compatibility 'attack_type' param
        if self.strategy_history and self.strategy_history.strategy_config.attack_type:
            client = self.strategy_history.get_client_by_id(self.client_id)
            current_attack_keywords = client.get_current_attack_keywords(current_round)
            if "gradient_scaling" in current_attack_keywords:
                attack_config = {
                    "attack_type": "gradient_scaling",
                    "scale_factor": self.strategy_history.strategy_config.scale_factor
                }
                local_params = apply_model_poisoning_attack(
                    local_params, global_params_ndarrays, attack_config
                )
            

        return local_params, len(self.trainloader), {"loss": epoch_loss, "accuracy": epoch_acc, "eta_eff": eta_eff}

    def evaluate(self, parameters, config):
        self.set_parameters(self.net, parameters)

        if self.model_type == "transformer":
            out = self.test(self.net, self.valloader, config)

            # test() may return (loss, acc) or (loss, acc, mm)
            if isinstance(out, tuple) and len(out) == 3:
                loss, accuracy, mm = out
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
            else:
                loss, accuracy = out
                metrics = {"accuracy": float(accuracy)}

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return float(loss), len(self.valloader), metrics
        
        elif self.model_type == "linear_regression":
            out = self.test(self.net, self.valloader)

            # for linear_regression, test() returns (loss, ss_res, ss_tot, r2)
            if isinstance(out, tuple) and len(out) == 4:
                loss, ss_res, ss_tot, r2 = out
                metrics = {
                    "ss_res": float(ss_res),
                    "ss_tot": float(ss_tot),
                    "r2": float(r2)
                }
            # should never get to else
            else:
                loss, accuracy = out
                metrics = {}
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return float(loss), len(self.valloader.dataset), metrics

        elif self.model_type == "lstm":
            out = self.test(self.net, self.valloader)

            # for lstm, test() returns (loss, loss_baseline, mare, mare_baseline)
            if isinstance(out, tuple) and len(out) == 5:
                loss, loss_baseline, mare, mare_baseline, num_elements = out
                metrics = {
                    "loss": float(loss),
                    "loss_baseline": float(loss_baseline),
                    "mare": float(mare),
                    "mare_baseline": float(mare_baseline),
                    "num_elements": int(num_elements)

                }
            # should never get to else
            else:
                loss, accuracy = out
                metrics = {}
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return float(loss), len(self.valloader.dataset), metrics

        else:
            loss, accuracy = self.test(self.net, self.valloader)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
