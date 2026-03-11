"""
Utility functions for applying poisoning attacks to datasets.
"""

import logging
import random
import math
from typing import Optional, Tuple

import torch


def apply_label_flipping(
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Apply bijective label flipping (each class maps to exactly one other class)."""
    perm = torch.randperm(num_classes).tolist()

    # Ensure no class maps to itself
    for i in range(num_classes):
        if perm[i] == i:
            swap_idx = (i + 1) % num_classes
            perm[i], perm[swap_idx] = perm[swap_idx], perm[i]

    modified_labels = labels.clone()
    for src_class in range(num_classes):
        modified_labels[labels == src_class] = perm[src_class]

    return modified_labels


def apply_gaussian_noise(
    images: torch.Tensor,
    mean: float = 0.0,
    std: float = 0.1,
    target_noise_snr: Optional[float] = None,
    attack_ratio: float = 1.0,
) -> torch.Tensor:
    """Add Gaussian noise to images. Use target_noise_snr (dB) or mean/std."""
    num_samples = images.shape[0]
    num_to_poison = int(num_samples * attack_ratio)

    if num_to_poison == 0:
        return images

    indices = torch.randperm(num_samples)[:num_to_poison]
    poisoned_images = images.clone()

    if target_noise_snr is not None:
        signal_power = torch.mean(
            poisoned_images[indices] ** 2, dim=(1, 2, 3), keepdim=True
        )
        noise_power = signal_power / (10 ** (target_noise_snr / 10))
        noise = torch.randn_like(poisoned_images[indices]) * torch.sqrt(noise_power)
    else:
        noise = torch.randn_like(poisoned_images[indices]) * std + mean

    poisoned_images[indices] = torch.clamp(poisoned_images[indices] + noise, 0, 1)
    return poisoned_images


# TODO: Genralize this for any regression task
def apply_target_flipping(
        targets: torch.Tensor,
        degree_of_poisoning: float = 0.1,
) -> torch.Tensor:
    substitute_fraction = 0.25  # Changing subset size doesn't change effectiveness of attack
    '''
    At a minimum use 25% of the client dataset as the subset.
    If user want a degree_of_poisoning that is greater than 25%,
        set the subset size to the degree_of_poisoning that the user wants.
    '''
    if (degree_of_poisoning > substitute_fraction):
        substitute_fraction = degree_of_poisoning

    # Auto-generate seed from attack parameters for reproducibility
    seed_string = substitute_fraction / degree_of_poisoning if degree_of_poisoning >= 0.01 else 123
    seed = math.ceil(seed_string) % (2 ** 32)
    rng = random.Random(seed)

    # Determine feasibility domain 
    min_target = targets.min()
    max_target = targets.max()

    # Take subset of dataset to poison from
    dataset_size = len(targets)
    subset_size = int(math.ceil(dataset_size * substitute_fraction))
    # Determine number of data points to poison of original dataset
    # (must be size of subset or less)
    num_of_poison_points = math.ceil(dataset_size * degree_of_poisoning) if math.ceil(dataset_size * degree_of_poisoning) <= subset_size else subset_size
    random_indexes = rng.sample(range(dataset_size), subset_size)
    sub_targets = [(i, targets[i].item()) for i in random_indexes]

    # Begin Flip algorithm
    distances = []
    for i, target in sub_targets:
        distance = max(target-min_target.item(), max_target.item()-target)
        distances.append((i,distance))
    sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)
    poison_indexes = [i for i, _ in sorted_distances[:num_of_poison_points]]
    
    for i in poison_indexes:
        if targets[i] > ((0.5*(max_target-min_target)) + min_target):
            targets[i] = min_target
        else:
            targets[i] = max_target
    
    return targets


def should_poison_this_round(
    current_round: int, client_id: int, attack_schedule: Optional[list]
) -> Tuple[bool, list]:
    """Check if client should be dynamically poisoned this round based on attack_schedule."""
    if not attack_schedule:
        return False, []

    attacks_by_type = {}

    for attack_entry in attack_schedule:
        start_round = attack_entry.get("start_round", 1)
        end_round = attack_entry.get("end_round", float("inf"))

        if not (start_round <= current_round <= end_round):
            continue

        selection_strategy = attack_entry.get("selection_strategy", "specific")
        is_match = False

        if selection_strategy == "specific":
            malicious_client_ids = attack_entry.get("malicious_client_ids", [])
            if client_id in malicious_client_ids:
                is_match = True

        elif selection_strategy == "random":
            targeted_clients = attack_entry.get("_selected_clients", [])
            if client_id in targeted_clients:
                is_match = True

        elif selection_strategy == "percentage":
            targeted_clients = attack_entry.get("_selected_clients", [])
            if client_id in targeted_clients:
                is_match = True

        if is_match:
            attack_type = attack_entry.get("attack_type")
            if attack_type and attack_type not in attacks_by_type:
                attacks_by_type[attack_type] = attack_entry

    return len(attacks_by_type) > 0, list(attacks_by_type.values())


def apply_poisoning_attack(
    data: torch.Tensor,
    labels: torch.Tensor,
    attack_config: dict,
    tokenizer=None,
    num_classes: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply poisoning attack based on config. Supports label_flipping and gaussian_noise."""
    if "params" in attack_config or (
        "type" in attack_config and "attack_type" not in attack_config
    ):
        raise ValueError(
            "Detected old nested attack config format. "
            "Please use flat attack_schedule format instead. "
            "Example: {'attack_type': 'label_flipping'} "
            "See docs/attack-scheduling.md guide."
        )

    attack_type = attack_config.get("attack_type")

    if attack_type == "label_flipping":
        if num_classes is None:
            raise ValueError(
                "Label flipping attack requires 'num_classes' parameter to be provided."
            )
        labels = apply_label_flipping(labels, num_classes=num_classes)

    elif attack_type == "gaussian_noise":
        target_noise_snr = attack_config.get("target_noise_snr")
        attack_ratio = attack_config.get("attack_ratio", 1.0)

        if target_noise_snr is not None:
            data = apply_gaussian_noise(
                data, target_noise_snr=target_noise_snr, attack_ratio=attack_ratio
            )
        else:
            data = apply_gaussian_noise(
                data,
                mean=attack_config.get("mean", 0.0),
                std=attack_config.get("std", 0.1),
                attack_ratio=attack_ratio,
            )
    
    elif attack_type == "target_flipping":
        degree_of_poisoing = attack_config.get("attack_ratio", 0.1)
        labels = apply_target_flipping(
            labels, degree_of_poisoing
        )

    return data, labels
