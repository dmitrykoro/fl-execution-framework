"""
Utility functions for applying poisoning attacks to datasets.
"""

import logging
from typing import Optional, Tuple

import torch


def apply_label_flipping(
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Apply label flipping attack by remapping each class to a different random class.

    Each source class is randomly mapped to a different target class (not itself).
    This creates a class-level random mapping for the entire label set.

    Args:
        labels: Input label tensor of shape (N,)
        num_classes: Total number of classes in the dataset

    Returns:
        Modified label tensor with same shape, where each class is remapped to
        a different random class
    """
    unique_classes = torch.unique(labels)
    modified_labels = labels.clone()

    for source_class in unique_classes:
        source_val = source_class.item()
        valid_targets = [t for t in range(num_classes) if t != source_val]
        if not valid_targets:
            continue

        target_class = valid_targets[torch.randint(0, len(valid_targets), (1,)).item()]
        modified_labels[labels == source_val] = target_class

    return modified_labels


def apply_gaussian_noise(
    images: torch.Tensor,
    mean: float = 0.0,
    std: float = 0.1,
    target_noise_snr: Optional[float] = None,
    attack_ratio: float = 1.0,
) -> torch.Tensor:
    """Apply Gaussian noise poisoning attack to images.

    Adds Gaussian noise to a subset of images. Noise can be specified either via
    mean/std parameters or target SNR (signal-to-noise ratio) in dB.

    Args:
        images: Input image tensor of shape (N, C, H, W)
        mean: Mean of Gaussian noise distribution (default: 0.0)
        std: Standard deviation of Gaussian noise distribution (default: 0.1)
        target_noise_snr: Target SNR in dB. If provided, overrides mean/std parameters
        attack_ratio: Fraction of samples to poison, range [0.0, 1.0] (default: 1.0)

    Returns:
        Poisoned image tensor with same shape as input, values clamped to [0, 1]

    Note:
        If target_noise_snr is specified, noise power is calculated to achieve
        the specified SNR relative to signal power.
    """
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


def should_poison_this_round(
    current_round: int, client_id: int, attack_schedule: Optional[list]
) -> Tuple[bool, list]:
    """Determine if a client should be poisoned in the current round.

    Checks attack schedule to see if the client is selected for poisoning
    in the specified round based on the attack configuration.

    Args:
        current_round: Current training round number
        client_id: ID of the client to check
        attack_schedule: List of attack configuration dicts with round ranges
            and client selection strategies (optional)

    Returns:
        Tuple of (should_poison: bool, active_attacks: list)
        - should_poison: True if client should be poisoned this round
        - active_attacks: List of attack configs active for this client/round

    Note:
        Returns (False, []) if no attack_schedule provided or no attacks
        are active for this client in this round.
    """
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
    """Apply poisoning attack to dataset based on attack configuration.

    Main entry point for applying attacks. Supports label_flipping
    and gaussian_noise attacks. Can apply multiple attacks sequentially
    if attack_config is a list.

    Args:
        data: Input data tensor (images or token IDs)
        labels: Input label tensor
        attack_config: Attack configuration dict or list of dicts
        tokenizer: HuggingFace tokenizer (unused, kept for compatibility)
        num_classes: Number of classes for label_flipping attacks (optional)

    Returns:
        Tuple of (poisoned_data, poisoned_labels)

    Raises:
        ValueError: If old nested attack config format detected
        ValueError: If unsupported attack type specified
        ValueError: If required parameters missing for attack type

    Note:
        For multiple attacks, specify attack_config as list of dicts.
        Each attack is applied sequentially to the output of the previous attack.
    """
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

    return data, labels
