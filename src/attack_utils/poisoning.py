"""
Data poisoning attack utilities for federated learning simulations.

This module provides modular attack functions that can be applied dynamically
during client training loops. Supports both image and text-based attacks.
"""

from typing import Optional, Tuple

import torch


def apply_label_flipping(
    labels: torch.Tensor,
    flip_fraction: float = 0.5,
    num_classes: int = 10,
    target_class: Optional[int] = None,
) -> torch.Tensor:
    """
    Flip a fraction of labels to random or targeted class.

    Args:
        labels: Tensor of shape (batch_size,)
        flip_fraction: Fraction of labels to flip (0.0-1.0)
        num_classes: Total number of classes in dataset
        target_class: If specified, flip all to this class (targeted attack)

    Returns:
        torch.Tensor: Labels with flipped values
    """
    if flip_fraction <= 0:
        return labels

    num_to_flip = int(len(labels) * flip_fraction)
    if num_to_flip == 0:
        return labels

    flip_indices = torch.randperm(len(labels))[:num_to_flip]

    if target_class is not None:
        labels[flip_indices] = target_class
    else:
        # Random flipping to different classes
        labels[flip_indices] = torch.randint(0, num_classes, (num_to_flip,))

    return labels


def apply_gaussian_noise(
    images: torch.Tensor, mean: float = 0.0, std: float = 0.1
) -> torch.Tensor:
    """
    Add Gaussian noise to image pixels.

    Args:
        images: Tensor of shape (batch_size, C, H, W)
        mean: Mean of Gaussian distribution
        std: Standard deviation

    Returns:
        torch.Tensor: Images with added noise, clamped to [0, 1]
    """
    noise = torch.randn_like(images) * std + mean
    return torch.clamp(images + noise, 0, 1)


def apply_brightness_attack(images: torch.Tensor, factor: float = 0.5) -> torch.Tensor:
    """
    Modify image brightness by a multiplicative factor.

    Args:
        images: Tensor of shape (batch_size, C, H, W)
        factor: Brightness multiplier (0.0 = black, 1.0 = unchanged, >1.0 = brighter)

    Returns:
        torch.Tensor: Images with modified brightness, clamped to [0, 1]
    """
    return torch.clamp(images * factor, 0, 1)


def apply_token_replacement(
    tokens: torch.Tensor, replacement_prob: float = 0.2, vocab_size: int = 30522
) -> torch.Tensor:
    """
    Replace tokens with random tokens from vocabulary.

    Args:
        tokens: Tensor of token IDs (batch_size, seq_len)
        replacement_prob: Probability of replacing each token
        vocab_size: Size of vocabulary (default: BERT vocab size)

    Returns:
        torch.Tensor: Tokens with random replacements
    """
    if replacement_prob <= 0:
        return tokens

    mask = torch.rand_like(tokens, dtype=torch.float32) < replacement_prob
    random_tokens = torch.randint(0, vocab_size, tokens.shape)
    return torch.where(mask, random_tokens, tokens)


def should_poison_this_round(
    current_round: int, client_id: int, attack_schedule: Optional[list]
) -> Tuple[bool, list[dict]]:
    """
    Check if client should be poisoned in current round.

    Returns all matching attack configs, deduplicated by attack type.
    If multiple schedules specify the same attack type, the last match wins.

    Args:
        current_round: Current training round (1-indexed)
        client_id: Client identifier
        attack_schedule: List of attack phase configurations

    Returns:
        Tuple[bool, list[dict]]: (should_poison, list_of_attack_configs)
            - should_poison: True if any attacks match
            - list_of_attack_configs: List of unique attack configs (by type)
    """
    if not attack_schedule:
        return False, []

    attack_configs_by_type = {}
    matching_phase_count = 0

    for attack_phase in attack_schedule:
        start_round = attack_phase.get("start_round", 1)
        end_round = attack_phase.get("end_round", float("inf"))

        if not (start_round <= current_round <= end_round):
            continue

        selection_strategy = attack_phase.get("selection_strategy", "specific")
        is_match = False

        if selection_strategy == "specific":
            client_ids = attack_phase.get("client_ids", [])
            if client_id in client_ids:
                is_match = True

        elif selection_strategy == "random":
            targeted_clients = attack_phase.get("_selected_clients", [])
            if client_id in targeted_clients:
                is_match = True

        elif selection_strategy == "percentage":
            targeted_clients = attack_phase.get("_selected_clients", [])
            if client_id in targeted_clients:
                is_match = True

        if is_match:
            matching_phase_count += 1
            attack_config = attack_phase.get("attack_config")
            if attack_config:
                attack_type = attack_config.get("type")
                attack_configs_by_type[attack_type] = attack_config

    return len(attack_configs_by_type) > 0, list(attack_configs_by_type.values())


def apply_poisoning_attack(
    data: torch.Tensor, labels: torch.Tensor, attack_config: dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply poisoning attack based on configuration.

    Args:
        data: Input data (images or tokens)
        labels: Labels
        attack_config: Attack configuration dict with 'type' and 'params'

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (poisoned_data, poisoned_labels)
    """
    attack_type = attack_config.get("type")
    params = attack_config.get("params", {})

    if attack_type == "label_flipping":
        labels = apply_label_flipping(
            labels,
            flip_fraction=params.get("flip_fraction", 0.5),
            num_classes=params.get("num_classes", 10),
            target_class=params.get("target_class"),
        )

    elif attack_type == "gaussian_noise":
        data = apply_gaussian_noise(
            data, mean=params.get("mean", 0.0), std=params.get("std", 0.1)
        )

    elif attack_type == "brightness":
        data = apply_brightness_attack(data, factor=params.get("factor", 0.5))

    elif attack_type == "token_replacement":
        data = apply_token_replacement(
            data,
            replacement_prob=params.get("replacement_prob", 0.2),
            vocab_size=params.get("vocab_size", 30522),
        )

    else:
        # Unknown attack type - skip poisoning
        pass

    return data, labels
