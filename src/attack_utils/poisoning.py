"""
Data poisoning attack utilities for federated learning simulations.

This module provides modular attack functions that can be applied dynamically
during client training loops. Supports both image and text-based attacks.
"""

from typing import Optional, Tuple
import logging

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
    images: torch.Tensor,
    mean: float = 0.0,
    std: float = 0.1,
    target_noise_snr: Optional[float] = None,
) -> torch.Tensor:
    """
    Add Gaussian noise to image pixels.

    Args:
        images: Tensor of shape (batch_size, C, H, W), normalized to [0, 1]
        mean: Mean of Gaussian distribution (if target_noise_snr not provided)
        std: Standard deviation (if target_noise_snr not provided)
        target_noise_snr: Target signal-to-noise ratio in dB (if provided, overrides mean/std)

    Returns:
        torch.Tensor: Images with added noise, clamped to [0, 1]
    """
    if target_noise_snr is not None:
        # SNR-based approach
        # SNR(dB) = 10 * log10(signal_power / noise_power)
        # → noise_power = signal_power / (10^(SNR/10))
        signal_power = torch.mean(images**2, dim=(1, 2, 3), keepdim=True)
        noise_power = signal_power / (10 ** (target_noise_snr / 10))
        noise = torch.randn_like(images) * torch.sqrt(noise_power)
    else:
        # Direct mean/std noise
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
) -> Tuple[bool, list]:
    """
    Check if client should be poisoned in current round.

    Returns all matching attack configs, deduplicated by attack type.
    If multiple schedules specify the same attack type, the first match wins per type.
    This allows stacking different attack types (e.g., label_flipping + gaussian_noise).

    Args:
        current_round: Current training round (1-indexed)
        client_id: Client identifier
        attack_schedule: List of attack schedule entries

    Returns:
        Tuple[bool, list[dict]]: (should_poison, list_of_attack_entries)
            - should_poison: True if any attacks match
            - list_of_attack_entries: List of unique attack entries (by attack_type)
    """
    if not attack_schedule:
        return False, []

    # Track matching attacks by attack_type to prevent duplicates
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
            # Random selection handled at strategy level
            # Marked clients stored in _selected_clients
            targeted_clients = attack_entry.get("_selected_clients", [])
            if client_id in targeted_clients:
                is_match = True

        elif selection_strategy == "percentage":
            # Percentage selection handled at strategy level
            targeted_clients = attack_entry.get("_selected_clients", [])
            if client_id in targeted_clients:
                is_match = True

        # If matched, add to attacks_by_type (first match per type wins)
        if is_match:
            attack_type = attack_entry.get("attack_type")
            if attack_type and attack_type not in attacks_by_type:
                attacks_by_type[attack_type] = attack_entry

    return len(attacks_by_type) > 0, list(attacks_by_type.values())


def apply_poisoning_attack(
    data: torch.Tensor, labels: torch.Tensor, attack_config: dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply poisoning attack based on attack_schedule configuration.

    Args:
        data: Input data (images or tokens)
        labels: Labels
        attack_config: Attack entry from attack_schedule

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (poisoned_data, poisoned_labels)

    Raises:
        ValueError: If old nested config format is detected
    """
    # Detect old nested format and provide helpful error
    if "params" in attack_config or (
        "type" in attack_config and "attack_type" not in attack_config
    ):
        raise ValueError(
            "Detected old nested attack config format. "
            "Please use flat attack_schedule format instead. "
            "Example: {'attack_type': 'label_flipping', 'flip_fraction': 0.7} "
            "See docs/attack-scheduling.md guide."
        )

    attack_type = attack_config.get("attack_type")

    if attack_type == "label_flipping":
        labels = apply_label_flipping(
            labels,
            flip_fraction=attack_config.get("flip_fraction", 0.5),
            num_classes=attack_config.get("num_classes", 10),
            target_class=attack_config.get("target_class"),
        )

    elif attack_type == "gaussian_noise":
        # Use SNR-based approach if available
        target_noise_snr = attack_config.get("target_noise_snr")
        if target_noise_snr is not None:
            data = apply_gaussian_noise(data, target_noise_snr=target_noise_snr)
        else:
            # Fallback to std/mean
            data = apply_gaussian_noise(
                data,
                mean=attack_config.get("mean", 0.0),
                std=attack_config.get("std", 0.1),
            )

    elif attack_type == "brightness":
        data = apply_brightness_attack(data, factor=attack_config.get("factor", 0.5))

    elif attack_type == "token_replacement":
        data = apply_token_replacement(
            data,
            replacement_prob=attack_config.get("replacement_prob", 0.2),
            vocab_size=attack_config.get("vocab_size", 30522),
        )

    return data, labels
