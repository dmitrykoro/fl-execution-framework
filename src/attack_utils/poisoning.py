"""
Data poisoning attack utilities for federated learning simulations.

This module provides modular attack functions that can be applied dynamically
during client training loops. Supports both image and text-based attacks.
"""

import logging
import torch
from typing import Optional, Tuple
from src.attack_utils.token_vocabularies import get_replacement_strategy, get_vocabulary


def apply_label_flipping(
    labels: torch.Tensor,
    flip_fraction: float = 0.5,
    target_class: int = 0,
) -> torch.Tensor:
    """
    Flip a fraction of labels to targeted class.

    Args:
        labels: Tensor of shape (batch_size,)
        flip_fraction: Fraction of labels to flip (0.0-1.0)
        target_class: Target class to flip labels to

    Returns:
        torch.Tensor: Labels with flipped values
    """
    if flip_fraction <= 0:
        return labels

    num_to_flip = int(len(labels) * flip_fraction)
    if num_to_flip == 0:
        return labels

    flip_indices = torch.randperm(len(labels))[:num_to_flip]
    labels[flip_indices] = target_class

    return labels


def apply_gaussian_noise(
    images: torch.Tensor,
    mean: float = 0.0,
    std: float = 0.1,
    target_noise_snr: Optional[float] = None,
    attack_ratio: float = 1.0,
) -> torch.Tensor:
    """
    Add Gaussian noise to image pixels.

    Args:
        images: Tensor of shape (batch_size, C, H, W), normalized to [0, 1]
        mean: Mean of Gaussian distribution (if target_noise_snr not provided)
        std: Standard deviation (if target_noise_snr not provided)
        target_noise_snr: Target signal-to-noise ratio in dB (overrides mean/std)
        attack_ratio: Fraction of samples in batch to poison (0.0-1.0)

    Returns:
        torch.Tensor: Images with added noise, clamped to [0, 1]
    """
    num_samples = images.shape[0]
    num_to_poison = int(num_samples * attack_ratio)

    # If no samples to poison, return unchanged
    if num_to_poison == 0:
        return images

    # Randomly select indices to poison
    indices = torch.randperm(num_samples)[:num_to_poison]

    # Clone images to avoid modifying original
    poisoned_images = images.clone()

    if target_noise_snr is not None:
        # SNR-based approach
        # SNR(dB) = 10 * log10(signal_power / noise_power)
        # → noise_power = signal_power / (10^(SNR/10))
        signal_power = torch.mean(
            poisoned_images[indices] ** 2, dim=(1, 2, 3), keepdim=True
        )
        noise_power = signal_power / (10 ** (target_noise_snr / 10))
        noise = torch.randn_like(poisoned_images[indices]) * torch.sqrt(noise_power)
    else:
        # Fallback: Direct mean/std noise
        noise = torch.randn_like(poisoned_images[indices]) * std + mean

    poisoned_images[indices] = torch.clamp(poisoned_images[indices] + noise, 0, 1)
    return poisoned_images


def _encode_vocabulary_sequences(tokenizer, vocabulary: list) -> dict:
    """
    Encode vocabulary words as complete token ID sequences.

    Handles multi-token words properly (e.g., "treatment" → [2487, 4567]).

    Args:
        tokenizer: Tokenizer instance
        vocabulary: List of vocabulary words (strings)

    Returns:
        Dict mapping token ID tuples to original words
        Example: {(2487, 4567): "treatment", (8291,): "scan"}
    """
    vocab_sequences = {}
    single_token_count = 0
    multi_token_count = 0

    for word in vocabulary:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if token_ids:
            key = tuple(token_ids)
            vocab_sequences[key] = word

            if len(token_ids) == 1:
                single_token_count += 1
            else:
                multi_token_count += 1

    logging.info(
        f"Encoded {len(vocab_sequences)} vocabulary words: "
        f"{single_token_count} single-token, {multi_token_count} multi-token"
    )

    return vocab_sequences


def _apply_sequence_replacement(
    tokens: torch.Tensor,
    replacement_prob: float,
    target_sequences: dict,
    replacement_token_ids: list,
) -> torch.Tensor:
    """
    Apply token replacement using multi-token sequence matching.

    Uses sliding window to find and replace complete word sequences.

    Args:
        tokens: Tensor of token IDs (batch_size, seq_len)
        replacement_prob: Probability of replacing each matched sequence
        target_sequences: Dict of {(token_id_tuple): word} mappings
        replacement_token_ids: List of replacement token IDs

    Returns:
        Modified token tensor with sequences replaced
    """
    modified_tokens = tokens.clone()
    max_seq_length = max(len(seq) for seq in target_sequences.keys())

    total_replacements = 0
    total_matches = 0

    for batch_idx in range(tokens.shape[0]):
        seq_idx = 0

        while seq_idx < tokens.shape[1]:
            matched = False

            # Try matching sequences from longest to shortest (greedy)
            for length in range(max_seq_length, 0, -1):
                if seq_idx + length > tokens.shape[1]:
                    continue

                # Extract current window
                window = tuple(
                    tokens[batch_idx, seq_idx : seq_idx + length].tolist()
                )

                # Check if this sequence matches a target
                if window in target_sequences:
                    total_matches += 1

                    # Replace with probability
                    if torch.rand(1).item() < replacement_prob:
                        # Choose random replacement token
                        replacement_id = replacement_token_ids[
                            torch.randint(0, len(replacement_token_ids), (1,)).item()
                        ]

                        # Replace first token in sequence
                        modified_tokens[batch_idx, seq_idx] = replacement_id

                        # Shift remaining tokens left if multi-token sequence
                        if length > 1:
                            # Shift tokens after the sequence to the left
                            remaining_len = tokens.shape[1] - (seq_idx + length)
                            if remaining_len > 0:
                                modified_tokens[
                                    batch_idx,
                                    seq_idx + 1 : seq_idx + 1 + remaining_len,
                                ] = tokens[
                                    batch_idx,
                                    seq_idx + length : seq_idx + length + remaining_len,
                                ]

                            # Pad the end with pad tokens (0)
                            modified_tokens[batch_idx, -(length - 1) :] = 0

                        total_replacements += 1

                    matched = True
                    seq_idx += 1  # Move past the replaced token
                    break

            if not matched:
                seq_idx += 1

    if total_matches > 0:
        replacement_rate = total_replacements / total_matches * 100
        logging.debug(
            f"Token replacement: {total_replacements}/{total_matches} matches replaced "
            f"({replacement_rate:.1f}%)"
        )

    return modified_tokens


def _apply_single_token_replacement(
    tokens: torch.Tensor,
    replacement_prob: float,
    target_token_ids: list,
    replacement_token_ids: list,
) -> torch.Tensor:
    """
    Legacy single-token replacement (backward compatibility).

    Only matches individual token IDs, not sequences.
    """
    modified_tokens = tokens.clone()

    for batch_idx in range(tokens.shape[0]):
        for seq_idx in range(tokens.shape[1]):
            token_id = tokens[batch_idx, seq_idx].item()

            # Check if this token is in our target list
            if token_id in target_token_ids:
                # Replace with probability
                if torch.rand(1).item() < replacement_prob:
                    # Choose random replacement from replacement list
                    replacement_id = replacement_token_ids[
                        torch.randint(0, len(replacement_token_ids), (1,)).item()
                    ]
                    modified_tokens[batch_idx, seq_idx] = replacement_id

    return modified_tokens


def apply_token_replacement(
    tokens: torch.Tensor,
    replacement_prob: float = 0.2,
    target_token_ids: Optional[list] = None,
    replacement_token_ids: Optional[list] = None,
    target_sequences: Optional[dict] = None,
) -> torch.Tensor:
    """
    Replace specific target tokens with replacement tokens.

    Uses vocabulary-based targeted replacement with multi-token sequence support.
    Example: "treatment" (2 tokens) → "avoid" (1 token), "doctor" → "harmful"

    Args:
        tokens: Tensor of token IDs (batch_size, seq_len)
        replacement_prob: Probability of replacing each target token/sequence
        target_token_ids: Single-token IDs
        replacement_token_ids: List of token IDs to use as replacements
        target_sequences: Dict mapping token ID tuples to words (for sequence matching)

    Returns:
        torch.Tensor: Tokens with targeted replacements applied
    """
    if replacement_prob <= 0:
        return tokens

    if not replacement_token_ids:
        return tokens

    # Use sequence matching if available, else fall back to single-token
    if target_sequences:
        return _apply_sequence_replacement(
            tokens, replacement_prob, target_sequences, replacement_token_ids
        )
    elif target_token_ids:
        return _apply_single_token_replacement(
            tokens, replacement_prob, target_token_ids, replacement_token_ids
        )
    else:
        return tokens


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
    data: torch.Tensor, labels: torch.Tensor, attack_config: dict, tokenizer=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply poisoning attack based on attack_schedule configuration.

    Args:
        data: Input data (images or tokens)
        labels: Labels
        attack_config: Attack entry from attack_schedule
        tokenizer: Optional tokenizer for converting target/replacement token strings to IDs

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
        target_class = attack_config.get("target_class")
        if target_class is None:
            raise ValueError(
                "label_flipping attack requires 'target_class' parameter. "
                "Example: {'attack_type': 'label_flipping', 'flip_fraction': 1.0, 'target_class': 7}"
            )
        labels = apply_label_flipping(
            labels,
            flip_fraction=attack_config.get("flip_fraction", 0.5),
            target_class=target_class,
        )

    elif attack_type == "gaussian_noise":
        # Use SNR-based approach if available
        target_noise_snr = attack_config.get("target_noise_snr")
        attack_ratio = attack_config.get("attack_ratio", 1.0)

        if target_noise_snr is not None:
            data = apply_gaussian_noise(
                data, target_noise_snr=target_noise_snr, attack_ratio=attack_ratio
            )
        else:
            # Fallback to std/mean
            data = apply_gaussian_noise(
                data,
                mean=attack_config.get("mean", 0.0),
                std=attack_config.get("std", 0.1),
                attack_ratio=attack_ratio,
            )

    elif attack_type == "token_replacement":
        # Convert target/replacement tokens to IDs if provided
        target_token_ids = attack_config.get("target_token_ids")
        replacement_token_ids = attack_config.get("replacement_token_ids")
        target_sequences = None

        if tokenizer and not target_token_ids:
            # Vocabulary-based token replacement
            if "target_vocabulary" not in attack_config:
                raise ValueError(
                    "Token replacement attack requires 'target_vocabulary' in attack_config. "
                    "Use predefined vocabularies from token_vocabularies.py (e.g., 'medical', "
                    "'financial', 'legal')."
                    "See src/attack_utils/token_vocabularies.py for available vocabularies."
                )

            vocab_name = attack_config.get("target_vocabulary")
            strategy_name = attack_config.get("replacement_strategy", "negative")

            target_tokens = get_vocabulary(vocab_name)
            replacement_tokens = get_replacement_strategy(strategy_name)

            logging.info(
                f"Using vocabulary '{vocab_name}' with '{strategy_name}' replacement strategy"
            )
            logging.info(
                f"Loaded {len(target_tokens)} target tokens, {len(replacement_tokens)} replacement tokens"
            )

            # Encode tokens to sequences
            if target_tokens and replacement_tokens:
                # Use new sequence-based encoding
                target_sequences = _encode_vocabulary_sequences(
                    tokenizer, target_tokens
                )

                # Encode replacement tokens (single tokens)
                replacement_token_ids = [
                    tokenizer.encode(token, add_special_tokens=False)[0]
                    for token in replacement_tokens
                    if tokenizer.encode(token, add_special_tokens=False)
                ]

                logging.info(
                    f"Sequence replacement: {len(target_sequences)} target sequences, "
                    f"{len(replacement_token_ids)} replacement IDs"
                )

        data = apply_token_replacement(
            data,
            replacement_prob=attack_config.get(
                "replacement_prob", attack_config.get("replacement_probability", 0.2)
            ),
            target_token_ids=target_token_ids,
            replacement_token_ids=replacement_token_ids,
            target_sequences=target_sequences,
        )

    return data, labels
