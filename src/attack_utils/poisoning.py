"""
Utility functions for applying poisoning attacks to datasets.
"""

import logging
from typing import Optional, Tuple

import torch
from src.attack_utils.vocabularies.registry import (
    get_replacement_strategy,
    get_vocabulary,
)


def apply_label_flipping(
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Applies label flipping attack by remapping each class to a different class."""
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
    """"""
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


def _encode_vocabulary_sequences(tokenizer, vocabulary: list) -> dict:
    """Encodes vocabulary words into token sequences."""
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
    """Applies sequence replacement attack."""
    modified_tokens = tokens.clone()
    max_seq_length = max(len(seq) for seq in target_sequences.keys())

    total_replacements = 0
    total_matches = 0

    for batch_idx in range(tokens.shape[0]):
        seq_idx = 0

        while seq_idx < tokens.shape[1]:
            matched = False

            for length in range(max_seq_length, 0, -1):
                if seq_idx + length > tokens.shape[1]:
                    continue

                window = tuple(tokens[batch_idx, seq_idx : seq_idx + length].tolist())

                if window in target_sequences:
                    total_matches += 1

                    if torch.rand(1).item() < replacement_prob:
                        replacement_id = replacement_token_ids[
                            torch.randint(0, len(replacement_token_ids), (1,)).item()
                        ]

                        modified_tokens[batch_idx, seq_idx] = replacement_id

                        if length > 1:
                            remaining_len = tokens.shape[1] - (seq_idx + length)
                            if remaining_len > 0:
                                modified_tokens[
                                    batch_idx,
                                    seq_idx + 1 : seq_idx + 1 + remaining_len,
                                ] = tokens[
                                    batch_idx,
                                    seq_idx + length : seq_idx + length + remaining_len,
                                ]
                            modified_tokens[batch_idx, -(length - 1) :] = 0
                        total_replacements += 1
                    matched = True
                    seq_idx += 1
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
    """Applies single token replacement attack."""
    modified_tokens = tokens.clone()

    for batch_idx in range(tokens.shape[0]):
        for seq_idx in range(tokens.shape[1]):
            token_id = tokens[batch_idx, seq_idx].item()

            if token_id in target_token_ids:
                if torch.rand(1).item() < replacement_prob:
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
    """Applies token replacement attack."""
    if replacement_prob <= 0:
        return tokens

    if not replacement_token_ids:
        return tokens

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
    """Determines if a client should be poisoned in the current round."""
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
    """Applies poisoning attack to the dataset."""
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

    elif attack_type == "token_replacement":
        target_token_ids = attack_config.get("target_token_ids")
        replacement_token_ids = attack_config.get("replacement_token_ids")
        target_sequences = None

        if tokenizer and not target_token_ids:
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

            if target_tokens and replacement_tokens:
                target_sequences = _encode_vocabulary_sequences(
                    tokenizer, target_tokens
                )

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
