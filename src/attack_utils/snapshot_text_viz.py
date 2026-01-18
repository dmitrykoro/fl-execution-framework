"""
Text visualization utilities for attack snapshots.
"""
from pathlib import Path
from typing import List, Optional, Union

import numpy as np


def _extract_attack_param(
    attack_config: Union[dict, List[dict]], *attack_parameters: str, default: any = "?"
) -> any:
    """Extract attack parameter value from attack configuration.

    Searches for the first matching parameter name in the config and returns its value.
    Handles both single dict configs and list of dict configs.

    Args:
        attack_config: Attack configuration dict or list of dicts
        *attack_parameters: Variable number of parameter names to search for
        default: Default value to return if parameter not found (default: "?")

    Returns:
        Parameter value if found, otherwise the default value
    """
    config = (
        attack_config[0]
        if isinstance(attack_config, list) and attack_config
        else attack_config
    )

    if isinstance(config, dict):
        for attack_parameter in attack_parameters:
            if attack_parameter in config:
                return config[attack_parameter]

    return default


def _extract_attack_type(attack_config: Union[dict, List[dict]]) -> str:
    """Extract attack type from attack configuration.

    For list configs (composite attacks), joins attack types with underscore.
    For single dict configs, returns the attack_type field.

    Args:
        attack_config: Attack configuration dict or list of dicts

    Returns:
        Attack type string, or "unknown" if not found. Multiple attacks
        are joined with underscores (e.g., "label_flipping_gaussian_noise")
    """
    if isinstance(attack_config, list):
        if attack_config:
            attack_types = [
                cfg.get("attack_type") or cfg.get("type", "unknown")
                for cfg in attack_config
            ]
            return "_".join(attack_types)
        else:
            return "unknown"
    else:
        return attack_config.get("attack_type") or attack_config.get("type", "unknown")


def save_text_samples(
    labels: np.ndarray,
    original_labels: np.ndarray,
    filepath: Path,
    attack_config: Optional[Union[dict, List[dict]]] = None,
    tokenizer=None,
    input_ids_original: Optional[np.ndarray] = None,
    input_ids_poisoned: Optional[np.ndarray] = None,
) -> None:
    """Save text attack samples to a visualization file.

    Creates a text file showing original vs poisoned text samples with token
    replacement statistics and label changes. If tokenizer and token IDs are
    provided, shows detailed token-level changes. Otherwise, shows only labels.

    Args:
        labels: Poisoned labels array of shape (N,) or (N, seq_len)
        original_labels: Original labels array of shape (N,) or (N, seq_len)
        filepath: Output file path for the text visualization
        attack_config: Attack configuration dict or list of dicts (optional)
        tokenizer: HuggingFace tokenizer for decoding token IDs (optional)
        input_ids_original: Original token IDs of shape (N, seq_len) (optional)
        input_ids_poisoned: Poisoned token IDs of shape (N, seq_len) (optional)

    Note:
        If tokenizer and token IDs are not provided, only saves label changes.
        Skips samples where no tokens were replaced to reduce file size.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        if (
            tokenizer is not None
            and input_ids_original is not None
            and input_ids_poisoned is not None
        ):
            f.write("=" * 80 + "\n")
            f.write("TOKEN REPLACEMENT ATTACK VISUALIZATION\n")
            f.write("=" * 80 + "\n\n")

            if attack_config:
                attack_type = _extract_attack_type(attack_config)
                f.write(f"Attack Type: {attack_type}\n")

                if attack_type == "token_replacement":
                    target_vocab = _extract_attack_param(
                        attack_config, "target_vocabulary", default="unknown"
                    )
                    replacement_strategy = _extract_attack_param(
                        attack_config, "replacement_strategy", default="negative"
                    )
                    replacement_prob = _extract_attack_param(
                        attack_config, "replacement_probability", default=1.0
                    )

                    f.write(f"Target Vocabulary: {target_vocab}\n")
                    f.write(f"Replacement Strategy: {replacement_strategy}\n")
                    f.write(f"Replacement Probability: {replacement_prob}\n")

                f.write("\n" + "=" * 80 + "\n\n")

            samples_shown = 0
            samples_skipped = 0

            for i in range(len(labels)):
                original_tokens = input_ids_original[i]
                poisoned_tokens = input_ids_poisoned[i]

                try:
                    original_text = tokenizer.decode(
                        original_tokens, skip_special_tokens=True
                    )
                    poisoned_text = tokenizer.decode(
                        poisoned_tokens, skip_special_tokens=True
                    )

                    if original_text == poisoned_text:
                        samples_skipped += 1
                        continue

                    f.write(f"--- Sample {i} ---\n\n")

                    f.write(f'ORIGINAL: "{original_text}"\n')
                    f.write(f'POISONED: "{poisoned_text}"\n')

                    num_replaced = np.sum(original_tokens != poisoned_tokens)
                    total_tokens = len(original_tokens)
                    replacement_rate = (
                        num_replaced / total_tokens * 100 if total_tokens > 0 else 0
                    )

                    f.write(
                        f"          {num_replaced}/{total_tokens} tokens replaced ({replacement_rate:.1f}%)\n"
                    )
                    f.write("          " + "^" * 15 + "[REPLACED]" + "^" * 15 + "\n")

                    current_label = labels[i]
                    original_label = original_labels[i]

                    if isinstance(current_label, np.ndarray) and current_label.size > 1:
                        label_changed = not np.array_equal(
                            current_label, original_label
                        )
                        num_masked_original = np.sum(original_label != -100)
                        num_masked_current = np.sum(current_label != -100)

                        if label_changed:
                            f.write(
                                f"Labels: {num_masked_original} masked tokens → {num_masked_current} masked tokens (CHANGED)\n"
                            )
                        else:
                            f.write(
                                f"Labels: {num_masked_original} masked tokens (unchanged)\n"
                            )
                    else:
                        if np.array_equal(current_label, original_label):
                            f.write(f"Label: {current_label} (unchanged)\n")
                        else:
                            f.write(
                                f"Label: {original_label} → {current_label} (FLIPPED)\n"
                            )

                    samples_shown += 1

                except Exception as e:
                    f.write(f"--- Sample {i} ---\n\n")
                    f.write(f"[Decoding error: {e}]\n")
                    f.write(
                        f"Original token IDs (first 10): {original_tokens[:10].tolist()}\n"
                    )
                    f.write(
                        f"Poisoned token IDs (first 10): {poisoned_tokens[:10].tolist()}\n"
                    )
                    samples_shown += 1

                f.write("\n")

            if samples_skipped > 0:
                f.write("=" * 80 + "\n")
                f.write(
                    f"SUMMARY: {samples_shown} samples modified, {samples_skipped} samples unchanged (skipped)\n"
                )
                f.write("=" * 80 + "\n")
        else:
            f.write("Sample Labels (Poisoned vs Original)\n")
            f.write("=" * 40 + "\n")
            for i, (label, orig) in enumerate(zip(labels, original_labels)):
                f.write(f"Sample {i}: {label} (was {orig})\n")
