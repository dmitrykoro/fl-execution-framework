"""
Lightweight attack data snapshot logging.

Provides utilities for saving small snapshots of poisoned data for inspection
and debugging.
"""

import json
import logging
import math
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt


def _extract_attack_type(attack_config: Union[dict, List[dict]]) -> str:
    """
    Extract attack type from attack config, handling both dict and list inputs.

    Args:
        attack_config: Either a single attack config dict or a list of attack config dicts

    Returns:
        Attack type string (composite if multiple attacks)
    """
    if isinstance(attack_config, list):
        if attack_config:
            # Create composite attack type from all configs
            attack_types = [
                cfg.get("attack_type") or cfg.get("type", "unknown")
                for cfg in attack_config
            ]
            return "_".join(attack_types)
        else:
            return "unknown"
    else:
        return attack_config.get("attack_type") or attack_config.get("type", "unknown")


def _get_snapshot_dir(
    output_dir: str, client_id: int, round_num: int, strategy_number: int = 0
) -> Path:
    """Get or create snapshot directory path."""
    snapshots_base = Path(output_dir) / f"attack_snapshots_{strategy_number}"
    snapshot_dir = snapshots_base / f"client_{client_id}" / f"round_{round_num}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_dir


def _create_snapshot_metadata(
    client_id: int,
    round_num: int,
    attack_type: str,
    attack_config: Union[dict, List[dict]],
    num_samples: int,
    data_shape: Optional[list] = None,
    labels_shape: Optional[list] = None,
    experiment_info: Optional[Dict[str, Any]] = None,
) -> dict:
    """Create metadata dictionary for snapshot."""
    metadata = {
        "client_id": client_id,
        "round_num": round_num,
        "attack_type": attack_type,
        "attack_config": attack_config,
        "num_samples": num_samples,
        "timestamp": datetime.now().isoformat(),
    }

    if data_shape:
        metadata["data_shape"] = data_shape
    if labels_shape:
        metadata["labels_shape"] = labels_shape
    if experiment_info:
        metadata["experiment_info"] = experiment_info

    return metadata


def _extract_attack_param(
    attack_config: Union[dict, List[dict]], *attack_parameters: str, default: Any = "?"
) -> Any:
    """Extract parameter from attack config, handling dict/list cases."""
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


def save_attack_snapshot(
    client_id: int,
    round_num: int,
    attack_config: Union[dict, List[dict]],
    data_sample: torch.Tensor,
    labels_sample: torch.Tensor,
    original_labels_sample: Optional[torch.Tensor],
    output_dir: str,
    max_samples: int = 5,
    save_format: str = "pickle",
    experiment_info: Optional[Dict[str, Any]] = None,
    strategy_number: int = 0,
) -> None:
    """
    Save lightweight snapshot of poisoned data for inspection.

    Args:
        client_id: ID of the attacking client
        round_num: Current training round
        attack_config: Attack configuration dict or list of dicts (for multiple attacks)
        data_sample: Poisoned data tensor (first N samples from batch)
        labels_sample: Poisoned labels tensor (first N samples from batch)
        original_labels_sample: Original labels before poisoning (None for transformer/text models)
        output_dir: Base output directory (e.g., "out/api_run_...")
        max_samples: Maximum number of samples to save (default: 5)
        save_format: Format to save ('pickle', 'json', or 'both')
        experiment_info: Optional experiment metadata (run_id, total_clients, total_rounds)
        strategy_number: Strategy number for multi-strategy runs (default: 0)
    """
    attack_type = _extract_attack_type(attack_config)
    snapshot_dir = _get_snapshot_dir(output_dir, client_id, round_num, strategy_number)

    data_sample = data_sample[:max_samples]
    labels_sample = labels_sample[:max_samples]
    if original_labels_sample is not None:
        original_labels_sample = original_labels_sample[:max_samples]

    metadata = _create_snapshot_metadata(
        client_id=client_id,
        round_num=round_num,
        attack_type=attack_type,
        attack_config=attack_config,
        num_samples=len(data_sample),
        data_shape=list(data_sample.shape),
        labels_shape=list(labels_sample.shape),
        experiment_info=experiment_info,
    )

    try:
        if save_format == "both":
            # Save both pickle and JSON formats
            pickle_path = snapshot_dir / f"{attack_type}.pickle"
            json_path = snapshot_dir / f"{attack_type}_metadata.json"

            snapshot = {
                "metadata": metadata,
                "data": data_sample.cpu().numpy(),
                "labels": labels_sample.cpu().numpy(),
                "original_labels": (
                    original_labels_sample.cpu().numpy()
                    if original_labels_sample is not None
                    else None
                ),
            }
            with open(pickle_path, "wb") as f:
                pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)

            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logging.debug(
                f"Saved {attack_type} attack snapshot: client {client_id}, round {round_num} "
                f"({len(data_sample)} samples) -> {pickle_path} and {json_path}"
            )

        elif save_format == "pickle":
            filepath = snapshot_dir / f"{attack_type}.pickle"
            snapshot = {
                "metadata": metadata,
                "data": data_sample.cpu().numpy(),
                "labels": labels_sample.cpu().numpy(),
                "original_labels": (
                    original_labels_sample.cpu().numpy()
                    if original_labels_sample is not None
                    else None
                ),
            }
            with open(filepath, "wb") as f:
                pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)

            logging.debug(
                f"Saved {attack_type} attack snapshot: client {client_id}, round {round_num} "
                f"({len(data_sample)} samples) -> {filepath}"
            )

        elif save_format == "json":
            filepath = snapshot_dir / f"{attack_type}_metadata.json"
            with open(filepath, "w") as f:
                json.dump(metadata, f, indent=2)

            logging.debug(
                f"Saved {attack_type} attack snapshot metadata: client {client_id}, round {round_num} -> {filepath}"
            )

    except Exception as e:
        logging.warning(f"Failed to save attack snapshot for client {client_id}: {e}")


def load_attack_snapshot(filepath: str) -> Optional[dict]:
    """
    Load an attack snapshot for inspection.

    Args:
        filepath: Path to snapshot file

    Returns:
        Dict containing metadata and (optionally) data/labels if pickle format
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logging.error(f"Snapshot file not found: {filepath}")
        return None

    try:
        if filepath.suffix == ".pickle":
            with open(filepath, "rb") as f:
                return pickle.load(f)
        elif filepath.suffix == ".json":
            with open(filepath, "r") as f:
                return json.load(f)
        else:
            logging.error(f"Unsupported snapshot format: {filepath.suffix}")
            return None

    except Exception as e:
        logging.error(f"Failed to load snapshot {filepath}: {e}")
        return None


def list_attack_snapshots(output_dir: str, strategy_number: int = 0) -> list:
    """
    List all attack snapshots in an output directory.

    Args:
        output_dir: Base output directory
        strategy_number: Strategy number for multi-strategy runs (default: 0)

    Returns:
        List of snapshot file paths (pickle files preferred, JSON metadata as fallback)
    """
    snapshots_dir = Path(output_dir) / f"attack_snapshots_{strategy_number}"
    if not snapshots_dir.exists():
        return []

    # First, find pickle files because they contain full data
    pickle_snapshots = list(snapshots_dir.glob("client_*/round_*/*.pickle"))

    if pickle_snapshots:
        return sorted(pickle_snapshots)

    # Fallback: find JSON metadata files if no pickle files exist
    json_snapshots = list(snapshots_dir.glob("client_*/round_*/*_metadata.json"))
    return sorted(json_snapshots)


def get_snapshot_summary(output_dir: str, strategy_number: int = 0) -> dict:
    """
    Get summary statistics for all attack snapshots in a run.

    Args:
        output_dir: Base output directory
        strategy_number: Strategy number for multi-strategy runs (default: 0)

    Returns:
        Dict with summary statistics (client IDs, rounds, attack types, etc.)
    """
    snapshots = list_attack_snapshots(output_dir, strategy_number)

    summary = {
        "total_snapshots": len(snapshots),
        "clients_attacked": set(),
        "rounds_with_attacks": set(),
        "attack_types": set(),
    }

    for snapshot_path in snapshots:
        snapshot = load_attack_snapshot(str(snapshot_path))
        if snapshot:
            metadata = snapshot.get("metadata", snapshot)
            summary["clients_attacked"].add(metadata.get("client_id"))
            summary["rounds_with_attacks"].add(metadata.get("round_num"))
            summary["attack_types"].add(metadata.get("attack_type"))

    # Convert sets to sorted lists for JSON serialization
    summary["clients_attacked"] = sorted(list(summary["clients_attacked"]))
    summary["rounds_with_attacks"] = sorted(list(summary["rounds_with_attacks"]))
    summary["attack_types"] = sorted(list(summary["attack_types"]))

    return summary


def save_visual_snapshot(
    client_id: int,
    round_num: int,
    attack_config: Union[dict, List[dict]],
    data_sample: np.ndarray,
    labels_sample: np.ndarray,
    original_labels_sample: np.ndarray,
    output_dir: str,
    experiment_info: Optional[Dict[str, Any]] = None,
    strategy_number: int = 0,
    tokenizer=None,
    original_data_sample: Optional[np.ndarray] = None,
) -> None:
    """
    Save visual PNG/TXT files alongside pickle for viewing.

    Args:
        client_id: ID of the attacking client
        round_num: Current training round
        attack_config: Attack configuration dict or list of dicts for multiple attacks
        data_sample: Poisoned data as numpy array
        labels_sample: Poisoned labels as numpy array
        original_labels_sample: Original labels before poisoning
        output_dir: Base output directory
        experiment_info: Optional experiment metadata
        strategy_number: Strategy number for multi-strategy runs
        tokenizer: Optional tokenizer for decoding text
        original_data_sample: Optional original data before poisoning
    """
    attack_type = _extract_attack_type(attack_config)
    snapshot_dir = _get_snapshot_dir(output_dir, client_id, round_num, strategy_number)

    try:
        if len(data_sample.shape) == 4:  # (N, C, H, W) Image data
            filename = f"{attack_type}_visual.png"
            _save_image_grid(
                data_sample,
                labels_sample,
                original_labels_sample,
                snapshot_dir / filename,
                attack_config,
            )
        else:  # Text data
            filename = f"{attack_type}_samples.txt"
            _save_text_samples(
                labels_sample,
                original_labels_sample,
                snapshot_dir / filename,
                attack_config=attack_config,
                tokenizer=tokenizer,
                input_ids_original=original_data_sample,
                input_ids_poisoned=data_sample,
            )

        metadata = _create_snapshot_metadata(
            client_id=client_id,
            round_num=round_num,
            attack_type=attack_type,
            attack_config=attack_config,
            num_samples=len(data_sample),
            experiment_info=experiment_info,
        )

        metadata_path = snapshot_dir / f"{attack_type}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logging.debug(
            f"Saved {attack_type} visual snapshot: client {client_id}, round {round_num} -> {snapshot_dir}"
        )

    except Exception as e:
        logging.warning(f"Failed to save visual snapshot for client {client_id}: {e}")


def _save_image_grid(
    images: np.ndarray,
    labels: np.ndarray,
    original_labels: np.ndarray,
    filepath: Path,
    attack_config: Union[dict, List[dict]],
) -> None:
    """Save image samples as PNG grid with attack-specific annotations."""
    matplotlib.use("Agg")

    num_images = len(images)

    max_cols = 8
    if num_images <= max_cols:
        rows, cols = 1, num_images
    else:
        cols = max_cols
        rows = math.ceil(num_images / cols)

    figsize = (4 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, layout="constrained")

    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else axes

    attack_type = _extract_attack_type(attack_config)

    for i in range(num_images):
        ax = axes[i]

        if images.shape[1] == 1:  # Grayscale
            ax.imshow(images[i, 0], cmap="gray")
        else:  # RGB
            ax.imshow(images[i].transpose(1, 2, 0))

        # Handle composite attacks (multiple attacks stacked)
        if isinstance(attack_config, list) and len(attack_config) > 1:
            # Build title from all attack types
            title_parts = []
            for cfg in attack_config:
                cfg_type = cfg.get("attack_type", "unknown")
                if cfg_type == "label_flipping":
                    title_parts.append(
                        f"Label Flip: {labels[i]} (was {original_labels[i]})"
                    )
                elif cfg_type == "gaussian_noise":
                    snr = cfg.get("target_noise_snr", "?")
                    title_parts.append(f"Noise (SNR: {snr}dB)")
                elif cfg_type == "brightness":
                    delta = cfg.get("brightness_delta", cfg.get("factor", "?"))
                    title_parts.append(f"Brightness: {delta}")
                elif cfg_type == "token_replacement":
                    title_parts.append("Token poisoned")
            title = (
                "\n".join(title_parts)
                if title_parts
                else f"{attack_type}\nLabel: {labels[i]}"
            )
        else:  # Single attack
            if attack_type == "label_flipping":
                title = f"Label: {labels[i]}\n(was {original_labels[i]})"
            elif attack_type == "gaussian_noise":
                snr = _extract_attack_param(attack_config, "target_noise_snr")
                title = f"Noisy (SNR: {snr}dB)\nLabel: {labels[i]}"
            elif attack_type == "brightness":
                delta = _extract_attack_param(
                    attack_config, "brightness_delta", "factor"
                )
                title = f"Brightness: {delta}\nLabel: {labels[i]}"
            elif attack_type == "token_replacement":
                title = f"Token poisoned\nLabel: {labels[i]}"
            else:
                title = f"{attack_type}\nLabel: {labels[i]}"

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

    total_subplots = rows * cols
    for i in range(num_images, total_subplots):
        axes[i].axis("off")

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


def _save_text_samples(
    labels: np.ndarray,
    original_labels: np.ndarray,
    filepath: Path,
    attack_config: Optional[Union[dict, List[dict]]] = None,
    tokenizer=None,
    input_ids_original: Optional[np.ndarray] = None,
    input_ids_poisoned: Optional[np.ndarray] = None,
) -> None:
    """
    Save text sample visualization as TXT.

    When tokenizer and input_ids are provided, creates side-by-side comparison
    Otherwise, falls back to simple label comparison.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        if (
            tokenizer is not None
            and input_ids_original is not None
            and input_ids_poisoned is not None
        ):
            # Visualization with decoded text
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

            # Show each sample with decoded text
            for i in range(len(labels)):
                f.write(f"--- Sample {i} ---\n\n")

                original_tokens = input_ids_original[i]
                poisoned_tokens = input_ids_poisoned[i]

                try:
                    original_text = tokenizer.decode(
                        original_tokens, skip_special_tokens=True
                    )
                    poisoned_text = tokenizer.decode(
                        poisoned_tokens, skip_special_tokens=True
                    )

                    f.write(f'ORIGINAL: "{original_text}"\n')
                    f.write(f'POISONED: "{poisoned_text}"\n')

                    # Highlight differences
                    if original_text != poisoned_text:
                        f.write(
                            "          " + "^" * 15 + "[REPLACED]" + "^" * 15 + "\n"
                        )

                    current_label = labels[i]
                    original_label = original_labels[i]

                    # Check if labels are arrays (MLM) or scalars (classification)
                    if isinstance(current_label, np.ndarray) and current_label.size > 1:
                        # MLM case: show summary statistics
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
                        # Classification case: show actual label values
                        if np.array_equal(current_label, original_label):
                            f.write(f"Label: {current_label} (unchanged)\n")
                        else:
                            f.write(
                                f"Label: {original_label} → {current_label} (FLIPPED)\n"
                            )

                except Exception as e:
                    # Fallback if decoding fails
                    f.write(f"[Decoding error: {e}]\n")
                    f.write(
                        f"Original token IDs (first 10): {original_tokens[:10].tolist()}\n"
                    )
                    f.write(
                        f"Poisoned token IDs (first 10): {poisoned_tokens[:10].tolist()}\n"
                    )

                f.write("\n")

        else:
            # Fallback: simple label comparison
            f.write("Sample Labels (Poisoned vs Original)\n")
            f.write("=" * 40 + "\n")
            for i, (label, orig) in enumerate(zip(labels, original_labels)):
                f.write(f"Sample {i}: {label} (was {orig})\n")
