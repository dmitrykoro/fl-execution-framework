"""
Lightweight attack data snapshot logging.

Provides utilities for saving small snapshots of poisoned data.
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


def _save_metadata_json(filepath: Path, metadata: dict) -> None:
    """Save metadata as JSON file."""
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)


def _save_pickle_snapshot(
    snapshot_dir: Path,
    attack_type: str,
    data_sample: torch.Tensor,
    labels_sample: torch.Tensor,
    original_labels_sample: Optional[torch.Tensor],
    metadata: dict,
    client_id: int,
    round_num: int,
) -> None:
    """Save snapshot as pickle + JSON metadata."""
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

    _save_metadata_json(json_path, metadata)

    logging.debug(
        f"Saved {attack_type} attack snapshot: client {client_id}, round {round_num} "
        f"({len(data_sample)} samples) -> {pickle_path} and {json_path}"
    )


def _display_image(ax, image: np.ndarray) -> None:
    """Display image on axis, handling grayscale vs RGB."""
    if image.shape[0] == 1:  # Grayscale (C, H, W)
        ax.imshow(image[0], cmap="gray")
    else:  # RGB
        ax.imshow(image.transpose(1, 2, 0))


def _normalize_axes(axes, rows: int, cols: int):
    """Normalize matplotlib axes to consistent 2D array format."""
    if rows == 1 and cols == 1:
        return [[axes]]
    elif rows == 1:
        return [axes]
    elif cols == 1:
        return [[ax] for ax in axes]
    else:
        return axes


def _build_single_attack_title(
    attack_config: Union[dict, List[dict]],
    attack_type: str,
    labels: np.ndarray,
    original_labels: np.ndarray,
    index: int,
    style: str,
) -> str:
    """Build title for single attack type."""
    if attack_type == "label_flipping":
        if style == "side_by_side":
            return f"Poisoned\nLabel: {labels[index]}"
        return f"Label: {labels[index]}\n(was {original_labels[index]})"

    elif attack_type == "gaussian_noise":
        snr = _extract_attack_param(attack_config, "target_noise_snr")
        if style == "side_by_side":
            return f"Poisoned (Noise)\nSNR: {snr}dB\nLabel: {labels[index]}"
        return f"Noisy (SNR: {snr}dB)\nLabel: {labels[index]}"

    elif attack_type == "token_replacement":
        return f"Token poisoned\nLabel: {labels[index]}"

    else:
        prefix = f"Poisoned ({attack_type})" if style == "side_by_side" else attack_type
        return f"{prefix}\nLabel: {labels[index]}"


def _build_attack_title(
    attack_config: Union[dict, List[dict]],
    attack_type: str,
    labels: np.ndarray,
    original_labels: np.ndarray,
    index: int,
    style: str = "side_by_side",
) -> str:
    """Build title for poisoned image based on attack type."""
    if isinstance(attack_config, list) and len(attack_config) > 1:
        # Composite attacks
        title_parts = ["Poisoned"] if style == "side_by_side" else []

        for cfg in attack_config:
            cfg_type = cfg.get("attack_type", "unknown")
            if cfg_type == "label_flipping":
                if style == "side_by_side":
                    title_parts.append(f"Label: {labels[index]}")
                else:
                    title_parts.append(
                        f"Label Flip: {labels[index]} (was {original_labels[index]})"
                    )
            elif cfg_type == "gaussian_noise":
                snr = cfg.get("target_noise_snr", "?")
                if style == "side_by_side":
                    title_parts.append(f"Noise: {snr}dB")
                else:
                    title_parts.append(f"Noise (SNR: {snr}dB)")
            elif cfg_type == "token_replacement" and style == "fallback":
                title_parts.append("Token poisoned")

        return (
            "\n".join(title_parts)
            if title_parts
            else f"{attack_type}\nLabel: {labels[index]}"
        )
    else:
        # Single attack
        return _build_single_attack_title(
            attack_config, attack_type, labels, original_labels, index, style
        )


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
        if save_format in ("pickle_and_visual", "pickle"):
            _save_pickle_snapshot(
                snapshot_dir,
                attack_type,
                data_sample,
                labels_sample,
                original_labels_sample,
                metadata,
                client_id,
                round_num,
            )

        elif save_format == "visual":
            json_path = snapshot_dir / f"{attack_type}_metadata.json"
            _save_metadata_json(json_path, metadata)

            logging.debug(
                f"Saved {attack_type} attack snapshot metadata: client {client_id}, round {round_num} -> {json_path}"
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
        List of snapshot file paths
    """
    snapshots_dir = Path(output_dir) / f"attack_snapshots_{strategy_number}"
    if not snapshots_dir.exists():
        return []

    pickle_snapshots = list(snapshots_dir.glob("client_*/round_*/*.pickle"))

    if pickle_snapshots:
        return sorted(pickle_snapshots)

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
                original_images=original_data_sample,
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
    original_images: Optional[np.ndarray] = None,
) -> None:
    """
    Save image samples as PNG grid with attack-specific annotations.

    If original_images provided, creates side-by-side comparison grid.
    Otherwise, shows only poisoned images.
    """
    matplotlib.use("Agg")

    num_samples = len(images)
    attack_type = _extract_attack_type(attack_config)

    # Side-by-side comparison
    if original_images is not None:
        # 8 columns: 4 pairs of [Original | Poisoned] per row
        pairs_per_row = 4
        cols = pairs_per_row * 2
        rows = math.ceil(num_samples / pairs_per_row)
        figsize = (3 * cols, 3 * rows)

        # Spacing between pairs for visual separation
        fig, axes = plt.subplots(
            rows, cols, figsize=figsize,
            gridspec_kw={'wspace': 0.3, 'hspace': 0.5}
        )

        # Normalize axes for consistent indexing
        axes = _normalize_axes(axes, rows, cols)

        for i in range(num_samples):
            pair_idx = i % pairs_per_row  # Which pair in the row (0-3)
            row_idx = i // pairs_per_row  # Which row
            col_original = pair_idx * 2  # Original column (0, 2, 4, 6)
            col_poisoned = pair_idx * 2 + 1  # Poisoned column (1, 3, 5, 7)

            # Original image
            ax_original = axes[row_idx][col_original]
            _display_image(ax_original, original_images[i])

            ax_original.set_title(
                f"Original\nLabel: {original_labels[i]}",
                fontsize=10,
                fontweight="bold",
                color="#2c3e50"
            )
            ax_original.axis("off")

            # Poisoned image
            ax_poisoned = axes[row_idx][col_poisoned]
            _display_image(ax_poisoned, images[i])

            # Build title for poisoned image
            title = _build_attack_title(
                attack_config, attack_type, labels, original_labels, i, "side_by_side"
            )

            ax_poisoned.set_title(
                title,
                fontsize=10,
                fontweight="bold",
                color="#c0392b"
            )
            ax_poisoned.axis("off")

        # Hide unused subplots
        total_pairs_needed = num_samples
        total_subplots = rows * pairs_per_row
        for i in range(total_pairs_needed, total_subplots):
            pair_idx = i % pairs_per_row
            row_idx = i // pairs_per_row
            col_original = pair_idx * 2
            col_poisoned = pair_idx * 2 + 1
            axes[row_idx][col_original].axis("off")
            axes[row_idx][col_poisoned].axis("off")

        # Add vertical separators between pairs
        for row_idx in range(rows):
            for pair_idx in range(pairs_per_row - 1):  # Don't add after last pair
                # Draw vertical line after each poisoned column (columns 1, 3, 5)
                col_poisoned = pair_idx * 2 + 1
                ax_poisoned = axes[row_idx][col_poisoned]

                # Get the position of this subplot
                bbox = ax_poisoned.get_position()

                # Add vertical line in figure coordinates
                line = plt.Line2D(
                    [bbox.x1 + 0.015, bbox.x1 + 0.015],  # x position (slightly right of subplot)
                    [bbox.y0, bbox.y1],  # y position (full height)
                    transform=fig.transFigure,
                    color='#95a5a6',
                    linewidth=2,
                    linestyle='-',
                    alpha=0.6
                )
                fig.add_artist(line)

    else:
        # Fallback: original behavior (poisoned images only)
        max_cols = 8
        if num_samples <= max_cols:
            rows, cols = 1, num_samples
        else:
            cols = max_cols
            rows = math.ceil(num_samples / cols)

        figsize = (4 * cols, 4 * rows)
        fig, axes = plt.subplots(rows, cols, figsize=figsize, layout="constrained")

        if num_samples == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, "flatten") else axes

        for i in range(num_samples):
            ax = axes[i]

            _display_image(ax, images[i])

            # Build title using helper
            title = _build_attack_title(
                attack_config, attack_type, labels, original_labels, i, "fallback"
            )

            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.axis("off")

        total_subplots = rows * cols
        for i in range(num_samples, total_subplots):
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

                    # Skip unchanged samples
                    if original_text == poisoned_text:
                        samples_skipped += 1
                        continue

                    f.write(f"--- Sample {i} ---\n\n")

                    f.write(f'ORIGINAL: "{original_text}"\n')
                    f.write(f'POISONED: "{poisoned_text}"\n')

                    # Calculate token-level replacement statistics
                    num_replaced = np.sum(original_tokens != poisoned_tokens)
                    total_tokens = len(original_tokens)
                    replacement_rate = num_replaced / total_tokens * 100 if total_tokens > 0 else 0

                    f.write(
                        f"          {num_replaced}/{total_tokens} tokens replaced ({replacement_rate:.1f}%)\n"
                    )
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

                    samples_shown += 1

                except Exception as e:
                    # Fallback if decoding fails
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

            # Add summary footer
            if samples_skipped > 0:
                f.write("=" * 80 + "\n")
                f.write(f"SUMMARY: {samples_shown} samples modified, {samples_skipped} samples unchanged (skipped)\n")
                f.write("=" * 80 + "\n")
        else:
            # Fallback: simple label comparison
            f.write("Sample Labels (Poisoned vs Original)\n")
            f.write("=" * 40 + "\n")
            for i, (label, orig) in enumerate(zip(labels, original_labels)):
                f.write(f"Sample {i}: {label} (was {orig})\n")
