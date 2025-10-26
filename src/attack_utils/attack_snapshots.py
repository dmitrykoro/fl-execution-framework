"""
Lightweight attack data snapshot logging.

Provides utilities for saving small snapshots of poisoned data for inspection
and debugging.
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

import numpy as np
import torch


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


def save_attack_snapshot(
    client_id: int,
    round_num: int,
    attack_config: Union[dict, List[dict]],
    data_sample: torch.Tensor,
    labels_sample: torch.Tensor,
    original_labels_sample: torch.Tensor,
    output_dir: str,
    max_samples: int = 5,
    save_format: str = "pickle",
    experiment_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save lightweight snapshot of poisoned data for inspection.

    Args:
        client_id: ID of the attacking client
        round_num: Current training round
        attack_config: Attack configuration dict or list of dicts (for multiple attacks)
        data_sample: Poisoned data tensor (first N samples from batch)
        labels_sample: Poisoned labels tensor (first N samples from batch)
        original_labels_sample: Original labels before poisoning
        output_dir: Base output directory (e.g., "out/api_run_...")
        max_samples: Maximum number of samples to save (default: 5)
        save_format: Format to save ('pickle' or 'json' for metadata only)
        experiment_info: Optional experiment metadata (run_id, total_clients, total_rounds)
    """
    attack_type = _extract_attack_type(attack_config)

    snapshots_base = Path(output_dir) / "attack_snapshots"
    snapshot_dir = snapshots_base / f"client_{client_id}" / f"round_{round_num}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    data_sample = data_sample[:max_samples]
    labels_sample = labels_sample[:max_samples]
    original_labels_sample = original_labels_sample[:max_samples]

    metadata = {
        "client_id": client_id,
        "round_num": round_num,
        "attack_type": attack_type,
        "attack_config": attack_config,
        "num_samples": len(data_sample),
        "data_shape": list(data_sample.shape),
        "labels_shape": list(labels_sample.shape),
        "timestamp": datetime.now().isoformat(),
    }

    if experiment_info:
        metadata["experiment_info"] = experiment_info

    filename = f"{attack_type}.{save_format}"
    filepath = snapshot_dir / filename

    try:
        if save_format == "pickle":
            snapshot = {
                "metadata": metadata,
                "data": data_sample.cpu().numpy(),
                "labels": labels_sample.cpu().numpy(),
                "original_labels": original_labels_sample.cpu().numpy(),
            }
            with open(filepath, "wb") as f:
                pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)

        elif save_format == "json":
            with open(filepath, "w") as f:
                json.dump(metadata, f, indent=2)

        logging.debug(
            f"Saved {attack_type} attack snapshot: client {client_id}, round {round_num} "
            f"({len(data_sample)} samples) -> {filepath}"
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


def list_attack_snapshots(output_dir: str) -> list:
    """
    List all attack snapshots in an output directory.

    Args:
        output_dir: Base output directory

    Returns:
        List of snapshot file paths (hierarchical: client_*/round_*/*.pickle)
    """
    snapshots_dir = Path(output_dir) / "attack_snapshots"
    if not snapshots_dir.exists():
        return []

    snapshots = list(snapshots_dir.glob("client_*/round_*/*.pickle"))
    return sorted(snapshots)


def get_snapshot_summary(output_dir: str) -> dict:
    """
    Get summary statistics for all attack snapshots in a run.

    Args:
        output_dir: Base output directory

    Returns:
        Dict with summary statistics (client IDs, rounds, attack types, etc.)
    """
    snapshots = list_attack_snapshots(output_dir)

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
) -> None:
    """
    Save visual PNG/TXT files alongside pickle for viewing.

    Args:
        client_id: ID of the attacking client
        round_num: Current training round
        attack_config: Attack configuration dict or list of dicts (for multiple attacks)
        data_sample: Poisoned data as numpy array
        labels_sample: Poisoned labels as numpy array
        original_labels_sample: Original labels before poisoning
        output_dir: Base output directory
        experiment_info: Optional experiment metadata (run_id, total_clients, total_rounds)
    """
    attack_type = _extract_attack_type(attack_config)

    snapshots_base = Path(output_dir) / "attack_snapshots"
    snapshot_dir = snapshots_base / f"client_{client_id}" / f"round_{round_num}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    try:
        if len(data_sample.shape) == 4:  # (N, C, H, W)
            filename = f"{attack_type}_visual.png"
            _save_image_grid(
                data_sample,
                labels_sample,
                original_labels_sample,
                snapshot_dir / filename,
                attack_config,
            )
        else:
            filename = f"{attack_type}_samples.txt"
            _save_text_samples(
                labels_sample,
                original_labels_sample,
                snapshot_dir / filename,
            )

        metadata = {
            "client_id": client_id,
            "round_num": round_num,
            "attack_type": attack_type,
            "num_samples": len(data_sample),
            "attack_config": attack_config,
            "timestamp": datetime.now().isoformat(),
        }

        if experiment_info:
            metadata["experiment_info"] = experiment_info

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
    import matplotlib.pyplot as plt
    import math

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

        if attack_type == "label_flipping":
            title = f"Label: {labels[i]}\n(was {original_labels[i]})"
        elif attack_type == "gaussian_noise":
            # Handle both dict and list cases
            if isinstance(attack_config, list):
                snr = attack_config[0].get("target_noise_snr", "?") if attack_config else "?"
            else:
                snr = attack_config.get("target_noise_snr", "?")
            title = f"Noisy (SNR: {snr}dB)\nLabel: {labels[i]}"
        elif attack_type == "brightness":
            # Handle both dict and list cases
            if isinstance(attack_config, list):
                delta = attack_config[0].get("brightness_delta", attack_config[0].get("factor", "?")) if attack_config else "?"
            else:
                delta = attack_config.get(
                    "brightness_delta", attack_config.get("factor", "?")
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
) -> None:
    """Save text sample labels as TXT."""
    with open(filepath, "w") as f:
        f.write("Sample Labels (Poisoned vs Original)\n")
        f.write("=" * 40 + "\n")
        for i, (label, orig) in enumerate(zip(labels, original_labels)):
            f.write(f"Sample {i}: {label} (was {orig})\n")
