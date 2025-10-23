"""
Lightweight attack data snapshot logging.

Provides utilities for saving small snapshots of poisoned data for inspection
and debugging.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import torch


def save_attack_snapshot(
    client_id: int,
    round_num: int,
    attack_config: dict,
    data_sample: torch.Tensor,
    labels_sample: torch.Tensor,
    output_dir: str,
    max_samples: int = 5,
    save_format: str = "pickle",
) -> None:
    """
    Save lightweight snapshot of poisoned data for inspection.

    Args:
        client_id: ID of the attacking client
        round_num: Current training round
        attack_config: Attack configuration dict (flat or nested)
        data_sample: Poisoned data tensor (first N samples from batch)
        labels_sample: Poisoned labels tensor (first N samples from batch)
        output_dir: Base output directory (e.g., "out/api_run_...")
        max_samples: Maximum number of samples to save (default: 5)
        save_format: Format to save ('pickle' or 'json' for metadata only)
    """
    # Create snapshots directory
    snapshots_dir = Path(output_dir) / "attack_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # Limit to max_samples
    data_sample = data_sample[:max_samples]
    labels_sample = labels_sample[:max_samples]

    # Prepare metadata
    metadata = {
        "client_id": client_id,
        "round_num": round_num,
        "attack_type": attack_config.get("attack_type") or attack_config.get("type"),
        "attack_config": attack_config,
        "num_samples": len(data_sample),
        "data_shape": list(data_sample.shape),
        "labels_shape": list(labels_sample.shape),
    }

    # Generate filename
    filename = f"client_{client_id}_round_{round_num}.{save_format}"
    filepath = snapshots_dir / filename

    try:
        if save_format == "pickle":
            # Save full snapshot with data (compact binary format)
            snapshot = {
                "metadata": metadata,
                "data": data_sample.cpu().numpy(),
                "labels": labels_sample.cpu().numpy(),
            }
            with open(filepath, "wb") as f:
                pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)

        elif save_format == "json":
            # Save metadata only (human-readable)
            with open(filepath, "w") as f:
                json.dump(metadata, f, indent=2)

        logging.debug(
            f"Saved attack snapshot: client {client_id}, round {round_num} "
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
        List of snapshot file paths
    """
    snapshots_dir = Path(output_dir) / "attack_snapshots"
    if not snapshots_dir.exists():
        return []

    snapshots = list(snapshots_dir.glob("client_*_round_*.*"))
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
