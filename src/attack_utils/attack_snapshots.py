"""
Utility functions for saving and loading attack snapshots.
"""

import json
import logging
import pickle

from pathlib import Path
from typing import Optional


def load_attack_snapshot(filepath: str) -> Optional[dict]:
    """Load an attack snapshot for inspection.

    Args:
        filepath: Path to the pickle snapshot file

    Returns:
        Dictionary containing snapshot data and metadata, or None if load fails
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
    """List all attack snapshots in an output directory.

    Args:
        output_dir: Base output directory
        strategy_number: Strategy index to search (default: 0)

    Returns:
        List of dictionaries with snapshot info (client_id, round, path, metadata)
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
    """Get a summary of all attack snapshots in an output directory.

    Args:
        output_dir: Base output directory
        strategy_number: Strategy index to search (default: 0)

    Returns:
        Dictionary with summary statistics (total_snapshots, clients, rounds, attack_types)
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

    summary["clients_attacked"] = sorted(list(summary["clients_attacked"]))
    summary["rounds_with_attacks"] = sorted(list(summary["rounds_with_attacks"]))
    summary["attack_types"] = sorted(list(summary["attack_types"]))

    return summary
