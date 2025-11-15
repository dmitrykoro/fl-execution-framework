"""
Utility functions for saving and loading attack snapshots.
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from .snapshot_image_viz import save_image_grid
from .snapshot_text_viz import save_text_samples


def _extract_attack_type(attack_config: Union[dict, List[dict]]) -> str:
    """Extract attack type(s) from attack configuration."""
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


def _get_snapshot_dir(
    output_dir: str, client_id: int, round_num: int, strategy_number: int = 0
) -> Path:
    """Get or create the directory for saving snapshots."""
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
    """Create metadata dictionary for attack snapshot."""
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


def _save_metadata_json(filepath: Path, metadata: dict) -> None:
    """Save metadata dictionary as JSON file."""
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
    """Save attack snapshot as a pickle file."""
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
    """Save attack snapshot for inspection."""
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
    """Load an attack snapshot for inspection."""
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
    """List all attack snapshots in an output directory."""
    snapshots_dir = Path(output_dir) / f"attack_snapshots_{strategy_number}"
    if not snapshots_dir.exists():
        return []

    pickle_snapshots = list(snapshots_dir.glob("client_*/round_*/*.pickle"))

    if pickle_snapshots:
        return sorted(pickle_snapshots)

    json_snapshots = list(snapshots_dir.glob("client_*/round_*/*_metadata.json"))
    return sorted(json_snapshots)


def get_snapshot_summary(output_dir: str, strategy_number: int = 0) -> dict:
    """Get a summary of all attack snapshots in an output directory."""
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
    """Save visual snapshot of attack samples for inspection."""
    attack_type = _extract_attack_type(attack_config)
    snapshot_dir = _get_snapshot_dir(output_dir, client_id, round_num, strategy_number)

    try:
        if len(data_sample.shape) == 4:
            filename = f"{attack_type}_visual.png"
            save_image_grid(
                data_sample,
                labels_sample,
                original_labels_sample,
                snapshot_dir / filename,
                attack_config,
                original_images=original_data_sample,
            )
        else:
            filename = f"{attack_type}_samples.txt"
            save_text_samples(
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
