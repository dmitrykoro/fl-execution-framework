import logging
import torch
import numpy as np
import pickle
import json
from datetime import datetime

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.attack_utils.snapshot_image_viz import save_image_grid
from src.attack_utils.snapshot_text_viz import save_text_samples, save_extra_attack_info


def save_attack_snapshots(
        attack_snapshot_format,
        tokenizer,
        model_type,
        save_attack_snapshots,
        client_id,
        output_dir,
        max_samples,
        save_format,
        experiment_info,
        strategy_number,
        current_round,
        attack_configs,
        data_sample,
        labels_sample,
        original_data_sample=None,
        original_labels_sample=None
):
    """
    Helper function to save attack snapshots for both CNN and transformer models.
    """

    if not (save_attack_snapshots and output_dir):
        return

    # Save pickle/JSON snapshot
    save_attack_snapshot(
        client_id=client_id,
        round_num=current_round,
        attack_config=attack_configs,
        data_sample=data_sample,
        labels_sample=labels_sample,
        original_labels_sample=original_labels_sample,
        output_dir=output_dir,
        max_samples=max_samples,
        save_format=save_format,
        experiment_info=experiment_info,
        strategy_number=strategy_number,
    )

    # Save visual snapshot (PNG for CNN, TXT for transformer)
    if attack_snapshot_format in ["visual", "pickle_and_visual"]:
        if model_type == "cnn" and original_data_sample is not None:
            save_visual_snapshot(
                client_id=client_id,
                round_num=current_round,
                attack_config=attack_configs,
                data_sample=data_sample.cpu().numpy(),
                labels_sample=labels_sample.cpu().numpy(),
                original_labels_sample=original_labels_sample.cpu().numpy() if original_labels_sample is not None else labels_sample.cpu().numpy(),
                output_dir=output_dir,
                model_type=model_type,
                experiment_info=experiment_info,
                strategy_number=strategy_number,
                original_data_sample=original_data_sample.cpu().numpy(),
            )
        elif model_type == "transformer" and original_data_sample is not None and tokenizer is not None:
            save_visual_snapshot(
                client_id=client_id,
                round_num=current_round,
                attack_config=attack_configs,
                data_sample=data_sample.cpu().numpy(),
                labels_sample=labels_sample.cpu().numpy(),
                original_labels_sample=original_labels_sample.cpu().numpy() if original_labels_sample is not None else labels_sample.cpu().numpy(),
                output_dir=output_dir,
                model_type=model_type,
                experiment_info=experiment_info,
                strategy_number=strategy_number,
                tokenizer=tokenizer,
                original_data_sample=original_data_sample.cpu().numpy(),
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
    """Save attack snapshot for inspection.

    Main entry point for saving snapshots. Supports pickle, visual, or both formats.

    Args:
        client_id: Client ID
        round_num: Round number
        attack_config: Attack configuration dict or list of dicts
        data_sample: Sample data tensor
        labels_sample: Sample labels tensor
        original_labels_sample: Original labels before poisoning (optional)
        output_dir: Base output directory
        max_samples: Maximum samples to include in snapshot (default: 5)
        save_format: Format to save - "pickle", "visual", or "pickle_and_visual" (default: "pickle")
        experiment_info: Additional experiment metadata (optional)
        strategy_number: Strategy index for multi-strategy runs (default: 0)

    Note:
        Creates snapshot directory structure and saves metadata JSON file.
        Visual snapshots delegated to save_visual_snapshot function.
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


def save_visual_snapshot(
        client_id: int,
        round_num: int,
        attack_config: Union[dict, List[dict]],
        data_sample: np.ndarray,
        labels_sample: np.ndarray,
        original_labels_sample: np.ndarray,
        output_dir: str,
        model_type: Optional[str] = None,
        experiment_info: Optional[Dict[str, Any]] = None,
        strategy_number: int = 0,
        tokenizer=None,
        original_data_sample: Optional[np.ndarray] = None,
        specific_attack_type: Optional[str] = None,
) -> None:
    """Save visual snapshot of attack samples for inspection.

    Creates visual representation of poisoned samples. For images, creates
    side-by-side grids. For text (transformers), creates text file with samples.

    Args:
        client_id: Client ID
        round_num: Round number
        attack_config: Attack configuration dict or list of dicts
        data_sample: Sample data array
        labels_sample: Sample labels array
        original_labels_sample: Original labels array
        output_dir: Base output directory
        model_type: Type of model being trained (optional)
        experiment_info: Additional experiment metadata (optional)
        strategy_number: Strategy index for multi-strategy runs (default: 0)
        tokenizer: HuggingFace tokenizer for text visualization (optional)
        original_data_sample: Original data before poisoning for comparison (optional)
        specific_attack_type: Used for when an visual attack snapshot needs to be made outside of the normal pipeline (optional)

    Note:
        Automatically detects data type (images vs text) and uses appropriate
        visualization method.
    """
    attack_type = _extract_attack_type(attack_config)
    snapshot_dir = _get_snapshot_dir(output_dir, client_id, round_num, strategy_number)

    try:
        if model_type == "cnn":
            filename = f"{attack_type}_visual.png"
            save_image_grid(
                data_sample,
                labels_sample,
                original_labels_sample,
                snapshot_dir / filename,
                attack_config,
                original_images=original_data_sample,
            )
        elif model_type == "transformer":
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

        if specific_attack_type == "little_is_enough" and experiment_info is not None:
            filename = f"{specific_attack_type}_attack_info.txt"
            save_extra_attack_info(
                client_id=client_id,
                round_num=round_num,
                filepath=snapshot_dir / filename,
                attack_type=specific_attack_type,
                experiment_info=experiment_info,
            )


        logging.debug(
            f"Saved {attack_type} visual snapshot: client {client_id}, round {round_num} -> {snapshot_dir}"
        )

    except Exception as e:
        logging.warning(f"Failed to save visual snapshot for client {client_id}: {e}")


def _extract_attack_type(attack_config: Union[dict, List[dict]]) -> str:
    """Extract attack type(s) from attack configuration.

    Handles both single attack and composite multi-attack configs.

    Args:
        attack_config: Attack configuration dict or list of dicts

    Returns:
        Attack type string, or composite type for multiple attacks
        (e.g., "label_flipping" or "label_flipping_gaussian_noise")
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


def _get_snapshot_dir(
        output_dir: str, client_id: int, round_num: int, strategy_number: int = 0
) -> Path:
    """Get or create the directory for saving snapshots.

    Creates nested directory structure: output_dir/attack_snapshots/strategy_{N}/

    Args:
        output_dir: Base output directory path
        client_id: Client ID for the snapshot
        round_num: Round number for the snapshot
        strategy_number: Strategy index for multi-strategy runs (default: 0)

    Returns:
        Path object for the snapshot directory
    """
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
    """Create metadata dictionary for attack snapshot.

    Args:
        client_id: Client ID for the snapshot
        round_num: Round number for the snapshot
        attack_type: Attack type string
        attack_config: Attack configuration dict or list of dicts
        num_samples: Number of samples in the snapshot
        data_shape: Shape of data tensor as list (optional)
        labels_shape: Shape of labels tensor as list (optional)
        experiment_info: Additional experiment metadata (optional)

    Returns:
        Dictionary containing snapshot metadata
    """
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
    """Save attack snapshot as a pickle file.

    Args:
        snapshot_dir: Directory where pickle file will be saved
        attack_type: Attack type string for filename
        data_sample: Sample data tensor
        labels_sample: Sample labels tensor
        original_labels_sample: Original labels before poisoning (optional)
        metadata: Snapshot metadata dict
        client_id: Client ID
        round_num: Round number
    """
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


def _save_metadata_json(filepath: Path, metadata: dict) -> None:
    """Save metadata dictionary as JSON file.

    Args:
        filepath: Path where JSON file will be saved
        metadata: Metadata dictionary to serialize
    """
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)
