"""
Utility functions for saving and visualizing weight poisoning snapshots.

Creates histograms and statistics showing weight distributions before
and after poisoning attacks for debugging and analysis.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


def _get_weight_snapshot_dir(
    output_dir: str,
    client_id: int,
    round_num: int,
    strategy_number: int = 0,
) -> Path:
    """Get or create directory for weight snapshots.

    Args:
        output_dir: Base output directory path.
        client_id: Client ID for the snapshot.
        round_num: Round number for the snapshot.
        strategy_number: Strategy index for multi-strategy runs.

    Returns:
        Path object for the weight snapshot directory.
    """
    snapshots_base = Path(output_dir) / f"weight_snapshots_{strategy_number}"
    snapshot_dir = snapshots_base / f"client_{client_id}" / f"round_{round_num}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_dir


def _compute_weight_statistics(parameters: List[NDArray]) -> Dict[str, float]:
    """Compute summary statistics for model parameters.

    Args:
        parameters: List of parameter arrays.

    Returns:
        Dictionary with statistics (mean, std, min, max, l2_norm, sparsity).
    """
    all_weights = np.concatenate([p.flatten() for p in parameters])

    return {
        "mean": float(np.mean(all_weights)),
        "std": float(np.std(all_weights)),
        "min": float(np.min(all_weights)),
        "max": float(np.max(all_weights)),
        "l2_norm": float(np.linalg.norm(all_weights)),
        "sparsity": float(np.mean(np.abs(all_weights) < 1e-6)),
        "num_parameters": int(len(all_weights)),
        "num_layers": len(parameters),
    }


def _compute_weight_diff_statistics(
    params_before: List[NDArray],
    params_after: List[NDArray],
) -> Dict[str, float]:
    """Compute statistics on the difference between weight sets.

    Args:
        params_before: Parameters before poisoning.
        params_after: Parameters after poisoning.

    Returns:
        Dictionary with difference statistics.
    """
    diffs = [after - before for before, after in zip(params_before, params_after)]
    all_diffs = np.concatenate([d.flatten() for d in diffs])

    return {
        "diff_mean": float(np.mean(all_diffs)),
        "diff_std": float(np.std(all_diffs)),
        "diff_min": float(np.min(all_diffs)),
        "diff_max": float(np.max(all_diffs)),
        "diff_l2_norm": float(np.linalg.norm(all_diffs)),
        "num_changed": int(np.sum(np.abs(all_diffs) > 1e-10)),
        "pct_changed": float(np.mean(np.abs(all_diffs) > 1e-10) * 100),
    }


def save_weight_snapshot(
    parameters_before: List[NDArray],
    parameters_after: List[NDArray],
    attack_type: str,
    attack_config: dict,
    client_id: int,
    round_num: int,
    output_dir: str,
    strategy_number: int = 0,
    experiment_info: Optional[Dict[str, Any]] = None,
    save_histogram: bool = True,
) -> None:
    """Save weight distribution snapshot before/after poisoning.

    Creates JSON metadata file with statistics and optionally a
    histogram visualization of weight distributions.

    Args:
        parameters_before: Model parameters before poisoning.
        parameters_after: Model parameters after poisoning.
        attack_type: Type of weight attack applied.
        attack_config: Attack configuration dictionary.
        client_id: Client ID.
        round_num: Round number.
        output_dir: Base output directory.
        strategy_number: Strategy index for multi-strategy runs.
        experiment_info: Additional experiment metadata.
        save_histogram: Whether to save histogram plot (requires matplotlib).
    """
    snapshot_dir = _get_weight_snapshot_dir(
        output_dir, client_id, round_num, strategy_number
    )

    # Compute statistics
    stats_before = _compute_weight_statistics(parameters_before)
    stats_after = _compute_weight_statistics(parameters_after)
    diff_stats = _compute_weight_diff_statistics(parameters_before, parameters_after)

    metadata = {
        "client_id": client_id,
        "round_num": round_num,
        "attack_type": attack_type,
        "attack_config": attack_config,
        "timestamp": datetime.now().isoformat(),
        "statistics": {
            "before": stats_before,
            "after": stats_after,
            "difference": diff_stats,
        },
    }

    if experiment_info:
        metadata["experiment_info"] = experiment_info

    # Save metadata JSON
    json_path = snapshot_dir / f"{attack_type}_weight_metadata.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.debug(
        f"Saved {attack_type} weight snapshot: client {client_id}, round {round_num} "
        f"-> {json_path}"
    )

    # Save histogram if requested
    if save_histogram:
        try:
            _save_weight_histogram(
                parameters_before,
                parameters_after,
                attack_type,
                snapshot_dir,
                client_id,
                round_num,
            )
        except ImportError:
            logger.warning("matplotlib not available, skipping histogram visualization")
        except Exception as e:
            logger.warning(f"Failed to save weight histogram: {e}")


def _save_weight_histogram(
    params_before: List[NDArray],
    params_after: List[NDArray],
    attack_type: str,
    snapshot_dir: Path,
    client_id: int,
    round_num: int,
) -> None:
    """Save histogram comparing weight distributions.

    Args:
        params_before: Parameters before poisoning.
        params_after: Parameters after poisoning.
        attack_type: Attack type for filename.
        snapshot_dir: Directory to save histogram.
        client_id: Client ID for title.
        round_num: Round number for title.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weights_before = np.concatenate([p.flatten() for p in params_before])
    weights_after = np.concatenate([p.flatten() for p in params_after])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Before histogram
    axes[0].hist(weights_before, bins=100, alpha=0.7, color="blue", density=True)
    axes[0].set_title("Before Poisoning")
    axes[0].set_xlabel("Weight Value")
    axes[0].set_ylabel("Density")
    axes[0].set_yscale("log")

    # After histogram
    axes[1].hist(weights_after, bins=100, alpha=0.7, color="red", density=True)
    axes[1].set_title("After Poisoning")
    axes[1].set_xlabel("Weight Value")
    axes[1].set_ylabel("Density")
    axes[1].set_yscale("log")

    # Overlay comparison
    axes[2].hist(
        weights_before, bins=100, alpha=0.5, color="blue", density=True, label="Before"
    )
    axes[2].hist(
        weights_after, bins=100, alpha=0.5, color="red", density=True, label="After"
    )
    axes[2].set_title("Comparison")
    axes[2].set_xlabel("Weight Value")
    axes[2].set_ylabel("Density")
    axes[2].set_yscale("log")
    axes[2].legend()

    fig.suptitle(
        f"Weight Distribution - Client {client_id}, Round {round_num} ({attack_type})"
    )
    plt.tight_layout()

    histogram_path = snapshot_dir / f"{attack_type}_weight_histogram.png"
    plt.savefig(histogram_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.debug(f"Saved weight histogram: {histogram_path}")


def load_weight_snapshot(filepath: str) -> Optional[dict]:
    """Load a weight snapshot metadata file.

    Args:
        filepath: Path to the JSON metadata file.

    Returns:
        Dictionary containing snapshot metadata, or None if load fails.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error(f"Weight snapshot file not found: {filepath}")
        return None

    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load weight snapshot {filepath}: {e}")
        return None


def list_weight_snapshots(output_dir: str, strategy_number: int = 0) -> List[Path]:
    """List all weight snapshots in an output directory.

    Args:
        output_dir: Base output directory.
        strategy_number: Strategy index to search.

    Returns:
        List of paths to weight snapshot metadata files.
    """
    snapshots_dir = Path(output_dir) / f"weight_snapshots_{strategy_number}"
    if not snapshots_dir.exists():
        return []

    return sorted(snapshots_dir.glob("client_*/round_*/*_weight_metadata.json"))
