"""
Image visualization utilities for attack snapshots.
"""

import math
from pathlib import Path
from typing import List, Optional, Union

import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def _display_image(ax, image: np.ndarray) -> None:
    """Display an image on the given axes."""
    if image.shape[0] == 1:
        ax.imshow(image[0], cmap="gray")
    else:
        ax.imshow(image.transpose(1, 2, 0))


def _normalize_axes(axes, rows: int, cols: int):
    """Normalize axes for plotting."""
    if rows == 1 and cols == 1:
        return [[axes]]
    elif rows == 1:
        return [axes]
    elif cols == 1:
        return [[ax] for ax in axes]
    else:
        return axes


def _extract_attack_param(
    attack_config: Union[dict, List[dict]], *attack_parameters: str, default: any = "?"
) -> any:
    """Extract attack parameters from config."""
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
    """Extract attack type from config."""
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


def _build_single_attack_title(
    attack_config: Union[dict, List[dict]],
    attack_type: str,
    labels: np.ndarray,
    original_labels: np.ndarray,
    index: int,
    style: str,
) -> str:
    """Build a title for a single attack."""
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
    """Build a title for an attack."""
    if isinstance(attack_config, list) and len(attack_config) > 1:
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
        return _build_single_attack_title(
            attack_config, attack_type, labels, original_labels, index, style
        )


def save_image_grid(
    images: np.ndarray,
    labels: np.ndarray,
    original_labels: np.ndarray,
    filepath: Path,
    attack_config: Union[dict, List[dict]],
    original_images: Optional[np.ndarray] = None,
) -> None:
    """Save an image grid to a file."""
    matplotlib.use("Agg")

    num_samples = len(images)
    attack_type = _extract_attack_type(attack_config)

    if original_images is not None:
        pairs_per_row = 4
        cols = pairs_per_row * 2
        rows = math.ceil(num_samples / pairs_per_row)
        figsize = (3 * cols, 3 * rows)

        fig, axes = plt.subplots(
            rows, cols, figsize=figsize, gridspec_kw={"wspace": 0.3, "hspace": 0.5}
        )

        axes = _normalize_axes(axes, rows, cols)

        for i in range(num_samples):
            pair_idx = i % pairs_per_row
            row_idx = i // pairs_per_row
            col_original = pair_idx * 2
            col_poisoned = pair_idx * 2 + 1

            ax_original = axes[row_idx][col_original]
            _display_image(ax_original, original_images[i])

            ax_original.set_title(
                f"Original\nLabel: {original_labels[i]}",
                fontsize=10,
                fontweight="bold",
                color="#2c3e50",
            )
            ax_original.axis("off")

            ax_poisoned = axes[row_idx][col_poisoned]
            _display_image(ax_poisoned, images[i])

            title = _build_attack_title(
                attack_config, attack_type, labels, original_labels, i, "side_by_side"
            )

            ax_poisoned.set_title(
                title, fontsize=10, fontweight="bold", color="#c0392b"
            )
            ax_poisoned.axis("off")

        total_pairs_needed = num_samples
        total_subplots = rows * pairs_per_row
        for i in range(total_pairs_needed, total_subplots):
            pair_idx = i % pairs_per_row
            row_idx = i // pairs_per_row
            col_original = pair_idx * 2
            col_poisoned = pair_idx * 2 + 1
            axes[row_idx][col_original].axis("off")
            axes[row_idx][col_poisoned].axis("off")

        for row_idx in range(rows):
            for pair_idx in range(pairs_per_row - 1):
                col_poisoned = pair_idx * 2 + 1
                ax_poisoned = axes[row_idx][col_poisoned]

                bbox = ax_poisoned.get_position()

                line = plt.Line2D(
                    [bbox.x1 + 0.015, bbox.x1 + 0.015],
                    [bbox.y0, bbox.y1],
                    transform=fig.transFigure,
                    color="#95a5a6",
                    linewidth=2,
                    linestyle="-",
                    alpha=0.6,
                )
                fig.add_artist(line)

    else:
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
