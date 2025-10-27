"""
HTML and JSON reporting utilities for attack snapshots.

Provides functions for generating interactive HTML indexes and JSON summaries
of attack snapshot data.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .attack_snapshots import (
    list_attack_snapshots,
    load_attack_snapshot,
    get_snapshot_summary,
)


def _get_snapshots_dir_checked(
    output_dir: str, strategy_number: int = 0
) -> Optional[Path]:
    """Get snapshots directory and verify it exists."""
    snapshots_dir = Path(output_dir) / f"attack_snapshots_{strategy_number}"
    if not snapshots_dir.exists():
        logging.warning(
            f"No attack_snapshots_{strategy_number} directory found in {output_dir}"
        )
        return None
    return snapshots_dir


def _extract_attack_params_for_display(
    attack_type: str, attack_config: dict
) -> list:
    """Extract attack parameters formatted for HTML display."""
    html_attack_params = []
    if attack_type == "label_flipping":
        html_attack_params.append(f"flip_fraction={attack_config.get('flip_fraction', '?')}")
        target_class = attack_config.get("target_class")
        if target_class is not None:
            html_attack_params.append(f"target_class={target_class}")
    elif attack_type == "gaussian_noise":
        html_attack_params.append(f"SNR={attack_config.get('target_noise_snr', '?')}dB")
        html_attack_params.append(f"ratio={attack_config.get('attack_ratio', '?')}")
    elif attack_type == "brightness":
        html_attack_params.append(f"delta={attack_config.get('brightness_delta', '?')}")
        html_attack_params.append(f"ratio={attack_config.get('attack_ratio', '?')}")
    return html_attack_params


def generate_summary_json(
    output_dir: str, run_config: Optional[dict] = None, strategy_number: int = 0
) -> None:
    """
    Generate summary.json with experiment metadata and attack timeline.

    Args:
        output_dir: Base output directory
        run_config: Optional strategy configuration dict
        strategy_number: Strategy number for multi-strategy runs (default: 0)
    """
    snapshots_dir = _get_snapshots_dir_checked(output_dir, strategy_number)
    if not snapshots_dir:
        return

    summary = get_snapshot_summary(output_dir, strategy_number)

    run_id = Path(output_dir).name
    full_summary = {
        "experiment": {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
        },
        "attack_summary": summary,
        "attack_timeline": {},
    }

    if run_config:
        full_summary["experiment"]["total_clients"] = run_config.get("num_of_clients")
        full_summary["experiment"]["total_rounds"] = run_config.get("num_of_rounds")
        full_summary["experiment"]["config_file"] = f"strategy_config_{strategy_number}.json"

    # Build attack timeline: {client_id: {round_num: [attack_types]}}
    snapshots = list_attack_snapshots(output_dir, strategy_number)
    for snapshot_path in snapshots:
        snapshot = load_attack_snapshot(str(snapshot_path))
        if snapshot:
            metadata = snapshot.get("metadata", snapshot)
            client_id = str(metadata.get("client_id"))
            round_num = str(metadata.get("round_num"))
            attack_type = metadata.get("attack_type")

            if client_id not in full_summary["attack_timeline"]:
                full_summary["attack_timeline"][client_id] = {}

            if round_num not in full_summary["attack_timeline"][client_id]:
                full_summary["attack_timeline"][client_id][round_num] = []

            full_summary["attack_timeline"][client_id][round_num].append(attack_type)

    summary_path = snapshots_dir / "summary.json"
    try:
        with open(summary_path, "w") as f:
            json.dump(full_summary, f, indent=2)
        logging.info(f"Generated attack summary: {summary_path}")
    except Exception as e:
        logging.warning(f"Failed to generate summary.json: {e}")


def generate_snapshot_index(
    output_dir: str, run_config: Optional[dict] = None, strategy_number: int = 0
) -> None:
    """
    Generate interactive HTML index for attack snapshots.

    Creates an index.html file with sortable/filterable table of all snapshots.

    Args:
        output_dir: Base output directory
        run_config: Optional strategy configuration dict for metadata
        strategy_number: Strategy number for multi-strategy runs (default: 0)
    """
    snapshots_dir = _get_snapshots_dir_checked(output_dir, strategy_number)
    if not snapshots_dir:
        return

    # Gather all snapshot metadata
    snapshots = list_attack_snapshots(output_dir, strategy_number)
    snapshot_data = []

    for snapshot_path in snapshots:
        snapshot = load_attack_snapshot(str(snapshot_path))
        if snapshot:
            metadata = snapshot.get("metadata", snapshot)
            attack_config = metadata.get("attack_config", {})

            # Handle list case (multiple attacks combined)
            if isinstance(attack_config, list):
                attack_config = attack_config[0] if attack_config else {}

            # Extract relevant parameters based on attack type
            attack_type = metadata.get("attack_type", "unknown")
            attack_parameters = _extract_attack_params_for_display(attack_type, attack_config)

            # Get relative paths for links
            rel_path = snapshot_path.relative_to(snapshots_dir)
            client_id = metadata.get("client_id")
            round_num = metadata.get("round_num")

            snapshot_data.append(
                {
                    "client": client_id,
                    "round": round_num,
                    "attack_type": attack_type,
                    "samples": metadata.get("num_samples", 0),
                    "parameters": ", ".join(attack_parameters) if attack_parameters else "N/A",
                    "pickle_path": str(rel_path),
                    "visual_path": f"client_{client_id}/round_{round_num}/{attack_type}_visual.png",
                    "metadata_path": f"client_{client_id}/round_{round_num}/{attack_type}_metadata.json",
                }
            )

    # Sort by client, then round
    snapshot_data.sort(key=lambda x: (x["client"], x["round"]))

    # Generate HTML
    html_content = _generate_index_html(snapshot_data, output_dir, run_config)

    # Save index.html
    index_path = snapshots_dir / "index.html"
    try:
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logging.info(f"Generated attack snapshot index: {index_path}")
    except Exception as e:
        logging.warning(f"Failed to generate index.html: {e}")


def _generate_index_html(
    snapshot_data: list, output_dir: str, run_config: Optional[dict] = None
) -> str:
    """Generate HTML content for snapshot index using Jinja2 template."""
    # Set up Jinja2 environment
    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(['html', 'xml'])
    )
    template = env.get_template("snapshot_index.html.jinja")

    # Prepare context data
    run_id = Path(output_dir).name
    total_clients = run_config.get("num_of_clients", "?") if run_config else "?"
    total_rounds = run_config.get("num_of_rounds", "?") if run_config else "?"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate unique values for stats
    unique_clients = sorted(set(s['client'] for s in snapshot_data))
    unique_rounds = sorted(set(s['round'] for s in snapshot_data))
    unique_attack_types = sorted(set(s['attack_type'] for s in snapshot_data))

    # Render template
    context = {
        "run_id": run_id,
        "total_clients": total_clients,
        "total_rounds": total_rounds,
        "timestamp": timestamp,
        "snapshot_data": snapshot_data,
        "unique_clients": unique_clients,
        "unique_rounds": unique_rounds,
        "unique_attack_types": unique_attack_types,
    }

    return template.render(context)
