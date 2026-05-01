"""
HTML and JSON reporting utilities for attack snapshots.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .attack_snapshots import (
    get_snapshot_summary,
    list_attack_snapshots,
    load_attack_snapshot,
)


def _get_snapshots_dir_checked(
    output_dir: str, strategy_number: int = 0
) -> Optional[Path]:
    """Get snapshots directory and verify it exists.

    Args:
        output_dir: Base output directory
        strategy_number: Strategy index (default: 0)

    Returns:
        Path to snapshots directory, or None if it doesn not exist
    """
    snapshots_dir = Path(output_dir) / f"attack_snapshots_{strategy_number}"
    if not snapshots_dir.exists():
        logging.warning(
            f"No attack_snapshots_{strategy_number} directory found in {output_dir}"
        )
        snapshots_dir.mkdir(parents=True, exist_ok=True)
    return snapshots_dir


def _extract_attack_params_for_display(attack_type: str, attack_config: dict) -> list:
    """Extract attack parameters formatted for HTML display.

    Args:
        attack_type: Attack type string
        attack_config: Attack configuration dict

    Returns:
        List of tuples (param_name, param_value) for display
    """
    html_attack_params = []
    if attack_type == "label_flipping":
        pass
    elif attack_type == "gaussian_noise":
        html_attack_params.append(f"SNR={attack_config.get('target_noise_snr', '?')}dB")
        html_attack_params.append(f"ratio={attack_config.get('attack_ratio', '?')}")
    elif attack_type == "token_replacement":
        vocab = attack_config.get("target_vocabulary")
        strategy = attack_config.get("replacement_strategy")
        prob = attack_config.get("replacement_probability")
        if vocab:
            html_attack_params.append(f"vocab={vocab}")
        if strategy:
            html_attack_params.append(f"strategy={strategy}")
        if prob is not None:
            html_attack_params.append(f"prob={prob}")
    elif attack_type == "target_flipping":
        degree_of_posoning = attack_config.get("attack_ratio", 0.1)
        substitute_fraction = 0.25
        if degree_of_posoning > substitute_fraction:
            substitute_fraction = degree_of_posoning
        html_attack_params.append(f"substitute={substitute_fraction}")
        html_attack_params.append(f"poison={degree_of_posoning}")
    return html_attack_params


def _split_composite_attack_info(attack_type: str, attack_configs: list) -> list:
    """Split composite attack type into individual attack entries for display.

    Args:
        attack_type: Composite attack type string (e.g., "label_flipping_gaussian_noise")
        attack_configs: List of attack configuration dicts

    Returns:
        List of dicts with attack_type and params for each individual attack
    """
    attack_info = []
    for config in attack_configs:
        config_type = config.get("attack_type", "unknown")
        params = _extract_attack_params_for_display(config_type, config)
        attack_info.append({"type": config_type, "params": params})

    return attack_info


def _build_visuals_list(attack_types:list[str], base_dir: Path) -> list:
    """
    Build list of visuals for HTML report.
    
    Args:
        attack_type: List of attack types
        base_dir: Path to base directory

    Returns:
        List of visuals
    """
    visuals = []
    
    for attack_type in attack_types:
        if attack_type == "token_replacement":
            visuals.append(
                {
                    "path": (base_dir / f"{attack_type}_samples.txt").as_posix(),
                    "type": "text",
                    "icon": "📝",
                    "title": "View text samples",
                }
            )
        elif attack_type == "little_is_enough":
            visuals.append(
                {
                    "path": (base_dir / f"{attack_type}_attack_info.txt").as_posix(),
                    "type": "text",
                    "icon": "📑",
                    "title": "View extra info",
                }
            )
        elif attack_type == "target_flipping":
                visuals.append(
                    {
                        "path": (base_dir / f"{attack_type}_samples.txt").as_posix(),
                        "type": "text",
                        "icon": "📝",
                        "title": "View text samples",
                    }
                )
                visuals.append(
                    {
                        "path": (base_dir / f"{attack_type}_scatter.png").as_posix(),
                        "type": "image",
                        "icon": "📊",
                        "title": "View scatter plot",
                    }
                )
        else:            
            visuals.append(
                {
                    "path": (base_dir / f"{attack_type}_visual.png").as_posix(),
                    "type": "image",
                    "icon": "🖼️",
                    "title": "View visual",
                }
            )

    return visuals


def generate_summary_json(
    output_dir: str, run_config: Optional[dict] = None, strategy_number: int = 0
) -> None:
    """Generate summary.json for attack snapshots.

    Args:
        output_dir: Base output directory
        run_config: Run configuration dict (optional)
        strategy_number: Strategy index (default: 0)

    Note:
        Creates summary.json file in the snapshots directory with metadata
        about all snapshots, grouped by round and client.
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
        full_summary["experiment"]["config_file"] = (
            f"strategy_config_{strategy_number}.json"
        )

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
    """Generate snapshot index HTML report.

    Creates an interactive HTML report showing all attack snapshots with
    filtering, navigation, and visual previews.

    Args:
        output_dir: Base output directory
        run_config: Run configuration dict (optional)
        strategy_number: Strategy index (default: 0)

    Note:
        Generates both summary.json and index.html files in snapshots directory.
    """
    snapshots_dir = _get_snapshots_dir_checked(output_dir, strategy_number)
    if not snapshots_dir:
        return

    snapshots = list_attack_snapshots(output_dir, strategy_number)
    snapshot_data = []

    for snapshot_path in snapshots:
        snapshot = load_attack_snapshot(str(snapshot_path))
        if snapshot:
            metadata = snapshot.get("metadata", snapshot)
            attack_config = metadata.get("attack_config", {})
            attack_type = metadata.get("attack_type", "unknown")
            client_id = metadata.get("client_id")
            round_num = metadata.get("round_num")

            base_dir = Path(f"client_{client_id}") / f"round_{round_num}"

            if isinstance(attack_config, list):
                attack_info_list = _split_composite_attack_info(
                    attack_type, attack_config
                )

                attack_types = [info["type"] for info in attack_info_list]
                all_params = []
                for info in attack_info_list:
                    if info["params"]:
                        all_params.extend(info["params"])

                visuals = _build_visuals_list(attack_types, base_dir)

                snapshot_data.append(
                    {
                        "client": client_id,
                        "round": round_num,
                        "attack_types": attack_types,
                        "is_stacked": True,
                        "samples": metadata.get("num_samples", 0),
                        "parameters": ", ".join(all_params) if all_params else "N/A",
                        "pickle_path": snapshot_path.relative_to(
                            snapshots_dir
                        ).as_posix(),
                        "visuals": visuals,
                        "metadata_path": (base_dir / f"{attack_type}_metadata.json").as_posix(),
                    }
                )
            else:

                attack_parameters = _extract_attack_params_for_display(
                    attack_type, attack_config
                )

                visuals = _build_visuals_list([attack_type], base_dir)

                snapshot_data.append(
                    {
                        "client": client_id,
                        "round": round_num,
                        "attack_types": [attack_type],
                        "is_stacked": False,
                        "samples": metadata.get("num_samples", 0),
                        "parameters": (
                            ", ".join(attack_parameters) if attack_parameters else "N/A"
                        ),
                        "pickle_path": snapshot_path.relative_to(
                            snapshots_dir
                        ).as_posix(),
                        "visuals": visuals,
                        "metadata_path": (base_dir / f"{attack_type}_metadata.json").as_posix(),
                    }
                )

    snapshot_data.sort(key=lambda x: (x["client"], x["round"]))

    html_content = _generate_index_html(snapshot_data, output_dir, run_config)

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
    """Generate snapshot index for HTML report."""
    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("snapshot_index.html.jinja")

    run_id = Path(output_dir).name
    total_clients = run_config.get("num_of_clients", "?") if run_config else "?"
    total_rounds = run_config.get("num_of_rounds", "?") if run_config else "?"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    unique_clients = sorted(set(s["client"] for s in snapshot_data))
    unique_rounds = sorted(set(s["round"] for s in snapshot_data))
    all_attack_types = []
    for s in snapshot_data:
        all_attack_types.extend(s["attack_types"])
    unique_attack_types = sorted(set(all_attack_types))

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


def generate_main_dashboard(output_dir: str) -> None:
    """Generate main dashboard for output directory.

    Creates a simple HTML dashboard that serves as an attack snapshot browser
    and quick navigation hub for all simulation outputs.

    Args:
        output_dir: Base output directory containing simulation outputs
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        logging.warning(f"Output directory does not exist: {output_dir}")
        return

    # Scan directory for outputs
    run_id = output_path.name
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Find attack snapshot directories
    attack_snapshot_dirs = sorted(output_path.glob("attack_snapshots_*"))
    snapshot_info = []
    for snap_dir in attack_snapshot_dirs:
        strategy_num = snap_dir.name.split("_")[-1]
        index_html = snap_dir / "index.html"
        summary_json = snap_dir / "summary.json"

        num_clients = 0
        num_rounds = 0
        total_snapshots = 0

        if summary_json.exists():
            try:
                with open(summary_json) as f:
                    summary = json.load(f)
                    num_clients = len(summary.get("attack_timeline", {}))
                    timeline = summary.get("attack_timeline", {})
                    for client_rounds in timeline.values():
                        num_rounds = max(num_rounds, len(client_rounds))
                    total_snapshots = summary.get("attack_summary", {}).get("total_snapshots", 0)
            except Exception:
                pass

        snapshot_info.append({
            "strategy_num": strategy_num,
            "dir_name": snap_dir.name,
            "has_index": index_html.exists(),
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "total_snapshots": total_snapshots,
        })

    # Find plots (organized by category)
    all_plots = sorted(output_path.glob("*.pdf"))
    plot_categories = {
        "Performance Metrics": [],
        "Attack Detection": [],
        "Client Analysis": [],
        "System Metrics": [],
    }

    for plot in all_plots:
        name = plot.name
        if any(x in name for x in ["accuracy", "loss", "average", 
                                   "r2",    # For air_quality_uci dataset 
                                   "mare"   # For smores dataset
                                   ]):
            plot_categories["Performance Metrics"].append(name)
        elif any(x in name for x in ["removal", "precision", "recall", "f1", "fp", "fn"]):
            plot_categories["Attack Detection"].append(name)
        elif any(x in name for x in ["distance", "criterion"]):
            plot_categories["Client Analysis"].append(name)
        elif any(x in name for x in ["time", "score_calculation"]):
            plot_categories["System Metrics"].append(name)
        else:
            plot_categories["Performance Metrics"].append(name)

    # Find CSV files
    csv_files = sorted(output_path.glob("csv/*.csv"))
    if not csv_files:
        csv_files = sorted(output_path.glob("*.csv"))

    # Find config files
    config_files = sorted(output_path.glob("strategy_config_*.json"))

    # Generate HTML
    html_content = _generate_main_dashboard_html(
        run_id=run_id,
        timestamp=timestamp,
        snapshot_info=snapshot_info,
        plot_categories=plot_categories,
        csv_files=[f.relative_to(output_path).as_posix() for f in csv_files],
        config_files=[f.relative_to(output_path).as_posix() for f in config_files],
    )

    index_path = output_path / "index.html"
    try:
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logging.info(f"Generated main dashboard: {index_path}")
    except Exception as e:
        logging.warning(f"Failed to generate main dashboard: {e}")


def _generate_main_dashboard_html(
    run_id: str,
    timestamp: str,
    snapshot_info: list,
    plot_categories: dict,
    csv_files: list,
    config_files: list,
) -> str:
    """Generate HTML content for main dashboard."""

    total_snapshots = sum(s["total_snapshots"] for s in snapshot_info)
    num_strategies = len(snapshot_info)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FL Simulation Results - {run_id}</title>
    <style>
        :root {{
            color-scheme: light;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            --bg: #f4f4f5;
            --panel: #ffffff;
            --panel-muted: #fafafa;
            --border: #e4e4e7;
            --border-soft: #f1f5f9;
            --text: #18181b;
            --text-muted: #71717a;
            --text-soft: #52525b;
            --accent: #18181b;
            --accent-hover: #27272a;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            min-height: 100vh;
            background: var(--bg);
            color: var(--text);
            padding: 24px;
            line-height: 1.6;
        }}
        a {{
            color: inherit;
        }}
        .page {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .shell {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 24px;
            box-shadow: 0 1px 2px rgba(24, 24, 27, 0.06);
            overflow: hidden;
        }}
        .header {{
            padding: 32px;
            background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
            border-bottom: 1px solid var(--border);
        }}
        .eyebrow {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 999px;
            background: var(--panel);
            border: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        .header h1 {{
            margin-top: 14px;
            font-size: 32px;
            line-height: 1.1;
            letter-spacing: -0.03em;
        }}
        .header .meta {{
            margin-top: 10px;
            font-size: 14px;
            color: var(--text-muted);
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            padding: 24px 32px;
            background: var(--panel-muted);
            border-bottom: 1px solid var(--border);
            flex-wrap: wrap;
        }}
        .stat {{
            min-width: 150px;
            padding: 18px;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 20px;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: 600;
            letter-spacing: -0.03em;
        }}
        .stat-label {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 6px;
        }}
        .section {{
            padding: 28px 32px 32px;
            border-bottom: 1px solid var(--border);
        }}
        .section:last-child {{ border-bottom: none; }}
        .section-title {{
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 18px;
            color: var(--text);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .section-subtitle {{
            margin-top: -8px;
            margin-bottom: 18px;
            font-size: 13px;
            color: var(--text-muted);
        }}
        .snapshot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 16px;
        }}
        .snapshot-card {{
            background: var(--panel-muted);
            padding: 20px;
            border-radius: 20px;
            border: 1px solid var(--border);
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        }}
        .snapshot-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(24, 24, 27, 0.06);
            border-color: #d4d4d8;
        }}
        .snapshot-card h3 {{
            font-size: 16px;
            margin-bottom: 8px;
            color: var(--text);
        }}
        .snapshot-card .info {{
            font-size: 13px;
            color: var(--text-muted);
            margin-bottom: 15px;
        }}
        .snapshot-card a {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 10px 14px;
            background: var(--accent);
            color: white;
            text-decoration: none;
            border-radius: 14px;
            font-size: 13px;
            font-weight: 500;
            transition: background 0.2s ease, transform 0.2s ease;
        }}
        .snapshot-card a:hover {{
            background: var(--accent-hover);
            transform: translateY(-1px);
        }}
        .file-list {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 12px;
        }}
        .file-item {{
            padding: 12px 14px;
            background: var(--panel-muted);
            border-radius: 16px;
            font-size: 13px;
            color: var(--text);
            border: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            transition: background 0.2s ease, border-color 0.2s ease;
        }}
        .file-item:hover {{
            background: #f4f4f5;
            border-color: #d4d4d8;
        }}
        .file-label {{
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 0;
        }}
        .file-name {{
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .file-item a {{
            color: var(--text);
            text-decoration: none;
            font-weight: 500;
            padding: 8px 12px;
            border-radius: 12px;
            border: 1px solid var(--border);
            background: var(--panel);
            white-space: nowrap;
        }}
        .file-item a:hover {{
            background: var(--panel-muted);
        }}
        .file-icon {{
            font-size: 16px;
        }}
        .category-section {{
            margin-bottom: 25px;
        }}
        .category-title {{
            font-size: 14px;
            font-weight: 600;
            color: var(--text-soft);
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        .empty-state {{
            padding: 24px;
            border: 1px dashed var(--border);
            border-radius: 20px;
            background: var(--panel-muted);
            color: var(--text-muted);
            font-size: 14px;
        }}
        @media (max-width: 900px) {{
            body {{
                padding: 16px;
            }}
            .header,
            .stats,
            .section {{
                padding-left: 18px;
                padding-right: 18px;
            }}
            .header h1 {{
                font-size: 28px;
            }}
        }}
    </style>
</head>
<body>
    <div class="page">
        <div class="shell">
        <div class="header">
            <div class="eyebrow">Experiment Export</div>
            <h1>FL Simulation Results</h1>
            <div class="meta">
                <strong>Run ID:</strong> {run_id} &nbsp;|&nbsp;
                <strong>Generated:</strong> {timestamp}
            </div>
        </div>

        <div class="stats">
            <div class="stat">
                <div class="stat-value">{num_strategies}</div>
                <div class="stat-label">{"Strategy" if num_strategies == 1 else "Strategies"}</div>
            </div>
            <div class="stat">
                <div class="stat-value">{total_snapshots}</div>
                <div class="stat-label">Attack Snapshots</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len([p for cat in plot_categories.values() for p in cat])}</div>
                <div class="stat-label">Plots Generated</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(csv_files)}</div>
                <div class="stat-label">Data Files</div>
            </div>
        </div>
"""

    # Attack Snapshots Section (primary focus)
    if snapshot_info:
        html += """
        <div class="section">
            <div class="section-title">Attack Snapshot Viewers</div>
            <div class="section-subtitle">Open per-strategy drill-down pages that mirror the main research UI styling.</div>
            <div class="snapshot-grid">
"""
        for snap in snapshot_info:
            strategy_label = f"Strategy {snap['strategy_num']}" if num_strategies > 1 else "Attack Snapshots"
            html += f"""
                <div class="snapshot-card">
                    <h3>{strategy_label}</h3>
                    <div class="info">
                        {snap['total_snapshots']} snapshots • {snap['num_clients']} clients • {snap['num_rounds']} rounds
                    </div>
                    <a href="{snap['dir_name']}/index.html">Open Snapshot Browser →</a>
                </div>
"""
        html += """
            </div>
        </div>
"""
    else:
        html += """
        <div class="section">
            <div class="section-title">Attack Snapshot Viewers</div>
            <div class="empty-state">No attack snapshot directories were generated for this experiment.</div>
        </div>
"""

    # Plots Section
    total_plots = sum(len(plots) for plots in plot_categories.values())
    if total_plots > 0:
        html += """
        <div class="section">
            <div class="section-title">Visualization Plots</div>
"""
        for category, plots in plot_categories.items():
            if plots:
                html += f"""
            <div class="category-section">
                <div class="category-title">{category}</div>
                <div class="file-list">
"""
                for plot in plots:
                    html += f"""
                    <div class="file-item">
                        <span class="file-label"><span class="file-icon">📊</span><span class="file-name">{plot}</span></span>
                        <a href="{plot}" target="_blank">View</a>
                    </div>
"""
                html += """
                </div>
            </div>
"""
        html += """
        </div>
"""
    else:
        html += """
        <div class="section">
            <div class="section-title">Visualization Plots</div>
            <div class="empty-state">No PDF plots were found in this output directory.</div>
        </div>
"""

    # Data Files Section
    if csv_files:
        html += """
        <div class="section">
            <div class="section-title">Data Exports (CSV)</div>
            <div class="file-list">
"""
        for csv_file in csv_files:
            filename = Path(csv_file).name
            html += f"""
                <div class="file-item">
                    <span class="file-label"><span class="file-icon">📄</span><span class="file-name">{filename}</span></span>
                    <a href="{csv_file}" download>Download</a>
                </div>
"""
        html += """
            </div>
        </div>
"""

    # Config Files Section
    if config_files:
        html += """
        <div class="section">
            <div class="section-title">Strategy Configurations</div>
            <div class="file-list">
"""
        for config_file in config_files:
            filename = Path(config_file).name
            html += f"""
                <div class="file-item">
                    <span class="file-label"><span class="file-icon">📋</span><span class="file-name">{filename}</span></span>
                    <a href="{config_file}" target="_blank">View</a>
                </div>
"""
        html += """
            </div>
        </div>
"""

    html += """
        </div>
    </div>
</body>
</html>
"""

    return html
