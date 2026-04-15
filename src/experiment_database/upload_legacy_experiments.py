"""
One-off upload script for a set of strategy_config_{n}.json,
csv/round_metrics_{n}.csv, and csv/per_client_metrics_{n}.csv files.

Prompts for full name and email before importing.
Posts data directly to the online HTTP server.

Payload format matches SimulationStrategyHistory.to_dict() so the server
can handle it identically to live experiment uploads.
"""

import csv
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from src.utils.common import obtain_env_var

import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Script lives at src/server/upload_legacy_experiments.py — parents[2] resolves to project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_URL = "http://localhost:8080"
ROUND_COL_OPTIONS = {"round #", "round"}
SKIP_COLS = ROUND_COL_OPTIONS
PARTICIPATION_METRIC = "aggregation_participation_history"
CLIENT_COL_RE = re.compile(r"^client_(\d+)_(.+)$")

db_mode = obtain_env_var("EXPERIMENT_DB_MODE")
if db_mode == "server":
    BASE_URL = obtain_env_var("EXPERIMENT_SERVER_URL")
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FOLDER_DATETIME_FORMATS = [
    "%m-%d-%Y_%H-%M-%S",
    "%m-%d-%Y",
    "%Y-%m-%d_%H-%M-%S",
    "%Y-%m-%d",
]


def parse_folder_datetime(folder_name: str) -> datetime | None:
    for fmt in FOLDER_DATETIME_FORMATS:
        try:
            naive_dt = datetime.strptime(folder_name, fmt)
            return naive_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    log.warning(f"  Could not parse datetime from folder name: {folder_name!r}")
    return None


def prompt_user_info() -> tuple[str, str, str]:
    load_dotenv(PROJECT_ROOT / ".env.private")
    token = os.getenv("EXPERIMENT_SERVER_TOKEN")
    if not token:
        raise RuntimeError(
            "EXPERIMENT_SERVER_TOKEN not found in .env.private, please make sure to run generate_db_token"
        )
    full_name = input("Full name: ").strip().upper()
    email = input("Email: ").strip().upper()
    return full_name, email, token


def resolve_malicious_ids(cfg: dict) -> tuple[set[int], int]:
    """
    Returns (malicious_ids, num_malicious).

    If no specific client IDs are provided in attack_schedule or
    specific_malicious_clients, assumes malicious clients start from 0
    up to num_of_malicious_clients.
    """
    malicious_ids = set()
    for attack in cfg.get("attack_schedule", []):
        malicious_ids.update(attack.get("malicious_client_ids", []))

    num_malicious = (
        len(malicious_ids) if malicious_ids else cfg.get("num_of_malicious_clients", 0)
    )

    # No specific IDs given — assume clients 0..num_malicious-1
    if not malicious_ids and num_malicious > 0:
        malicious_ids = set(range(num_malicious))

    return malicious_ids, num_malicious


def get_attack_type_for_client(
    cfg: dict, client_number: int, round_number: int
) -> str | None:
    """Return attack type from attack_schedule if present, else flat attack_type field."""
    for attack in cfg.get("attack_schedule", []):
        in_client_list = client_number in attack.get("malicious_client_ids", [])
        in_round_range = (
            attack.get("start_round", 1)
            <= round_number
            <= attack.get("end_round", float("inf"))
        )
        if in_client_list and in_round_range:
            return attack.get("attack_type")

    # Fallback: flat attack_type field with no schedule.
    # If no specific IDs given, assume malicious clients start from 0.
    attack_type = cfg.get("attack_type")
    if attack_type:
        num_malicious = cfg.get("num_of_malicious_clients", 0)
        assumed_malicious = set(range(num_malicious))
        if client_number in assumed_malicious:
            return attack_type

    return None


# CSV parsers
def get_round_col(row: dict) -> str:
    for col in ROUND_COL_OPTIONS:
        if col in row:
            return col
    raise KeyError(
        f"No round column found in row. Available columns: {list(row.keys())}"
    )


def parse_round_metrics(csv_path: Path) -> list[dict]:
    """Return list of {metric_name, round_number, value} dicts."""
    entries = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            round_number = int(row[get_round_col(row)])
            for col, raw in row.items():
                if col in SKIP_COLS:
                    continue
                try:
                    value = float(raw)
                except (ValueError, TypeError):
                    continue
                entries.append(
                    {
                        "metric_name": col,
                        "metric_type": "round_global",
                        "round_number": round_number,
                        "value": value,
                        "client_number": None,
                    }
                )
    return entries


def parse_per_client_metrics(
    csv_path: Path, cfg: dict
) -> tuple[list[dict], list[dict]]:
    """
    Returns:
        metric_entries: list of {metric_name, metric_type, round_number, value, client_number}
        clients_at_round: list of {round_number, client_number, is_aggregated, attack_type}
    """
    metric_entries = []
    clients_at_round = []
    num_clients = cfg["num_of_clients"]

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            round_number = int(row[get_round_col(row)])

            # Collect participation flags: {client_number: bool}
            participation = {}
            for col, raw in row.items():
                if col in SKIP_COLS:
                    continue
                match = CLIENT_COL_RE.match(col)
                if not match or match.group(2) != PARTICIPATION_METRIC:
                    continue
                try:
                    participation[int(match.group(1))] = bool(int(float(raw)))
                except (ValueError, TypeError):
                    pass

            # ClientAtRound rows
            for client_number in range(num_clients):
                clients_at_round.append(
                    {
                        "round_number": round_number,
                        "client_number": client_number,
                        "is_aggregated": participation.get(client_number, True),
                        "attack_type": get_attack_type_for_client(
                            cfg, client_number, round_number
                        ),
                    }
                )

            # MetricEntry rows
            for col, raw in row.items():
                if col in SKIP_COLS:
                    continue
                match = CLIENT_COL_RE.match(col)
                if not match:
                    continue
                metric_name = match.group(2)
                if metric_name == PARTICIPATION_METRIC:
                    continue
                client_number = int(match.group(1))
                try:
                    value = float(raw)
                except (ValueError, TypeError):
                    continue
                metric_entries.append(
                    {
                        "metric_name": metric_name,
                        "metric_type": "client_round",
                        "round_number": round_number,
                        "value": value,
                        "client_number": client_number,
                    }
                )

    return metric_entries, clients_at_round


# Build payload
# Round metric column names as they appear in round_metrics_*.csv
ROUND_METRIC_COLS = {
    "aggregated_loss_history": "aggregated_loss_history",
    "score_calculation_time_nanos_history": "score_calculation_time_nanos_history",
    "removal_threshold_history": "removal_threshold_history",
    "average_accuracy_history": "average_accuracy_history",
    "tp_history": "tp_history",
    "tn_history": "tn_history",
    "fp_history": "fp_history",
    "fn_history": "fn_history",
}


def build_rounds_history(
    round_metric_lookup: dict, num_rounds: int, csv_cols: set
) -> dict:
    """
    Build a rounds_history dict matching RoundsInfo.to_dict() exactly.
    Columns present in the CSV are filled from data; all others default to
    a list of 0.0 so the server never receives None for a numeric field.
    """

    def round_series(col_name: str) -> list:
        if col_name in csv_cols:
            return [
                round_metric_lookup.get((r, col_name), 0.0)
                for r in range(1, num_rounds + 1)
            ]
        return [0.0] * num_rounds

    return {
        "score_calculation_time_nanos_history": round_series(
            "score_calculation_time_nanos_history"
        ),
        "removal_threshold_history": round_series("removal_threshold_history"),
        "aggregated_loss_history": round_series("aggregated_loss_history"),
        "average_accuracy_history": round_series("average_accuracy_history"),
        "average_accuracy_std_history": round_series("average_accuracy_std_history"),
        "tp_history": round_series("tp_history"),
        "tn_history": round_series("tn_history"),
        "fp_history": round_series("fp_history"),
        "fn_history": round_series("fn_history"),
        "total_fp_and_fn_history": round_series("total_fp_and_fn_history"),
        "removal_accuracy_history": round_series("removal_accuracy_history"),
        "removal_precision_history": round_series("removal_precision_history"),
        "removal_recall_history": round_series("removal_recall_history"),
        "removal_f1_history": round_series("removal_f1_history"),
        "mention_precision_history": round_series("mention_precision_history"),
        "mention_recall_history": round_series("mention_recall_history"),
        "mention_f1_history": round_series("mention_f1_history"),
        "document_precision_history": round_series("document_precision_history"),
        "document_recall_history": round_series("document_recall_history"),
        "document_f1_history": round_series("document_f1_history"),
    }


def build_simulation_history_dict(
    cfg: dict,
    round_metrics: list[dict],
    client_metrics: list[dict],
    clients_at_round: list[dict],
) -> dict:
    """
    Build a dict matching SimulationStrategyHistory.to_dict() format so the
    server can handle it identically to a live experiment upload.
    """
    malicious_ids, num_malicious = resolve_malicious_ids(cfg)

    num_clients = cfg["num_of_clients"]
    num_rounds = cfg["num_of_rounds"]

    # Index client metrics by (client_number, round_number, metric_name)
    client_metric_lookup: dict[tuple, float] = {}
    for entry in client_metrics:
        key = (entry["client_number"], entry["round_number"], entry["metric_name"])
        client_metric_lookup[key] = entry["value"]

    # Index clients_at_round by (client_number, round_number)
    participation_lookup: dict[tuple, dict] = {}
    for entry in clients_at_round:
        key = (entry["client_number"], entry["round_number"])
        participation_lookup[key] = entry

    # --- Per-client histories ---
    clients = []
    for client_id in range(num_clients):
        loss_history = []
        accuracy_history = []
        removal_criterion_history = []
        absolute_distance_history = []
        aggregation_participation_history = []

        for round_num in range(num_rounds):
            r = round_num + 1
            car = participation_lookup.get((client_id, r), {})

            loss_history.append(
                client_metric_lookup.get((client_id, r, "loss_history"))
            )
            accuracy_history.append(
                client_metric_lookup.get((client_id, r, "accuracy_history"))
            )
            removal_criterion_history.append(
                client_metric_lookup.get((client_id, r, "removal_criterion_history"))
            )
            absolute_distance_history.append(
                client_metric_lookup.get((client_id, r, "absolute_distance_history"))
            )
            aggregation_participation_history.append(
                1 if car.get("is_aggregated", True) else 0
            )

        # Build attack_plan_keywords per round, matching what ClientInfo.__post_init__ produces
        attack_plan_keywords = [[] for _ in range(num_rounds)]
        legacy_attack_type = cfg.get("attack_type")
        if legacy_attack_type:
            attack_plan_keywords = [[legacy_attack_type]] * num_rounds
        else:
            for attack in cfg.get("attack_schedule", []):
                if client_id not in attack.get("malicious_client_ids", []):
                    continue
                attack_type = attack["attack_type"]
                start_round = attack["start_round"]
                end_round = attack["end_round"]
                for idx, r in enumerate(range(1, num_rounds + 1)):
                    if start_round <= r <= end_round:
                        attack_plan_keywords[idx] = list(attack_plan_keywords[idx]) + [
                            attack_type
                        ]

        clients.append(
            {
                "client_id": client_id,
                "num_of_rounds": num_rounds,
                "is_malicious": client_id in malicious_ids,
                "attack_plan_keywords": attack_plan_keywords,
                "attack_plan_configs": [
                    attack
                    for attack in cfg.get("attack_schedule", [])
                    if client_id in attack.get("malicious_client_ids", [])
                ],
                "loss_history": loss_history,
                "accuracy_history": accuracy_history,
                "removal_criterion_history": removal_criterion_history,
                "absolute_distance_history": absolute_distance_history,
                "aggregation_participation_history": aggregation_participation_history,
            }
        )

    # --- Per-round history ---
    round_metric_lookup: dict[tuple, float] = {}
    for entry in round_metrics:
        key = (entry["round_number"], entry["metric_name"])
        round_metric_lookup[key] = entry["value"]

    def round_series(col_name: str) -> list:
        return [
            round_metric_lookup.get((r, col_name), 0.0)
            for r in range(1, num_rounds + 1)
        ]

    # Collect which columns are actually present in the CSV data
    csv_round_cols = {col for (_, col) in round_metric_lookup.keys()}
    rounds_history = build_rounds_history(
        round_metric_lookup, num_rounds, csv_round_cols
    )

    return {
        "strategy_config": cfg,
        "total_malicious_clients": num_malicious,
        "malicious_clients_ids": list(malicious_ids),
        "rounds_history": rounds_history,
        "clients": clients,
        "training_device_type": "cpu",
        "training_device_name": "cpu",
        "training_device_ram": None,
        "training_device_arch": None,
        "host_cpu": None,
        "host_ram": None,
    }


# Upload


def upload_strategy(
    history_dict: dict, full_name: str, email: str, token: str, created_at: datetime
) -> None:
    payload = {
        "strategy_history": history_dict,
        "total_exec_time": None,
        "cpu": history_dict.get("host_cpu"),
        "arch": history_dict.get("training_device_arch"),
        "ram": history_dict.get("host_ram"),
        "gpu_name": history_dict.get("training_device_name"),
        "gpu_ram": history_dict.get("training_device_ram"),
        "is_executed_on": history_dict.get("training_device_type"),
        "full_author_name": full_name,
        "author_email": email,
        "created_at": created_at.isoformat(),
    }
    response = requests.post(
        f"{BASE_URL}/experiments",
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        timeout=30,
    )
    if response.status_code in (200, 201):
        log.info(f"    Uploaded successfully — {response.json()}")
    else:
        log.error(f"    Upload failed: {response.status_code} — {response.text}")
        response.raise_for_status()


# Entry point


def generate_clients_at_round_from_config(cfg: dict) -> list[dict]:
    """Generate ClientAtRound entries from config when no CSV data is available."""
    num_rounds = cfg.get("num_of_rounds", 0)
    num_clients = cfg.get("num_of_clients", 0)
    clients_at_round = []
    for round_number in range(1, num_rounds + 1):
        for client_number in range(num_clients):
            attack_type = get_attack_type_for_client(cfg, client_number, round_number)
            clients_at_round.append(
                {
                    "round_number": round_number,
                    "client_number": client_number,
                    "is_aggregated": True,
                    "attack_type": attack_type,
                }
            )
    return clients_at_round


def upload_folder(
    folder: Path,
    full_name: str,
    email: str,
    created_at: datetime,
    token: str,
) -> None:
    strategy_configs = sorted(folder.glob("strategy_config_*.json"))
    output_log = folder / "output.log"

    if not strategy_configs:
        if output_log.exists():
            log.warning(
                f"  Found output.log but no strategy_config_*.json in {folder.name} — "
                f"experiment was unsuccessful, skipping upload."
            )
        else:
            log.warning(
                f"  No strategy_config_*.json files found in {folder.name} — skipping."
            )
        return

    log.info(f"  Found {len(strategy_configs)} strategy config(s).")

    for config_path in strategy_configs:
        match = re.search(r"_(\d+)\.json$", config_path.name)
        if not match:
            continue
        n = int(match.group(1))
        log.info(f"    Processing strategy {n}...")

        cfg = json.loads(config_path.read_text())

        csv_dir = folder / "csv"
        round_csv = csv_dir / f"round_metrics_{n}.csv"
        per_client_csv = csv_dir / f"per_client_metrics_{n}.csv"
        has_csvs = round_csv.exists() and per_client_csv.exists()

        if not has_csvs:
            log.warning(f"    CSVs missing for strategy {n} — marking as unsuccessful.")
            round_metrics, client_metrics = [], []
            clients_at_round = generate_clients_at_round_from_config(cfg)
        else:
            round_metrics = parse_round_metrics(round_csv)
            client_metrics, clients_at_round = parse_per_client_metrics(
                per_client_csv, cfg
            )

        history_dict = build_simulation_history_dict(
            cfg=cfg,
            round_metrics=round_metrics,
            client_metrics=client_metrics,
            clients_at_round=clients_at_round,
        )

        upload_strategy(history_dict, full_name, email, token, created_at)


def run_upload(files_dir: Path) -> None:
    print("Please enter your details before uploading.\n")
    full_name, email, token = prompt_user_info()

    folders = sorted([f for f in files_dir.iterdir() if f.is_dir()])
    if not folders:
        log.error(f"No subfolders found in {files_dir}")
        return

    log.info(f"Found {len(folders)} folder(s).")
    total_ok = 0
    total_skip = 0

    for folder in folders:
        log.info(f"Processing: {folder.name}")
        created_at = parse_folder_datetime(folder.name)
        if not created_at:
            log.warning(f"  Could not parse datetime from {folder.name!r} — skipping.")
            total_skip += 1
            continue
        try:
            upload_folder(folder, full_name, email, created_at, token)
            total_ok += 1
        except Exception:
            log.exception(f"  Failed to upload {folder.name}.")
            total_skip += 1

    log.info(f"Done. {total_ok} uploaded, {total_skip} skipped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload experiment CSVs and configs to the remote HTTP server."
    )
    parser.add_argument(
        "--files-dir",
        type=Path,
        default=PROJECT_ROOT / "out",
        help="Directory containing dated subfolders with strategy_config_*.json and csv/*.csv files",
    )
    args = parser.parse_args()
    run_upload(args.files_dir)
