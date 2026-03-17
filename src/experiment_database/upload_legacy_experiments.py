"""
One-off upload script for a set of strategy_config_{n}.json,
csv/round_metrics_{n}.csv, and csv/per_client_metrics_{n}.csv files.

Prompts for full name and email before importing.
Posts data directly to the online HTTP server.
"""

import csv
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Script lives at src/server/upload_direct.py — parents[2] resolves to project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_URL = "http://localhost:8080"
ROUND_COL_OPTIONS = {"round #", "round"}
SKIP_COLS = ROUND_COL_OPTIONS
PARTICIPATION_METRIC = "aggregation_participation_history"
CLIENT_COL_RE = re.compile(r"^client_(\d+)_(.+)$")


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

    # Fallback: flat attack_type field (no schedule, assume all rounds)
    return cfg.get("attack_type")


# ---------------------------------------------------------------------------
# CSV parsers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Build payload
# ---------------------------------------------------------------------------


def build_strategy_payload(
    cfg: dict,
    round_metrics: list[dict],
    client_metrics: list[dict],
    clients_at_round: list[dict],
    full_name: str,
    email: str,
    created_at: datetime,
    is_successful: bool,
) -> dict:
    malicious_ids = set()
    for attack in cfg.get("attack_schedule", []):
        malicious_ids.update(attack.get("malicious_client_ids", []))

    # Fallback for flat num_of_malicious_clients field
    num_malicious = (
        len(malicious_ids) if malicious_ids else cfg.get("num_of_malicious_clients", 0)
    )

    return {
        "full_author_name": full_name,
        "author_email": email,
        "created_at": created_at.isoformat(),
        "is_successful": is_successful,
        "executed_config": {
            "aggregation_strategy_keyword": cfg["aggregation_strategy_keyword"],
            "dataset_keyword": cfg["dataset_keyword"],
            "num_rounds": str(cfg["num_of_rounds"]),
            "num_clients": str(cfg["num_of_clients"]),
            "num_malicious_clients": str(num_malicious),
            "json": cfg,
        },
        "clients": [
            {
                "client_number": i,
                "was_ever_malicious": i in malicious_ids,
            }
            for i in range(cfg["num_of_clients"])
        ],
        "clients_at_round": clients_at_round,
        "metric_entries": round_metrics + client_metrics,
    }


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def upload_strategy(payload: dict, token: str) -> None:
    response = requests.post(
        f"{BASE_URL}/experiments/import",
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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
    if not strategy_configs:
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
            # Still generate ClientAtRound from config so we have client records
            clients_at_round = generate_clients_at_round_from_config(cfg)
        else:
            round_metrics = parse_round_metrics(round_csv)
            client_metrics, clients_at_round = parse_per_client_metrics(
                per_client_csv, cfg
            )

        payload = build_strategy_payload(
            cfg=cfg,
            round_metrics=round_metrics,
            client_metrics=client_metrics,
            clients_at_round=clients_at_round,
            full_name=full_name,
            email=email,
            created_at=created_at,
            is_successful=has_csvs,
        )

        upload_strategy(payload, token)


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
        description="Upload experiment CSVs and configs to the local HTTP server."
    )
    parser.add_argument(
        "--files-dir",
        type=Path,
        default=PROJECT_ROOT / "out",
        help="Directory containing strategy_config_*.json and csv_*.csv files",
    )
    args = parser.parse_args()
    run_upload(args.files_dir)
