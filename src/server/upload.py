import csv
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.experiment_database.models import (
    ClientAtRound,
    ExecutedConfig,
    ExperimentClient,
    ExperimentEvent,
    MetricEntry,
    MetricInfo,
)
from src.experiment_database.session import LocalDatabaseSession

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("out")
FOLDER_DATETIME_FORMAT = "%m-%d-%Y_%H-%M-%S"
SKIP_COLS = {"round"}
PARTICIPATION_METRIC = "aggregation_participation_history"
CLIENT_COL_RE = re.compile(r"^client_(\d+)_(.+)$")

def get_or_create_metric_info(session, name: str, type_: str) -> MetricInfo:
    info = session.query(MetricInfo).filter_by(name=name, type=type_).first()
    if not info:
        info = MetricInfo(name=name, type=type_)
        session.add(info)
        session.flush()
    return info


def parse_folder_datetime(folder_name: str) -> datetime | None:
    try:
        naive_dt = datetime.strptime(folder_name, FOLDER_DATETIME_FORMAT)
        return naive_dt.replace(tzinfo=timezone.utc)
    except ValueError:
        log.warning(f"  Could not parse datetime from folder name: {folder_name!r}")
        return None


def get_attack_type_for_client(
    cfg: dict, client_number: int, round_number: int
) -> str | None:
    """Return the attack type for a given client at a given round, or None."""
    for attack in cfg.get("attack_schedule", []):
        in_client_list = client_number in attack.get("malicious_client_ids", [])
        in_round_range = (
            attack.get("start_round", 1)
            <= round_number
            <= attack.get("end_round", float("inf"))
        )
        if in_client_list and in_round_range:
            return attack.get("attack_type")
    return None


def create_executed_config(session, cfg: dict) -> ExecutedConfig:
    malicious_ids = set()
    for attack in cfg.get("attack_schedule", []):
        malicious_ids.update(attack.get("malicious_client_ids", []))

    executed_config = ExecutedConfig(
        id=str(uuid.uuid4()),
        json=cfg,
        aggregation_strategy_keyword=cfg["aggregation_strategy_keyword"],
        dataset_keyword=cfg["dataset_keyword"],
        num_rounds=str(cfg["num_of_rounds"]),
        num_clients=str(cfg["num_of_clients"]),
        num_malicious_clients=str(len(malicious_ids)),
    )
    session.add(executed_config)
    session.flush()
    return executed_config


def create_experiment_event(
    session,
    executed_config: ExecutedConfig,
    created_at: datetime,
    username: str,
) -> ExperimentEvent:
    event = ExperimentEvent(
        created_at=created_at,
        username=username,
        is_successful=True,
        executed_config_id=executed_config.id,
        hardware_id=None,
        total_exec_time=None,
    )
    session.add(event)
    session.flush()
    return event


def create_experiment_clients(
    session, event: ExperimentEvent, cfg: dict
) -> dict:
    """Create ExperimentClient rows and return a client_number -> client map."""
    malicious_ids = set()
    for attack in cfg.get("attack_schedule", []):
        malicious_ids.update(attack.get("malicious_client_ids", []))

    client_map = {}
    for i in range(cfg["num_of_clients"]):
        client = ExperimentClient(
            id=str(uuid.uuid4()),
            client_number=i,
            was_ever_malicious=(i in malicious_ids),
            experiment_event_id=event.id,
        )
        session.add(client)
        client_map[i] = client

    session.flush()
    return client_map

def import_round_metrics(session, csv_path: Path, event_id: int) -> None:
    count = 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            round_number = int(row["round"])
            for col, raw in row.items():
                if col in SKIP_COLS:
                    continue
                try:
                    value = float(raw)
                except (ValueError, TypeError):
                    continue
                info = get_or_create_metric_info(
                    session, name=col, type_="round_global"
                )
                session.add(MetricEntry(
                    id=str(uuid.uuid4()),
                    metric_info_id=info.id,
                    experiment_event_id=event_id,
                    client_id=None,
                    value=value,
                    round_number=round_number,
                ))
                count += 1
    log.info(f"      round_metrics: {count} entries")


def import_per_client_metrics(
    session, csv_path: Path, event_id: int, client_map: dict, cfg: dict
) -> None:
    metric_count = 0
    client_at_round_count = 0

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            round_number = int(row["round"])

            # Collect participation flags for this round: {client_number: bool}
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

            # Insert ClientAtRound rows for this round
            for client_number, client in client_map.items():
                is_aggregated = participation.get(client_number, True)
                attack_type = get_attack_type_for_client(
                    cfg, client_number, round_number
                )
                session.add(ClientAtRound(
                    experiment_event_id=event_id,
                    experiment_client_id=client.id,
                    round_number=round_number,
                    client_is_aggregated=is_aggregated,
                    attack_type=attack_type,
                ))
                client_at_round_count += 1

            # Insert MetricEntry rows for all non-participation metrics
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
                client = client_map.get(client_number)
                info = get_or_create_metric_info(
                    session, name=metric_name, type_="client_round"
                )
                session.add(MetricEntry(
                    id=str(uuid.uuid4()),
                    metric_info_id=info.id,
                    experiment_event_id=event_id,
                    client_id=client.id if client else None,
                    value=value,
                    round_number=round_number,
                ))
                metric_count += 1

    log.info(
        f"      per_client_metrics: {metric_count} MetricEntry, "
        f"{client_at_round_count} ClientAtRound"
    )

def import_strategy(
    session,
    folder: Path,
    strategy_num: int,
    created_at: datetime,
    username: str,
) -> None:
    config_path = folder / f"strategy_config_{strategy_num}.json"
    if not config_path.exists():
        log.warning(f"    strategy_config_{strategy_num}.json not found — skipping.")
        return

    cfg = json.loads(config_path.read_text())
    log.info(f"    Creating DB records for strategy {strategy_num}...")

    executed_config = create_executed_config(session, cfg)
    event = create_experiment_event(session, executed_config, created_at, username)
    client_map = create_experiment_clients(session, event, cfg)

    log.info(f"    ExperimentEvent id={event.id}, {len(client_map)} clients created.")

    csv_dir = folder / "csv"
    has_csv_files = csv_dir.exists() and any(csv_dir.glob("*.csv"))

    if not has_csv_files:
        log.warning(f"    No CSV files found for strategy {strategy_num} — marking as unsuccessful.")
        event.is_successful = False
        return

    round_csv = csv_dir / f"round_metrics_{strategy_num}.csv"
    if round_csv.exists():
        import_round_metrics(session, round_csv, event.id)

    client_csv = csv_dir / f"per_client_metrics_{strategy_num}.csv"
    if client_csv.exists():
        import_per_client_metrics(session, client_csv, event.id, client_map, cfg)


def import_experiment_folder(session, folder: Path, username: str) -> None:
    created_at = parse_folder_datetime(folder.name)
    if not created_at:
        return

    strategy_configs = sorted(folder.glob("strategy_config_*.json"))
    if not strategy_configs:
        log.warning(f"  No strategy_config_*.json files found — skipping.")
        return

    log.info(f"  Found {len(strategy_configs)} strategy config(s).")
    for config_path in strategy_configs:
        match = re.search(r"_(\d+)\.json$", config_path.name)
        if not match:
            continue
        strategy_num = int(match.group(1))
        log.info(f"  Processing strategy {strategy_num}...")
        import_strategy(session, folder, strategy_num, created_at, username)


def run_bulk_import(out_dir: Path = OUT_DIR, username: str = None) -> None:
    if username is None:
        username = (
            input("Enter username to associate with all imported experiments: ").strip()
            or None
        )

    folders = sorted([f for f in out_dir.iterdir() if f.is_dir()])
    log.info(f"Found {len(folders)} experiment folder(s) in {out_dir}")

    total_ok = 0
    total_skip = 0

    for folder in folders:
        log.info(f"Processing: {folder.name}")
        session = LocalDatabaseSession()
        try:
            import_experiment_folder(session, folder, username)
            session.commit()
            total_ok += 1
        except Exception:
            session.rollback()
            log.exception(f"  Failed to import {folder.name} — rolled back.")
            total_skip += 1
        finally:
            session.close()

    log.info(f"Done. {total_ok} imported, {total_skip} skipped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Bulk import experiment output folders into the DB."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Path to the out/ directory",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="Username to tag all imported experiments with",
    )
    args = parser.parse_args()
    run_bulk_import(args.out_dir, args.username)