from sqlalchemy.orm import Session

from src.data_models.round_info import RoundsInfo
from src.experiment_database.models import (
    ClientAtRound,
    ExecutedConfig,
    ExperimentClient,
    ExperimentEvent,
    HostHardwareSpec,
    MetricEntry,
    MetricInfo,
)


def create_executed_config(
    session: Session,
    strategy_config,
    num_of_malicious_clients,
) -> ExecutedConfig:
    """Create an ExecutedConfig row holding the base FL experiment configuration."""
    cfg = ExecutedConfig(
        json=strategy_config.to_json(),
        aggregation_strategy_keyword=strategy_config.aggregation_strategy_keyword,
        dataset_keyword=strategy_config.dataset_keyword,
        num_rounds=strategy_config.num_of_rounds,
        num_clients=strategy_config.num_of_clients,
        num_malicious_clients=num_of_malicious_clients,
    )
    session.add(cfg)
    session.flush()
    return cfg


def get_or_create_hardware(
    session: Session,
    cpu: str,
    arch: str,
    ram: str,
    gpu_name: str,
    is_executed_on: str,
    gpu_ram: str,
) -> HostHardwareSpec:
    """Return an existing HostHardwareSpec row or create one if it does not exist."""
    hw = session.query(HostHardwareSpec).filter_by(
        cpu_name=cpu,
        cpu_arch=arch,
        ram_amount=ram,
        gpu_name=gpu_name if is_executed_on == "gpu" else None,
        gpu_ram=gpu_ram if is_executed_on == "gpu" else None,
        is_executed_on=is_executed_on,
    ).one_or_none()

    if hw:
        return hw

    hw = HostHardwareSpec(
        cpu_name=cpu,
        cpu_arch=arch,
        ram_amount=ram,
        gpu_name=gpu_name if is_executed_on == "gpu" else None,
        gpu_ram=gpu_ram if is_executed_on == "gpu" else None,
        is_executed_on=is_executed_on,
    )
    session.add(hw)
    session.flush()
    return hw


def create_experiment_event(
    session: Session,
    executed_config: ExecutedConfig,
    hardware: HostHardwareSpec | None = None,
    wandb_url=None,
    username=None,
) -> ExperimentEvent:
    """Create an ExperimentEvent row linking a config and hardware spec to a single run."""
    event = ExperimentEvent(
        executed_config_id=executed_config.id,
        hardware_id=hardware.id if hardware else None,
        is_successful=False,
        wandb_url=wandb_url,
        username=username,
    )
    session.add(event)
    session.flush()
    return event


def create_clients(
    session: Session,
    experiment_event: ExperimentEvent,
    num_clients: int,
    malicious_ids: set[int],
):
    """Create ExperimentClient rows for all clients in the experiment.

    Returns a tuple of (clients list, mapping of client_number to database ID).
    """
    clients = []

    for i in range(num_clients):
        client = ExperimentClient(
            experiment_event_id=experiment_event.id,
            client_number=i,
            was_ever_malicious=i in malicious_ids,
        )
        session.add(client)
        clients.append(client)

    session.flush()

    experiment_client_id_map = {c.client_number: c.id for c in clients}
    return clients, experiment_client_id_map


def load_metric_info_map(session: Session) -> dict[str, MetricInfo]:
    """Return a mapping of metric name to MetricInfo for all rows in the database."""
    return {mi.name: mi for mi in session.query(MetricInfo).all()}


def persist_round_metrics(
    session: Session,
    experiment_event: ExperimentEvent,
    rounds_info: RoundsInfo,
) -> None:
    """Persist global per-round metrics from RoundsInfo into MetricEntry rows."""
    metric_info_map = load_metric_info_map(session)

    for metric_name in rounds_info.savable_metrics:
        if metric_name not in metric_info_map:
            mi = MetricInfo(name=metric_name, type="round_global")
            session.add(mi)
            session.flush()
            metric_info_map[metric_name] = mi

        values = getattr(rounds_info, metric_name)

        for round_number, value in enumerate(values, start=1):
            session.add(
                MetricEntry(
                    experiment_event_id=experiment_event.id,
                    metric_info_id=metric_info_map[metric_name].id,
                    round_number=round_number,
                    value=float(value),
                )
            )

    session.commit()


def persist_client_round_metrics(
    session: Session,
    experiment_event: ExperimentEvent,
    clients_info,
    experiment_client_id_map,
) -> None:
    """Persist per-client per-round metrics into MetricEntry rows."""
    metric_info_map = load_metric_info_map(session)

    for client in clients_info:
        experiment_client_db_id = experiment_client_id_map[client.client_id]

        for metric_name in client.savable_metrics:
            if metric_name not in metric_info_map:
                mi = MetricInfo(name=metric_name, type="client_round")
                session.add(mi)
                session.flush()
                metric_info_map[metric_name] = mi

            values = getattr(client, metric_name)

            for round_number, value in enumerate(values, start=1):
                session.add(
                    MetricEntry(
                        experiment_event_id=experiment_event.id,
                        client_id=experiment_client_db_id,
                        metric_info_id=metric_info_map[metric_name].id,
                        round_number=round_number,
                        value=float(value) if value is not None else 0.0,
                    )
                )

    session.flush()


def persist_clients_at_round(
    session: Session,
    experiment_event: ExperimentEvent,
    clients_info: list,
    experiment_client_id_map: dict[int, str],
) -> None:
    """Persist ClientAtRound rows for every client across all rounds.

    Stores participation and attack metadata only. Metrics are written
    separately via persist_client_round_metrics.

    Raises:
        RuntimeError: If a simulation client_id is missing from experiment_client_id_map.
    """
    for client in clients_info:
        if client.client_id not in experiment_client_id_map:
            raise RuntimeError(
                f"Simulation client_id {client.client_id} missing from "
                "experiment_client_id_map"
            )

        experiment_client_db_id = experiment_client_id_map[client.client_id]

        for r, round_number in enumerate(client.rounds):
            participated = (
                client.aggregation_participation_history[r]
                if r < len(client.aggregation_participation_history)
                else 0
            )

            attack_keywords = None
            if r < len(client.attack_plan_keywords):
                keywords = client.attack_plan_keywords[r]
                attack_keywords = ",".join(keywords) if keywords else None

            if attack_keywords is None:
                session.add(
                    ClientAtRound(
                        experiment_event_id=experiment_event.id,
                        experiment_client_id=experiment_client_db_id,
                        round_number=round_number,
                        client_is_aggregated=participated,
                        attack_type=attack_keywords,
                    )
                )
            else:
                for key in keywords:
                    session.add(
                        ClientAtRound(
                            experiment_event_id=experiment_event.id,
                            experiment_client_id=experiment_client_db_id,
                            round_number=round_number,
                            client_is_aggregated=participated,
                            attack_type=key,
                        )
                    )

    session.flush()