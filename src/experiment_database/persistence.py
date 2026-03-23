from sqlalchemy.orm import Session

from src.data_models.round_info import RoundsInfo
from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.experiment_database.models import (
    ClientAtRound,
    ExecutedConfig,
    ExperimentClient,
    ExperimentEvent,
    ExperimentAuthor,
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
        wandb_url: str | None = None,
        full_author_name: str | None = None,
        author_email: str | None = None,
        created_at=None,
) -> ExperimentEvent:
    """Create an ExperimentEvent row linking a config, hardware spec, and author to a single run."""

    author = None

    normalized_full_name = full_author_name.upper() if full_author_name else None
    normalized_email = author_email.upper() if author_email else None

    if normalized_email is not None:
        author = (
            session.query(ExperimentAuthor)
            .filter(ExperimentAuthor.author_email == normalized_email)
            .first()
        )

        if author is None:
            author = ExperimentAuthor(
                full_author_name=normalized_full_name,
                author_email=normalized_email,
            )
            session.add(author)
            session.flush()

        elif normalized_full_name and author.full_author_name != normalized_full_name:
            author.full_author_name = normalized_full_name

    event = ExperimentEvent(
        executed_config_id=executed_config.id,
        hardware_id=hardware.id if hardware else None,
        author_id=author.id if author else None,
        is_successful=False,
        wandb_url=wandb_url,
    )
    if created_at is not None:
        event.created_at = created_at
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
            mi = MetricInfo(name=metric_name, type="per_round")
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
                    value=float(value) if value is not None else None,
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
                mi = MetricInfo(name=metric_name, type="per_client")
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


def fill_all_tables(session, strategy_history: SimulationStrategyHistory, total_exec_time, full_author_name, author_email, created_at=None):
    """
    Fill tables with full experiment information.
    """

    hardware = get_or_create_hardware(
        session,
        cpu=strategy_history.host_cpu,
        arch=strategy_history.training_device_arch,
        ram=strategy_history.host_ram,
        gpu_name=strategy_history.training_device_name,
        gpu_ram=strategy_history.training_device_ram,
        is_executed_on=strategy_history.training_device_type,
    )
    executed_config = create_executed_config(
        session=session,
        strategy_config=strategy_history.strategy_config,
        num_of_malicious_clients=strategy_history.total_malicious_clients,
    )
    experiment_event = create_experiment_event(
        session=session,
        executed_config=executed_config,
        hardware=hardware,
        full_author_name=full_author_name,
        author_email=author_email,
        created_at=created_at,
    )
    persist_round_metrics(
        session=session,
        experiment_event=experiment_event,
        rounds_info=strategy_history.rounds_history,
    )
    _, experiment_client_id_map = create_clients(
        session=session,
        experiment_event=experiment_event,
        num_clients=strategy_history.strategy_config.num_of_clients,
        malicious_ids=strategy_history.get_malicious_client_ids(),
    )
    persist_clients_at_round(
        session=session,
        experiment_event=experiment_event,
        clients_info=strategy_history.get_all_clients(),
        experiment_client_id_map=experiment_client_id_map,
    )
    persist_client_round_metrics(
        session=session,
        experiment_event=experiment_event,
        clients_info=strategy_history.get_all_clients(),
        experiment_client_id_map=experiment_client_id_map,
    )
    experiment_event.is_successful = True
    experiment_event.total_exec_time = total_exec_time
    session.commit()
