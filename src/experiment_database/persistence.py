from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import Null

from src.experiment_database.models import (
    ExecutedConfig,
    ExperimentEvent,
    HostHardwareSpec,
    ExperimentClient,
    ClientAtRound,
    MetricInfo,
    MetricEntry,
)
from src.data_models.round_info import RoundsInfo

# -----------------------------
# Executed Config
# -----------------------------

def create_executed_config(session: Session, strategy_config, num_of_malicious_clients) -> ExecutedConfig:
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


# -----------------------------
# Hardware
# -----------------------------

def get_or_create_hardware(session: Session, cpu: str, arch: str, ram: str) -> HostHardwareSpec:
    hw = session.query(HostHardwareSpec).filter_by(
        cpu_name=cpu,
        cpu_arch=arch,
        ram_amount=ram,
    ).one_or_none()

    if hw:
        return hw

    hw = HostHardwareSpec(
        cpu_name=cpu,
        cpu_arch=arch,
        ram_amount=ram,
    )
    session.add(hw)
    session.flush()
    return hw


# -----------------------------
# Experiment Event
# -----------------------------

def create_experiment_event(
    session: Session,
    executed_config: ExecutedConfig,
    hardware: HostHardwareSpec | None = None,
) -> ExperimentEvent:
    event = ExperimentEvent(
        executed_config_id=executed_config.id,
        hardware_id=hardware.id if hardware else None,
        is_successful=False,
    )
    session.add(event)
    session.flush()
    return event


# -----------------------------
# Clients
# -----------------------------

def create_clients(
    session: Session,
    experiment_event: ExperimentEvent,
    num_clients: int,
    malicious_ids: set[int],
):
    clients = []

    for i in range(num_clients):
        client = ExperimentClient(
            experiment_event_id=experiment_event.id,
            client_number=i,
            was_ever_malicious=i in malicious_ids, # TODO: determine based on strategy_history
        )
        session.add(client)
        clients.append(client)

    session.flush()

    experiment_client_id_map = {c.client_number: c.id for c in clients}
    return clients, experiment_client_id_map




# -----------------------------
# Metrics
# -----------------------------

METRICS = [
    ("average_accuracy", "round_global", "Mean accuracy per round"),
    ("aggregated_loss", "round_global", "Aggregated loss"),
    ("tp", "round_global", "True positives"),
    ("fp", "round_global", "False positives"),
    ("removal_f1", "round_global", "Removal F1 score"),
]

def seed_metric_info(session):
    for name, type_, desc in METRICS:
        exists = session.query(MetricInfo).filter_by(name=name).first()
        if not exists:
            session.add(
                MetricInfo(
                    name=name,
                    type=type_,
                    desc=desc,
                )
            )
    session.commit()



def load_metric_info_map(session: Session) -> dict[str, MetricInfo]:
    return {mi.name: mi for mi in session.query(MetricInfo).all()}


def persist_round_metrics(
    session: Session,
    experiment_event: ExperimentEvent,
    rounds_info: RoundsInfo,
):
    if not isinstance(rounds_info, RoundsInfo):
        raise TypeError("Expected RoundsInfo instance")

    metric_info_map = load_metric_info_map(session)

    for metric_name in rounds_info.savable_metrics:
        if metric_name not in metric_info_map:
            mi = MetricInfo(
                name=metric_name,
                type="round_global",
                desc=f"{metric_name} recorded per round",
            )
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



# -----------------------------
# Clients at Round
# -----------------------------

# -----------------------------
# persist_clients_at_round
# -----------------------------
def persist_clients_at_round(
    session: Session,
    experiment_event,
    round_clients_info: list,
    experiment_client_id_map: dict[int, str],
    attack_type: str,
):
    """
    Persist ClientAtRound rows for all clients in a round.
    """

    for client in round_clients_info:
        # get the DB PK using the simulation client_id
        if client.client_id not in experiment_client_id_map:
            raise RuntimeError(
                f"Simulation client_id {client.client_id} not found in DB mapping"
            )

        experiment_client_db_id = experiment_client_id_map[client.client_id]

        for i, round_number in enumerate(client.rounds):

            session.add(
                ClientAtRound(
                    experiment_event_id=experiment_event.id,
                    experiment_client_id=experiment_client_db_id,  # ✅ DB PK
                    round_number=round_number,
                    client_is_aggregated=client.aggregation_participation_history[i],
                    attack_type=client.get_current_attack_keywords(round_number) or None,  # NOT NULL safe
                )
            )
