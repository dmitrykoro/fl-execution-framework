import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, ForeignKey, Text, JSON, Float
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class ExecutedConfig(Base):
    __tablename__ = "executed_config"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    json = Column(JSON)
    aggregation_strategy_keyword = Column(Text, nullable=False)
    dataset_keyword = Column(Text, nullable=False)
    num_rounds = Column(Text, nullable=False)
    num_clients = Column(Text, nullable=False)
    num_malicious_clients = Column(Text, nullable=False)

    # relationships
    experiment_events = relationship("ExperimentEvent", back_populates="executed_config")


class HostHardwareSpec(Base):
    __tablename__ = "host_hardware_spec"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    cpu_name = Column(Text, nullable=False)
    cpu_arch = Column(Text, nullable=False)
    ram_amount = Column(Text, nullable=False)
    gpu_name = Column(Text, nullable=True)
    gpu_ram = Column(Text, nullable=True)
    is_executed_on = Column(Text, nullable=False)

    # relationships
    experiment_events = relationship("ExperimentEvent", back_populates="hardware")


class ExperimentEvent(Base):
    __tablename__ = "experiment_event"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    wandb_url = Column(String(500), nullable=True)
    user_name = Column(String(500), nullable=True)
    is_successful = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    total_exec_time = Column(Text, nullable=True)

    executed_config_id = Column(String(36), ForeignKey("executed_config.id"), nullable=False)
    executed_config = relationship("ExecutedConfig", back_populates="experiment_events")

    hardware_id = Column(String(36), ForeignKey("host_hardware_spec.id"), nullable=True)
    hardware = relationship("HostHardwareSpec", back_populates="experiment_events")

    # relationships
    clients = relationship("ExperimentClient", back_populates="experiment_event")
    metric_entries = relationship("MetricEntry", back_populates="experiment_event")
    clients_at_round = relationship("ClientAtRound", back_populates="experiment_event")


class ExperimentClient(Base):
    __tablename__ = "experiment_client"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    client_number = Column(Integer, nullable=False)

    was_ever_malicious = Column(Boolean, default=False)

    experiment_event_id = Column(
        Integer,
        ForeignKey("experiment_event.id", ondelete="CASCADE"),
        nullable=False
    )
    username=Column(String, nullabel=True)
    experiment_event = relationship(
        "ExperimentEvent",
        back_populates="clients"
    )

    metric_entries = relationship(
        "MetricEntry",
        back_populates="client",
        cascade="all, delete-orphan"
    )

    rounds = relationship(
        "ClientAtRound",
        back_populates="client",
        cascade="all, delete-orphan"
    )

class ClientAtRound(Base):
    __tablename__ = "client_at_round"

    id = Column(Integer, primary_key=True, autoincrement=True)

    experiment_event_id = Column(Integer, ForeignKey("experiment_event.id"), nullable=False)
    experiment_client_id = Column(String(36), ForeignKey("experiment_client.id"), nullable=False)

    experiment_event = relationship("ExperimentEvent", back_populates="clients_at_round")
    client = relationship("ExperimentClient", back_populates="rounds")

    round_number = Column(Integer, default=1, nullable=False)
    client_is_aggregated = Column(Boolean, default=True)
    attack_type = Column(Text, nullable=True)


class MetricInfo(Base):
    __tablename__ = "metric_info"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    type = Column(Text, nullable=False)

    metric_entries = relationship("MetricEntry", back_populates="metric_info")


class MetricEntry(Base):
    __tablename__ = "metric_entry"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    metric_info_id = Column(Integer, ForeignKey("metric_info.id"), nullable=False)
    metric_info = relationship("MetricInfo", back_populates="metric_entries")

    experiment_event_id = Column(Integer, ForeignKey("experiment_event.id"), nullable=False)
    experiment_event = relationship("ExperimentEvent", back_populates="metric_entries")

    client_id = Column(String(36), ForeignKey("experiment_client.id"), nullable=True)
    client = relationship("ExperimentClient", back_populates="metric_entries")

    value = Column(Float, nullable=False)
    tolerance = Column(Float, nullable=True)
    round_number = Column(Integer, nullable=True)
