"""Create SQLAlchemy engine and session."""

import time
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from src.utils.common import obtain_env_var


db_type = obtain_env_var("EXPERIMENT_DB_TYPE")
db_port = obtain_env_var("EXPERIMENT_DB_PORT")
db_mode = obtain_env_var("EXPERIMENT_DB_MODE")
local_db_name = obtain_env_var("LOCAL_DB_NAME")
local_db_password = obtain_env_var("LOCAL_DB_PASSWORD")
local_db_host = obtain_env_var("LOCAL_DB_HOST")

if db_type == "postgres":
    local_engine = create_engine(
        f"postgresql+psycopg2://postgres:{local_db_password}"
        f"@{local_db_host}:{db_port}/{local_db_name}",
        future=True,
    )
else:
    raise NotImplementedError(
        f"Connection engine for database '{db_type}' is not implemented. "
        "Implement it in src/experiment_database/session.py"
    )

LocalDatabaseSession = sessionmaker(bind=local_engine, expire_on_commit=False)

if db_mode == "online":
    db_name = obtain_env_var("EXPERIMENT_DB_NAME")
    db_password = obtain_env_var("EXPERIMENT_DB_PASSWORD")
    db_host = obtain_env_var("EXPERIMENT_DB_HOST")
    db_user = obtain_env_var("EXPERIMENT_DB_USER")

    neon_engine = create_engine(
        f"postgresql+psycopg2://{db_user}:{db_password}"
        f"@{db_host}:{db_port}/{db_name}",
        connect_args={"sslmode": "require"},
        future=True,
    )
    NeonDatabaseSession = sessionmaker(bind=neon_engine, expire_on_commit=False)
else:
    NeonDatabaseSession = None


@contextmanager
def get_db():
    """Yield a LocalDatabaseSession and ensure it is closed on exit."""
    session = LocalDatabaseSession()
    try:
        yield session
    finally:
        session.close()