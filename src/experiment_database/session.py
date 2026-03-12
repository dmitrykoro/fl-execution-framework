"""Create SQLAlchemy engine and session."""

import logging

from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.utils.common import obtain_env_var

db_mode = obtain_env_var("EXPERIMENT_DB_MODE")

if db_mode in ("direct", "neon"):
    db_type = obtain_env_var("EXPERIMENT_DB_TYPE")
    db_host = obtain_env_var("EXPERIMENT_DB_HOST")
    db_port = obtain_env_var("EXPERIMENT_DB_PORT")
    db_name = obtain_env_var("EXPERIMENT_DB_NAME")
    db_user = obtain_env_var("EXPERIMENT_DB_USER")
    db_password = obtain_env_var("EXPERIMENT_DB_PASSWORD")

if db_mode == "direct":
    if db_type == "postgres":
        engine = create_engine(
            f"postgresql+psycopg2://{db_user}:{db_password}"
            f"@{db_host}:{db_port}/{db_name}",
            future=True,
        )
        LocalDatabaseSession = sessionmaker(bind=engine, expire_on_commit=False)
    else:
        raise NotImplementedError(
            f"Connection engine for database '{db_type}' is not implemented. "
            "Implement it in src/experiment_database/session.py"
        )

elif db_mode == "neon":
    try:
        engine = create_engine(
            f"postgresql+psycopg2://{db_user}:{db_password}"
            f"@{db_host}:{db_port}/{db_name}",
            connect_args={"sslmode": "require"},
            future=True,
        )
        LocalDatabaseSession = sessionmaker(bind=engine, expire_on_commit=False)
    except Exception as e:  # noqa
        logging.error(f"Error while connecting to NEON database: {e}")

elif db_mode == "server":
    pass

else:
    raise NotImplementedError(
        f"Not supported EXPERIMENT_DB_MODE: {db_mode}"
    )


@contextmanager
def get_db():
    """Yield a LocalDatabaseSession and ensure it is closed on exit."""
    session = LocalDatabaseSession()

    try:
        yield session

    finally:
        session.close()
