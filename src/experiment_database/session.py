"""Create SQLAlchemy engine and session."""

import logging
import os

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from src.experiment_database.models import Base
from src.utils.common import obtain_env_var

DEFAULT_DB_MODE = "local"
DEFAULT_LOCAL_DB_PATH = "out/experiments.db"

# Modes backed by a SQLAlchemy engine in this process.
ENGINE_DB_MODES = ("local", "direct", "neon")
SUPPORTED_DB_MODES = ENGINE_DB_MODES + ("server",)

_engine: Engine | None = None
_session_factory: sessionmaker | None = None
_db_mode_override: str | None = None


def set_db_mode(db_mode: str) -> None:
    """
    Force the database mode for the rest of this process and drop any cached engine.

    Used when the configured mode turns out to be unusable at runtime, e.g. when
    `server` mode falls back to the local database because no credentials are available.
    """

    global _db_mode_override, _engine, _session_factory

    if db_mode not in SUPPORTED_DB_MODES:
        raise NotImplementedError(
            f"Not supported EXPERIMENT_DB_MODE: {db_mode}. "
            f"Supported modes: {', '.join(SUPPORTED_DB_MODES)}"
        )

    if _engine is not None:
        _engine.dispose()

    _db_mode_override = db_mode
    _engine = None
    _session_factory = None


def resolve_db_mode() -> str:
    """Return the effective EXPERIMENT_DB_MODE, defaulting to the local SQLite database."""

    if _db_mode_override is not None:
        return _db_mode_override

    db_mode = obtain_env_var(
        "EXPERIMENT_DB_MODE",
        default=DEFAULT_DB_MODE,
        prompt_hint=f"'{'/'.join(SUPPORTED_DB_MODES)}', empty for '{DEFAULT_DB_MODE}'",
    )

    if not db_mode:
        return DEFAULT_DB_MODE

    if db_mode not in SUPPORTED_DB_MODES:
        raise NotImplementedError(
            f"Not supported EXPERIMENT_DB_MODE: {db_mode}. "
            f"Supported modes: {', '.join(SUPPORTED_DB_MODES)}"
        )

    return db_mode


def local_db_path() -> Path:
    """Return the filesystem path of the local SQLite experiment database."""

    configured_path = obtain_env_var(
        "EXPERIMENT_DB_LOCAL_PATH", default=DEFAULT_LOCAL_DB_PATH
    )

    return Path(configured_path or DEFAULT_LOCAL_DB_PATH).expanduser()


def _create_local_engine() -> Engine:
    """
    Build an engine for the local SQLite database, creating the schema if needed.

    The local database is not managed by Alembic: it is created from the current
    models on first use, which is what a fresh checkout without access to the
    research group's PostgreSQL server needs.
    """

    db_path = local_db_path()
    is_new_database = not db_path.exists()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    local_engine = create_engine(
        f"sqlite:///{db_path}",
        future=True,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(local_engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
        """Enable WAL and foreign keys, which SQLite disables per connection by default."""
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

    Base.metadata.create_all(bind=local_engine)

    if is_new_database:
        _stamp_alembic_head()

    logging.info(f"Using local SQLite experiment database at {db_path.resolve()}")

    return local_engine


def _stamp_alembic_head() -> None:
    """
    Record a freshly created local database as being at the latest migration(s).

    Its schema comes from Base.metadata.create_all rather than from replaying the
    migration chain, so without this stamp `alembic upgrade` would try to create
    tables that already exist.
    """

    alembic_ini = Path("alembic.ini")

    if not alembic_ini.exists():
        logging.debug("alembic.ini not found, leaving the local database unstamped.")
        return

    previous_db_mode = os.environ.get("EXPERIMENT_DB_MODE")
    os.environ["EXPERIMENT_DB_MODE"] = "local"

    try:
        from alembic import command
        from alembic.config import Config

        command.stamp(Config(str(alembic_ini)), "heads")

    except Exception as e:  # noqa
        logging.debug(f"Could not stamp the local database at the latest migration: {e}")

    finally:
        if previous_db_mode is None:
            os.environ.pop("EXPERIMENT_DB_MODE", None)
        else:
            os.environ["EXPERIMENT_DB_MODE"] = previous_db_mode


def _create_postgres_engine(db_mode: str) -> Engine:
    """Build an engine for a remote PostgreSQL database (`direct` or `neon` mode)."""

    db_type = obtain_env_var("EXPERIMENT_DB_TYPE", default="postgres")
    db_host = obtain_env_var("EXPERIMENT_DB_HOST")
    db_port = obtain_env_var("EXPERIMENT_DB_PORT")
    db_name = obtain_env_var("EXPERIMENT_DB_NAME")
    db_user = obtain_env_var("EXPERIMENT_DB_USER")
    db_password = obtain_env_var("EXPERIMENT_DB_PASSWORD")

    if db_type != "postgres":
        raise NotImplementedError(
            f"Connection engine for database '{db_type}' is not implemented. "
            "Implement it in src/experiment_database/session.py"
        )

    if not all((db_host, db_port, db_name, db_user, db_password)):
        raise RuntimeError(
            f"EXPERIMENT_DB_MODE='{db_mode}' requires EXPERIMENT_DB_HOST, EXPERIMENT_DB_PORT, "
            "EXPERIMENT_DB_NAME, EXPERIMENT_DB_USER and EXPERIMENT_DB_PASSWORD to be set. "
            f"Use EXPERIMENT_DB_MODE='{DEFAULT_DB_MODE}' to store results in a local SQLite file instead."
        )

    connect_args = {"sslmode": "require"} if db_mode == "neon" else {}

    return create_engine(
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
        connect_args=connect_args,
        future=True,
    )


def create_db_engine(db_mode: str | None = None) -> Engine:
    """Build a new engine for the given mode without caching it."""

    db_mode = db_mode or resolve_db_mode()

    if db_mode == "local":
        return _create_local_engine()

    if db_mode in ("direct", "neon"):
        return _create_postgres_engine(db_mode)

    raise RuntimeError(
        f"EXPERIMENT_DB_MODE='{db_mode}' has no local database engine: results are sent to the "
        "experiment server over HTTP. Use 'local', 'direct' or 'neon' to open a database directly."
    )


def get_engine() -> Engine:
    """Return the process-wide engine, creating it on first use."""

    global _engine

    if _engine is None:
        _engine = create_db_engine()

    return _engine


def get_session_factory() -> sessionmaker:
    """Return the process-wide session factory, creating it on first use."""

    global _session_factory

    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)

    return _session_factory


def __getattr__(name: str):
    """
    Resolve `engine` and `LocalDatabaseSession` lazily.

    They used to be module-level objects, so importing this module connected to the
    database as a side effect. Keeping them as attributes preserves the old imports
    while deferring the connection until something actually queries.
    """

    if name == "engine":
        return get_engine()

    if name == "LocalDatabaseSession":
        return get_session_factory()

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@contextmanager
def get_db():
    """Yield a database session and ensure it is closed on exit."""
    session = get_session_factory()()

    try:
        yield session

    finally:
        session.close()
