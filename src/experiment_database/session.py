"""
Create sqlalchemy engine and session.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError
import time

from src.utils.common import obtain_env_var

db_type = obtain_env_var("EXPERIMENT_DB_TYPE")
db_port = obtain_env_var("EXPERIMENT_DB_PORT")
db_mode = obtain_env_var("EXPERIMENT_DB_MODE")

local_db_name = obtain_env_var("LOCAL_DB_NAME")
local_db_password = obtain_env_var("LOCAL_DB_PASSWORD")
local_db_host = obtain_env_var("LOCAL_DB_HOST")

if db_type == "postgres":
    local_engine = create_engine(
        f"postgresql+psycopg2://postgres:{local_db_password}@{local_db_host}:{db_port}/{local_db_name}",
        future=True,
    )
else:
    raise NotImplementedError(
        f"Connection engine for database {db_type} is not implemented. "
        f"Implement it in src/experiment_database/session.py"
    )

LocalDatabaseSession = sessionmaker(bind=local_engine, expire_on_commit=False)
"""
    This is for Neon, an online test server I was working on, don't change env.public
"""
if db_mode == "online":
    db_name = obtain_env_var("EXPERIMENT_DB_NAME")
    db_password = obtain_env_var("EXPERIMENT_DB_PASSWORD")
    db_host = obtain_env_var("EXPERIMENT_DB_HOST")
    db_user = obtain_env_var("EXPERIMENT_DB_USER")

    neon_engine = create_engine(
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
        connect_args={"sslmode": "require"},
        future=True,
    )
    NeonDatabaseSession = sessionmaker(bind=neon_engine, expire_on_commit=False)
else:
    NeonDatabaseSession = None

