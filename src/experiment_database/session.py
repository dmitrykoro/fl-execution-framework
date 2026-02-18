"""
Create sqlalchemy engine and session.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.utils.common import obtain_env_var

db_type = obtain_env_var("EXPERIMENT_DB_TYPE")
db_name = obtain_env_var("EXPERIMENT_DB_NAME")
db_password = obtain_env_var("EXPERIMENT_DB_PASSWORD")
db_host = obtain_env_var("EXPERIMENT_DB_HOST")
db_port = obtain_env_var("EXPERIMENT_DB_PORT")


if db_type == "postgres":
    engine = create_engine(
        F"postgresql+psycopg2://postgres:{db_password}@{db_host}:{db_port}/{db_name}",
        future=True,
    )
else:
    raise NotImplementedError(
        f"Connection engine for database {db_type} is not implemented. "
        f"Implement it in src/experiment_database/session.py"
    )

LocalDatabaseSession = sessionmaker(
    bind=engine,
    expire_on_commit=False,
)
