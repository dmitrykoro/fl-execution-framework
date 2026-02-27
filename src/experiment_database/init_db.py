from src.experiment_database.models import Base
from src.experiment_database.session import local_engine


def init_db() -> None:
    """Drop and recreate all tables on the local database engine."""
    Base.metadata.drop_all(bind=local_engine)
    Base.metadata.create_all(bind=local_engine)


if __name__ == "__main__":
    init_db()