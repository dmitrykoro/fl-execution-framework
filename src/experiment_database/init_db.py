from src.experiment_database.models import Base
from src.experiment_database.session import engine


def init_db() -> None:
    """Drop and recreate all tables on the local database engine."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()