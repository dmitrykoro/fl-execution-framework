from src.experiment_database.session import local_engine, neon_engine
from src.experiment_database.models import Base

def init_db():
    for engine in [local_engine]:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()