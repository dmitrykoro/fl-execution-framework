from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(
    "postgresql+psycopg2://postgres:Soravic2202!@localhost:5432/fldb",
    future=True,
)


SessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False,
)