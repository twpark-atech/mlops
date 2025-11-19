# src/common/db.py
from sqlalchemy import create_engine

def create_pg_engine(
    user: str,
    password: str,
    host: str,
    port: int,
    database: str,
    echo: bool = False
):
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(url, echo=echo, future=True)
    return engine