from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()  # carga variables desde .env

usuario = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")
host = os.getenv("DB_HOST")
puerto = os.getenv("POSTGRES_PORT", 5432)
basedatos = os.getenv("POSTGRES_DB")

SQLALCHEMY_DATABASE_URL = (
    f"postgresql://{usuario}:{password}@{host}:{puerto}/{basedatos}"
)

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
