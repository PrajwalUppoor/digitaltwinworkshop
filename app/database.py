import os
import time
from typing import Iterator
from sqlmodel import SQLModel, create_engine, Session
from sqlmodel import select
from .models import Item

# Read DATABASE_URL from environment, fallback to a local Postgres URL for development
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres"
)

# Create synchronous SQLAlchemy engine via SQLModel
# enable pool_pre_ping to avoid stale connections
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)


def init_db(retries: int = 20, delay: float = 1.0):
    """Create database tables, retrying until Postgres is reachable.

    This function will attempt to connect and create tables. If the DB is not
    ready yet (common when using docker-compose), it will retry for a number
    of times before raising the last exception.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            SQLModel.metadata.create_all(engine)
            # If the items table is empty, insert some example rows so the
            # app / dashboard shows data on first startup. This is a small
            # convenience for development; in production you may want to
            # remove or replace this with a proper migration/seed mechanism.
            try:
                with Session(engine) as session:
                    result = session.exec(select(Item)).all()
                    if len(result) == 0:
                        sample = [
                            Item(name="Sample A", value=1.23),
                            Item(name="Sample B", value=4.56),
                            Item(name="Sample C", value=7.89),
                        ]
                        for it in sample:
                            session.add(it)
                        session.commit()
                        print("Seeded items table with example rows")
            except Exception as seed_err:
                # Non-fatal: log but don't fail DB init if seeding fails
                print(f"Warning: seeding example data failed: {seed_err}")
            return
        except Exception as e:
            last_exc = e
            print(f"init_db: attempt {attempt} failed, retrying in {delay}s: {e}")
            time.sleep(delay)
    # if we get here, all retries failed
    raise last_exc


def get_session() -> Iterator[Session]:
    """Yield a SQLModel Session for FastAPI dependency injection."""
    with Session(engine) as session:
        yield session
