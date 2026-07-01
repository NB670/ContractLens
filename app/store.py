"""Persistent contract store (SQLite via SQLModel).

Checkpoint 2 originally used a process-local in-memory dict; this replaces it
with a SQLite-backed store so ingested contracts survive a server restart.
The public API (`store.new_id`, `store.add`, `store.get`, `store.list_ids`)
is unchanged, so `app/main.py` and `app/pipeline.py` don't need to change.
"""

from __future__ import annotations

import os
import uuid

from sqlalchemy.pool import StaticPool
from sqlmodel import Field, Session, SQLModel, create_engine, select

from app.models.contract import Contract


class ContractRecord(SQLModel, table=True):
    """A single serialized Contract, stored as one JSON blob per row."""

    id: str = Field(primary_key=True)
    filename: str
    source_format: str
    payload_json: str


class ContractStore:
    def __init__(self, database_url: str | None = None) -> None:
        if database_url is None:
            db_path = os.environ.get("CONTRACTLENS_DB_PATH", "contractlens.db")
            database_url = f"sqlite:///{db_path}"

        engine_kwargs: dict = {}
        if database_url.startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}
        if database_url in ("sqlite://", "sqlite:///:memory:"):
            engine_kwargs["poolclass"] = StaticPool

        self._engine = create_engine(database_url, **engine_kwargs)
        SQLModel.metadata.create_all(self._engine)

    def new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def add(self, contract: Contract) -> None:
        record = ContractRecord(
            id=contract.id,
            filename=contract.filename,
            source_format=contract.source_format,
            payload_json=contract.model_dump_json(),
        )
        with Session(self._engine) as session:
            session.merge(record)
            session.commit()

    def get(self, contract_id: str) -> Contract | None:
        with Session(self._engine) as session:
            record = session.get(ContractRecord, contract_id)
            if record is None:
                return None
            return Contract.model_validate_json(record.payload_json)

    def list_ids(self) -> list[str]:
        with Session(self._engine) as session:
            return list(session.exec(select(ContractRecord.id)).all())


store = ContractStore()
