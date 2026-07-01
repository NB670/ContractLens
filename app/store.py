"""In-memory contract store.

A deliberately tiny placeholder for Checkpoint 2 so the API has somewhere to keep
ingested contracts within a single process. A persistent store (and the vector
index for retrieval) arrives in later checkpoints.
"""

from __future__ import annotations

import threading
import uuid

from app.models.contract import Contract


class ContractStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._contracts: dict[str, Contract] = {}

    def new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def add(self, contract: Contract) -> None:
        with self._lock:
            self._contracts[contract.id] = contract

    def get(self, contract_id: str) -> Contract | None:
        with self._lock:
            return self._contracts.get(contract_id)

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._contracts.keys())


store = ContractStore()
