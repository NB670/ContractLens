"""Tests for the SQLite-backed contract store."""

from app.models.contract import Contract, ContractMetadata
from app.store import ContractStore


def _sample_contract(contract_id: str = "c1") -> Contract:
    return Contract(
        id=contract_id,
        filename="nda.txt",
        source_format="txt",
        metadata=ContractMetadata(
            contract_type="Non-Disclosure Agreement",
            parties=["Acme", "Globex"],
            num_clauses=1,
            num_chars=42,
        ),
        clauses=[],
    )


def test_add_then_get_round_trips_contract():
    store = ContractStore(database_url="sqlite://")
    contract = _sample_contract()

    store.add(contract)
    fetched = store.get(contract.id)

    assert fetched == contract


def test_get_missing_contract_returns_none():
    store = ContractStore(database_url="sqlite://")
    assert store.get("does-not-exist") is None


def test_list_ids_returns_all_added_contracts():
    store = ContractStore(database_url="sqlite://")
    store.add(_sample_contract("c1"))
    store.add(_sample_contract("c2"))

    assert sorted(store.list_ids()) == ["c1", "c2"]


def test_new_store_instance_does_not_share_state():
    store_a = ContractStore(database_url="sqlite://")
    store_b = ContractStore(database_url="sqlite://")
    store_a.add(_sample_contract("only-in-a"))

    assert store_b.get("only-in-a") is None
