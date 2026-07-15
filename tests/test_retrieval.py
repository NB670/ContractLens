"""Tests for semantic clause retrieval: embedder, cosine similarity, and index."""

import pytest

from app.models.contract import Clause, Contract
from app.retrieval.embedder import HashingEmbedder, cosine_similarity
from app.retrieval.index import ClauseIndex


def _clause(index: int, text: str, category: str = "Unclassified") -> Clause:
    return Clause(
        index=index, heading=None, text=text, category=category, confidence=1.0,
        start_offset=0, end_offset=len(text),
    )


def _contract(contract_id: str, texts: list[str]) -> Contract:
    return Contract(
        id=contract_id,
        filename=f"{contract_id}.txt",
        source_format="txt",
        clauses=[_clause(i, t) for i, t in enumerate(texts)],
    )


def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0], [1.0])


def test_hashing_embedder_is_deterministic():
    embedder = HashingEmbedder()
    assert embedder.embed("Confidential Information") == embedder.embed("Confidential Information")


def test_hashing_embedder_similar_text_scores_higher_than_unrelated_text():
    embedder = HashingEmbedder()
    query = embedder.embed("Each party shall keep all Confidential Information secret.")
    similar = embedder.embed("The parties agree to keep Confidential Information confidential.")
    unrelated = embedder.embed("Payment shall be made net thirty days after invoice.")

    assert cosine_similarity(query, similar) > cosine_similarity(query, unrelated)


def test_clause_index_search_ranks_by_similarity():
    index = ClauseIndex(embedder=HashingEmbedder())
    index.add_contract(_contract("c1", [
        "Each party shall keep all Confidential Information secret and confidential.",
        "Payment shall be made net thirty days after invoice.",
    ]))

    hits = index.search("confidential information secrecy obligations", k=2)

    assert len(hits) == 2
    assert hits[0].clause_index == 0
    assert hits[0].score >= hits[1].score


def test_clause_index_search_respects_category_filter():
    index = ClauseIndex(embedder=HashingEmbedder())
    contract = _contract("c1", ["Confidential Information clause.", "Payment terms clause."])
    contract.clauses[0].category = "Confidentiality"
    contract.clauses[1].category = "Payment Terms"
    index.add_contract(contract)

    hits = index.search("clause", k=5, category="Payment Terms")

    assert len(hits) == 1
    assert hits[0].category == "Payment Terms"


def test_clause_index_most_similar_to_excludes_self():
    index = ClauseIndex(embedder=HashingEmbedder())
    index.add_contract(_contract("c1", [
        "Each party shall keep all Confidential Information secret.",
        "The parties agree to keep Confidential Information confidential at all times.",
        "Payment shall be made net thirty days after invoice.",
    ]))

    hits = index.most_similar_to("c1", 0, k=2)

    assert all(not (h.contract_id == "c1" and h.clause_index == 0) for h in hits)
    assert hits[0].clause_index == 1


def test_clause_index_size_tracks_added_clauses():
    index = ClauseIndex(embedder=HashingEmbedder())
    index.add_contract(_contract("c1", ["one", "two", "three"]))
    assert index.size == 3


def test_clause_index_search_on_empty_index_returns_empty_list():
    index = ClauseIndex(embedder=HashingEmbedder())
    assert index.search("anything") == []
