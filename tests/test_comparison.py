"""Tests for contract comparison: optimal clause alignment and diffing."""

from app.comparison.comparator import compare_contracts
from app.models.analysis import CHANGE_ADDED, CHANGE_MODIFIED, CHANGE_REMOVED, CHANGE_UNCHANGED
from app.models.contract import Clause, Contract
from app.retrieval.embedder import HashingEmbedder


def _clause(index: int, text: str, category: str = "Unclassified") -> Clause:
    return Clause(
        index=index, heading=None, text=text, category=category, confidence=1.0,
        start_offset=0, end_offset=len(text),
    )


def _contract(contract_id: str, texts: list[str]) -> Contract:
    return Contract(
        id=contract_id, filename=f"{contract_id}.txt", source_format="txt",
        clauses=[_clause(i, t) for i, t in enumerate(texts)],
    )


def test_identical_contracts_are_all_unchanged():
    texts = [
        "Each party shall keep all Confidential Information secret and confidential.",
        "Payment shall be made net thirty days after invoice.",
    ]
    base = _contract("base", texts)
    revised = _contract("revised", texts)

    diff = compare_contracts(base, revised, embedder=HashingEmbedder())

    assert diff.summary[CHANGE_UNCHANGED] == 2
    assert diff.summary[CHANGE_ADDED] == 0
    assert diff.summary[CHANGE_REMOVED] == 0
    assert diff.summary[CHANGE_MODIFIED] == 0


def test_detects_added_removed_and_modified_clauses():
    base = _contract("base", [
        "Each party shall keep all Confidential Information secret and confidential.",
        "Payment shall be made net thirty days after invoice.",
        "This Agreement shall be governed by the laws of Delaware.",
    ])
    revised = _contract("revised", [
        "Payment shall be made net thirty days after invoice.",
        "This Agreement shall be governed by the laws of Delaware, without regard to conflicts of law.",
        "This is a brand new indemnification clause added in the revision.",
    ])

    diff = compare_contracts(base, revised, embedder=HashingEmbedder())

    assert diff.summary[CHANGE_REMOVED] == 1  # confidentiality clause dropped
    assert diff.summary[CHANGE_UNCHANGED] == 1  # payment clause identical
    assert diff.summary[CHANGE_MODIFIED] == 1  # governing law clause reworded
    assert diff.summary[CHANGE_ADDED] == 1  # new indemnification clause

    removed = [c for c in diff.changes if c.change_type == CHANGE_REMOVED][0]
    assert "Confidential" in removed.base_text

    added = [c for c in diff.changes if c.change_type == CHANGE_ADDED][0]
    assert "indemnification" in added.revised_text.lower()


def test_empty_base_contract_all_added():
    base = _contract("base", [])
    revised = _contract("revised", ["A brand new clause."])

    diff = compare_contracts(base, revised, embedder=HashingEmbedder())

    assert diff.summary[CHANGE_ADDED] == 1
    assert diff.summary[CHANGE_REMOVED] == 0


def test_empty_revised_contract_all_removed():
    base = _contract("base", ["An old clause that got dropped."])
    revised = _contract("revised", [])

    diff = compare_contracts(base, revised, embedder=HashingEmbedder())

    assert diff.summary[CHANGE_REMOVED] == 1
    assert diff.summary[CHANGE_ADDED] == 0


def test_rectangular_matrix_with_multiple_simultaneous_additions():
    """Non-square, non-degenerate case: 3 base clauses vs. 5 revised clauses.

    Exercises linear_sum_assignment on a genuinely rectangular cost matrix
    (more revised clauses than base clauses, with three simultaneous adds)
    rather than the square or fully-degenerate (0xN/Nx0) shapes covered by
    the other tests in this file.
    """
    base = _contract("base", [
        "Each party shall keep all Confidential Information secret and confidential.",
        "Payment shall be made net thirty days after invoice.",
        "This Agreement shall be governed by the laws of Delaware.",
    ])
    revised = _contract("revised", [
        "Each party shall keep all Confidential Information secret and confidential.",
        "Payment shall be made net thirty days after invoice.",
        "This is a brand new indemnification clause covering third-party claims.",
        "Either party may terminate this Agreement upon thirty days written notice for convenience.",
        "Neither party shall be liable for delays caused by events of force majeure beyond its control.",
    ])

    diff = compare_contracts(base, revised, embedder=HashingEmbedder())

    assert diff.summary[CHANGE_UNCHANGED] == 2  # confidentiality + payment carried over
    assert diff.summary[CHANGE_REMOVED] == 1  # governing law clause dropped
    assert diff.summary[CHANGE_ADDED] == 3  # indemnification, termination, force majeure
    assert diff.summary[CHANGE_MODIFIED] == 0

    removed = [c for c in diff.changes if c.change_type == CHANGE_REMOVED][0]
    assert "Delaware" in removed.base_text

    added_texts = {c.revised_text for c in diff.changes if c.change_type == CHANGE_ADDED}
    assert any("indemnification" in t.lower() for t in added_texts)
    assert any("terminate" in t.lower() for t in added_texts)
    assert any("force majeure" in t.lower() for t in added_texts)
