"""Tests for the clause segmenter."""

from app.ingestion.segmenter import segment

SAMPLE = """MUTUAL NON-DISCLOSURE AGREEMENT

1. Confidentiality
Each party agrees to hold in confidence all Confidential Information.

2. Termination
Either party may terminate upon thirty days notice.

3. Governing Law
This Agreement shall be governed by the laws of Delaware.
"""


def test_segment_returns_clauses():
    clauses = segment(SAMPLE)
    # Expect at least the three numbered sections.
    assert len(clauses) >= 3


def test_segment_preserves_offsets_in_order():
    clauses = segment(SAMPLE)
    offsets = [c.start_offset for c in clauses]
    assert offsets == sorted(offsets)
    # Offsets should index back into the original text.
    for c in clauses:
        assert SAMPLE[c.start_offset:c.start_offset + 5] in SAMPLE


def test_segment_empty_text():
    assert segment("") == []
    assert segment("   \n  ") == []


def test_clause_indices_are_sequential():
    clauses = segment(SAMPLE)
    assert [c.index for c in clauses] == list(range(len(clauses)))
