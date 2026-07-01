"""End-to-end ingestion pipeline: bytes -> structured Contract.

This wires together the three CP2 stages (parse -> segment -> classify) and a
small metadata-extraction step. It is framework-agnostic so it can be reused by
the FastAPI routes, the tests, and any future CLI.
"""

from __future__ import annotations

import re

from app.clauses.classifier import classify_clauses
from app.ingestion.parsers import parse
from app.ingestion.segmenter import segment
from app.models.contract import Contract, ContractMetadata
from app.store import store

# Rough contract-type detection from the opening text.
_TYPE_PATTERNS = [
    (r"non-?disclosure agreement|\bnda\b", "Non-Disclosure Agreement"),
    (r"master services agreement|\bmsa\b", "Master Services Agreement"),
    (r"employment agreement", "Employment Agreement"),
    (r"license agreement", "License Agreement"),
    (r"lease agreement", "Lease Agreement"),
    (r"purchase agreement", "Purchase Agreement"),
    (r"consulting agreement", "Consulting Agreement"),
    (r"supply agreement", "Supply Agreement"),
    (r"reseller agreement", "Reseller Agreement"),
    (r"franchise agreement", "Franchise Agreement"),
    (r"joint venture agreement", "Joint Venture Agreement"),
    (r"service agreement", "Service Agreement"),
]

# Two forms of party preamble:
#   - "by and between/among X, Y, and Z" (N parties, named group `n_party`)
#   - "between X and Y" (exactly 2 parties, named groups `party_a`/`party_b`)
# The first alternative is tried first so "by and between ..." text (which
# also contains the literal substring "between") is captured by the N-party
# path rather than falling through to the 2-party path.
_PARTIES_RE = re.compile(
    r"by and (?:between|among)\s+(?P<n_party>.+?)(?:\s*[\.\(]|\bdated\b|$)"
    r"|between\s+(?P<party_a>.+?)\s+and\s+(?P<party_b>.+?)(?:\s*[\.,\(]|\bdated\b|$)",
    re.IGNORECASE | re.DOTALL,
)


def _detect_type(text: str) -> str:
    head = text[:2000].lower()
    for pattern, label in _TYPE_PATTERNS:
        if re.search(pattern, head):
            return label
    return "Unknown"


def _split_party_list(span: str) -> list[str]:
    """Split a preamble party-list span like "A, B, and C" into parties."""
    span = span.strip(" ,.")
    # Normalize the final ", and " / " and " into a plain comma so a
    # straight split(",") below yields every party, including 2-party spans
    # with no comma at all ("A and B" -> "A, B").
    span = re.sub(r"\s*,?\s+and\s+(?=[^,]*$)", ", ", span, count=1)
    parts = [p.strip(" ,.") for p in span.split(",")]
    return [" ".join(p.split())[:120] for p in parts if p.strip()]


def _detect_parties(text: str) -> list[str]:
    match = _PARTIES_RE.search(text[:2000])
    if not match:
        return []

    n_party_span = match.group("n_party")
    if n_party_span is not None:
        return _split_party_list(n_party_span)

    parties = []
    for group in (match.group("party_a"), match.group("party_b")):
        cleaned = " ".join(group.split())[:120].strip(" ,.")
        if cleaned:
            parties.append(cleaned)
    return parties


def ingest(filename: str, data: bytes) -> Contract:
    """Parse, segment, classify, and store a contract; return the structured form."""
    source_format, text = parse(filename, data)

    clauses = segment(text)
    classify_clauses(clauses)

    metadata = ContractMetadata(
        contract_type=_detect_type(text),
        parties=_detect_parties(text),
        num_clauses=len(clauses),
        num_chars=len(text),
    )

    contract = Contract(
        id=store.new_id(),
        filename=filename,
        source_format=source_format,
        metadata=metadata,
        clauses=clauses,
    )
    store.add(contract)
    return contract
