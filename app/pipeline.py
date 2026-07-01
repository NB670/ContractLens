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
    (r"service agreement", "Service Agreement"),
]

# "by and between X and Y" / "between X and Y"
_PARTIES_RE = re.compile(
    r"between\s+(.+?)\s+and\s+(.+?)(?:\s*[\.,\(]|\bdated\b|$)",
    re.IGNORECASE | re.DOTALL,
)


def _detect_type(text: str) -> str:
    head = text[:2000].lower()
    for pattern, label in _TYPE_PATTERNS:
        if re.search(pattern, head):
            return label
    return "Unknown"


def _detect_parties(text: str) -> list[str]:
    match = _PARTIES_RE.search(text[:2000])
    if not match:
        return []
    parties = []
    for group in match.groups():
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
