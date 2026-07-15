"""Semantic contract comparison (Checkpoint 3).

Given a base contract and a revised contract, align their clauses by
embedding similarity and classify each into one of: ``unchanged``,
``modified``, ``added``, ``removed``.

Alignment uses scipy's Hungarian algorithm (``linear_sum_assignment``) to
find the globally optimal one-to-one pairing over the clause-similarity
matrix -- a deliberate departure from a greedy highest-similarity-first
match (which can lock in a suboptimal pairing when a clause's best partner
is claimed by a slightly-better competing pair elsewhere in the matrix).
Pairs below ``match_threshold`` are rejected even if the optimal assignment
would otherwise include them, so unrelated clauses are never forced
together.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.models.analysis import (
    CHANGE_ADDED,
    CHANGE_MODIFIED,
    CHANGE_REMOVED,
    CHANGE_UNCHANGED,
    ClauseChange,
    ContractDiff,
)
from app.models.contract import Clause, Contract
from app.retrieval.embedder import Embedder, cosine_similarity, get_embedder

DEFAULT_MATCH_THRESHOLD = 0.60
DEFAULT_IDENTICAL_THRESHOLD = 0.995


def _pick_category(base: Clause, revised: Clause) -> str:
    """Prefer a classified category over the 'Unclassified' placeholder."""
    if revised.category and revised.category != "Unclassified":
        return revised.category
    return base.category


def compare_contracts(
    base: Contract,
    revised: Contract,
    embedder: Embedder | None = None,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    identical_threshold: float = DEFAULT_IDENTICAL_THRESHOLD,
) -> ContractDiff:
    """Align two contracts' clauses and return a structured ``ContractDiff``."""
    embedder = embedder or get_embedder()

    base_vecs = [embedder.embed(c.text) for c in base.clauses]
    revised_vecs = [embedder.embed(c.text) for c in revised.clauses]

    matched_base: dict[int, tuple[int, float]] = {}
    matched_revised: set[int] = set()

    if base_vecs and revised_vecs:
        similarity = np.zeros((len(base_vecs), len(revised_vecs)))
        for i, bv in enumerate(base_vecs):
            for j, rv in enumerate(revised_vecs):
                similarity[i, j] = cosine_similarity(bv, rv)

        # linear_sum_assignment minimizes cost; negate similarity to maximize it.
        row_idx, col_idx = linear_sum_assignment(-similarity)
        for i, j in zip(row_idx, col_idx):
            sim = float(similarity[i, j])
            if sim < match_threshold:
                continue
            matched_base[i] = (j, sim)
            matched_revised.add(j)

    changes: list[ClauseChange] = []

    for i, (j, sim) in sorted(matched_base.items()):
        base_clause, revised_clause = base.clauses[i], revised.clauses[j]
        identical = (
            sim >= identical_threshold
            or base_clause.text.strip() == revised_clause.text.strip()
        )
        changes.append(
            ClauseChange(
                change_type=CHANGE_UNCHANGED if identical else CHANGE_MODIFIED,
                category=_pick_category(base_clause, revised_clause),
                base_index=base_clause.index,
                revised_index=revised_clause.index,
                similarity=round(sim, 4),
                base_text=base_clause.text,
                revised_text=revised_clause.text,
            )
        )

    for i, base_clause in enumerate(base.clauses):
        if i not in matched_base:
            changes.append(
                ClauseChange(
                    change_type=CHANGE_REMOVED,
                    category=base_clause.category,
                    base_index=base_clause.index,
                    revised_index=None,
                    similarity=0.0,
                    base_text=base_clause.text,
                    revised_text=None,
                )
            )

    for j, revised_clause in enumerate(revised.clauses):
        if j not in matched_revised:
            changes.append(
                ClauseChange(
                    change_type=CHANGE_ADDED,
                    category=revised_clause.category,
                    base_index=None,
                    revised_index=revised_clause.index,
                    similarity=0.0,
                    base_text=None,
                    revised_text=revised_clause.text,
                )
            )

    _ORDER = {CHANGE_MODIFIED: 0, CHANGE_ADDED: 1, CHANGE_REMOVED: 2, CHANGE_UNCHANGED: 3}
    changes.sort(
        key=lambda c: (
            _ORDER[c.change_type],
            c.revised_index if c.revised_index is not None else c.base_index or 0,
        )
    )

    summary: dict[str, int] = {
        CHANGE_ADDED: 0,
        CHANGE_REMOVED: 0,
        CHANGE_MODIFIED: 0,
        CHANGE_UNCHANGED: 0,
    }
    for change in changes:
        summary[change.change_type] += 1

    return ContractDiff(
        base_contract_id=base.id,
        revised_contract_id=revised.id,
        changes=changes,
        summary=summary,
    )
