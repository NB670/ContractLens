"""In-memory semantic index over contract clauses (Checkpoint 3).

``ClauseIndex`` embeds every clause of every added contract and answers
similarity queries with an exact (brute-force) cosine nearest-neighbour
search. The public surface (``add_contract`` / ``search`` / ``most_similar_to``)
is deliberately small so a FAISS/Chroma backend could be dropped in behind it
later without touching callers.

``app/main.py`` maintains one process-wide ``ClauseIndex`` instance and calls
``add_contract`` once per upload, rather than rebuilding the index from the
store on every request -- clauses are embedded exactly once, not once per
query.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.models.analysis import RetrievalHit
from app.models.contract import Contract
from app.retrieval.embedder import Embedder, cosine_similarity, get_embedder


@dataclass
class _Entry:
    contract_id: str
    clause_index: int
    category: str
    heading: str | None
    text: str
    vector: list[float]


@dataclass
class ClauseIndex:
    """A brute-force cosine index over clauses drawn from one or more contracts."""

    embedder: Embedder = field(default_factory=get_embedder)
    _entries: list[_Entry] = field(default_factory=list)

    def add_contract(self, contract: Contract) -> int:
        """Embed and index every clause of ``contract``; return #clauses added."""
        added = 0
        for clause in contract.clauses:
            if not clause.text or not clause.text.strip():
                continue
            self._entries.append(
                _Entry(
                    contract_id=contract.id,
                    clause_index=clause.index,
                    category=clause.category,
                    heading=clause.heading,
                    text=clause.text,
                    vector=self.embedder.embed(clause.text),
                )
            )
            added += 1
        return added

    @property
    def size(self) -> int:
        return len(self._entries)

    def search(
        self,
        query: str,
        k: int = 5,
        category: str | None = None,
        exclude: tuple[str, int] | None = None,
    ) -> list[RetrievalHit]:
        """Return the ``k`` clauses most similar to ``query``."""
        if not query or not query.strip() or not self._entries:
            return []

        query_vec = self.embedder.embed(query)
        scored: list[tuple[float, _Entry]] = []
        for entry in self._entries:
            if category is not None and entry.category != category:
                continue
            if exclude is not None and (entry.contract_id, entry.clause_index) == exclude:
                continue
            scored.append((cosine_similarity(query_vec, entry.vector), entry))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            RetrievalHit(
                contract_id=entry.contract_id,
                clause_index=entry.clause_index,
                category=entry.category,
                heading=entry.heading,
                text=entry.text,
                score=round(score, 4),
            )
            for score, entry in scored[: max(0, k)]
        ]

    def most_similar_to(
        self, contract_id: str, clause_index: int, k: int = 5
    ) -> list[RetrievalHit]:
        """Find the clauses most similar to one already-indexed clause."""
        for entry in self._entries:
            if entry.contract_id == contract_id and entry.clause_index == clause_index:
                return self.search(entry.text, k=k, exclude=(contract_id, clause_index))
        return []
