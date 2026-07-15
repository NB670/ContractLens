"""Text embedding for semantic retrieval and comparison (Checkpoint 3).

Two backends, mirroring the two-backend pattern already used for
classification (`app/clauses/classifier.py`):

  * ``HashingEmbedder`` (dependency-free fallback) -- a deterministic
    bag-of-words hashing embedding. No model download, works everywhere.

  * ``SentenceTransformerEmbedder`` (default) -- a small sentence-embedding
    model purpose-trained for cosine-similarity retrieval. This is a
    deliberate departure from reusing LegalBERT's mean-pooled tokens (as the
    reference scaffold does): LegalBERT is a masked-LM encoder never trained
    for sentence similarity, whereas sentence-transformers models are
    trained specifically for it via contrastive/triplet losses over sentence
    pairs -- the right tool for retrieval specifically. Falls back to
    ``HashingEmbedder`` if the dependency or model is unavailable, so
    nothing hard-fails on a missing optional dependency.

Both return a plain ``list[float]`` vector, and ``cosine_similarity`` operates
on those lists, so no heavy numeric dependency is required at the retrieval
layer itself.
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Protocol

from app.config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Dimensionality of the hashing embedding. Large enough to keep collisions
# rare for clause-length text, small enough to keep pure-python cosine cheap.
_HASH_DIM = 512


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors, in [-1, 1]."""
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class Embedder(Protocol):
    """Anything that turns text into a fixed-length vector."""

    def embed(self, text: str) -> list[float]: ...  # pragma: no cover - protocol


class HashingEmbedder:
    """Deterministic, dependency-free bag-of-words hashing embedder.

    Each token is hashed into one of ``dim`` buckets and its (sub-linear) term
    frequency is accumulated there. The resulting vector is L2-normalized. Two
    texts that share vocabulary land close under cosine similarity; disjoint
    texts land near zero. Deterministic across runs and processes (uses a
    stable hash, not Python's salted ``hash()``).
    """

    def __init__(self, dim: int = _HASH_DIM) -> None:
        self.dim = dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokenize(text):
            vec[self._bucket(token)] += 1.0
        vec = [math.log1p(v) for v in vec]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec


class SentenceTransformerEmbedder:
    """Sentence-embedding backend for semantic retrieval.

    Loads the model lazily on first ``embed()`` call (same lazy-load timing
    as ``LegalBertClassifier``). Falls back to ``HashingEmbedder`` if the
    dependency or model fails to load, so the pipeline never hard-fails on a
    missing optional dependency.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.retrieval_model
        self._model = None
        self._load_failed = False
        self._fallback = HashingEmbedder()

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        try:  # pragma: no cover - depends on optional heavy dep
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            return True
        except Exception:
            self._load_failed = True
            return False

    def embed(self, text: str) -> list[float]:
        if not self._ensure_model():
            return self._fallback.embed(text)
        try:  # pragma: no cover - depends on optional heavy dep
            vector = self._model.encode(text)
            return [float(x) for x in vector.tolist()]
        except Exception:
            return self._fallback.embed(text)


@lru_cache(maxsize=None)
def get_embedder() -> Embedder:
    """Return the process-wide configured embedder instance.

    Cached so every caller (the retrieval index, the comparator, the eval
    scripts) shares one instance -- and, for ``SentenceTransformerEmbedder``,
    one lazily-loaded model -- instead of each reloading it independently.
    """
    if settings.retrieval_backend == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder()
