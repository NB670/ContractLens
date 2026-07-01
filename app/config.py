"""Runtime configuration for ContractLens.

Everything is local-first by design: nothing here points at an external service,
in keeping with the project's privacy-preserving goal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Maximum upload size accepted by the /upload endpoint (bytes).
    max_upload_bytes: int = int(os.environ.get("CONTRACTLENS_MAX_UPLOAD", 25 * 1024 * 1024))

    # File extensions the ingestion layer knows how to parse.
    supported_extensions: tuple[str, ...] = (".pdf", ".docx", ".txt")

    # Clause-classifier backend: "rule" (default, no model download) or "legalbert".
    classifier_backend: str = os.environ.get("CONTRACTLENS_CLASSIFIER", "rule")

    # HuggingFace model id used when classifier_backend == "legalbert".
    legalbert_model: str = os.environ.get(
        "CONTRACTLENS_LEGALBERT_MODEL", "nlpaueb/legal-bert-base-uncased"
    )


settings = Settings()
