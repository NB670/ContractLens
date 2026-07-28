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

    # Clause-classifier backend: "rule" (default, no model download),
    # "legalbert" (zero-shot cosine similarity, no training), or
    # "legalbert-finetuned" (a real fine-tuned classification head trained
    # via scripts/finetune_legalbert.py -- see app/clauses/classifier.py's
    # FineTunedLegalBertClassifier).
    classifier_backend: str = os.environ.get("CONTRACTLENS_CLASSIFIER", "rule")

    # HuggingFace model id used when classifier_backend == "legalbert".
    legalbert_model: str = os.environ.get(
        "CONTRACTLENS_LEGALBERT_MODEL", "nlpaueb/legal-bert-base-uncased"
    )

    # Local directory used when classifier_backend == "legalbert-finetuned".
    # Produced by `python -m scripts.finetune_legalbert` (gitignored --
    # weights aren't committed, only the training script is).
    legalbert_finetuned_dir: str = os.environ.get(
        "CONTRACTLENS_LEGALBERT_FINETUNED_DIR", "models/legalbert-finetuned/final"
    )

    # Retrieval embedding backend: "sentence" (default, semantic) or "hashing"
    # (dependency-free fallback, same posture as classifier_backend).
    retrieval_backend: str = os.environ.get("CONTRACTLENS_RETRIEVAL_BACKEND", "sentence")

    # sentence-transformers model id used when retrieval_backend == "sentence".
    retrieval_model: str = os.environ.get(
        "CONTRACTLENS_RETRIEVAL_MODEL", "all-MiniLM-L6-v2"
    )

    # Chatbot answer backend (Checkpoint 4): "local" (default; a local
    # transformers text-generation model, with an extractive fallback if the
    # dependency/model is unavailable) or "extractive" (dependency-free,
    # deterministic answers composed from retrieved clauses).
    chat_backend: str = os.environ.get("CONTRACTLENS_CHAT_BACKEND", "local")

    # Local HuggingFace instruction-tuned model id used when
    # chat_backend == "local". Small enough to run on CPU; instruction-tuned
    # so it actually follows the "answer only from these clauses" prompt,
    # unlike a raw completion model.
    chat_model: str = os.environ.get(
        "CONTRACTLENS_CHAT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
    )


settings = Settings()
