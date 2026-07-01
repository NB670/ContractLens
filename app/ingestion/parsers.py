"""Document parsing: extract raw text from PDF / DOCX / TXT contracts.

PDF and DOCX extraction depend on optional third-party libraries (pypdf,
python-docx). They are imported lazily so that the rest of the pipeline — and the
TXT path used by the tests and the sample contract — works even if those
libraries are not installed yet.
"""

from __future__ import annotations

import io
from pathlib import Path


class UnsupportedFormatError(ValueError):
    """Raised when an uploaded file has an extension we cannot parse."""


def detect_format(filename: str) -> str:
    """Return a normalized format string ("pdf" | "docx" | "txt")."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "docx"
    if ext in (".txt", ".text"):
        return "txt"
    raise UnsupportedFormatError(f"Unsupported file extension: {ext!r}")


def parse_txt(data: bytes) -> str:
    """Decode a plain-text contract."""
    return data.decode("utf-8", errors="replace")


def parse_pdf(data: bytes) -> str:
    """Extract text from a PDF using pypdf (imported lazily)."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "pypdf is required to parse PDF files. Install it with `pip install pypdf`."
        ) from exc

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def parse_docx(data: bytes) -> str:
    """Extract text from a DOCX using python-docx (imported lazily)."""
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "python-docx is required to parse DOCX files. "
            "Install it with `pip install python-docx`."
        ) from exc

    document = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in document.paragraphs)


def parse(filename: str, data: bytes) -> tuple[str, str]:
    """Parse an uploaded contract.

    Returns a (source_format, raw_text) tuple. Raises UnsupportedFormatError for
    unknown extensions.
    """
    fmt = detect_format(filename)
    if fmt == "txt":
        return fmt, parse_txt(data)
    if fmt == "pdf":
        return fmt, parse_pdf(data)
    if fmt == "docx":
        return fmt, parse_docx(data)
    raise UnsupportedFormatError(fmt)  # pragma: no cover - unreachable
