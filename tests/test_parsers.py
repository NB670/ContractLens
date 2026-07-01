"""Tests for the document parsing layer (TXT path; PDF/DOCX need optional deps)."""

import pytest

from app.ingestion.parsers import (
    UnsupportedFormatError,
    detect_format,
    parse,
    parse_txt,
)


def test_detect_format():
    assert detect_format("contract.pdf") == "pdf"
    assert detect_format("contract.DOCX") == "docx"
    assert detect_format("contract.txt") == "txt"


def test_detect_format_unsupported():
    with pytest.raises(UnsupportedFormatError):
        detect_format("contract.rtf")


def test_parse_txt_roundtrip():
    text = "Hello, contract."
    assert parse_txt(text.encode("utf-8")) == text


def test_parse_dispatches_txt():
    fmt, text = parse("nda.txt", b"1. Confidentiality\nKeep it secret.")
    assert fmt == "txt"
    assert "Confidentiality" in text
