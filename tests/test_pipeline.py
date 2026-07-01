"""Tests for contract-type and party detection in the ingestion pipeline."""

from app.pipeline import _detect_parties, _detect_type


def test_detect_type_new_patterns():
    assert _detect_type("This Consulting Agreement is entered into by the parties.") == "Consulting Agreement"
    assert _detect_type("This Supply Agreement governs the sale of goods.") == "Supply Agreement"
    assert _detect_type("This Reseller Agreement authorizes resale of products.") == "Reseller Agreement"
    assert _detect_type("This Franchise Agreement grants a franchise to operate.") == "Franchise Agreement"
    assert _detect_type("This Joint Venture Agreement forms a joint venture between the parties.") == "Joint Venture Agreement"


def test_detect_parties_two_party_by_and_between():
    text = (
        "This Agreement is made and entered into by and between Acme "
        "Corporation and Globex LLC, dated as of January 1, 2026."
    )
    assert _detect_parties(text) == ["Acme Corporation", "Globex LLC"]


def test_detect_parties_two_party_plain_between():
    text = "This Agreement is entered into between Acme Corporation and Globex LLC."
    assert _detect_parties(text) == ["Acme Corporation", "Globex LLC"]


def test_detect_parties_n_party_by_and_among():
    text = (
        "This Agreement is entered into by and among Acme Corporation, "
        "Beta Industries, and Globex LLC, dated as of January 1, 2026."
    )
    assert _detect_parties(text) == ["Acme Corporation", "Beta Industries", "Globex LLC"]


def test_detect_parties_no_match_returns_empty_list():
    assert _detect_parties("This document has no party preamble at all.") == []
