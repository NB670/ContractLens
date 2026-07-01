"""Canonical clause categories.

These follow the CUAD (Contract Understanding Atticus Dataset) taxonomy, trimmed
to the high-value categories called out in the project plan (confidentiality,
termination, liability, indemnification, intellectual property, ...). The
rule-based classifier keys its keyword sets off these names; a LegalBERT backend
can later map model labels onto the same set so downstream tasks stay stable.
"""

from __future__ import annotations

UNCLASSIFIED = "Unclassified"

# category -> indicative keywords / phrases (lowercased)
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Confidentiality": [
        "confidential",
        "confidentiality",
        "non-disclosure",
        "nondisclosure",
        "proprietary information",
        "trade secret",
    ],
    "Termination": [
        "terminate",
        "termination",
        "expiration of this agreement",
        "for convenience",
        "notice of termination",
    ],
    "Liability": [
        "limitation of liability",
        "liable",
        "liability",
        "consequential damages",
        "in no event",
    ],
    "Indemnification": [
        "indemnify",
        "indemnification",
        "hold harmless",
        "defend",
    ],
    "Intellectual Property": [
        "intellectual property",
        "copyright",
        "patent",
        "trademark",
        "work product",
        "ownership of",
        "license",
    ],
    "Governing Law": [
        "governing law",
        "governed by the laws",
        "jurisdiction",
        "venue",
    ],
    "Payment Terms": [
        "payment",
        "fees",
        "invoice",
        "compensation",
        "net 30",
    ],
    "Warranty": [
        "warranty",
        "warrants",
        "as is",
        "merchantability",
        "fitness for a particular purpose",
    ],
    "Assignment": [
        "assign",
        "assignment",
        "successors and assigns",
    ],
    "Force Majeure": [
        "force majeure",
        "act of god",
        "beyond the reasonable control",
    ],
}

# Stable ordering used by the visualization view.
CATEGORIES: list[str] = list(CATEGORY_KEYWORDS.keys())
