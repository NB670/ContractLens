"""Segment a contract's raw text into individual clauses / sections.

This is intentionally a transparent, heuristic-first segmenter (no model
download required). It recognizes the common ways contract sections are
delimited:

  - numbered headings:        "1.", "1.1", "12.3.4", "Section 4", "Article V"
  - ALL-CAPS heading lines:   "CONFIDENTIALITY"
  - title-case heading lines: "Limitation of Liability"

Each detected boundary starts a new clause; text between boundaries is the clause
body. Character offsets into the original document are preserved so the UI can
link a clause back to its source.
"""

from __future__ import annotations

import re

from app.models.contract import Clause

# A line that is purely a numbered/lettered section marker, optionally followed
# by a short heading on the same line.
_NUMBERED_HEADING = re.compile(
    r"^\s*(?:section|article)?\s*"
    r"(?:\d+(?:\.\d+)*|[IVXLCDM]+|[A-Z])"
    r"[\.\)]?\s+\S",
    re.IGNORECASE,
)

# A line that is a standalone heading (ALL CAPS, or short Title Case, no
# sentence-ending punctuation).
_CAPS_HEADING = re.compile(r"^\s*[A-Z][A-Z0-9 ,/&'\-]{2,60}\s*$")
_TITLE_HEADING = re.compile(r"^\s*(?:[A-Z][a-zA-Z]+)(?:\s+(?:of|and|the|[A-Z][a-zA-Z]+)){0,6}\s*$")


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    if stripped.endswith((".", ";", ":")) and not _NUMBERED_HEADING.match(stripped):
        # Looks like a sentence, not a heading.
        if not _NUMBERED_HEADING.match(stripped):
            return False
    if _NUMBERED_HEADING.match(stripped):
        return True
    if _CAPS_HEADING.match(stripped):
        return True
    if _TITLE_HEADING.match(stripped) and len(stripped.split()) <= 7:
        return True
    return False


def _extract_heading(block_lines: list[str]) -> tuple[str | None, str]:
    """Given the lines of one clause block, split off a leading heading."""
    if not block_lines:
        return None, ""
    first = block_lines[0].strip()
    # If the first line is short and the next lines exist, treat it as a heading.
    if len(block_lines) > 1 and (_CAPS_HEADING.match(first) or _TITLE_HEADING.match(first)):
        body = "\n".join(line for line in block_lines[1:]).strip()
        return first, body
    return None, "\n".join(block_lines).strip()


def segment(text: str) -> list[Clause]:
    """Split contract text into Clause objects with offsets preserved."""
    if not text or not text.strip():
        return []

    lines = text.splitlines(keepends=True)

    # Compute char offset of the start of each line.
    offsets: list[int] = []
    running = 0
    for line in lines:
        offsets.append(running)
        running += len(line)

    # Find indices of lines that begin a new clause.
    boundaries = [i for i, line in enumerate(lines) if _is_heading(line)]
    if not boundaries or boundaries[0] != 0:
        boundaries = [0] + boundaries

    clauses: list[Clause] = []
    for idx, start_line in enumerate(boundaries):
        end_line = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(lines)
        block_lines = [lines[i].rstrip("\n") for i in range(start_line, end_line)]
        block_text = "".join(lines[start_line:end_line]).strip()
        if not block_text:
            continue

        heading, _body = _extract_heading(block_lines)
        start_offset = offsets[start_line]
        end_offset = (
            offsets[end_line] if end_line < len(offsets) else running
        )

        clauses.append(
            Clause(
                index=len(clauses),
                heading=heading,
                text=block_text,
                start_offset=start_offset,
                end_offset=end_offset,
            )
        )

    return clauses
