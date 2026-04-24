"""Parse Obsidian daily notes and extract Reading / Arxiv entries."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from rss_filter.models import NoteEntry

# Matches a markdown link at the start of a list item:
#   - [Title](url) -> optional tags
_LINK_RE = re.compile(r"^\s*-\s+\[([^\]]+)\]\(([^)]+)\)")

# Date filename pattern YYYY-MM-DD.md
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Section heading patterns
_READING_RE = re.compile(r"^##\s+Reading\s*$", re.IGNORECASE)
_ARXIV_RE = re.compile(r"^##\s+\[\[Arxiv\]\]\s+monitoring\s*$", re.IGNORECASE)
_ANY_H2_RE = re.compile(r"^##\s+")


def _strip_frontmatter(lines: list[str]) -> list[str]:
    """Remove YAML frontmatter delimited by --- lines."""
    if not lines or lines[0].strip() != "---":
        return lines
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return lines[i + 1 :]
    return lines


def parse_note(path: Path, date: str) -> List[NoteEntry]:
    """Parse a single daily note file and return all NoteEntry objects."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    lines = _strip_frontmatter(lines)

    entries: List[NoteEntry] = []
    current_source: str | None = None
    current_url: str | None = None
    current_context_lines: list[str] = []

    def _flush():
        if current_url is not None and current_source is not None:
            entries.append(
                NoteEntry(
                    url=current_url,
                    context=" ".join(current_context_lines).strip(),
                    source=current_source,
                    date=date,
                )
            )

    for line in lines:
        # Detect section headings
        if _READING_RE.match(line):
            _flush()
            current_source = "reading"
            current_url = None
            current_context_lines = []
            continue
        if _ARXIV_RE.match(line):
            _flush()
            current_source = "arxiv"
            current_url = None
            current_context_lines = []
            continue
        # Any other h2 heading ends current section
        if _ANY_H2_RE.match(line):
            _flush()
            current_source = None
            current_url = None
            current_context_lines = []
            continue

        if current_source is None:
            continue

        link_match = _LINK_RE.match(line)
        if link_match:
            # Save previous entry before starting new one
            _flush()
            title = link_match.group(1)
            url = link_match.group(2)
            current_url = url
            current_context_lines = [title]
        elif current_url is not None and line.strip():
            # Indented context line — strip leading whitespace / tab
            current_context_lines.append(line.strip())

    _flush()
    return entries


def parse_vault(daily_notes_path: Path) -> List[NoteEntry]:
    """Scan a folder of YYYY-MM-DD.md files and return all NoteEntry objects."""
    entries: List[NoteEntry] = []
    for md_file in sorted(daily_notes_path.glob("*.md")):
        stem = md_file.stem
        if not _DATE_RE.match(stem):
            continue
        entries.extend(parse_note(md_file, date=stem))
    return entries
