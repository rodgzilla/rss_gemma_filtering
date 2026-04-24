"""Tests for rss_filter.notes_parser"""

import textwrap
from pathlib import Path

import pytest

from rss_filter.notes_parser import parse_note, parse_vault


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

READING_NOTE = textwrap.dedent("""\
    ---
    tags:
      - daily
    ---
    [[Daily note]]

    ## Reading

    - [Probabilistic Tic-Tac-Toe](https://www.csun.io/2024/06/08/probabilistic-tic-tac-toe.html) -> [[Game]], [[Probability]]
    \tI spend a lot of time thinking about the role of random chance in our lives.
    - [Another Article](https://example.com/article) -> [[Tech]]
    \tShort context here.
""")

ARXIV_NOTE = textwrap.dedent("""\
    ---
    tags:
      - daily
    ---
    [[Daily note]]

    ## [[Arxiv]] monitoring

    - [Masking Improves Contrastive SSL](https://arxiv.org/abs/2309.12757) -> [[SSL]], [[CNN]]
    \tWhile image data starts to enjoy the simple-but-effective SSL scheme...
""")

BOTH_NOTE = textwrap.dedent("""\
    ---
    tags:
      - daily
    ---
    [[Daily note]]

    ## Reading

    - [Some Blog Post](https://blog.example.com/post) -> [[Tech]]
    \tThis is a blog post about technology.

    ## [[Arxiv]] monitoring

    - [A Research Paper](https://arxiv.org/abs/1234.56789) -> [[ML]]
    \tThis paper explores machine learning.
""")

EMPTY_NOTE = textwrap.dedent("""\
    ---
    tags:
      - daily
    weight: 80.0
    ---
    [[Daily note]]
""")

FRONTMATTER_ONLY_NOTE = textwrap.dedent("""\
    ---
    tags:
      - daily
    ---
""")


# ---------------------------------------------------------------------------
# parse_note tests
# ---------------------------------------------------------------------------


def test_parse_note_reading_section_returns_reading_entries(tmp_path):
    note_file = tmp_path / "2024-06-11.md"
    note_file.write_text(READING_NOTE)

    entries = parse_note(note_file, date="2024-06-11")

    assert len(entries) == 2
    assert entries[0].source == "reading"
    assert entries[1].source == "reading"


def test_parse_note_reading_section_extracts_url(tmp_path):
    note_file = tmp_path / "2024-06-11.md"
    note_file.write_text(READING_NOTE)

    entries = parse_note(note_file, date="2024-06-11")

    assert (
        entries[0].url
        == "https://www.csun.io/2024/06/08/probabilistic-tic-tac-toe.html"
    )
    assert entries[1].url == "https://example.com/article"


def test_parse_note_reading_section_extracts_context(tmp_path):
    note_file = tmp_path / "2024-06-11.md"
    note_file.write_text(READING_NOTE)

    entries = parse_note(note_file, date="2024-06-11")

    assert "random chance" in entries[0].context
    assert "Short context" in entries[1].context


def test_parse_note_reading_section_stores_date(tmp_path):
    note_file = tmp_path / "2024-06-11.md"
    note_file.write_text(READING_NOTE)

    entries = parse_note(note_file, date="2024-06-11")

    assert entries[0].date == "2024-06-11"


def test_parse_note_arxiv_section_returns_arxiv_entries(tmp_path):
    note_file = tmp_path / "2024-06-11.md"
    note_file.write_text(ARXIV_NOTE)

    entries = parse_note(note_file, date="2024-06-11")

    assert len(entries) == 1
    assert entries[0].source == "arxiv"
    assert entries[0].url == "https://arxiv.org/abs/2309.12757"


def test_parse_note_both_sections_returns_all_entries(tmp_path):
    note_file = tmp_path / "2024-06-11.md"
    note_file.write_text(BOTH_NOTE)

    entries = parse_note(note_file, date="2024-06-11")

    sources = [e.source for e in entries]
    assert "reading" in sources
    assert "arxiv" in sources
    assert len(entries) == 2


def test_parse_note_no_sections_returns_empty(tmp_path):
    note_file = tmp_path / "2024-06-11.md"
    note_file.write_text(EMPTY_NOTE)

    entries = parse_note(note_file, date="2024-06-11")

    assert entries == []


def test_parse_note_frontmatter_only_returns_empty(tmp_path):
    note_file = tmp_path / "2024-06-11.md"
    note_file.write_text(FRONTMATTER_ONLY_NOTE)

    entries = parse_note(note_file, date="2024-06-11")

    assert entries == []


# ---------------------------------------------------------------------------
# parse_vault tests
# ---------------------------------------------------------------------------


def test_parse_vault_returns_entries_from_all_notes(tmp_path):
    (tmp_path / "2024-06-11.md").write_text(READING_NOTE)
    (tmp_path / "2025-04-07.md").write_text(ARXIV_NOTE)
    (tmp_path / "2026-01-02.md").write_text(EMPTY_NOTE)

    entries = parse_vault(tmp_path)

    assert len(entries) == 3  # 2 from reading note + 1 from arxiv note


def test_parse_vault_tags_entries_with_correct_dates(tmp_path):
    (tmp_path / "2024-06-11.md").write_text(READING_NOTE)
    (tmp_path / "2025-04-07.md").write_text(ARXIV_NOTE)

    entries = parse_vault(tmp_path)

    dates = {e.date for e in entries}
    assert "2024-06-11" in dates
    assert "2025-04-07" in dates


def test_parse_vault_ignores_non_date_files(tmp_path):
    (tmp_path / "2024-06-11.md").write_text(READING_NOTE)
    (tmp_path / "README.md").write_text("# Not a daily note\n")

    entries = parse_vault(tmp_path)

    assert len(entries) == 2  # only from 2024-06-11.md


def test_parse_vault_uses_mock_vault(tmp_path):
    """Smoke test against the real mock_vault provided in the repo."""
    repo_root = Path(__file__).parent.parent
    vault_daily = repo_root / "mock_vault" / "Daily notes"

    entries = parse_vault(vault_daily)

    # We know the mock vault has notes with Reading and Arxiv sections
    assert len(entries) > 0
    sources = {e.source for e in entries}
    assert "reading" in sources
    assert "arxiv" in sources
