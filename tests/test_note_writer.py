"""Tests for rss_filter.note_writer"""

from pathlib import Path

import pytest

from rss_filter.models import FilterResult, RSSEntry
from rss_filter.note_writer import (
    build_note_content,
    get_output_path,
    render_arxiv_section,
    render_reading_section,
    write_note,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    title: str,
    url: str,
    is_arxiv: bool,
    reason: str = "Relevant topic",
    exemplars: list[dict] | None = None,
) -> FilterResult:
    entry = RSSEntry(
        title=title,
        url=url,
        summary="Some summary.",
        feed_name="Test Feed",
        is_arxiv=is_arxiv,
        guid=url,
    )
    return FilterResult(
        entry=entry,
        keep=True,
        reason=reason,
        exemplars=exemplars or [],
    )


# ---------------------------------------------------------------------------
# get_output_path
# ---------------------------------------------------------------------------


def test_get_output_path_returns_correct_path(tmp_path):
    result = get_output_path(tmp_path, "2025-01-15")
    assert result == tmp_path / "Filtered feed" / "RSS-2025-01-15.md"


def test_get_output_path_uses_filtered_feed_subfolder(tmp_path):
    result = get_output_path(tmp_path, "2026-04-24")
    assert result.parent.name == "Filtered feed"


# ---------------------------------------------------------------------------
# render_reading_section
# ---------------------------------------------------------------------------


def test_render_reading_section_contains_title_and_url():
    results = [_make_result("Cool Post", "https://example.com/post", is_arxiv=False)]
    section = render_reading_section(results)

    assert "Cool Post" in section
    assert "https://example.com/post" in section


def test_render_reading_section_contains_exemplar():
    results = [
        _make_result(
            "Post",
            "https://x.com",
            is_arxiv=False,
            exemplars=[{"text": "vault note about games", "score": 0.87}],
        )
    ]
    section = render_reading_section(results)

    assert "vault note about games" in section
    assert "0.8700" in section


def test_render_reading_section_has_heading():
    results = [_make_result("A", "https://a.com", is_arxiv=False)]
    section = render_reading_section(results)

    assert "## Reading" in section


def test_render_reading_section_empty_returns_empty_string():
    assert render_reading_section([]) == ""


# ---------------------------------------------------------------------------
# render_arxiv_section
# ---------------------------------------------------------------------------


def test_render_arxiv_section_contains_title_and_url():
    results = [
        _make_result("A Paper", "https://arxiv.org/abs/1234.5678", is_arxiv=True)
    ]
    section = render_arxiv_section(results)

    assert "A Paper" in section
    assert "https://arxiv.org/abs/1234.5678" in section


def test_render_arxiv_section_has_heading():
    results = [_make_result("P", "https://arxiv.org/abs/1", is_arxiv=True)]
    section = render_arxiv_section(results)

    assert "## Arxiv monitoring" in section


def test_render_arxiv_section_empty_returns_empty_string():
    assert render_arxiv_section([]) == ""


# ---------------------------------------------------------------------------
# build_note_content
# ---------------------------------------------------------------------------


def test_build_note_content_includes_date_in_title():
    content = build_note_content([], [], date="2025-01-15")
    assert "2025-01-15" in content


def test_build_note_content_includes_both_sections():
    reading = [_make_result("Blog", "https://blog.com", is_arxiv=False)]
    arxiv = [_make_result("Paper", "https://arxiv.org/abs/1", is_arxiv=True)]
    content = build_note_content(reading, arxiv, date="2025-01-15")

    assert "## Reading" in content
    assert "## Arxiv monitoring" in content


def test_build_note_content_omits_empty_sections():
    reading = [_make_result("Blog", "https://blog.com", is_arxiv=False)]
    content = build_note_content(reading, [], date="2025-01-15")

    assert "## Reading" in content
    assert "## Arxiv monitoring" not in content


def test_build_note_content_has_yaml_frontmatter():
    content = build_note_content([], [], date="2025-01-15")
    assert content.startswith("---\n")
    assert "tags:" in content


def test_build_note_content_frontmatter_contains_rss_and_embedding_tags():
    content = build_note_content([], [], date="2025-01-15")
    assert "- rss" in content
    assert "- embedding" in content


def test_build_note_content_frontmatter_has_cssclasses_tracker():
    content = build_note_content([], [], date="2025-01-15")
    assert "cssclasses:" in content
    assert "- tracker" in content


def test_build_note_content_frontmatter_has_up_daily_note_link():
    content = build_note_content([], [], date="2025-01-15")
    assert "up:" in content
    assert "[[2025-01-15]]" in content


def test_build_note_content_frontmatter_closes_before_heading():
    """The closing --- must appear before the # heading."""
    content = build_note_content([], [], date="2025-01-15")
    close_pos = content.index("---\n", 1)  # second occurrence closes frontmatter
    heading_pos = content.index("# RSS Digest")
    assert close_pos < heading_pos


def test_build_note_content_no_viz_block_when_viz_filename_not_given():
    content = build_note_content([], [], date="2025-01-15")
    assert "dataviewjs" not in content
    assert "<iframe" not in content


def test_build_note_content_includes_viz_block_when_viz_filename_given():
    content = build_note_content(
        [], [], date="2025-01-15", viz_filename="RSS-2025-01-15-scores.html"
    )
    assert "dataviewjs" in content


def test_build_note_content_viz_block_uses_getResourcePath():
    """Block must use app.vault.getResourcePath for portable app:// resolution."""
    content = build_note_content(
        [], [], date="2025-01-15", viz_filename="RSS-2025-01-15-scores.html"
    )
    assert "getResourcePath" in content
    assert "getAbstractFileByPath" in content


def test_build_note_content_viz_block_derives_filename_from_note_name():
    """Filename must be derived from dv.current().file.name, not hardcoded."""
    content = build_note_content(
        [], [], date="2025-01-15", viz_filename="RSS-2025-01-15-scores.html"
    )
    assert "dv.current().file.name" in content


def test_build_note_content_viz_block_appears_before_content():
    """DataviewJS viz block must come before the article sections."""
    reading = [
        FilterResult(
            entry=RSSEntry(
                title="Some Article",
                url="http://x.com",
                summary="s",
                feed_name="F",
                is_arxiv=False,
                guid="g1",
            ),
            keep=True,
            reason="ok",
            score=0.9,
        )
    ]
    content = build_note_content(
        reading, [], date="2025-01-15", viz_filename="RSS-2025-01-15-scores.html"
    )
    viz_pos = content.index("dataviewjs")
    heading_pos = content.index("## Reading")
    assert viz_pos < heading_pos


# ---------------------------------------------------------------------------
# write_note
# ---------------------------------------------------------------------------


def test_write_note_creates_file(tmp_path):
    reading = [_make_result("Blog", "https://blog.com", is_arxiv=False)]
    write_note(tmp_path, reading, [], date="2025-01-15")

    output_path = tmp_path / "Filtered feed" / "RSS-2025-01-15.md"
    assert output_path.exists()


def test_write_note_creates_parent_directory(tmp_path):
    reading = [_make_result("Blog", "https://blog.com", is_arxiv=False)]
    write_note(tmp_path, reading, [], date="2025-01-15")

    assert (tmp_path / "Filtered feed").is_dir()


def test_write_note_skips_when_both_lists_empty(tmp_path):
    write_note(tmp_path, [], [], date="2025-01-15")

    output_path = tmp_path / "Filtered feed" / "RSS-2025-01-15.md"
    assert not output_path.exists()


def test_write_note_file_contains_correct_content(tmp_path):
    reading = [
        _make_result(
            "Cool Blog",
            "https://cool.com",
            is_arxiv=False,
            exemplars=[{"text": "Cool tech article", "score": 0.91}],
        )
    ]
    write_note(tmp_path, reading, [], date="2025-01-15")

    content = (tmp_path / "Filtered feed" / "RSS-2025-01-15.md").read_text()
    assert "Cool Blog" in content
    assert "https://cool.com" in content
    assert "Cool tech article" in content


def test_write_note_passes_viz_filename_to_content(tmp_path):
    reading = [_make_result("Blog", "https://blog.com", is_arxiv=False)]
    write_note(
        tmp_path,
        reading,
        [],
        date="2025-01-15",
        viz_filename="RSS-2025-01-15-scores.html",
    )

    content = (tmp_path / "Filtered feed" / "RSS-2025-01-15.md").read_text()
    assert "dataviewjs" in content
    assert "dv.current().file.name" in content


def test_write_note_no_viz_block_without_viz_filename(tmp_path):
    reading = [_make_result("Blog", "https://blog.com", is_arxiv=False)]
    write_note(tmp_path, reading, [], date="2025-01-15")

    content = (tmp_path / "Filtered feed" / "RSS-2025-01-15.md").read_text()
    assert "dataviewjs" not in content


# ---------------------------------------------------------------------------
# postMessage click listener block
# ---------------------------------------------------------------------------


def test_build_note_content_includes_click_listener_when_viz_given():
    """When viz_filename is provided the note must include the postMessage listener block."""
    content = build_note_content(
        [], [], date="2025-01-15", viz_filename="RSS-2025-01-15-scores.html"
    )
    assert "rss-viz-click" in content
    assert "addEventListener" in content


def test_build_note_content_no_click_listener_without_viz():
    """Without a viz_filename, no postMessage listener block should appear."""
    content = build_note_content([], [], date="2025-01-15")
    assert "rss-viz-click" not in content


def test_build_note_content_click_listener_after_viz_block():
    """The postMessage listener block must appear after the iframe viz block."""
    content = build_note_content(
        [], [], date="2025-01-15", viz_filename="RSS-2025-01-15-scores.html"
    )
    viz_pos = content.index("getResourcePath")
    listener_pos = content.index("rss-viz-click")
    assert listener_pos > viz_pos


def test_build_note_content_click_listener_renders_link():
    """The listener block must contain code to render a link element."""
    content = build_note_content(
        [], [], date="2025-01-15", viz_filename="RSS-2025-01-15-scores.html"
    )
    assert "createEl('a'" in content
