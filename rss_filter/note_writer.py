"""Write filtered RSS digest notes into the Obsidian vault."""

from __future__ import annotations

from pathlib import Path
from typing import List

from rss_filter.models import FilterResult

_OUTPUT_FOLDER = "Filtered feed"
_DEFAULT_TAGS = ["rss", "embedding"]


def get_output_path(vault_path: Path, date: str) -> Path:
    """Return the Path where the output note should be written."""
    return vault_path / _OUTPUT_FOLDER / f"RSS-{date}.md"


def _render_exemplars(exemplars: list[dict]) -> list[str]:
    """Return indented lines listing k-NN exemplars with their similarity scores."""
    if not exemplars:
        return []
    lines = []
    for ex in exemplars:
        label = ex.get("title") or ex["text"]
        lines.append(f"  - `{ex['score']:.4f}` {label}")
    return lines


def _render_frontmatter(tags: list[str]) -> str:
    """Return a YAML frontmatter block with the given tags."""
    tag_lines = "\n".join(f"  - {t}" for t in tags)
    return f"---\ntags:\n{tag_lines}\n---\n"


def render_reading_section(results: List[FilterResult]) -> str:
    """Render the ## Reading section from a list of FilterResult objects."""
    if not results:
        return ""
    lines = ["## Reading", ""]
    for r in results:
        score_str = f" *(score: {r.score:.4f})*" if r.score is not None else ""
        lines.append(f"- [{r.entry.title}]({r.entry.url}){score_str}")
        lines.extend(_render_exemplars(r.exemplars))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_arxiv_section(results: List[FilterResult]) -> str:
    """Render the ## Arxiv monitoring section from a list of FilterResult objects."""
    if not results:
        return ""
    lines = ["## Arxiv monitoring", ""]
    for r in results:
        score_str = f" *(score: {r.score:.4f})*" if r.score is not None else ""
        lines.append(f"- [{r.entry.title}]({r.entry.url}){score_str}")
        lines.extend(_render_exemplars(r.exemplars))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_note_content(
    reading_results: List[FilterResult],
    arxiv_results: List[FilterResult],
    date: str,
    tags: list[str] | None = None,
    viz_filename: str | None = None,
) -> str:
    """Build the full note content string.

    Args:
        tags:         YAML frontmatter tags (default: ["rss", "embedding"]).
        viz_filename: Bare filename of the companion HTML visualisation (e.g.
                      "RSS-2026-04-28-scores.html").  When provided an iframe
                      block is appended so the chart is embedded in the note.
                      The path is kept as a bare filename (no directory prefix)
                      so it works on any machine where both files sit in the
                      same Obsidian folder.
    """
    effective_tags = tags if tags is not None else _DEFAULT_TAGS
    parts = [_render_frontmatter(effective_tags), f"# RSS Digest — {date}", ""]
    reading_section = render_reading_section(reading_results)
    if reading_section:
        parts.append(reading_section)
    arxiv_section = render_arxiv_section(arxiv_results)
    if arxiv_section:
        parts.append(arxiv_section)
    if viz_filename:
        parts.append(
            f'<iframe src="{viz_filename}" allow="fullscreen" allowfullscreen="" '
            f'style="height:100%;width:100%; aspect-ratio: 16 / 9; "></iframe>\n'
        )
    return "\n".join(parts)


def write_note(
    vault_path: Path,
    reading_results: List[FilterResult],
    arxiv_results: List[FilterResult],
    date: str,
    viz_filename: str | None = None,
) -> None:
    """Write the filtered digest note. Skips writing if both lists are empty."""
    if not reading_results and not arxiv_results:
        return
    output_path = get_output_path(vault_path, date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = build_note_content(
        reading_results, arxiv_results, date, viz_filename=viz_filename
    )
    output_path.write_text(content, encoding="utf-8")
