"""Build and cache an interest profile from Obsidian daily notes."""

from __future__ import annotations

from pathlib import Path
from typing import List

from rss_filter.models import NoteEntry


def build_profile_prompt(entries: List[NoteEntry]) -> str:
    """Build the LLM prompt used to generate an interest profile."""
    lines = [
        "Below is a collection of articles and research papers that a person has saved "
        "as interesting over time. Each item includes a URL and the context or summary "
        "they noted.",
        "",
        "Please analyse these items and write a concise interest profile (a few "
        "paragraphs) describing the recurring topics, themes, and subjects this person "
        "finds interesting. This profile will be used to filter new RSS feed entries.",
        "",
        "--- Saved items ---",
        "",
    ]
    for entry in entries:
        lines.append(f"Source: {entry.source}  |  Date: {entry.date}")
        lines.append(f"URL: {entry.url}")
        lines.append(f"Context: {entry.context}")
        lines.append("")
    lines.append("--- End of items ---")
    lines.append("")
    lines.append("Interest profile:")
    return "\n".join(lines)


def save_profile(profile_text: str, profile_path: Path) -> None:
    """Save the profile text to disk."""
    profile_path.write_text(profile_text, encoding="utf-8")


def load_or_build_profile(
    entries: List[NoteEntry],
    profile_path: Path,
    client,
    model: str,
    rebuild: bool = False,
) -> str:
    """Return the interest profile, loading from cache or calling the LLM."""
    if not rebuild and profile_path.exists():
        return profile_path.read_text(encoding="utf-8")

    prompt = build_profile_prompt(entries)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    profile_text = response.choices[0].message.content
    save_profile(profile_text, profile_path)
    return profile_text
