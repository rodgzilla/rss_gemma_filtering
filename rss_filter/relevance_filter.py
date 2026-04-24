"""Filter RSS entries for relevance using the LLM."""

from __future__ import annotations

import re

from rss_filter.models import FilterResult, RSSEntry

_RESPONSE_RE = re.compile(r"^(yes|no)\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL)


def build_filter_prompt(entry: RSSEntry, interest_profile: str) -> str:
    """Build the per-entry relevance prompt."""
    return (
        f"You are a content filter. Below is a user's interest profile followed by a "
        f"new RSS entry. Reply with exactly one line in the format:\n"
        f"  yes: <one-line reason>\n"
        f"or\n"
        f"  no: <one-line reason>\n\n"
        f"--- Interest profile ---\n"
        f"{interest_profile}\n\n"
        f"--- RSS entry ---\n"
        f"Title: {entry.title}\n"
        f"Summary: {entry.summary}\n\n"
        f"Is this entry relevant to the user's interests? (yes/no + reason)"
    )


def parse_llm_response(response: str) -> FilterResult:
    """Parse the LLM yes/no response into a partial FilterResult (no entry attached)."""
    match = _RESPONSE_RE.match(response.strip())
    if not match:
        return FilterResult(
            entry=None, keep=False, reason="parse error: unexpected response format"
        )  # type: ignore[arg-type]
    verdict = match.group(1).lower()
    reason = match.group(2).strip()
    keep = verdict == "yes"
    return FilterResult(entry=None, keep=keep, reason=reason)  # type: ignore[arg-type]


def filter_entry(
    entry: RSSEntry,
    interest_profile: str,
    client,
    model: str,
    temperature: float = 0.1,
) -> FilterResult:
    """Ask the LLM whether an RSS entry is relevant and return a FilterResult."""
    prompt = build_filter_prompt(entry, interest_profile)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    raw = response.choices[0].message.content
    partial = parse_llm_response(raw)
    return FilterResult(entry=entry, keep=partial.keep, reason=partial.reason)
