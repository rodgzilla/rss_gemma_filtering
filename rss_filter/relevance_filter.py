"""Filter RSS entries for relevance using the LLM."""

from __future__ import annotations

import re
from typing import List

from rss_filter.models import FilterResult, RSSEntry

_RESPONSE_RE = re.compile(r"^(yes|no)\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL)
# Matches a numbered line like "1. yes: reason" or "1) no: reason"
_BATCH_LINE_RE = re.compile(r"^(\d+)[.)]\s*(yes|no)\s*:\s*(.+)$", re.IGNORECASE)


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


def build_batch_filter_prompt(entries: List[RSSEntry], interest_profile: str) -> str:
    """Build a prompt that asks the LLM to evaluate multiple entries at once.

    Expected response format (one line per entry, in order):
        1. yes: <reason>
        2. no: <reason>
        ...
    """
    entries_text = ""
    for i, entry in enumerate(entries, 1):
        entries_text += (
            f"{i}. Title: {entry.title}\n   Summary: {entry.summary[:300]}\n\n"
        )
    return (
        f"You are a content filter. Below is a user's interest profile followed by "
        f"{len(entries)} RSS entries.\n\n"
        f"For EACH entry reply with exactly one numbered line:\n"
        f"  <n>. yes: <one-line reason>\n"
        f"or\n"
        f"  <n>. no: <one-line reason>\n\n"
        f"Output ONLY the {len(entries)} numbered lines, nothing else.\n\n"
        f"--- Interest profile ---\n"
        f"{interest_profile}\n\n"
        f"--- Entries ---\n"
        f"{entries_text}"
        f"Evaluate all {len(entries)} entries:"
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


def parse_batch_response(response: str, entries: List[RSSEntry]) -> List[FilterResult]:
    """Parse a numbered multi-entry LLM response into FilterResult objects.

    If a line is missing or malformed, that entry defaults to keep=False.
    """
    # Index parsed lines by their number
    parsed: dict[int, tuple[bool, str]] = {}
    for line in response.strip().splitlines():
        m = _BATCH_LINE_RE.match(line.strip())
        if m:
            idx = int(m.group(1))
            keep = m.group(2).lower() == "yes"
            reason = m.group(3).strip()
            parsed[idx] = (keep, reason)

    results = []
    for i, entry in enumerate(entries, 1):
        if i in parsed:
            keep, reason = parsed[i]
        else:
            keep, reason = False, "parse error: no response for this entry"
        results.append(FilterResult(entry=entry, keep=keep, reason=reason))
    return results


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


def filter_entries_batch(
    entries: List[RSSEntry],
    interest_profile: str,
    client,
    model: str,
    temperature: float = 0.1,
    batch_size: int = 10,
) -> List[FilterResult]:
    """Filter a list of entries using batched LLM calls.

    Entries are split into chunks of batch_size.  Each chunk is sent as a
    single LLM call.  Results are collected and returned in input order.
    """
    results: List[FilterResult] = []
    for start in range(0, len(entries), batch_size):
        batch = entries[start : start + batch_size]
        prompt = build_batch_filter_prompt(batch, interest_profile)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        raw = response.choices[0].message.content
        results.extend(parse_batch_response(raw, batch))
    return results
