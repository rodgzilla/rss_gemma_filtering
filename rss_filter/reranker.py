"""Re-rank kept FilterResults by relevance score using the LLM.

Strategy
--------
The kept entries (those with keep=True from the two-pass filter) are sent to
the LLM in batches.  For each entry the model assigns a relevance score from
1 to 10.  The entries are then sorted descending by score so the most relevant
items appear at the top of the Obsidian digest.

Prompt format (one numbered line per entry expected back):
    1. 8
    2. 3
    3. 10
    ...

Entries whose score cannot be parsed default to 5 (middle of the range) so
they are not artificially boosted or buried.
"""

from __future__ import annotations

import re
from typing import List, Optional

from tqdm import tqdm

from rss_filter.models import FilterResult

# Matches "1. 8" or "1) 8" with optional whitespace
_SCORE_LINE_RE = re.compile(r"^(\d+)[.)]\s*(\d+)", re.MULTILINE)

DEFAULT_SCORE = 5.0
DEFAULT_RERANK_BATCH_SIZE = 20


def build_rerank_prompt(results: List[FilterResult], interest_profile: str) -> str:
    """Build a prompt asking the LLM to score each entry from 1 to 10."""
    entries_text = ""
    for i, r in enumerate(results, 1):
        snippet = r.entry.summary[:200].strip()
        entries_text += f"{i}. Title: {r.entry.title}\n   Summary: {snippet}\n\n"
    n = len(results)
    return (
        f"You are a relevance scoring assistant. A user's interest profile is given "
        f"below, followed by {n} articles that have already been judged relevant.\n\n"
        f"For EACH article, output ONE numbered line with a relevance score from "
        f"1 (weakly relevant) to 10 (highly relevant).\n"
        f"Output ONLY the {n} numbered lines, nothing else. Example:\n"
        f"  1. 7\n"
        f"  2. 3\n\n"
        f"--- Interest profile ---\n"
        f"{interest_profile}\n\n"
        f"--- Articles ---\n"
        f"{entries_text}"
        f"Score all {n} articles:"
    )


def parse_rerank_response(
    response: str, results: List[FilterResult]
) -> List[FilterResult]:
    """Parse score lines and attach scores to FilterResult objects.

    Returns new FilterResult instances with the score field populated.
    Missing or invalid scores default to DEFAULT_SCORE.
    """
    scores: dict[int, float] = {}
    for m in _SCORE_LINE_RE.finditer(response):
        idx = int(m.group(1))
        raw_score = int(m.group(2))
        # Clamp to [1, 10] then normalise to [0, 1]
        clamped = max(1, min(10, raw_score))
        scores[idx] = clamped / 10.0

    updated = []
    for i, r in enumerate(results, 1):
        score = scores.get(i, DEFAULT_SCORE / 10.0)
        updated.append(
            FilterResult(entry=r.entry, keep=r.keep, reason=r.reason, score=score)
        )
    return updated


def rerank(
    results: List[FilterResult],
    interest_profile: str,
    client,
    model: str,
    temperature: float = 0.1,
    batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
) -> List[FilterResult]:
    """Score and sort a list of kept FilterResults by relevance.

    Args:
        results:          FilterResult objects with keep=True.
        interest_profile: Plain-text interest profile.
        client:           OpenAI-compatible client.
        model:            Model identifier.
        temperature:      Sampling temperature.
        batch_size:       Entries per LLM scoring call.

    Returns:
        The same FilterResult objects, scores populated and sorted
        descending by score (most relevant first).
    """
    if not results:
        return results

    scored: List[FilterResult] = []
    batches = [results[i : i + batch_size] for i in range(0, len(results), batch_size)]

    for batch in tqdm(batches, desc="Re-ranking", unit="batch"):
        prompt = build_rerank_prompt(batch, interest_profile)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        raw = response.choices[0].message.content
        scored.extend(parse_rerank_response(raw, batch))

    return sorted(scored, key=lambda r: r.score or 0.0, reverse=True)
