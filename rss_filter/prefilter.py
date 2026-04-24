"""Keyword-based pre-filter to reduce LLM calls.

Extracts keywords from the interest profile and scores each entry by how many
keywords appear in its title + summary.  Entries above a configurable threshold
are forwarded to the LLM; the rest are dropped without an LLM call.
"""

from __future__ import annotations

import re
import string
from typing import List, Set, Tuple

from rss_filter.models import RSSEntry

# Common English stop-words to ignore when extracting keywords
_STOPWORDS: Set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "need",
    "dare",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "as",
    "up",
    "out",
    "about",
    "into",
    "through",
    "than",
    "then",
    "so",
    "if",
    "not",
    "no",
    "new",
    "use",
    "using",
    "used",
    "based",
    "paper",
    "work",
    "model",
    "approach",
    "method",
    "results",
    "show",
    "shows",
    "also",
    "which",
    "we",
    "our",
    "their",
    "they",
    "he",
    "she",
    "you",
    "i",
    "my",
    "your",
}

_PUNCT_RE = re.compile(r"[^\w\s]")


def _tokenise(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace, drop stop-words."""
    text = _PUNCT_RE.sub(" ", text.lower())
    return [t for t in text.split() if len(t) > 2 and t not in _STOPWORDS]


def extract_keywords(interest_profile: str, top_n: int = 60) -> List[str]:
    """Return the top_n most-frequent content words in the interest profile."""
    tokens = _tokenise(interest_profile)
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    ranked = sorted(freq, key=lambda w: -freq[w])
    return ranked[:top_n]


def score_entry(entry: RSSEntry, keywords: List[str]) -> int:
    """Count how many keywords appear in the entry's title + summary."""
    text = _tokenise(f"{entry.title} {entry.summary}")
    text_set = set(text)
    return sum(1 for kw in keywords if kw in text_set)


def prefilter_entries(
    entries: List[RSSEntry],
    interest_profile: str,
    top_n_keywords: int = 60,
    min_score: int = 1,
) -> Tuple[List[RSSEntry], List[RSSEntry]]:
    """Split entries into (passed, rejected) based on keyword overlap.

    Args:
        entries: All candidate entries.
        interest_profile: Plain-text profile from the profiler.
        top_n_keywords: How many top keywords to extract from the profile.
        min_score: Entries scoring below this are rejected without LLM call.

    Returns:
        (passed, rejected) — passed entries go on to LLM filtering.
    """
    keywords = extract_keywords(interest_profile, top_n=top_n_keywords)
    passed, rejected = [], []
    for entry in entries:
        if score_entry(entry, keywords) >= min_score:
            passed.append(entry)
        else:
            rejected.append(entry)
    return passed, rejected
