"""Tests for rss_filter.prefilter"""

from rss_filter.models import RSSEntry
from rss_filter.prefilter import extract_keywords, prefilter_entries, score_entry


def _entry(title: str, summary: str = "", guid: str = "g") -> RSSEntry:
    return RSSEntry(
        title=title,
        url="https://x.com",
        summary=summary,
        feed_name="F",
        is_arxiv=False,
        guid=guid,
    )


# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------


def test_extract_keywords_returns_list():
    profile = "machine learning deep learning neural networks transformers"
    kws = extract_keywords(profile, top_n=10)
    assert isinstance(kws, list)
    assert len(kws) <= 10


def test_extract_keywords_drops_stopwords():
    profile = "the and or but machine learning"
    kws = extract_keywords(profile, top_n=20)
    assert "the" not in kws
    assert "and" not in kws


def test_extract_keywords_returns_most_frequent_first():
    profile = "rust rust rust python python go"
    kws = extract_keywords(profile, top_n=3)
    assert kws[0] == "rust"
    assert kws[1] == "python"


def test_extract_keywords_respects_top_n():
    profile = " ".join(f"word{i}" for i in range(100))
    kws = extract_keywords(profile, top_n=10)
    assert len(kws) == 10


# ---------------------------------------------------------------------------
# score_entry
# ---------------------------------------------------------------------------


def test_score_entry_counts_keyword_matches():
    keywords = ["rust", "compiler", "memory"]
    entry = _entry("Rust compiler improvements", "Better memory safety in Rust")
    assert score_entry(entry, keywords) == 3


def test_score_entry_zero_for_no_match():
    keywords = ["python", "tensorflow"]
    entry = _entry("Cooking pasta at home")
    assert score_entry(entry, keywords) == 0


def test_score_entry_is_case_insensitive():
    keywords = ["rust"]
    entry = _entry("RUST programming language")
    assert score_entry(entry, keywords) == 1


def test_score_entry_counts_each_keyword_once():
    keywords = ["rust"]
    entry = _entry("Rust rust RUST")
    # keyword appears 3 times in text but should count as 1
    assert score_entry(entry, keywords) == 1


# ---------------------------------------------------------------------------
# prefilter_entries
# ---------------------------------------------------------------------------


def test_prefilter_passes_matching_entries():
    profile = "rust systems programming memory safety"
    entries = [
        _entry("Rust 2.0 released", guid="a"),
        _entry("Baking sourdough bread", guid="b"),
    ]
    passed, rejected = prefilter_entries(
        entries, profile, top_n_keywords=10, min_score=1
    )
    guids_passed = {e.guid for e in passed}
    guids_rejected = {e.guid for e in rejected}
    assert "a" in guids_passed
    assert "b" in guids_rejected


def test_prefilter_returns_all_when_min_score_zero():
    profile = "anything"
    entries = [_entry("Unrelated topic", guid="a"), _entry("Also unrelated", guid="b")]
    passed, rejected = prefilter_entries(entries, profile, min_score=0)
    assert len(passed) == 2
    assert len(rejected) == 0


def test_prefilter_empty_input():
    passed, rejected = prefilter_entries([], "some profile")
    assert passed == []
    assert rejected == []


def test_prefilter_total_equals_input():
    profile = "machine learning transformers attention"
    entries = [_entry(f"Entry {i}", guid=str(i)) for i in range(20)]
    passed, rejected = prefilter_entries(entries, profile)
    assert len(passed) + len(rejected) == 20
