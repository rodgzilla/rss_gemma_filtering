"""Tests for rss_filter.reranker"""

from unittest.mock import MagicMock

import pytest

from rss_filter.models import FilterResult, RSSEntry
from rss_filter.reranker import (
    DEFAULT_SCORE,
    build_rerank_prompt,
    parse_rerank_response,
    rerank,
)

INTEREST_PROFILE = (
    "The user is interested in machine learning, systems programming, "
    "and local LLM tooling."
)


def _entry(title: str, summary: str = "", guid: str = "g") -> RSSEntry:
    return RSSEntry(
        title=title,
        url="https://example.com",
        summary=summary,
        feed_name="F",
        is_arxiv=False,
        guid=guid,
    )


def _result(title: str, summary: str = "", guid: str = "g") -> FilterResult:
    return FilterResult(
        entry=_entry(title, summary, guid), keep=True, reason="relevant"
    )


def _mock_client(response_text: str) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = response_text
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    return client


# ---------------------------------------------------------------------------
# build_rerank_prompt
# ---------------------------------------------------------------------------


def test_build_rerank_prompt_contains_profile():
    results = [_result("Title A")]
    prompt = build_rerank_prompt(results, INTEREST_PROFILE)
    assert INTEREST_PROFILE in prompt


def test_build_rerank_prompt_contains_all_titles():
    results = [_result("Title A"), _result("Title B")]
    prompt = build_rerank_prompt(results, INTEREST_PROFILE)
    assert "Title A" in prompt
    assert "Title B" in prompt


def test_build_rerank_prompt_mentions_score_range():
    results = [_result("T")]
    prompt = build_rerank_prompt(results, INTEREST_PROFILE)
    assert "1" in prompt and "10" in prompt


def test_build_rerank_prompt_truncates_summary():
    long_summary = "x" * 500
    results = [_result("T", long_summary)]
    prompt = build_rerank_prompt(
        results,
        INTEREST_PROFILE,
    )
    # 200-char truncation from reranker
    assert "x" * 200 in prompt
    assert "x" * 201 not in prompt


# ---------------------------------------------------------------------------
# parse_rerank_response
# ---------------------------------------------------------------------------


def test_parse_rerank_response_attaches_scores():
    results = [_result("A"), _result("B"), _result("C")]
    response = "1. 8\n2. 3\n3. 10"
    updated = parse_rerank_response(response, results)
    assert updated[0].score == pytest.approx(0.8)
    assert updated[1].score == pytest.approx(0.3)
    assert updated[2].score == pytest.approx(1.0)


def test_parse_rerank_response_clamps_above_10():
    results = [_result("A")]
    response = "1. 99"
    updated = parse_rerank_response(response, results)
    assert updated[0].score == pytest.approx(1.0)


def test_parse_rerank_response_clamps_below_1():
    results = [_result("A")]
    response = "1. 0"
    updated = parse_rerank_response(response, results)
    assert updated[0].score == pytest.approx(0.1)


def test_parse_rerank_response_missing_line_uses_default():
    results = [_result("A"), _result("B")]
    response = "1. 7"  # entry 2 missing
    updated = parse_rerank_response(response, results)
    assert updated[1].score == pytest.approx(DEFAULT_SCORE / 10.0)


def test_parse_rerank_response_preserves_entry_references():
    r = _result("A")
    updated = parse_rerank_response("1. 6", [r])
    assert updated[0].entry is r.entry


def test_parse_rerank_response_preserves_keep_and_reason():
    r = FilterResult(entry=_entry("A"), keep=True, reason="my reason")
    updated = parse_rerank_response("1. 5", [r])
    assert updated[0].keep is True
    assert updated[0].reason == "my reason"


def test_parse_rerank_response_accepts_parenthesis_separator():
    results = [_result("A")]
    response = "1) 9"
    updated = parse_rerank_response(response, results)
    assert updated[0].score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# rerank
# ---------------------------------------------------------------------------


def test_rerank_returns_sorted_descending():
    results = [_result("A", guid="a"), _result("B", guid="b"), _result("C", guid="c")]
    # A=3, B=9, C=6 → expected order B, C, A
    client = _mock_client("1. 3\n2. 9\n3. 6")

    ranked = rerank(results, INTEREST_PROFILE, client, model="m")

    assert [r.entry.guid for r in ranked] == ["b", "c", "a"]


def test_rerank_populates_score_field():
    results = [_result("A")]
    client = _mock_client("1. 7")

    ranked = rerank(results, INTEREST_PROFILE, client, model="m")

    assert ranked[0].score is not None


def test_rerank_empty_list_returns_empty():
    client = _mock_client("")
    result = rerank([], INTEREST_PROFILE, client, model="m")
    assert result == []
    client.chat.completions.create.assert_not_called()


def test_rerank_splits_into_batches():
    results = [_result(f"T{i}", guid=str(i)) for i in range(5)]
    responses = ["1. 5\n2. 8\n3. 2", "1. 9\n2. 4"]
    call_idx = 0

    def fake_create(**kwargs):
        nonlocal call_idx
        r = MagicMock()
        r.choices[0].message.content = responses[call_idx]
        call_idx += 1
        return r

    client = MagicMock()
    client.chat.completions.create.side_effect = fake_create

    ranked = rerank(results, INTEREST_PROFILE, client, model="m", batch_size=3)

    assert call_idx == 2  # ceil(5/3) = 2 LLM calls
    assert len(ranked) == 5
    # highest score is 9 (entry index 3 in batch 2, guid="3")
    assert ranked[0].entry.guid == "3"
