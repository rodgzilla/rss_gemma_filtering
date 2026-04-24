"""Tests for rss_filter.relevance_filter"""

from unittest.mock import MagicMock

import pytest

from rss_filter.models import FilterResult, RSSEntry
from rss_filter.relevance_filter import (
    build_batch_filter_prompt,
    build_filter_prompt,
    filter_entries_batch,
    filter_entry,
    parse_batch_response,
    parse_llm_response,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INTEREST_PROFILE = (
    "The user is interested in machine learning, self-supervised learning, "
    "board games, probability theory, and local LLM tooling."
)


def _make_entry(
    title: str, summary: str, is_arxiv: bool = False, guid: str = "guid-test"
) -> RSSEntry:
    return RSSEntry(
        title=title,
        url="https://example.com/entry",
        summary=summary,
        feed_name="Test Feed",
        is_arxiv=is_arxiv,
        guid=guid,
    )


# ---------------------------------------------------------------------------
# build_filter_prompt
# ---------------------------------------------------------------------------


def test_build_filter_prompt_contains_interest_profile():
    entry = _make_entry("Cool ML Post", "A post about machine learning techniques.")
    prompt = build_filter_prompt(entry, INTEREST_PROFILE)

    assert INTEREST_PROFILE in prompt


def test_build_filter_prompt_contains_entry_title():
    entry = _make_entry("Cool ML Post", "A post about machine learning techniques.")
    prompt = build_filter_prompt(entry, INTEREST_PROFILE)

    assert "Cool ML Post" in prompt


def test_build_filter_prompt_contains_entry_summary():
    entry = _make_entry("Title", "A post about machine learning techniques.")
    prompt = build_filter_prompt(entry, INTEREST_PROFILE)

    assert "machine learning techniques" in prompt


def test_build_filter_prompt_requests_yes_no_format():
    entry = _make_entry("Title", "Summary.")
    prompt = build_filter_prompt(entry, INTEREST_PROFILE)

    lower = prompt.lower()
    assert "yes" in lower and "no" in lower


# ---------------------------------------------------------------------------
# parse_llm_response
# ---------------------------------------------------------------------------


def test_parse_llm_response_yes_returns_keep_true():
    result = parse_llm_response("yes: Matches interest in machine learning")

    assert result.keep is True


def test_parse_llm_response_yes_extracts_reason():
    result = parse_llm_response("yes: Matches interest in machine learning")

    assert result.reason == "Matches interest in machine learning"


def test_parse_llm_response_no_returns_keep_false():
    result = parse_llm_response("no: Unrelated to tracked interests")

    assert result.keep is False


def test_parse_llm_response_no_extracts_reason():
    result = parse_llm_response("no: Unrelated to tracked interests")

    assert result.reason == "Unrelated to tracked interests"


def test_parse_llm_response_case_insensitive():
    result = parse_llm_response("Yes: Some reason here")

    assert result.keep is True


def test_parse_llm_response_malformed_defaults_to_not_keep():
    result = parse_llm_response("I am not sure about this entry.")

    assert result.keep is False


def test_parse_llm_response_malformed_has_parse_error_reason():
    result = parse_llm_response("Something totally unexpected")

    assert "parse error" in result.reason.lower()


def test_parse_llm_response_empty_string_defaults_to_not_keep():
    result = parse_llm_response("")

    assert result.keep is False


# ---------------------------------------------------------------------------
# filter_entry (LLM mocked)
# ---------------------------------------------------------------------------


def _make_mock_client(response_text: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices[0].message.content = response_text
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def test_filter_entry_returns_filter_result(mocker):
    entry = _make_entry("ML Post", "About machine learning.")
    client = _make_mock_client("yes: Directly relevant to ML interests")

    result = filter_entry(entry, INTEREST_PROFILE, client, model="gemma-4-e4b")

    assert isinstance(result, FilterResult)


def test_filter_entry_propagates_entry(mocker):
    entry = _make_entry("ML Post", "About machine learning.")
    client = _make_mock_client("yes: Directly relevant to ML interests")

    result = filter_entry(entry, INTEREST_PROFILE, client, model="gemma-4-e4b")

    assert result.entry is entry


def test_filter_entry_keep_true_on_yes_response(mocker):
    entry = _make_entry("ML Post", "About machine learning.")
    client = _make_mock_client("yes: Directly relevant to ML interests")

    result = filter_entry(entry, INTEREST_PROFILE, client, model="gemma-4-e4b")

    assert result.keep is True


def test_filter_entry_keep_false_on_no_response(mocker):
    entry = _make_entry("Celebrity Gossip", "Hollywood news.")
    client = _make_mock_client("no: Unrelated to any tracked interest")

    result = filter_entry(entry, INTEREST_PROFILE, client, model="gemma-4-e4b")

    assert result.keep is False


# ---------------------------------------------------------------------------
# build_batch_filter_prompt
# ---------------------------------------------------------------------------


def test_build_batch_filter_prompt_contains_profile():
    entries = [_make_entry("Title A", "Summary A"), _make_entry("Title B", "Summary B")]
    prompt = build_batch_filter_prompt(entries, INTEREST_PROFILE)
    assert INTEREST_PROFILE in prompt


def test_build_batch_filter_prompt_contains_all_titles():
    entries = [_make_entry("Title A", "Summary A"), _make_entry("Title B", "Summary B")]
    prompt = build_batch_filter_prompt(entries, INTEREST_PROFILE)
    assert "Title A" in prompt
    assert "Title B" in prompt


def test_build_batch_filter_prompt_contains_entry_count():
    entries = [_make_entry(f"T{i}", f"S{i}") for i in range(5)]
    prompt = build_batch_filter_prompt(entries, INTEREST_PROFILE)
    assert "5" in prompt


# ---------------------------------------------------------------------------
# parse_batch_response
# ---------------------------------------------------------------------------


def test_parse_batch_response_parses_all_lines():
    entries = [_make_entry("A", ""), _make_entry("B", ""), _make_entry("C", "")]
    response = "1. yes: Relevant\n2. no: Off-topic\n3. yes: Matches interest"
    results = parse_batch_response(response, entries)
    assert len(results) == 3
    assert results[0].keep is True
    assert results[1].keep is False
    assert results[2].keep is True


def test_parse_batch_response_extracts_reasons():
    entries = [_make_entry("A", "")]
    response = "1. yes: Great match for ML interest"
    results = parse_batch_response(response, entries)
    assert results[0].reason == "Great match for ML interest"


def test_parse_batch_response_attaches_entries():
    entry = _make_entry("A", "")
    results = parse_batch_response("1. yes: reason", [entry])
    assert results[0].entry is entry


def test_parse_batch_response_missing_line_defaults_to_false():
    entries = [_make_entry("A", ""), _make_entry("B", "")]
    response = "1. yes: reason"  # entry 2 missing
    results = parse_batch_response(response, entries)
    assert results[1].keep is False
    assert "parse error" in results[1].reason


def test_parse_batch_response_accepts_parenthesis_separator():
    entries = [_make_entry("A", "")]
    response = "1) no: Not relevant"
    results = parse_batch_response(response, entries)
    assert results[0].keep is False


# ---------------------------------------------------------------------------
# filter_entries_batch (LLM mocked)
# ---------------------------------------------------------------------------


def test_filter_entries_batch_returns_one_result_per_entry():
    entries = [_make_entry(f"T{i}", f"S{i}", guid=f"g{i}") for i in range(3)]
    llm_response = "1. yes: r1\n2. no: r2\n3. yes: r3"
    client = _make_mock_client(llm_response)

    results = filter_entries_batch(
        entries, INTEREST_PROFILE, client, model="gemma-4-e4b", batch_size=10
    )

    assert len(results) == 3


def test_filter_entries_batch_splits_into_batches():
    entries = [_make_entry(f"T{i}", f"S{i}") for i in range(5)]
    # Return two-line response for each batch of 2, one-line for remainder
    responses = [
        "1. yes: r1\n2. no: r2",
        "1. yes: r3\n2. no: r4",
        "1. yes: r5",
    ]
    call_count = 0

    def fake_create(**kwargs):
        nonlocal call_count
        resp = MagicMock()
        resp.choices[0].message.content = responses[call_count]
        call_count += 1
        return resp

    client = MagicMock()
    client.chat.completions.create.side_effect = fake_create

    results = filter_entries_batch(
        entries, INTEREST_PROFILE, client, model="gemma-4-e4b", batch_size=2
    )

    assert call_count == 3  # ceil(5/2) = 3 LLM calls
    assert len(results) == 5
