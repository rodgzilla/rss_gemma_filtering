"""Tests for rss_filter.embedding_filter."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import numpy as np
import pytest

from rss_filter.embedding_filter import (
    build_filter_prompt,
    filter_entries_batch,
    parse_filter_response,
)
from rss_filter.models import FilterResult, RSSEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(
    title: str, summary: str = "Some summary.", is_arxiv: bool = False
) -> RSSEntry:
    return RSSEntry(
        title=title,
        url=f"http://example.com/{title.replace(' ', '_')}",
        summary=summary,
        feed_name="Test Feed",
        is_arxiv=is_arxiv,
        guid=title,
    )


def _exemplars(n: int = 2) -> list[dict]:
    return [
        {
            "text": f"Exemplar title {i}",
            "url": f"http://ex.com/{i}",
            "score": 0.9 - i * 0.1,
        }
        for i in range(n)
    ]


def _mock_llm_response(text: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = text
    return response


# ---------------------------------------------------------------------------
# build_filter_prompt
# ---------------------------------------------------------------------------


class TestBuildFilterPrompt:
    def test_single_entry_contains_title(self):
        entry = _entry("AI Safety Research")
        items = [{"entry": entry, "exemplars": _exemplars(2)}]
        prompt = build_filter_prompt(items)

        assert "AI Safety Research" in prompt
        assert "1." in prompt

    def test_multiple_entries_numbered(self):
        items = [
            {"entry": _entry("Article One"), "exemplars": _exemplars(1)},
            {"entry": _entry("Article Two"), "exemplars": _exemplars(1)},
        ]
        prompt = build_filter_prompt(items)

        assert "1." in prompt
        assert "2." in prompt
        assert "Article One" in prompt
        assert "Article Two" in prompt

    def test_exemplars_included_with_scores(self):
        exemplars = [{"text": "Past Article", "url": "http://ex.com", "score": 0.87}]
        items = [{"entry": _entry("New Article"), "exemplars": exemplars}]
        prompt = build_filter_prompt(items)

        assert "Past Article" in prompt
        assert "0.87" in prompt

    def test_summary_not_included_in_prompt(self):
        entry = _entry("Title", summary="This is the article summary.")
        items = [{"entry": entry, "exemplars": []}]
        prompt = build_filter_prompt(items)

        assert "This is the article summary." not in prompt

    def test_no_exemplars_shows_placeholder(self):
        items = [{"entry": _entry("Title"), "exemplars": []}]
        prompt = build_filter_prompt(items)

        assert "no similar articles found" in prompt

    def test_empty_list_returns_empty_string(self):
        assert build_filter_prompt([]) == ""


# ---------------------------------------------------------------------------
# parse_filter_response
# ---------------------------------------------------------------------------


class TestParseFilterResponse:
    def test_yes_line_parsed_correctly(self):
        entries = [_entry("Article A")]
        response = "1. yes: very relevant topic"
        results = parse_filter_response(response, entries)

        assert len(results) == 1
        assert results[0].keep is True
        assert results[0].reason == "very relevant topic"

    def test_no_line_parsed_correctly(self):
        entries = [_entry("Article A")]
        response = "1. no: unrelated to interests"
        results = parse_filter_response(response, entries)

        assert results[0].keep is False
        assert results[0].reason == "unrelated to interests"

    def test_multiple_entries_parsed_in_order(self):
        entries = [_entry("A"), _entry("B"), _entry("C")]
        response = "1. yes: good\n2. no: bad\n3. yes: great"
        results = parse_filter_response(response, entries)

        assert [r.keep for r in results] == [True, False, True]

    def test_missing_line_defaults_to_no(self):
        entries = [_entry("A"), _entry("B")]
        response = "1. yes: good"  # Line 2 missing
        results = parse_filter_response(response, entries)

        assert results[0].keep is True
        assert results[1].keep is False
        assert "no response" in results[1].reason

    def test_case_insensitive_yes_no(self):
        entries = [_entry("A"), _entry("B")]
        response = "1. YES: good\n2. NO: bad"
        results = parse_filter_response(response, entries)

        assert results[0].keep is True
        assert results[1].keep is False

    def test_preserves_entry_reference(self):
        entry = _entry("My Article")
        results = parse_filter_response("1. yes: ok", [entry])

        assert results[0].entry is entry

    def test_empty_response_all_default_to_no(self):
        entries = [_entry("A"), _entry("B")]
        results = parse_filter_response("", entries)

        assert all(not r.keep for r in results)


# ---------------------------------------------------------------------------
# filter_entries_batch
# ---------------------------------------------------------------------------


class TestFilterEntriesBatch:
    def _make_mocks(self, llm_text: str, embed_dim: int = 4):
        embed_client = MagicMock()
        embed_client.embed_batch.return_value = [np.zeros(embed_dim, dtype=np.float32)]

        store = MagicMock()
        store.query.return_value = _exemplars(2)

        llm_client = MagicMock()
        llm_client.chat.completions.create.return_value = _mock_llm_response(llm_text)

        return embed_client, store, llm_client

    def test_returns_filter_result_list(self):
        embed_client, store, llm_client = self._make_mocks("1. yes: relevant")
        embed_client.embed_batch.return_value = [np.zeros(4, dtype=np.float32)]

        results = filter_entries_batch(
            [_entry("Test Article")],
            store=store,
            embed_client=embed_client,
            llm_client=llm_client,
            model="gemma",
        )

        assert len(results) == 1
        assert isinstance(results[0], FilterResult)

    def test_empty_entries_returns_empty_list(self):
        embed_client, store, llm_client = self._make_mocks("")

        results = filter_entries_batch(
            [],
            store=store,
            embed_client=embed_client,
            llm_client=llm_client,
            model="gemma",
        )

        assert results == []
        embed_client.embed_batch.assert_not_called()

    def test_preserves_input_order(self):
        entries = [_entry(f"Article {i}") for i in range(3)]
        embed_client = MagicMock()
        embed_client.embed_batch.return_value = [
            np.zeros(4, dtype=np.float32) for _ in entries
        ]
        store = MagicMock()
        store.query.return_value = []
        llm_client = MagicMock()
        llm_client.chat.completions.create.return_value = _mock_llm_response(
            "1. yes: a\n2. no: b\n3. yes: c"
        )

        results = filter_entries_batch(
            entries,
            store=store,
            embed_client=embed_client,
            llm_client=llm_client,
            model="gemma",
            batch_size=10,
        )

        assert [r.entry for r in results] == entries

    def test_batching_splits_llm_calls(self):
        entries = [_entry(f"Article {i}") for i in range(5)]
        embed_client = MagicMock()
        embed_client.embed_batch.return_value = [
            np.zeros(4, dtype=np.float32) for _ in entries
        ]
        store = MagicMock()
        store.query.return_value = []

        llm_client = MagicMock()
        # First batch: 2 entries, second batch: 2, third: 1
        llm_client.chat.completions.create.side_effect = [
            _mock_llm_response("1. yes: a\n2. no: b"),
            _mock_llm_response("1. yes: c\n2. no: d"),
            _mock_llm_response("1. yes: e"),
        ]

        results = filter_entries_batch(
            entries,
            store=store,
            embed_client=embed_client,
            llm_client=llm_client,
            model="gemma",
            batch_size=2,
        )

        assert llm_client.chat.completions.create.call_count == 3
        assert len(results) == 5

    def test_embed_called_once_for_all_entries(self):
        entries = [_entry("A"), _entry("B")]
        embed_client = MagicMock()
        embed_client.embed_batch.return_value = [
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        ]
        store = MagicMock()
        store.query.return_value = []
        llm_client = MagicMock()
        llm_client.chat.completions.create.return_value = _mock_llm_response(
            "1. yes: a\n2. no: b"
        )

        filter_entries_batch(
            entries,
            store=store,
            embed_client=embed_client,
            llm_client=llm_client,
            model="gemma",
        )

        # embed_batch should be called exactly once with both texts
        embed_client.embed_batch.assert_called_once()
        texts_arg = embed_client.embed_batch.call_args[0][0]
        assert len(texts_arg) == 2
