"""Tests for the pure helpers of scripts/eval_prompt_style.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from rss_filter.models import NoteEntry, RSSEntry

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_prompt_style.py"
_spec = importlib.util.spec_from_file_location("eval_prompt_style", _SCRIPT)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def _note(url: str, date: str, source: str = "reading") -> NoteEntry:
    return NoteEntry(url=url, context=f"ctx {url}", source=source, date=date, title=url)


class TestRocAuc:
    def test_perfect_separation(self):
        assert ev.roc_auc([0.9, 0.8], [0.1, 0.2, 0.3]) == pytest.approx(1.0)

    def test_inverted(self):
        assert ev.roc_auc([0.1], [0.5, 0.6]) == pytest.approx(0.0)

    def test_all_tied_is_half(self):
        assert ev.roc_auc([0.5, 0.5], [0.5, 0.5, 0.5]) == pytest.approx(0.5)

    def test_matches_pairwise_definition(self):
        rng = np.random.default_rng(0)
        pos = list(np.round(rng.random(20), 1))
        neg = list(np.round(rng.random(30), 1))
        pairwise = np.mean(
            [1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg]
        )
        assert ev.roc_auc(pos, neg) == pytest.approx(pairwise)

    def test_empty_side_is_nan(self):
        assert np.isnan(ev.roc_auc([], [0.1]))


class TestPrecisionAtFraction:
    def test_top_fifth(self):
        # 10 items, top 2 by score: one positive (0.9), one negative (0.8).
        pos = [0.9, 0.1]
        neg = [0.8, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2, 0.0]
        assert ev.precision_at_fraction(pos, neg, 0.2) == pytest.approx(0.5)

    def test_at_least_one_item(self):
        assert ev.precision_at_fraction([0.9], [0.1], 0.01) == pytest.approx(1.0)


class TestHoldoutSplit:
    def test_most_recent_tenth_of_reading_held_out(self):
        entries = [_note(f"u{i}", f"2024-01-{i + 1:02d}") for i in range(20)]
        entries.append(_note("arx", "2024-02-01", source="arxiv"))
        store, holdout = ev.holdout_split(entries, 0.1)
        assert {e.url for e in holdout} == {"u19", "u18"}
        assert len(store) == 19
        assert "arx" in {e.url for e in store}

    def test_at_least_one_holdout(self):
        store, holdout = ev.holdout_split([_note("a", "2024-01-01")], 0.1)
        assert [e.url for e in holdout] == ["a"] and store == []

    def test_holdout_url_removed_from_store(self):
        entries = [_note("a", "2024-01-01"), _note("b", "2024-01-02")]
        entries.append(_note("b", "2023-01-01", source="arxiv"))
        store, holdout = ev.holdout_split(entries, 0.1)
        assert [e.url for e in holdout] == ["b"]
        assert [e.url for e in store] == ["a"]


class _FakeClient:
    """Embeds by keyword: texts mentioning 'cat' point one way, others another."""

    def embed_batch(self, texts):
        return [
            np.array([1.0, 0.0] if "cat" in t else [0.0, 1.0], dtype=np.float32)
            for t in texts
        ]


def test_evaluate_style_end_to_end():
    store_entries = [
        NoteEntry(url="s", context="cat s", source="reading", date="2024", title="cat")
    ]
    positives = [
        NoteEntry(url="p", context="cat p", source="reading", date="2025", title="cat")
    ]
    negatives = [
        RSSEntry(title="dog", url="n", summary="<b>dog</b>", feed_name="f",
                 is_arxiv=False, guid="n")
    ]
    row = ev.evaluate_style(
        "document", store_entries, positives, negatives, _FakeClient(), 3, 1.0
    )
    assert row["auc"] == pytest.approx(1.0)
    assert row["p_at"] == pytest.approx(1.0)
