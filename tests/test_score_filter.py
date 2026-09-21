"""Tests for rss_filter.score_filter."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from rss_filter.score_filter import (
    _exponential_decay_score,
    mrl_truncate,
    score_entries,
)
from rss_filter.models import RSSEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(title: str, is_arxiv: bool = False) -> RSSEntry:
    return RSSEntry(
        title=title,
        url=f"http://example.com/{title.replace(' ', '_')}",
        summary="A summary.",
        feed_name="Test Feed",
        is_arxiv=is_arxiv,
        guid=title,
    )


def _make_mocks(scores_per_entry: list[list[float]], embed_dim: int = 256):
    """Build mock EmbeddingClient and EmbeddingStore returning given per-entry scores."""
    embed_client = MagicMock()
    embed_client.embed_batch.return_value = [
        np.zeros(embed_dim, dtype=np.float32) for _ in scores_per_entry
    ]

    store = MagicMock()
    store.query.side_effect = [
        [
            {"text": f"ex{i}", "url": f"http://ex.com/{i}", "score": s}
            for i, s in enumerate(scores)
        ]
        for scores in scores_per_entry
    ]
    return embed_client, store


# ---------------------------------------------------------------------------
# _exponential_decay_score
# ---------------------------------------------------------------------------


class TestExponentialDecayScore:
    def test_single_similarity(self):
        # With one similarity, weight is exp(0)=1, so score == similarity.
        assert _exponential_decay_score([0.8], decay_lambda=1.0) == pytest.approx(0.8)

    def test_lambda_zero_equals_plain_mean(self):
        sims = [0.9, 0.7, 0.5]
        result = _exponential_decay_score(sims, decay_lambda=0.0)
        assert result == pytest.approx(np.mean(sims), abs=1e-6)

    def test_high_lambda_weights_top_match(self):
        sims = [0.9, 0.1, 0.1]
        # With very high lambda, first term dominates almost entirely.
        result = _exponential_decay_score(sims, decay_lambda=10.0)
        assert result == pytest.approx(0.9, abs=0.01)

    def test_empty_returns_zero(self):
        assert _exponential_decay_score([], decay_lambda=1.0) == 0.0

    def test_known_values(self):
        # Manually verify: sims=[0.8, 0.4], lambda=1
        # w0=exp(0)=1, w1=exp(-1)≈0.368
        # score = (0.8*1 + 0.4*0.368) / (1 + 0.368)
        w0, w1 = 1.0, np.exp(-1.0)
        expected = (0.8 * w0 + 0.4 * w1) / (w0 + w1)
        result = _exponential_decay_score([0.8, 0.4], decay_lambda=1.0)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_scores_bounded_when_sims_in_unit_range(self):
        sims = [0.95, 0.85, 0.75]
        result = _exponential_decay_score(sims, decay_lambda=1.0)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# score_entries
# ---------------------------------------------------------------------------


class TestScoreEntries:
    def test_empty_input_returns_empty(self):
        embed_client = MagicMock()
        store = MagicMock()
        results, meta, threshold_reading, threshold_arxiv = score_entries(
            [], store, embed_client
        )
        assert results == []
        assert meta == []
        assert threshold_reading == 0.0
        assert threshold_arxiv == 0.0

    def test_returns_correct_lengths(self):
        entries = [_entry("A"), _entry("B"), _entry("C")]
        embed_client, store = _make_mocks([[0.9, 0.8, 0.7]] * 3)
        results, meta, threshold_reading, threshold_arxiv = score_entries(
            entries, store, embed_client, top_k=3
        )
        assert len(results) == 3
        assert len(meta) == 3

    def test_scores_stored_in_filter_result(self):
        entries = [_entry("A")]
        embed_client, store = _make_mocks([[0.9, 0.8]])
        results, meta, _, _ = score_entries(entries, store, embed_client, top_k=2)
        assert results[0].score is not None
        assert results[0].score == pytest.approx(meta[0]["agg_score"], abs=1e-6)

    def test_threshold_in_reason_string(self):
        entries = [_entry("A"), _entry("B")]
        embed_client, store = _make_mocks([[0.9], [0.3]])
        results, _, threshold_reading, _ = score_entries(
            entries, store, embed_client, top_k=1, top_quantile=0.5
        )
        for r in results:
            assert f"{threshold_reading:.4f}" in r.reason

    def test_quantile_keeps_top_fraction(self):
        # 4 entries; top_quantile=0.5 should keep the top 2.
        entries = [_entry(f"E{i}") for i in range(4)]
        # Give them clearly separated scores so ranking is unambiguous.
        scores_per = [[0.9], [0.8], [0.3], [0.2]]
        embed_client, store = _make_mocks(scores_per)
        results, _, _, _ = score_entries(
            entries, store, embed_client, top_k=1, top_quantile=0.5
        )
        kept = [r.keep for r in results]
        assert sum(kept) == 2
        # The two highest-scored entries should be kept.
        assert kept[0] is True
        assert kept[1] is True
        assert kept[2] is False
        assert kept[3] is False

    def test_single_entry_always_kept(self):
        entries = [_entry("Solo")]
        embed_client, store = _make_mocks([[0.5]])
        results, _, _, _ = score_entries(entries, store, embed_client, top_k=1)
        assert results[0].keep is True

    def test_metadata_contains_embedding_128(self):
        entries = [_entry("A")]
        embed_dim = 256
        embed_client = MagicMock()
        embed_client.embed_batch.return_value = [np.ones(embed_dim, dtype=np.float32)]
        store = MagicMock()
        store.query.return_value = [{"text": "x", "url": "u", "score": 0.8}]
        _, meta, _, _ = score_entries(entries, store, embed_client, top_k=1)
        assert meta[0]["embedding_128"].shape == (128,)

    def test_metadata_embedding_128_is_unit_norm(self):
        entries = [_entry("A")]
        emb = np.r_[np.full(128, 0.1), np.ones(640)].astype(np.float32)
        embed_client = MagicMock()
        embed_client.embed_batch.return_value = [emb]
        store = MagicMock()
        store.query.return_value = [{"text": "x", "url": "u", "score": 0.8}]
        _, meta, _, _ = score_entries(entries, store, embed_client, top_k=1)
        assert np.linalg.norm(meta[0]["embedding_128"]) == pytest.approx(1.0, abs=1e-6)

    def test_input_order_preserved(self):
        entries = [_entry(f"E{i}") for i in range(5)]
        embed_client, store = _make_mocks([[0.5]] * 5)
        results, _, _, _ = score_entries(entries, store, embed_client, top_k=1)
        assert [r.entry for r in results] == entries

    def test_summary_html_is_stripped_before_embedding(self):
        entry = RSSEntry(
            title="T",
            url="http://example.com/t",
            summary="<p>Hi</p>",
            feed_name="F",
            is_arxiv=False,
            guid="t",
        )
        embed_client, store = _make_mocks([[0.5]])
        score_entries([entry], store, embed_client, top_k=1, prompt_style="none")
        embed_client.embed_batch.assert_called_once_with(["T Hi"])

    def test_document_prompt_style(self):
        entry = RSSEntry(
            title="T",
            url="http://example.com/t",
            summary="<p>Hi</p>",
            feed_name="F",
            is_arxiv=False,
            guid="t",
        )
        embed_client, store = _make_mocks([[0.5]])
        score_entries([entry], store, embed_client, top_k=1, prompt_style="document")
        embed_client.embed_batch.assert_called_once_with(["title: T | text: Hi"])


# ---------------------------------------------------------------------------
# mrl_truncate
# ---------------------------------------------------------------------------


class TestMrlTruncate:
    def test_truncated_vector_is_unit_norm(self):
        emb = np.r_[np.full(128, 0.1), np.ones(640)]
        out = mrl_truncate(emb)
        assert out.shape == (128,)
        assert np.linalg.norm(out) == pytest.approx(1.0)

    def test_direction_preserved(self):
        emb = np.r_[np.arange(1, 129, dtype=float), np.ones(640)]
        out = mrl_truncate(emb)
        np.testing.assert_allclose(out, emb[:128] / np.linalg.norm(emb[:128]))

    def test_zero_vector_returned_unchanged(self):
        out = mrl_truncate(np.zeros(768))
        assert out.shape == (128,)
        assert not out.any()
