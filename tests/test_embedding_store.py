"""Tests for rss_filter.embedding_store."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from rss_filter.embedding_store import EmbeddingStore
from rss_filter.models import NoteEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(vectors: list[np.ndarray] | None = None) -> MagicMock:
    """Return a mock EmbeddingClient whose embed_batch returns *vectors*."""
    client = MagicMock()
    if vectors is not None:
        client.embed_batch.return_value = vectors
    return client


def _unit(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def _note(
    url: str, context: str = "", source: str = "reading", date: str = "2024-01-01"
) -> NoteEntry:
    return NoteEntry(url=url, context=context, source=source, date=date)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    """EmbeddingStore backed by a temporary file."""
    db = EmbeddingStore(str(tmp_path / "test.db"))
    yield db
    db.close()


# ---------------------------------------------------------------------------
# build_or_update
# ---------------------------------------------------------------------------


class TestBuildOrUpdate:
    def test_inserts_new_entries(self, store):
        vecs = [_unit([1.0, 0.0]), _unit([0.0, 1.0])]
        client = _make_client(vecs)
        entries = [_note("http://a.com", "text A"), _note("http://b.com", "text B")]

        store.build_or_update(entries, client)

        assert store.count() == 2

    def test_skips_already_stored_entries(self, store):
        vec = _unit([1.0, 0.0])
        client = _make_client([vec])
        entry = _note("http://a.com", "text A")

        store.build_or_update([entry], client)
        assert store.count() == 1

        # Second call with same entry — should not insert again.
        store.build_or_update([entry], client)
        assert store.count() == 1
        assert client.embed_batch.call_count == 1  # only called once

    def test_force_rebuild_clears_table(self, store):
        vecs = [_unit([1.0, 0.0])]
        client = _make_client(vecs)
        entry = _note("http://a.com", "text A")

        store.build_or_update([entry], client)
        assert store.count() == 1

        # Rebuild with a different entry.
        client2 = _make_client([_unit([0.0, 1.0])])
        new_entry = _note("http://b.com", "text B")
        store.build_or_update([new_entry], client2, force_rebuild=True)

        assert store.count() == 1  # old entry gone, new one in

    def test_uses_url_as_fallback_when_no_context(self, store):
        vec = _unit([1.0, 0.0])
        client = _make_client([vec])
        entry = _note("http://fallback.com", context="")  # no context

        store.build_or_update([entry], client)

        # embed_batch should have been called with the URL as text
        call_args = client.embed_batch.call_args
        texts_passed = call_args[0][0]
        assert texts_passed == ["http://fallback.com"]

    def test_empty_entries_list_does_nothing(self, store):
        client = _make_client()
        store.build_or_update([], client)

        assert store.count() == 0
        client.embed_batch.assert_not_called()


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


class TestQuery:
    def test_returns_top_k_results(self, store):
        vecs = [_unit([1.0, 0.0]), _unit([0.0, 1.0]), _unit([1.0, 1.0])]
        client = _make_client(vecs)
        entries = [
            _note("http://a.com", "text A"),
            _note("http://b.com", "text B"),
            _note("http://c.com", "text C"),
        ]
        store.build_or_update(entries, client)

        query_vec = _unit([1.0, 0.0])
        results = store.query(query_vec, top_k=2)

        assert len(results) == 2
        # The most similar vector to [1,0] is [1,0] itself (score ~1.0)
        assert results[0]["url"] == "http://a.com"
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)

    def test_results_sorted_descending(self, store):
        vecs = [_unit([1.0, 0.0]), _unit([0.0, 1.0])]
        client = _make_client(vecs)
        entries = [_note("http://a.com", "A"), _note("http://b.com", "B")]
        store.build_or_update(entries, client)

        query_vec = _unit([1.0, 0.1])
        results = store.query(query_vec, top_k=2)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_store_returns_empty_list(self, store):
        results = store.query(_unit([1.0, 0.0]), top_k=3)
        assert results == []

    def test_top_k_capped_at_store_size(self, store):
        vecs = [_unit([1.0, 0.0])]
        client = _make_client(vecs)
        store.build_or_update([_note("http://a.com", "A")], client)

        results = store.query(_unit([1.0, 0.0]), top_k=10)
        assert len(results) == 1

    def test_result_dict_has_required_keys(self, store):
        vecs = [_unit([1.0, 0.0])]
        client = _make_client(vecs)
        store.build_or_update([_note("http://a.com", "context text")], client)

        results = store.query(_unit([1.0, 0.0]), top_k=1)
        assert set(results[0].keys()) == {"text", "url", "score", "title"}


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


class TestCount:
    def test_count_zero_on_empty_store(self, store):
        assert store.count() == 0

    def test_count_after_insert(self, store):
        vecs = [_unit([1.0, 0.0]), _unit([0.0, 1.0])]
        client = _make_client(vecs)
        entries = [_note("http://a.com"), _note("http://b.com")]
        store.build_or_update(entries, client)
        assert store.count() == 2
