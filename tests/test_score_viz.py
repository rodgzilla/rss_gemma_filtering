"""Tests for rss_filter.score_viz."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rss_filter.models import FilterResult, RSSEntry
from rss_filter.score_viz import _kde, write_score_viz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(title: str) -> RSSEntry:
    return RSSEntry(
        title=title,
        url=f"http://example.com/{title.replace(' ', '_')}",
        summary="Summary.",
        feed_name="Test Feed",
        is_arxiv=False,
        guid=title,
    )


def _result(title: str, keep: bool, score: float) -> FilterResult:
    return FilterResult(
        entry=_entry(title),
        keep=keep,
        reason=f"embedding score {score:.4f} (threshold 0.5000)",
        score=score,
    )


def _meta(score: float, embed_dim: int = 256) -> dict:
    return {
        "embedding": np.zeros(embed_dim, dtype=np.float32),
        "embedding_128": np.zeros(128, dtype=np.float32),
        "exemplars": [{"text": "Past article about topic", "score": score}],
        "agg_score": score,
    }


def _vault_2d(n: int = 5) -> np.ndarray:
    return np.random.rand(n, 2).astype(np.float32)


def _vault_docs(n: int = 5) -> list[dict]:
    return [
        {
            "url": f"http://v.com/{i}",
            "text": f"Vault doc {i}",
            "source_note": "2024-01-01.md",
            "date": "2024-01-01",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# _kde
# ---------------------------------------------------------------------------


class TestKde:
    def test_returns_equal_length_lists(self):
        x, y = _kde(np.array([0.1, 0.5, 0.9]))
        assert len(x) == len(y)

    def test_empty_input_returns_empty(self):
        x, y = _kde(np.array([]))
        assert x == []
        assert y == []

    def test_constant_input_returns_empty(self):
        x, y = _kde(np.array([0.5, 0.5, 0.5]))
        assert x == []
        assert y == []

    def test_kde_values_are_non_negative(self):
        x, y = _kde(np.array([0.2, 0.5, 0.8, 0.3]))
        assert all(v >= 0 for v in y)


# ---------------------------------------------------------------------------
# write_score_viz
# ---------------------------------------------------------------------------


class TestWriteScoreViz:
    def _basic_call(self, tmp_path: Path, with_umap: bool = True):
        results = [
            _result("Kept Article", keep=True, score=0.8),
            _result("Rejected Article", keep=False, score=0.3),
        ]
        metadata = [_meta(0.8), _meta(0.3)]
        vault_2d = _vault_2d()
        vault_docs = _vault_docs()
        umap_reducer = MagicMock() if with_umap else None
        if with_umap:
            umap_reducer.transform.return_value = np.random.rand(2, 2).astype(
                np.float32
            )

        out = tmp_path / "RSS-2024-01-01-scores.html"
        write_score_viz(
            results=results,
            entry_metadata=metadata,
            vault_2d=vault_2d,
            vault_docs=vault_docs,
            umap_reducer=umap_reducer,
            threshold=0.5,
            output_path=out,
        )
        return out

    def test_output_file_created(self, tmp_path):
        out = self._basic_call(tmp_path)
        assert out.exists()

    def test_html_contains_plotly_cdn(self, tmp_path):
        out = self._basic_call(tmp_path)
        html = out.read_text()
        assert "plotly" in html.lower()

    def test_html_contains_article_titles(self, tmp_path):
        out = self._basic_call(tmp_path)
        html = out.read_text()
        assert "Kept Article" in html
        assert "Rejected Article" in html

    def test_html_contains_threshold(self, tmp_path):
        out = self._basic_call(tmp_path)
        html = out.read_text()
        assert "0.5" in html

    def test_html_is_valid_structure(self, tmp_path):
        out = self._basic_call(tmp_path)
        html = out.read_text()
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "Plotly.newPlot" in html

    def test_empty_results_skips_write(self, tmp_path):
        out = tmp_path / "no-output.html"
        write_score_viz(
            results=[],
            entry_metadata=[],
            vault_2d=np.zeros((0, 2), dtype=np.float32),
            vault_docs=[],
            umap_reducer=None,
            threshold=0.5,
            output_path=out,
        )
        assert not out.exists()

    def test_no_umap_reducer_still_writes(self, tmp_path):
        out = self._basic_call(tmp_path, with_umap=False)
        assert out.exists()

    def test_parent_dirs_created(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "output.html"
        results = [_result("Article", keep=True, score=0.8)]
        metadata = [_meta(0.8)]
        write_score_viz(
            results=results,
            entry_metadata=metadata,
            vault_2d=_vault_2d(),
            vault_docs=_vault_docs(),
            umap_reducer=None,
            threshold=0.5,
            output_path=nested,
        )
        assert nested.exists()
