"""Tests for rss_filter.score_viz."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from rss_filter.models import FilterResult, RSSEntry
from rss_filter.score_viz import write_score_viz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(title: str, is_arxiv: bool = False) -> RSSEntry:
    return RSSEntry(
        title=title,
        url=f"http://example.com/{title.replace(' ', '_')}",
        summary="Summary.",
        feed_name="Test Feed",
        is_arxiv=is_arxiv,
        guid=title,
    )


def _result(
    title: str, keep: bool, score: float, is_arxiv: bool = False
) -> FilterResult:
    return FilterResult(
        entry=_entry(title, is_arxiv=is_arxiv),
        keep=keep,
        reason=f"embedding score {score:.4f}",
        score=score,
    )


def _meta(score: float, embed_dim: int = 256) -> dict:
    return {
        "embedding": np.zeros(embed_dim, dtype=np.float32),
        "embedding_128": np.random.rand(128).astype(np.float32),
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
# write_score_viz
# ---------------------------------------------------------------------------


class TestWriteScoreViz:
    def _basic_call(
        self,
        tmp_path: Path,
        with_umap: bool = True,
        n_entries: int = 2,
        vault_bg_max: int = 500,
        n_vault: int = 5,
        threshold_reading: float = 0.5,
        threshold_arxiv: float = 0.4,
    ):
        results = [
            _result(f"Article {i}", keep=(i % 2 == 0), score=0.8 if i % 2 == 0 else 0.3)
            for i in range(n_entries)
        ]
        metadata = [_meta(r.score) for r in results]
        vault_2d = _vault_2d(n_vault)
        vault_docs = _vault_docs(n_vault)
        umap_reducer = MagicMock() if with_umap else None
        if with_umap:
            umap_reducer.transform.return_value = np.random.rand(n_entries, 2).astype(
                np.float32
            )

        out = tmp_path / "RSS-2024-01-01-scores.html"
        write_score_viz(
            results=results,
            entry_metadata=metadata,
            vault_2d=vault_2d,
            vault_docs=vault_docs,
            umap_reducer=umap_reducer,
            threshold_reading=threshold_reading,
            threshold_arxiv=threshold_arxiv,
            output_path=out,
            vault_bg_max=vault_bg_max,
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
        assert "Article 0" in html
        assert "Article 1" in html

    def test_html_contains_both_thresholds(self, tmp_path):
        out = self._basic_call(tmp_path, threshold_reading=0.55, threshold_arxiv=0.40)
        html = out.read_text()
        assert "0.55" in html
        assert "0.40" in html

    def test_html_contains_reading_threshold_line(self, tmp_path):
        out = self._basic_call(tmp_path, threshold_reading=0.55, threshold_arxiv=0.40)
        html = out.read_text()
        # Reading threshold label must be present
        assert "reading" in html.lower()

    def test_html_contains_arxiv_threshold_line(self, tmp_path):
        out = self._basic_call(tmp_path, threshold_reading=0.55, threshold_arxiv=0.40)
        html = out.read_text()
        # arXiv threshold line should use distinct orange colour
        assert "#f39c12" in html

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
            threshold_reading=0.5,
            threshold_arxiv=0.4,
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
            threshold_reading=0.5,
            threshold_arxiv=0.4,
            output_path=nested,
        )
        assert nested.exists()

    def test_no_histogram_panel(self, tmp_path):
        """Panel 1 (histogram/KDE) must not appear in the output."""
        out = self._basic_call(tmp_path)
        html = out.read_text()
        assert '"type": "histogram"' not in html
        assert "Score Distribution" not in html

    def test_rss_only_umap_panel_present(self, tmp_path):
        """Panel 3 annotation for RSS-only UMAP should appear with ≥2 entries."""
        out = self._basic_call(tmp_path, n_entries=4)
        html = out.read_text()
        assert "RSS entries only" in html

    def test_vault_background_subsampled(self, tmp_path):
        """When vault is larger than vault_bg_max the file is still produced."""
        out = self._basic_call(tmp_path, n_vault=600, vault_bg_max=50)
        assert out.exists()
        html = out.read_text()
        assert "Plotly.newPlot" in html
