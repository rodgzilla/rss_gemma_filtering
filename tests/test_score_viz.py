"""Tests for rss_filter.score_viz."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from rss_filter.models import FilterResult, RSSEntry
from rss_filter.score_viz import _muted_colour, _FEED_PALETTE, write_score_viz


def test_feed_palette_has_ten_entries():
    assert len(_FEED_PALETTE) == 10


def test_muted_colour_is_paler():
    """Muted colour should be closer to #cccccc than the original."""
    intense = "#2980b9"  # blue: r=0x29=41, g=0x80=128, b=0xb9=185
    muted = _muted_colour(intense)
    # Muted red component should be higher than intense red (blended toward #cccccc=204)
    r_intense = int(intense[1:3], 16)
    r_muted = int(muted[1:3], 16)
    assert r_muted > r_intense


def test_muted_colour_returns_hex_string():
    muted = _muted_colour("#2980b9")
    assert muted.startswith("#")
    assert len(muted) == 7


def test_muted_colour_keep_zero_returns_grey():
    """keep=0.0 should return #cccccc."""
    result = _muted_colour("#e74c3c", keep=0.0)
    assert result == "#cccccc"


def test_muted_colour_keep_one_returns_original():
    """keep=1.0 should return the original colour."""
    result = _muted_colour("#2980b9", keep=1.0)
    assert result == "#2980b9"


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

    def test_no_rss_only_umap_panel(self, tmp_path):
        """RSS-only UMAP column must not appear in the output."""
        out = self._mixed_call(tmp_path, n_reading=2, n_arxiv=2)
        html = out.read_text()
        assert "RSS entries only" not in html

    def test_vault_background_subsampled(self, tmp_path):
        """When vault is larger than vault_bg_max the file is still produced."""
        out = self._basic_call(tmp_path, n_vault=600, vault_bg_max=50)
        assert out.exists()
        html = out.read_text()
        assert "Plotly.newPlot" in html

    # --- 2×2 layout tests ---

    def _mixed_call(self, tmp_path: Path, n_reading: int = 3, n_arxiv: int = 3):
        """Helper that produces a mix of reading and arXiv entries."""
        reading = [
            _result(f"Reading {i}", keep=(i % 2 == 0), score=0.8 if i % 2 == 0 else 0.3)
            for i in range(n_reading)
        ]
        arxiv = [
            _result(
                f"Arxiv {i}",
                keep=(i % 2 == 0),
                score=0.7 if i % 2 == 0 else 0.2,
                is_arxiv=True,
            )
            for i in range(n_arxiv)
        ]
        results = reading + arxiv
        metadata = [_meta(r.score) for r in results]
        umap_reducer = MagicMock()
        umap_reducer.transform.return_value = np.random.rand(len(results), 2).astype(
            np.float32
        )
        out = tmp_path / "RSS-2024-01-01-scores.html"
        write_score_viz(
            results=results,
            entry_metadata=metadata,
            vault_2d=_vault_2d(),
            vault_docs=_vault_docs(),
            umap_reducer=umap_reducer,
            threshold_reading=0.5,
            threshold_arxiv=0.4,
            output_path=out,
        )
        return out

    def test_html_has_reading_row_label(self, tmp_path):
        """HTML must contain a 'Reading' section label."""
        out = self._mixed_call(tmp_path)
        html = out.read_text()
        assert "Reading" in html

    def test_html_has_arxiv_row_label(self, tmp_path):
        """HTML must contain an 'arXiv' section label."""
        out = self._mixed_call(tmp_path)
        html = out.read_text()
        assert "arXiv" in html

    def test_html_has_four_axis_domains(self, tmp_path):
        """Layout must define 4 independent axis pairs for the 2×2 grid."""
        out = self._mixed_call(tmp_path)
        html = out.read_text()
        layout_match = re.search(r"var layout = ({.*?});\s*var div", html, re.DOTALL)
        assert layout_match, "Could not find layout JSON in HTML"
        layout = json.loads(layout_match.group(1))
        xaxis_keys = [k for k in layout if k.startswith("xaxis")]
        assert len(xaxis_keys) == 4, f"Expected 4 xaxis keys, got {xaxis_keys}"

    def test_no_scatter_on_x1(self, tmp_path):
        """x1/x3 axes must not carry scatter traces (they are bar chart axes now)."""
        out = self._mixed_call(tmp_path)
        html = out.read_text()
        traces_match = re.search(
            r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
        )
        assert traces_match, "Could not find traces JSON in HTML"
        traces = json.loads(traces_match.group(1))
        scatter_on_bar_axes = [
            t
            for t in traces
            if t.get("type") == "scatter" and t.get("xaxis") in ("x1", "x3")
        ]
        assert not scatter_on_bar_axes, (
            f"Unexpected scatter on bar axes: {scatter_on_bar_axes}"
        )

    def test_bar_traces_present_for_reading(self, tmp_path):
        """Bar traces for the reading row must be present on x1."""
        out = self._mixed_call(tmp_path)
        html = out.read_text()
        traces_match = re.search(
            r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
        )
        assert traces_match, "Could not find traces JSON in HTML"
        traces = json.loads(traces_match.group(1))
        bar_traces = [
            t for t in traces if t.get("type") == "bar" and t.get("xaxis") == "x1"
        ]
        assert bar_traces, "Expected bar traces on x1 for reading row"

    def test_bar_traces_present_for_arxiv(self, tmp_path):
        """Bar traces for the arXiv row must be present on x3."""
        out = self._mixed_call(tmp_path)
        html = out.read_text()
        traces_match = re.search(
            r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
        )
        assert traces_match, "Could not find traces JSON in HTML"
        traces = json.loads(traces_match.group(1))
        bar_traces = [
            t for t in traces if t.get("type") == "bar" and t.get("xaxis") == "x3"
        ]
        assert bar_traces, "Expected bar traces on x3 for arXiv row"

    def test_bar_traces_contain_feed_names(self, tmp_path):
        """Bar chart y-axis values must include the feed name from the entries."""
        out = self._mixed_call(tmp_path)
        html = out.read_text()
        traces_match = re.search(
            r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
        )
        traces = json.loads(traces_match.group(1))
        bar_traces = [t for t in traces if t.get("type") == "bar"]
        all_y = [y for t in bar_traces for y in (t.get("y") or [])]
        # _mixed_call uses feed_name="Test Feed"
        assert "Test Feed" in all_y, (
            f"Expected 'Test Feed' in bar y values, got {all_y}"
        )

    def test_bar_kept_colour_green(self, tmp_path):
        """Kept bar segment must use green colour #2ecc71."""
        out = self._mixed_call(tmp_path)
        html = out.read_text()
        traces_match = re.search(
            r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
        )
        traces = json.loads(traces_match.group(1))
        kept_bars = [
            t for t in traces if t.get("type") == "bar" and "Kept" in t.get("name", "")
        ]
        assert kept_bars, "No kept bar traces found"
        for t in kept_bars:
            assert t["marker"]["color"] == "#2ecc71", (
                f"Expected #2ecc71, got {t['marker']['color']}"
            )

    def test_reading_entries_in_reading_row(self, tmp_path):
        """Reading article titles must appear in traces assigned to the reading row."""
        out = self._mixed_call(tmp_path, n_reading=2, n_arxiv=2)
        html = out.read_text()
        traces_match = re.search(
            r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
        )
        traces = json.loads(traces_match.group(1))
        # Reading row uses x1 (bar) and x2 (UMAP)
        reading_row_text = " ".join(
            str(t.get("text", "") or "") + " ".join(str(v) for v in (t.get("y") or []))
            for t in traces
            if t.get("xaxis") in ("x1", "x2")
        )
        assert "Reading 0" in reading_row_text

    def test_arxiv_entries_in_arxiv_row(self, tmp_path):
        """arXiv article titles must appear in traces assigned to the arXiv row."""
        out = self._mixed_call(tmp_path, n_reading=2, n_arxiv=2)
        html = out.read_text()
        traces_match = re.search(
            r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
        )
        traces = json.loads(traces_match.group(1))
        # arXiv row uses x3 (bar) and x4 (UMAP)
        arxiv_row_text = " ".join(
            str(t.get("text", "") or "") + " ".join(str(v) for v in (t.get("y") or []))
            for t in traces
            if t.get("xaxis") in ("x3", "x4")
        )
        assert "Arxiv 0" in arxiv_row_text

    def test_umap_traces_have_customdata(self, tmp_path):
        """UMAP scatter traces for RSS entries must include customdata (URLs)."""
        out = self._mixed_call(tmp_path)
        html = out.read_text()

        traces_match = re.search(
            r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
        )
        assert traces_match, "Could not find traces JSON in HTML"
        traces = json.loads(traces_match.group(1))
        # UMAP RSS traces are on x2 (reading) and x4 (arXiv)
        umap_rss_traces = [
            t
            for t in traces
            if t.get("type") == "scatter"
            and t.get("xaxis") in ("x2", "x4")
            and t.get("name") in ("Kept", "Rejected")
        ]
        assert umap_rss_traces, "No UMAP RSS traces found on x2/x4"
        for t in umap_rss_traces:
            assert "customdata" in t, (
                f"Trace {t.get('name')} on {t.get('xaxis')} missing customdata"
            )
            assert len(t["customdata"]) == len(t["x"]), (
                "customdata length must match x length"
            )

    def test_html_contains_postmessage_click_handler(self, tmp_path):
        """Generated HTML must include a plotly_click postMessage handler."""
        out = self._mixed_call(tmp_path)
        html = out.read_text()
        assert "plotly_click" in html
        assert "postMessage" in html
        assert "rss-viz-click" in html
