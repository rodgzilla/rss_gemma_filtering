"""Tests for rss_filter.score_viz."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from rss_filter.models import FilterResult, RSSEntry
from rss_filter.score_viz import (
    _muted_colour,
    _FEED_PALETTE,
    write_score_viz,
    build_or_load_umap,
)


def test_feed_palette_has_twelve_entries():
    assert len(_FEED_PALETTE) == 12


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
        umap_reducer.transform.return_value = np.random.rand(
            max(len(results), 1), 2
        ).astype(np.float32)
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

    def test_no_bar_traces_on_x1(self, tmp_path):
        """x1/x3 axes must not carry bar traces — dot grid uses scatter."""
        out = self._mixed_call(tmp_path)
        html = out.read_text()
        traces_match = re.search(
            r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
        )
        assert traces_match
        traces = json.loads(traces_match.group(1))
        bar_on_dot_axes = [
            t
            for t in traces
            if t.get("type") == "bar" and t.get("xaxis") in ("x1", "x3")
        ]
        assert not bar_on_dot_axes, (
            f"Unexpected bar traces on dot axes: {bar_on_dot_axes}"
        )

    def test_dot_traces_present_for_reading(self, tmp_path):
        """Scatter traces for the reading row must be present on x1."""
        out = self._mixed_call(tmp_path)
        html = out.read_text()
        traces_match = re.search(
            r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
        )
        traces = json.loads(traces_match.group(1))
        dot_traces = [
            t for t in traces if t.get("type") == "scatter" and t.get("xaxis") == "x1"
        ]
        assert dot_traces, "Expected scatter dot traces on x1 for reading row"

    def test_dot_traces_present_for_arxiv(self, tmp_path):
        """Scatter traces for the arXiv row must be present on x3."""
        out = self._mixed_call(tmp_path)
        html = out.read_text()
        traces_match = re.search(
            r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
        )
        traces = json.loads(traces_match.group(1))
        dot_traces = [
            t for t in traces if t.get("type") == "scatter" and t.get("xaxis") == "x3"
        ]
        assert dot_traces, "Expected scatter dot traces on x3 for arXiv row"

    def test_dot_traces_x_positions_wrap_at_20_columns(self, tmp_path):
        """Dot x positions must be in range [0, 19] (20-column wrap)."""
        out = self._mixed_call(tmp_path, n_reading=25, n_arxiv=0)
        html = out.read_text()
        traces_match = re.search(
            r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
        )
        traces = json.loads(traces_match.group(1))
        dot_traces = [
            t for t in traces if t.get("type") == "scatter" and t.get("xaxis") == "x1"
        ]
        all_x = [x for t in dot_traces for x in t.get("x", [])]
        assert all_x, "No x positions found on x1 dot traces"
        assert max(all_x) <= 19, f"Max x should be <=19, got {max(all_x)}"
        assert min(all_x) >= 0

    def test_dot_size_8px_for_small_feed(self, tmp_path):
        """Dot marker size must be 8 when total entries <= 200."""
        out = self._mixed_call(tmp_path, n_reading=4, n_arxiv=0)
        html = out.read_text()
        traces_match = re.search(
            r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
        )
        traces = json.loads(traces_match.group(1))
        dot_traces = [
            t for t in traces if t.get("type") == "scatter" and t.get("xaxis") == "x1"
        ]
        for t in dot_traces:
            assert t["marker"]["size"] == 8, (
                f"Expected size 8, got {t['marker']['size']}"
            )

    def test_dot_size_6px_for_medium_feed(self, tmp_path):
        """Dot marker size must be 6 when total entries > 200 and <= 500."""
        out = self._mixed_call(tmp_path, n_reading=201, n_arxiv=0)
        html = out.read_text()
        traces_match = re.search(
            r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
        )
        traces = json.loads(traces_match.group(1))
        dot_traces = [
            t for t in traces if t.get("type") == "scatter" and t.get("xaxis") == "x1"
        ]
        for t in dot_traces:
            assert t["marker"]["size"] == 6, (
                f"Expected size 6, got {t['marker']['size']}"
            )

    def test_dot_size_4px_for_large_feed(self, tmp_path):
        """Dot marker size must be 4 when total entries > 500."""
        out = self._mixed_call(tmp_path, n_reading=501, n_arxiv=0)
        html = out.read_text()
        traces_match = re.search(
            r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
        )
        traces = json.loads(traces_match.group(1))
        dot_traces = [
            t for t in traces if t.get("type") == "scatter" and t.get("xaxis") == "x1"
        ]
        for t in dot_traces:
            assert t["marker"]["size"] == 4, (
                f"Expected size 4, got {t['marker']['size']}"
            )

    def test_dot_traces_feed_names_in_legend(self, tmp_path):
        """Each feed must appear as a named legend entry in the dot traces."""
        out = self._mixed_call(tmp_path)
        html = out.read_text()
        traces_match = re.search(
            r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
        )
        traces = json.loads(traces_match.group(1))
        dot_traces = [
            t for t in traces if t.get("type") == "scatter" and t.get("xaxis") == "x1"
        ]
        names = [t.get("name", "") for t in dot_traces]
        assert any("Test Feed" in n for n in names), (
            f"Expected 'Test Feed' in dot trace names, got {names}"
        )

    def test_reading_entries_in_reading_row(self, tmp_path):
        """Reading feed name must appear in traces assigned to the reading row."""
        out = self._mixed_call(tmp_path, n_reading=2, n_arxiv=2)
        html = out.read_text()
        traces_match = re.search(
            r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
        )
        traces = json.loads(traces_match.group(1))
        reading_row_text = " ".join(
            str(t.get("name", "")) + str(t.get("hovertemplate", ""))
            for t in traces
            if t.get("xaxis") in ("x1", "x2")
        )
        assert "Test Feed" in reading_row_text

    def test_arxiv_entries_in_arxiv_row(self, tmp_path):
        """arXiv feed name must appear in traces assigned to the arXiv row."""
        out = self._mixed_call(tmp_path, n_reading=2, n_arxiv=2)
        html = out.read_text()
        traces_match = re.search(
            r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
        )
        traces = json.loads(traces_match.group(1))
        arxiv_row_text = " ".join(
            str(t.get("name", "")) + str(t.get("hovertemplate", ""))
            for t in traces
            if t.get("xaxis") in ("x3", "x4")
        )
        assert "Test Feed" in arxiv_row_text

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


# ---------------------------------------------------------------------------
# build_or_load_umap with source_filter
# ---------------------------------------------------------------------------


def _make_mock_store(docs: list[dict]):
    """Return a MagicMock store whose get_all() returns docs."""
    store = MagicMock()
    store.get_all.return_value = docs
    return store


def _fake_vault_docs(n: int, source: str = "reading") -> list[dict]:
    """Return n fake vault docs with embeddings, tagged with source."""
    return [
        {
            "url": f"http://vault.com/{source}/{i}",
            "text": f"Vault {source} doc {i}",
            "source_note": source,
            "date": "2024-01-01",
            "embedding": np.random.rand(256).astype(np.float32),
        }
        for i in range(n)
    ]


class TestBuildOrLoadUmapSourceFilter:
    def test_source_filter_arxiv_only_fits_arxiv_docs(self, tmp_path):
        """With source_filter='arxiv', only arXiv docs are used."""
        reading_docs = _fake_vault_docs(20, source="reading")
        arxiv_docs = _fake_vault_docs(20, source="arxiv")
        store = _make_mock_store(reading_docs + arxiv_docs)
        model_path = str(tmp_path / "umap_arxiv.joblib")

        reducer, vault_2d, vault_docs = build_or_load_umap(
            store=store,
            model_path=model_path,
            source_filter="arxiv",
        )

        # Only arXiv-tagged docs should be returned
        assert len(vault_docs) == 20
        assert all(d["source_note"] == "arxiv" for d in vault_docs)
        assert vault_2d.shape == (20, 2)

    def test_source_filter_none_returns_all_docs(self, tmp_path):
        """Without source_filter, all vault docs are used (existing behaviour)."""
        reading_docs = _fake_vault_docs(20, source="reading")
        arxiv_docs = _fake_vault_docs(20, source="arxiv")
        store = _make_mock_store(reading_docs + arxiv_docs)
        model_path = str(tmp_path / "umap_all.joblib")

        reducer, vault_2d, vault_docs = build_or_load_umap(
            store=store,
            model_path=model_path,
            source_filter=None,
        )

        assert len(vault_docs) == 40
        assert vault_2d.shape == (40, 2)

    def test_source_filter_no_matching_docs_returns_empty(self, tmp_path):
        """If no docs match the filter, return (None, zeros(0,2), [])."""
        reading_docs = _fake_vault_docs(5, source="reading")
        store = _make_mock_store(reading_docs)
        model_path = str(tmp_path / "umap_arxiv_empty.joblib")

        reducer, vault_2d, vault_docs = build_or_load_umap(
            store=store,
            model_path=model_path,
            source_filter="arxiv",
        )

        assert reducer is None
        assert vault_2d.shape == (0, 2)
        assert vault_docs == []

    def test_source_filter_single_matching_doc_returns_empty(self, tmp_path):
        """If exactly 1 doc matches, return (None, zeros(1,2), []) — not crash."""
        docs = _fake_vault_docs(1, source="arxiv")
        store = _make_mock_store(docs)
        model_path = str(tmp_path / "umap_arxiv_one.joblib")

        reducer, vault_2d, vault_docs = build_or_load_umap(
            store=store,
            model_path=model_path,
            source_filter="arxiv",
        )

        assert reducer is None
        assert vault_2d.shape == (1, 2)
        assert vault_docs == []


class TestWriteScoreVizArxivVault:
    """Tests for arXiv-specific vault params in write_score_viz."""

    def _arxiv_vault_docs(self, n: int) -> list[dict]:
        return [
            {
                "url": f"http://arxiv.com/{i}",
                "text": f"arXiv vault {i}",
                "source_note": "arxiv",
                "date": "2024-06-01",
            }
            for i in range(n)
        ]

    def test_arxiv_vault_bg_uses_arxiv_coords(self, tmp_path):
        """When arxiv_vault_2d is provided, arXiv panel background uses those coords."""
        n_arxiv = 4
        arxiv_vault_2d = np.full((n_arxiv, 2), 99.0, dtype=np.float32)  # distinctive
        arxiv_vault_docs = self._arxiv_vault_docs(n_arxiv)

        results = [
            _result(f"arXiv {i}", keep=True, score=0.9, is_arxiv=True) for i in range(2)
        ]
        metadata = [_meta(0.9) for _ in results]

        arxiv_reducer = MagicMock()
        arxiv_reducer.transform.return_value = np.random.rand(2, 2).astype(np.float32)

        out = tmp_path / "test.html"
        write_score_viz(
            results=results,
            entry_metadata=metadata,
            vault_2d=np.zeros((3, 2), dtype=np.float32),
            vault_docs=[
                {
                    "url": "u",
                    "text": "t",
                    "source_note": "reading",
                    "date": "2024-01-01",
                }
            ]
            * 3,
            umap_reducer=MagicMock(
                transform=MagicMock(
                    return_value=np.random.rand(2, 2).astype(np.float32)
                )
            ),
            threshold_reading=0.5,
            threshold_arxiv=0.4,
            output_path=out,
            arxiv_vault_2d=arxiv_vault_2d,
            arxiv_vault_docs=arxiv_vault_docs,
            arxiv_umap_reducer=arxiv_reducer,
        )

        html = out.read_text()
        # The distinctive 99.0 value should appear in the JSON trace data
        assert "99.0" in html

    def test_arxiv_vault_none_falls_back_to_reading_vault(self, tmp_path):
        """Without arxiv args, arXiv panel uses reading vault (backward compat)."""
        results = [
            _result(f"arXiv {i}", keep=True, score=0.9, is_arxiv=True) for i in range(2)
        ]
        metadata = [_meta(0.9) for _ in results]

        out = tmp_path / "test_fallback.html"
        umap_reducer = MagicMock()
        umap_reducer.transform.return_value = np.random.rand(2, 2).astype(np.float32)

        write_score_viz(
            results=results,
            entry_metadata=metadata,
            vault_2d=np.zeros((3, 2), dtype=np.float32),
            vault_docs=[
                {
                    "url": "u",
                    "text": "t",
                    "source_note": "reading",
                    "date": "2024-01-01",
                }
            ]
            * 3,
            umap_reducer=umap_reducer,
            threshold_reading=0.5,
            threshold_arxiv=0.4,
            output_path=out,
            # No arxiv_vault_* args
        )

        assert out.exists()
