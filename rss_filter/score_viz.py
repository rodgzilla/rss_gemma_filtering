"""Interactive score visualisation: scatter and UMAP panels.

Generates a self-contained HTML file (Plotly.js via CDN) with three panels:
  1. Scatter plot: aggregated score vs top-1 similarity, kept/rejected coloured
  2. UMAP 2-D projection of new entries (foreground) + vault background (subsampled)
  3. UMAP 2-D projection fitted on RSS entries only (no vault background)

UMAP for panel 2 is fitted on vault embeddings (Matryoshka 128-dim) and the
fitted model is cached to disk so daily runs only need to *transform* new entries.
UMAP for panel 3 is fitted fresh on each run directly from the RSS batch.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import umap

from rss_filter.embedding_store import EmbeddingStore
from rss_filter.models import FilterResult
from rss_filter.score_filter import MATRYOSHKA_DIM

# ---------------------------------------------------------------------------
# UMAP caching
# ---------------------------------------------------------------------------

_UMAP_PARAMS = dict(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)


def _meta_path(model_path: str) -> Path:
    return Path(model_path).with_suffix(".meta.json")


def _load_umap_meta(model_path: str) -> dict:
    p = _meta_path(model_path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _save_umap_meta(model_path: str, meta: dict) -> None:
    with open(_meta_path(model_path), "w") as f:
        json.dump(meta, f)


def build_or_load_umap(
    store: EmbeddingStore,
    model_path: str,
    force_rebuild: bool = False,
    growth_threshold: float = 0.1,
) -> tuple[umap.UMAP, np.ndarray, list[dict]]:
    """Return a fitted UMAP model, the vault 2-D coordinates, and vault metadata.

    The fitted model and vault coordinates are cached to *model_path* and a
    sidecar ``.meta.json`` file.  Refit is triggered when:
    - ``force_rebuild`` is True, or
    - The cached model file does not exist, or
    - The vault has grown by more than ``growth_threshold`` (fraction) since
      the model was last fitted.

    Returns:
        reducer:        fitted umap.UMAP instance
        vault_2d:       np.ndarray of shape (N_vault, 2)
        vault_docs:     list of dicts {"url", "text", "source_note", "date"}
                        in the same order as vault_2d rows
    """
    vault_docs = store.get_all()
    n_current = len(vault_docs)

    meta = _load_umap_meta(model_path)
    n_fitted = meta.get("n_vault_docs", 0)
    model_exists = Path(model_path).exists()

    growth = (n_current - n_fitted) / max(n_fitted, 1)
    needs_rebuild = force_rebuild or not model_exists or growth > growth_threshold

    if needs_rebuild:
        print(
            f"  Fitting UMAP on {n_current} vault documents "
            f"(128-dim Matryoshka embeddings)…"
        )
        if n_current < 2:
            # Not enough points to fit UMAP; return dummy 2-D coords.
            reducer = None
            vault_2d = np.zeros((n_current, 2), dtype=np.float32)
        else:
            matrix = np.stack(
                [d["embedding"][:MATRYOSHKA_DIM] for d in vault_docs], axis=0
            ).astype(np.float32)
            reducer = umap.UMAP(**_UMAP_PARAMS)
            vault_2d = reducer.fit_transform(matrix).astype(np.float32)
            joblib.dump(reducer, model_path)

        vault_2d_list = vault_2d.tolist()
        meta = {
            "n_vault_docs": n_current,
            "fitted_at": date.today().isoformat(),
            "vault_2d": vault_2d_list,
            "vault_urls": [d["url"] for d in vault_docs],
        }
        _save_umap_meta(model_path, meta)
        print(f"  UMAP model saved to {model_path}.")
    else:
        print(f"  Loading cached UMAP model from {model_path}.")
        reducer = joblib.load(model_path)
        vault_2d = np.array(meta["vault_2d"], dtype=np.float32)
        # Re-order vault_docs to match cached order by URL.
        cached_urls = meta.get("vault_urls", [])
        url_to_doc = {d["url"]: d for d in vault_docs}
        vault_docs = [url_to_doc[u] for u in cached_urls if u in url_to_doc]

    # Strip embeddings from vault_docs before returning (not needed downstream).
    vault_docs_clean = [
        {
            "url": d["url"],
            "text": d["text"],
            "source_note": d["source_note"],
            "date": d["date"],
        }
        for d in vault_docs
    ]
    return reducer, vault_2d, vault_docs_clean


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_PLOTLY_CDN = (
    '<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>'
)


def _js_array(values: list) -> str:
    return json.dumps(values)


def _truncate(text: str, n: int = 80) -> str:
    return text[:n] + "…" if len(text) > n else text


def _build_hover(result: FilterResult, meta: dict) -> str:
    """Build a hover text string for an RSS entry."""
    entry = result.entry
    entry_type = "arXiv" if entry.is_arxiv else "reading"
    lines = [
        f"<b>{_truncate(entry.title, 100)}</b>",
        f"Feed: {entry.feed_name}",
        f"Type: {entry_type}",
        f"Score: {meta['agg_score']:.4f}",
        "---",
    ]
    for i, ex in enumerate(meta["exemplars"], 1):
        label = ex.get("title") or ex["text"]
        lines.append(f"#{i}: {_truncate(label, 80)} ({ex['score']:.3f})")
    return "<br>".join(lines)


def write_score_viz(
    results: list[FilterResult],
    entry_metadata: list[dict],
    vault_2d: np.ndarray,
    vault_docs: list[dict],
    umap_reducer,
    threshold_reading: float,
    threshold_arxiv: float,
    output_path: Path,
    vault_bg_max: int = 500,
) -> None:
    """Write a self-contained interactive HTML visualisation.

    Two rows × three columns:
      Row 1 (Reading): scatter, vault-fitted UMAP, RSS-only UMAP
      Row 2 (arXiv):   scatter, vault-fitted UMAP, RSS-only UMAP

    Args:
        threshold_reading: Score threshold used for general reading entries.
        threshold_arxiv:   Score threshold used for arXiv entries.
        vault_bg_max: Maximum number of vault background points shown in UMAP
                      panels.  If the vault is larger a random subsample is
                      drawn (seed=42).
    """
    if not results:
        print("  No entries to visualise — skipping HTML output.")
        return

    # --- Split into reading / arXiv groups ---
    reading_pairs = [
        (r, m) for r, m in zip(results, entry_metadata) if not r.entry.is_arxiv
    ]
    arxiv_pairs = [(r, m) for r, m in zip(results, entry_metadata) if r.entry.is_arxiv]

    def _group_arrays(pairs):
        if not pairs:
            return np.array([]), np.array([]), np.array([], dtype=bool), [], None
        rs, ms = zip(*pairs)
        scores = np.array([m["agg_score"] for m in ms])
        top1 = np.array(
            [m["exemplars"][0]["score"] if m["exemplars"] else 0.0 for m in ms]
        )
        kept = np.array([r.keep for r in rs])
        hovers = [_build_hover(r, m) for r, m in zip(rs, ms)]
        if len(ms) >= 2:
            matrix = np.stack([m["embedding_128"] for m in ms], axis=0).astype(
                np.float32
            )
        else:
            matrix = None
        return scores, top1, kept, hovers, matrix

    r_scores, r_top1, r_kept, r_hovers, r_matrix = _group_arrays(reading_pairs)
    a_scores, a_top1, a_kept, a_hovers, a_matrix = _group_arrays(arxiv_pairs)

    kept_colour = "#2ecc71"
    rejected_colour = "#bdc3c7"

    # --- Vault background (shared between rows) ---
    vault_2d_bg = vault_2d
    vault_docs_bg = vault_docs
    if len(vault_2d) > vault_bg_max:
        rng = np.random.default_rng(seed=42)
        idx_bg = rng.choice(len(vault_2d), vault_bg_max, replace=False)
        idx_bg.sort()
        vault_2d_bg = vault_2d[idx_bg]
        vault_docs_bg = [vault_docs[i] for i in idx_bg]

    def _vault_trace(xax, yax):
        vault_hover = [
            f"<b>{_truncate(d['text'], 100)}</b><br>Source: {d['source_note']}"
            for d in vault_docs_bg
        ]
        return {
            "type": "scatter",
            "x": vault_2d_bg[:, 0].tolist(),
            "y": vault_2d_bg[:, 1].tolist(),
            "mode": "markers",
            "name": "Vault",
            "showlegend": False,
            "marker": {
                "color": "#ecf0f1",
                "size": 4,
                "opacity": 0.5,
                "line": {"color": "#95a5a6", "width": 0.5},
            },
            "text": vault_hover,
            "hovertemplate": "%{text}<extra></extra>",
            "xaxis": xax,
            "yaxis": yax,
        }

    def _umap_scatter(xy, mask, hovers, name, colour, xax, yax, showlegend=False):
        idx = np.where(mask)[0].tolist()
        return {
            "type": "scatter",
            "x": xy[mask, 0].tolist(),
            "y": xy[mask, 1].tolist(),
            "mode": "markers",
            "name": name,
            "showlegend": showlegend,
            "marker": {
                "color": colour,
                "size": 10,
                "opacity": 0.5,
                "line": {"color": "white", "width": 1},
            },
            "text": [hovers[i] for i in idx],
            "hovertemplate": "%{text}<extra></extra>",
            "xaxis": xax,
            "yaxis": yax,
        }

    def _feed_bar_traces(pairs, xax, yax):
        """Build kept + rejected horizontal bar traces for a group of (result, meta) pairs."""
        if not pairs:
            return []

        kept_by_feed: dict[str, int] = defaultdict(int)
        rejected_by_feed: dict[str, int] = defaultdict(int)
        for r, _ in pairs:
            feed = r.entry.feed_name
            if r.keep:
                kept_by_feed[feed] += 1
            else:
                rejected_by_feed[feed] += 1
        all_feeds = sorted(
            set(kept_by_feed) | set(rejected_by_feed),
            key=lambda f: kept_by_feed[f],
            reverse=True,
        )
        kept_counts = [kept_by_feed[f] for f in all_feeds]
        rejected_counts = [rejected_by_feed[f] for f in all_feeds]
        kept_trace = {
            "type": "bar",
            "orientation": "h",
            "name": "Kept",
            "y": all_feeds,
            "x": kept_counts,
            "marker": {"color": kept_colour, "opacity": 0.75},
            "xaxis": xax,
            "yaxis": yax,
        }
        rejected_trace = {
            "type": "bar",
            "orientation": "h",
            "name": "Rejected",
            "y": all_feeds,
            "x": rejected_counts,
            "marker": {"color": rejected_colour, "opacity": 0.4},
            "xaxis": xax,
            "yaxis": yax,
        }
        return [kept_trace, rejected_trace]

    # --- Vault-fitted UMAP transform per group ---
    def _vault_umap_traces(matrix, kept, hovers, xax, yax):
        traces = []
        if vault_2d_bg is not None and len(vault_2d_bg) > 0:
            traces.append(_vault_trace(xax, yax))
        if matrix is not None and umap_reducer is not None:
            try:
                xy = umap_reducer.transform(matrix).astype(np.float32)[: len(matrix)]
                traces.append(
                    _umap_scatter(xy, kept, hovers, "Kept", kept_colour, xax, yax)
                )
                traces.append(
                    _umap_scatter(
                        xy, ~kept, hovers, "Rejected", rejected_colour, xax, yax
                    )
                )
            except Exception as e:
                print(f"  [WARN] UMAP transform failed: {e}")
        return traces

    all_traces: list[dict] = []

    # Row 1 — Reading bar chart (x1/y1)
    all_traces += _feed_bar_traces(reading_pairs, "x1", "y1")

    # Row 2 — arXiv bar chart (x3/y3)
    all_traces += _feed_bar_traces(arxiv_pairs, "x3", "y3")

    # Vault-fitted UMAP traces
    all_traces += _vault_umap_traces(r_matrix, r_kept, r_hovers, "x2", "y2")
    all_traces += _vault_umap_traces(a_matrix, a_kept, a_hovers, "x4", "y4")

    traces_json = json.dumps(all_traces)

    today = date.today().isoformat()
    n_kept = sum(r.keep for r in results)
    n_total = len(results)

    # --- Layout: 2 rows × 2 columns ---
    TOP_Y = [0.55, 1.00]
    BOT_Y = [0.00, 0.45]
    LEFT_X = [0.00, 0.46]
    RIGHT_X = [0.54, 1.00]

    layout = {
        "title": {"text": f"RSS Score Analysis — {today} ({n_kept}/{n_total} kept)"},
        "hovermode": "closest",
        "barmode": "stack",
        # Row 1 — Reading
        "xaxis": {"domain": LEFT_X, "title": "Entry count", "anchor": "y1"},
        "yaxis": {"domain": TOP_Y, "anchor": "x1", "title": "Feed"},
        "xaxis2": {"domain": RIGHT_X, "title": "UMAP dim 1", "anchor": "y2"},
        "yaxis2": {"domain": TOP_Y, "title": "UMAP dim 2", "anchor": "x2"},
        # Row 2 — arXiv
        "xaxis3": {"domain": LEFT_X, "title": "Entry count", "anchor": "y3"},
        "yaxis3": {"domain": BOT_Y, "anchor": "x3", "title": "Feed"},
        "xaxis4": {"domain": RIGHT_X, "title": "UMAP dim 1", "anchor": "y4"},
        "yaxis4": {"domain": BOT_Y, "title": "UMAP dim 2", "anchor": "x4"},
        "legend": {"orientation": "h", "y": -0.05},
        "paper_bgcolor": "#1e1e2e",
        "plot_bgcolor": "#2a2a3e",
        "font": {"color": "#cdd6f4"},
        "shapes": [],
        "annotations": [
            # Column headings
            {
                "text": "Entries per feed",
                "xref": "paper",
                "yref": "paper",
                "x": 0.23,
                "y": 1.04,
                "showarrow": False,
                "font": {"size": 12, "color": "#a6adc8"},
            },
            {
                "text": "UMAP — vault projection",
                "xref": "paper",
                "yref": "paper",
                "x": 0.77,
                "y": 1.04,
                "showarrow": False,
                "font": {"size": 12, "color": "#a6adc8"},
            },
            # Row labels
            {
                "text": "<b>Reading</b>",
                "xref": "paper",
                "yref": "paper",
                "x": -0.01,
                "y": (TOP_Y[0] + TOP_Y[1]) / 2,
                "showarrow": False,
                "textangle": -90,
                "font": {"size": 13, "color": "#cdd6f4"},
            },
            {
                "text": "<b>arXiv</b>",
                "xref": "paper",
                "yref": "paper",
                "x": -0.01,
                "y": (BOT_Y[0] + BOT_Y[1]) / 2,
                "showarrow": False,
                "textangle": -90,
                "font": {"size": 13, "color": "#cdd6f4"},
            },
        ],
    }
    layout_json = json.dumps(layout)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RSS Score Analysis — {today}</title>
  {_PLOTLY_CDN}
  <style>
    body {{ margin: 0; padding: 12px; background: #1e1e2e; font-family: sans-serif; }}
    #chart {{ width: 100%; height: 90vh; }}
  </style>
</head>
<body>
  <div id="chart"></div>
  <script>
    var traces = {traces_json};
    var layout = {layout_json};
    Plotly.newPlot('chart', traces, layout, {{responsive: true}});
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"  Visualisation written to: {output_path}")
