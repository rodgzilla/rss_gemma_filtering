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

    Three panels:
      1. Scatter: aggregated score vs top-1 similarity (kept/rejected), with
         separate threshold lines for reading entries (red) and arXiv (orange)
      2. UMAP 2-D projection: vault background (subsampled) + new entries foreground
      3. UMAP 2-D projection fitted on RSS entries only (no vault background)

    Args:
        threshold_reading: Score threshold used for general reading entries.
        threshold_arxiv:   Score threshold used for arXiv entries.
        vault_bg_max: Maximum number of vault background points shown in panel 2.
                      If the vault is larger a random subsample is drawn (seed=42).
    """
    if not results:
        print("  No entries to visualise — skipping HTML output.")
        return

    scores = np.array([m["agg_score"] for m in entry_metadata])
    top1_sims = np.array(
        [m["exemplars"][0]["score"] if m["exemplars"] else 0.0 for m in entry_metadata]
    )
    kept_mask = np.array([r.keep for r in results])
    hovers = [_build_hover(r, m) for r, m in zip(results, entry_metadata)]

    kept_colour = "#2ecc71"
    rejected_colour = "#bdc3c7"

    # --- Panel 1: score vs top-1 similarity scatter ---
    def _scatter(mask: np.ndarray, name: str, colour: str, xax: str, yax: str) -> dict:
        idx = np.where(mask)[0].tolist()
        return {
            "type": "scatter",
            "x": scores[mask].tolist(),
            "y": top1_sims[mask].tolist(),
            "mode": "markers",
            "name": name,
            "marker": {"color": colour, "size": 8, "opacity": 0.8},
            "text": [hovers[i] for i in idx],
            "hovertemplate": "%{text}<extra></extra>",
            "xaxis": xax,
            "yaxis": yax,
        }

    p1_kept = _scatter(kept_mask, "Kept", kept_colour, "x1", "y1")
    p1_rej = _scatter(~kept_mask, "Rejected", rejected_colour, "x1", "y1")

    # --- Panel 2: UMAP vault-fitted projection ---

    # Subsample vault background
    vault_2d_bg = vault_2d
    vault_docs_bg = vault_docs
    if len(vault_2d) > vault_bg_max:
        rng = np.random.default_rng(seed=42)
        idx_bg = rng.choice(len(vault_2d), vault_bg_max, replace=False)
        idx_bg.sort()
        vault_2d_bg = vault_2d[idx_bg]
        vault_docs_bg = [vault_docs[i] for i in idx_bg]

    # Transform new entries with the vault-fitted UMAP
    umap_xy: np.ndarray | None = None
    new_matrix: np.ndarray | None = None
    if len(entry_metadata) >= 2:
        new_matrix = np.stack(
            [m["embedding_128"] for m in entry_metadata], axis=0
        ).astype(np.float32)
        if umap_reducer is not None:
            try:
                umap_xy = umap_reducer.transform(new_matrix).astype(np.float32)
            except Exception as e:
                print(f"  [WARN] UMAP transform failed: {e}")
                umap_xy = None

    p2_traces: list[dict] = []
    if vault_2d_bg is not None and len(vault_2d_bg) > 0:
        vault_hover = [
            f"<b>{_truncate(d['text'], 100)}</b><br>Source: {d['source_note']}"
            for d in vault_docs_bg
        ]
        p2_traces.append(
            {
                "type": "scatter",
                "x": vault_2d_bg[:, 0].tolist(),
                "y": vault_2d_bg[:, 1].tolist(),
                "mode": "markers",
                "name": "Vault",
                "marker": {
                    "color": "#ecf0f1",
                    "size": 4,
                    "opacity": 0.5,
                    "line": {"color": "#95a5a6", "width": 0.5},
                },
                "text": vault_hover,
                "hovertemplate": "%{text}<extra></extra>",
                "xaxis": "x3",
                "yaxis": "y3",
            }
        )

    if umap_xy is not None:

        def _umap_scatter(mask: np.ndarray, name: str, colour: str) -> dict:
            idx = np.where(mask)[0].tolist()
            return {
                "type": "scatter",
                "x": umap_xy[mask, 0].tolist(),
                "y": umap_xy[mask, 1].tolist(),
                "mode": "markers",
                "name": name,
                "showlegend": False,
                "marker": {
                    "color": colour,
                    "size": 10,
                    "opacity": 0.9,
                    "line": {"color": "white", "width": 1},
                },
                "text": [hovers[i] for i in idx],
                "hovertemplate": "%{text}<extra></extra>",
                "xaxis": "x3",
                "yaxis": "y3",
            }

        p2_traces.append(_umap_scatter(kept_mask, "Kept", kept_colour))
        p2_traces.append(_umap_scatter(~kept_mask, "Rejected", rejected_colour))

    # --- Panel 3: UMAP fitted on RSS entries only ---
    p3_traces: list[dict] = []
    if new_matrix is not None and len(new_matrix) >= 2:
        try:
            print("  Fitting UMAP on RSS entries only…")
            rss_reducer = umap.UMAP(**_UMAP_PARAMS)
            rss_2d = rss_reducer.fit_transform(new_matrix).astype(np.float32)

            def _rss_scatter(mask: np.ndarray, name: str, colour: str) -> dict:
                idx = np.where(mask)[0].tolist()
                return {
                    "type": "scatter",
                    "x": rss_2d[mask, 0].tolist(),
                    "y": rss_2d[mask, 1].tolist(),
                    "mode": "markers",
                    "name": name,
                    "showlegend": False,
                    "marker": {
                        "color": colour,
                        "size": 10,
                        "opacity": 0.9,
                        "line": {"color": "white", "width": 1},
                    },
                    "text": [hovers[i] for i in idx],
                    "hovertemplate": "%{text}<extra></extra>",
                    "xaxis": "x5",
                    "yaxis": "y5",
                }

            p3_traces.append(_rss_scatter(kept_mask, "Kept", kept_colour))
            p3_traces.append(_rss_scatter(~kept_mask, "Rejected", rejected_colour))
        except Exception as e:
            print(f"  [WARN] RSS-only UMAP failed: {e}")

    all_traces = [p1_kept, p1_rej] + p2_traces + p3_traces
    traces_json = json.dumps(all_traces)

    today = date.today().isoformat()
    n_kept = int(kept_mask.sum())
    n_total = len(results)

    layout = {
        "title": {"text": f"RSS Score Analysis — {today} ({n_kept}/{n_total} kept)"},
        "hovermode": "closest",
        # Panel 1: left third — score vs top-1 scatter
        "xaxis": {"domain": [0.00, 0.30], "title": "Aggregated score", "anchor": "y1"},
        "yaxis": {"domain": [0.00, 1.00], "title": "Top-1 similarity", "anchor": "x1"},
        # Panel 2: middle third — vault-fitted UMAP
        "xaxis3": {"domain": [0.37, 0.63], "title": "UMAP dim 1", "anchor": "y3"},
        "yaxis3": {"domain": [0.00, 1.00], "title": "UMAP dim 2", "anchor": "x3"},
        # Panel 3: right third — RSS-only UMAP
        "xaxis5": {"domain": [0.70, 1.00], "title": "UMAP dim 1", "anchor": "y5"},
        "yaxis5": {"domain": [0.00, 1.00], "title": "UMAP dim 2", "anchor": "x5"},
        "legend": {"orientation": "h", "y": -0.05},
        "paper_bgcolor": "#1e1e2e",
        "plot_bgcolor": "#2a2a3e",
        "font": {"color": "#cdd6f4"},
        "shapes": [
            # Reading threshold line (red)
            {
                "type": "line",
                "xref": "x1",
                "yref": "paper",
                "x0": threshold_reading,
                "x1": threshold_reading,
                "y0": 0,
                "y1": 1,
                "line": {"color": "#e74c3c", "width": 2, "dash": "dash"},
            },
            # arXiv threshold line (orange)
            {
                "type": "line",
                "xref": "x1",
                "yref": "paper",
                "x0": threshold_arxiv,
                "x1": threshold_arxiv,
                "y0": 0,
                "y1": 1,
                "line": {"color": "#f39c12", "width": 2, "dash": "dash"},
            },
        ],
        "annotations": [
            {
                "text": "Score vs Top-1 Similarity",
                "xref": "paper",
                "yref": "paper",
                "x": 0.15,
                "y": 1.03,
                "showarrow": False,
                "font": {"size": 13},
            },
            {
                "text": "UMAP — vault projection",
                "xref": "paper",
                "yref": "paper",
                "x": 0.50,
                "y": 1.03,
                "showarrow": False,
                "font": {"size": 13},
            },
            {
                "text": "UMAP — RSS entries only",
                "xref": "paper",
                "yref": "paper",
                "x": 0.85,
                "y": 1.03,
                "showarrow": False,
                "font": {"size": 13},
            },
            {
                "text": f"reading = {threshold_reading:.4f}",
                "xref": "x1",
                "yref": "paper",
                "x": threshold_reading,
                "y": 0.97,
                "showarrow": False,
                "font": {"color": "#e74c3c", "size": 11},
            },
            {
                "text": f"arxiv = {threshold_arxiv:.4f}",
                "xref": "x1",
                "yref": "paper",
                "x": threshold_arxiv,
                "y": 0.91,
                "showarrow": False,
                "font": {"color": "#f39c12", "size": 11},
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
