# HTML Visualisation Redesign

**Date:** 2026-04-29
**Status:** Approved

## Problem

The current 2×3 Plotly grid (scatter + vault UMAP + RSS-only UMAP, repeated for Reading and arXiv rows) is embedded as an iframe in Obsidian. It has three issues:

1. The RSS-only UMAP column adds little value and wastes horizontal space.
2. The score-vs-top-1-similarity scatter plot is not very actionable.
3. The chart has dead space at top and bottom inside the Obsidian iframe; panels feel cramped.

## Design

### Layout

A **2 rows × 2 columns** Plotly subplot grid:

- **Row 1 — Reading:** left = per-feed bar chart, right = vault-fitted UMAP
- **Row 2 — arXiv:** left = per-feed bar chart, right = vault-fitted UMAP

Row labels ("Reading", "arXiv") remain as left-side annotations. Column headings ("Entries per feed", "UMAP — vault projection") remain at top.

The RSS-only UMAP column (previously column 3) is **removed entirely** from both `score_viz.py` and the Plotly layout.

### Panel: Per-feed stacked bar chart (replaces scatter)

- **Type:** Plotly horizontal bar, one bar per RSS feed name.
- **Data:** For each feed, two segments — kept count (green `#2ecc71`) and rejected count (grey `#bdc3c7`).
- **Sort order:** Descending by kept count (feed with most kept entries at top).
- **Annotations:** Each bar ends with a `"kept/total"` text label.
- **Computed from:** existing `results` + `entry_metadata` lists — no new pipeline data needed.
- Separate chart per row (Reading feeds only in Reading row, arXiv feeds only in arXiv row).

### Panel: Vault-fitted UMAP (unchanged functionally)

- Same Plotly scatter as current columns 2/5 (vault background + transformed RSS entries, kept/rejected coloured).
- No functional changes; just repositioned to columns 2/4 in the new layout.

### Height and spacing

- Overall chart height stays at `90vh` to fill the Obsidian iframe.
- Y-axis domains adjusted for 2 rows with a gap:
  - Top row: `[0.55, 1.00]`
  - Bottom row: `[0.00, 0.45]`
- X-axis domains adjusted for 2 columns:
  - Left: `[0.00, 0.46]`
  - Right: `[0.54, 1.00]`

### Threshold lines

The reading and arXiv threshold lines (currently on the scatter plots) are removed along with the scatter panels. They may be added as annotations in the bar chart panels in a future iteration if desired.

## Files changed

- `rss_filter/score_viz.py` — primary change: remove RSS-only UMAP traces and layout axes (x3/y3, x6/y6), replace scatter traces (x1/y1, x4/y4) with per-feed bar traces, update layout domains. The `threshold_reading` and `threshold_arxiv` parameters remain in the function signature for backwards compatibility but are no longer used to draw threshold lines.

## Out of scope

- No changes to the scoring pipeline (`score_filter.py`).
- No new data collected at scoring time.
- Interactivity (clicking a bar to filter the UMAP) is a future enhancement.
