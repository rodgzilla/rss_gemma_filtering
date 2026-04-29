# Spec: Unit Dot Grid — per-feed statistics panel

**Date:** 2026-04-29
**Branch:** feature/html-viz-redesign

## Summary

Replace the stacked horizontal bar chart in the left panel of each row (Reading, arXiv) with a unit dot grid. Each dot represents one article. Feed membership and kept/rejected status are encoded by color.

## Motivation

The stacked bar chart is hard to read at a glance: the absolute counts require reading axis labels, and comparing kept ratios across feeds requires mental arithmetic. A unit dot grid makes the volume and ratio per feed immediately visible — you can see both the number of articles and the proportion kept without any labels.

## Layout context

The overall 2×2 grid layout is unchanged:

```
[ dot grid (Reading) ] [ vault UMAP (Reading) ]
[ dot grid (arXiv)   ] [ vault UMAP (arXiv)   ]
```

The dot grid occupies the same subplot cell as the former bar chart (`x1`/`y1` for Reading, `x3`/`y3` for arXiv).

## Dot grid specification

### Article ordering

Articles are sorted into groups: for each feed (sorted by kept count descending), all kept articles come first, then all rejected articles. Feeds are concatenated in that order, so the grid reads: feed-1-kept, feed-1-rejected, feed-2-kept, feed-2-rejected, …

### Grid layout

Dots are placed left-to-right, wrapping at **20 columns**. Row and column indices are computed from the article's position in the ordered list:

```
col = position % 20
row = position // 20
```

The y-axis is inverted so row 0 is at the top.

### Dot size

- Default: **10px** marker size
- When the total article count for the row exceeds **200**: **6px** marker size

### Color scheme

A fixed palette of 10 distinct hues is defined in `score_viz.py`. Each feed is assigned a hue in order of first appearance (sorted by kept count descending). If there are more than 10 feeds, colors cycle.

For each feed:
- **Kept** articles: intense/saturated version of the feed hue
- **Rejected** articles: pale/desaturated version of the same hue (achieved by blending toward `#d0d0d0` or reducing saturation by ~60%)

### Legend

One Plotly legend entry per feed, using the intense color. The muted color is not given its own legend entry — the kept/rejected distinction is conveyed by saturation contrast within each feed's dot cluster.

A shared legend entry labeled "Kept" / "Rejected" using neutral swatches (green/grey) is added once (Reading row only, `showlegend=True`; arXiv row sets `showlegend=False` to avoid duplication), consistent with the existing UMAP legend pattern.

### Axes

Both axes on the dot grid subplot:
- `showticklabels: false`
- `showgrid: false`
- `zeroline: false`
- `fixedrange: true` (no zoom/pan)

The y-axis is inverted (`autorange: "reversed"`) so dots fill top-to-bottom.

## Implementation scope

### Files changed

- `rss_filter/score_viz.py`: replace `_feed_bar_traces()` with `_feed_dot_traces()`. The new function returns a list of Plotly `scatter` traces (one per feed × kept/rejected status = up to 2N traces for N feeds). Computed x/y positions replace bar chart orientation/value encoding.
- `tests/test_score_viz.py`: replace bar-chart-specific tests with dot-grid tests (column wrap, dot size scaling, color assignment, ordering).

### Files unchanged

- `note_writer.py` — no changes needed
- Layout subplot definitions — subplot cells are reused as-is; axes config changes are inside `_feed_dot_traces()`

## Color palette

```python
_FEED_PALETTE = [
    "#e74c3c",  # red
    "#e67e22",  # orange
    "#f1c40f",  # yellow
    "#27ae60",  # green
    "#1abc9c",  # teal
    "#2980b9",  # blue
    "#8e44ad",  # purple
    "#d35400",  # dark orange
    "#16a085",  # dark teal
    "#c0392b",  # dark red
]
```

Muted variant: linear interpolation 40% toward `#cccccc` in hex RGB space.

## Out of scope

- Clickable dots (no `postMessage` on dot grid — only UMAP dots are clickable)
- Tooltip on dots (hover is a nice-to-have but not required for this spec)
- Dynamic column count based on panel width
