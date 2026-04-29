# Unit Dot Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stacked horizontal bar chart in the left panel of each row with a unit dot grid — one dot per article, colored by feed (intense = kept, muted = rejected).

**Architecture:** Add a `_feed_dot_traces()` helper in `score_viz.py` that computes (x, y) dot positions using a 20-column wrap layout, assigns per-feed colors from a fixed palette, and returns Plotly `scatter` traces. Replace all calls to `_feed_bar_traces()` with `_feed_dot_traces()`. Update axis config for the left panels (hide tick labels, disable zoom, invert y). Update tests to match the new trace structure.

**Tech Stack:** Python, Plotly.js (JSON trace dicts), pytest

---

## File map

- **Modify:** `.worktrees/html-viz-redesign/rss_filter/score_viz.py`
  - Add `_FEED_PALETTE` module-level constant
  - Add `_muted_colour()` helper
  - Replace `_feed_bar_traces()` inner function with `_feed_dot_traces()`
  - Update layout dict: remove `barmode`, update x1/x3/y1/y3 axis config, remove `Entry count` x-axis title
- **Modify:** `.worktrees/html-viz-redesign/tests/test_score_viz.py`
  - Replace bar-chart-specific tests with dot-grid tests

---

## Task 1: Add `_FEED_PALETTE` and `_muted_colour()` to `score_viz.py`

**Files:**
- Modify: `.worktrees/html-viz-redesign/rss_filter/score_viz.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_score_viz.py` inside `class TestWriteScoreViz` (or as a standalone function at module level):

```python
from rss_filter.score_viz import _muted_colour, _FEED_PALETTE


def test_feed_palette_has_ten_entries():
    assert len(_FEED_PALETTE) == 10


def test_muted_colour_is_paler():
    """Muted colour should be closer to #cccccc than the original."""
    intense = "#e74c3c"
    muted = _muted_colour(intense)
    # Muted red component should be higher than intense red (blended toward #cccccc)
    r_intense = int(intense[1:3], 16)
    r_muted = int(muted[1:3], 16)
    assert r_muted > r_intense


def test_muted_colour_returns_hex_string():
    muted = _muted_colour("#2980b9")
    assert muted.startswith("#")
    assert len(muted) == 7
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/rodgzilla/Documents/rss_gemma_filtering/.worktrees/html-viz-redesign
pytest tests/test_score_viz.py::test_feed_palette_has_ten_entries tests/test_score_viz.py::test_muted_colour_is_paler tests/test_score_viz.py::test_muted_colour_returns_hex_string -v
```

Expected: `ImportError` or `FAILED` — `_muted_colour` and `_FEED_PALETTE` not yet defined.

- [ ] **Step 3: Add palette and helper to `score_viz.py`**

After the existing imports at the top of `score_viz.py` (after line 27, before `_UMAP_PARAMS`), add:

```python
# ---------------------------------------------------------------------------
# Per-feed dot grid colour palette
# ---------------------------------------------------------------------------

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


def _muted_colour(hex_colour: str, blend: float = 0.4) -> str:
    """Return a paler version of *hex_colour* by blending toward #cccccc.

    *blend* = fraction of the original hue to keep (0 = all grey, 1 = original).
    """
    base = 0xCC
    r = int(hex_colour[1:3], 16)
    g = int(hex_colour[3:5], 16)
    b = int(hex_colour[5:7], 16)
    mr = round(r * blend + base * (1 - blend))
    mg = round(g * blend + base * (1 - blend))
    mb = round(b * blend + base * (1 - blend))
    return f"#{mr:02x}{mg:02x}{mb:02x}"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_score_viz.py::test_feed_palette_has_ten_entries tests/test_score_viz.py::test_muted_colour_is_paler tests/test_score_viz.py::test_muted_colour_returns_hex_string -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Run the full test suite to check nothing is broken**

```bash
pytest tests/test_score_viz.py -v
```

Expected: all existing tests pass (no regressions).

- [ ] **Step 6: Commit**

```bash
cd /home/rodgzilla/Documents/rss_gemma_filtering/.worktrees/html-viz-redesign
git add rss_filter/score_viz.py tests/test_score_viz.py
git commit -m "feat: add _FEED_PALETTE and _muted_colour helper for dot grid"
```

---

## Task 2: Replace `_feed_bar_traces()` with `_feed_dot_traces()`

**Files:**
- Modify: `.worktrees/html-viz-redesign/rss_filter/score_viz.py`

- [ ] **Step 1: Write failing tests**

Replace the following tests in `tests/test_score_viz.py` (remove the old bar-chart tests and add new dot-grid tests). Remove these tests:
- `test_no_scatter_on_x1`
- `test_bar_traces_present_for_reading`
- `test_bar_traces_present_for_arxiv`
- `test_bar_traces_contain_feed_names`
- `test_bar_kept_colour_green`

Add these new tests inside `class TestWriteScoreViz`:

```python
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
        t for t in traces if t.get("type") == "bar" and t.get("xaxis") in ("x1", "x3")
    ]
    assert not bar_on_dot_axes, f"Unexpected bar traces on dot axes: {bar_on_dot_axes}"

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

def test_dot_size_10px_for_small_feed(self, tmp_path):
    """Dot marker size must be 10 when total entries <= 200."""
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
        assert t["marker"]["size"] == 10, (
            f"Expected size 10, got {t['marker']['size']}"
        )

def test_dot_size_6px_for_large_feed(self, tmp_path):
    """Dot marker size must be 6 when total entries > 200."""
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
```

Note: `_mixed_call` needs to support larger `n_reading` and `n_arxiv`. Update its signature to pass them through properly — the current helper already does (`n_reading`, `n_arxiv` params). The `umap_reducer.transform` mock must return `n_reading + n_arxiv` rows. Update `_mixed_call` to set the mock return size correctly:

```python
def _mixed_call(self, tmp_path: Path, n_reading: int = 3, n_arxiv: int = 3):
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
    umap_reducer.transform.return_value = np.random.rand(max(len(results), 1), 2).astype(
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/rodgzilla/Documents/rss_gemma_filtering/.worktrees/html-viz-redesign
pytest tests/test_score_viz.py::TestWriteScoreViz::test_no_bar_traces_on_x1 \
       tests/test_score_viz.py::TestWriteScoreViz::test_dot_traces_present_for_reading \
       tests/test_score_viz.py::TestWriteScoreViz::test_dot_traces_present_for_arxiv \
       tests/test_score_viz.py::TestWriteScoreViz::test_dot_traces_x_positions_wrap_at_20_columns \
       tests/test_score_viz.py::TestWriteScoreViz::test_dot_size_10px_for_small_feed \
       tests/test_score_viz.py::TestWriteScoreViz::test_dot_size_6px_for_large_feed \
       tests/test_score_viz.py::TestWriteScoreViz::test_dot_traces_feed_names_in_legend \
       -v
```

Expected: FAIL — bar traces still present, no scatter on x1/x3.

- [ ] **Step 3: Replace `_feed_bar_traces()` with `_feed_dot_traces()` in `score_viz.py`**

Inside `write_score_viz`, remove the `_feed_bar_traces` function and replace with:

```python
_COLS = 20

def _feed_dot_traces(pairs, xax, yax, showlegend=True):
    """Build unit dot grid scatter traces (one dot per article) for a row."""
    if not pairs:
        return []

    kept_by_feed: dict[str, int] = defaultdict(int)
    for r, _ in pairs:
        if r.keep:
            kept_by_feed[r.entry.feed_name] += 1

    all_feeds = sorted(
        {r.entry.feed_name for r, _ in pairs},
        key=lambda f: kept_by_feed[f],
        reverse=True,
    )

    # Assign palette colour per feed
    feed_colour = {
        f: _FEED_PALETTE[i % len(_FEED_PALETTE)]
        for i, f in enumerate(all_feeds)
    }

    dot_size = 6 if len(pairs) > 200 else 10

    # Order: for each feed (kept-count desc), kept articles then rejected
    ordered = []
    for feed in all_feeds:
        ordered += [(r, m) for r, m in pairs if r.entry.feed_name == feed and r.keep]
        ordered += [(r, m) for r, m in pairs if r.entry.feed_name == feed and not r.keep]

    # Compute (col, row) positions
    xs = [pos % _COLS for pos in range(len(ordered))]
    ys = [pos // _COLS for pos in range(len(ordered))]

    # Build one trace per feed (kept) and one per feed (rejected)
    traces = []
    for feed in all_feeds:
        intense = feed_colour[feed]
        muted = _muted_colour(intense)
        for status, colour, label_suffix in [
            (True, intense, " kept"),
            (False, muted, " rejected"),
        ]:
            indices = [
                i for i, (r, _) in enumerate(ordered)
                if r.entry.feed_name == feed and r.keep == status
            ]
            if not indices:
                continue
            trace_name = feed + label_suffix
            traces.append({
                "type": "scatter",
                "mode": "markers",
                "name": feed,
                "legendgroup": feed,
                "showlegend": showlegend and (status is True),
                "x": [xs[i] for i in indices],
                "y": [ys[i] for i in indices],
                "marker": {
                    "color": colour,
                    "size": dot_size,
                    "opacity": 0.85,
                    "line": {"color": "rgba(0,0,0,0)", "width": 0},
                },
                "hovertemplate": f"{trace_name}<extra></extra>",
                "xaxis": xax,
                "yaxis": yax,
            })
    return traces
```

- [ ] **Step 4: Update the layout dict in `write_score_viz`**

In the `layout` dict (around line 394), update the left-panel axes to suit the dot grid (remove `barmode`, strip bar-chart-specific titles, hide ticks, invert y, fix range):

```python
    layout = {
        "title": {"text": f"RSS Score Analysis — {today} ({n_kept}/{n_total} kept)"},
        "hovermode": "closest",
        # Row 1 — Reading dot grid (x1/y1)
        "xaxis": {
            "domain": LEFT_X,
            "anchor": "y1",
            "fixedrange": True,
            "showticklabels": False,
            "showgrid": False,
            "zeroline": False,
        },
        "yaxis": {
            "domain": TOP_Y,
            "anchor": "x1",
            "title": "Reading",
            "fixedrange": True,
            "showticklabels": False,
            "showgrid": False,
            "zeroline": False,
            "autorange": "reversed",
        },
        "xaxis2": {"domain": RIGHT_X, "title": "UMAP dim 1", "anchor": "y2"},
        "yaxis2": {"domain": TOP_Y, "title": "UMAP dim 2", "anchor": "x2"},
        # Row 2 — arXiv dot grid (x3/y3)
        "xaxis3": {
            "domain": LEFT_X,
            "anchor": "y3",
            "fixedrange": True,
            "showticklabels": False,
            "showgrid": False,
            "zeroline": False,
        },
        "yaxis3": {
            "domain": BOT_Y,
            "anchor": "x3",
            "title": "arXiv",
            "fixedrange": True,
            "showticklabels": False,
            "showgrid": False,
            "zeroline": False,
            "autorange": "reversed",
        },
        "xaxis4": {"domain": RIGHT_X, "title": "UMAP dim 1", "anchor": "y4"},
        "yaxis4": {"domain": BOT_Y, "title": "UMAP dim 2", "anchor": "x4"},
        "legend": {"orientation": "h", "y": -0.05, "x": 0.5, "xanchor": "center"},
        "paper_bgcolor": "#1e1e2e",
        "plot_bgcolor": "#2a2a3e",
        "font": {"color": "#cdd6f4"},
        "margin": {"l": 20},
        "shapes": [],
        "annotations": [
            {
                "text": "UMAP — vault projection",
                "xref": "paper",
                "yref": "paper",
                "x": 0.775,
                "y": 1.04,
                "showarrow": False,
                "font": {"size": 12, "color": "#a6adc8"},
            },
        ],
    }
```

Also update the two call sites (replace `_feed_bar_traces` → `_feed_dot_traces`):

```python
    # Row 1 — Reading dot grid (x1/y1)
    all_traces += _feed_dot_traces(reading_pairs, "x1", "y1")

    # Row 2 — arXiv dot grid (x3/y3)
    all_traces += _feed_dot_traces(arxiv_pairs, "x3", "y3", showlegend=False)
```

- [ ] **Step 5: Run all new dot-grid tests**

```bash
cd /home/rodgzilla/Documents/rss_gemma_filtering/.worktrees/html-viz-redesign
pytest tests/test_score_viz.py::TestWriteScoreViz::test_no_bar_traces_on_x1 \
       tests/test_score_viz.py::TestWriteScoreViz::test_dot_traces_present_for_reading \
       tests/test_score_viz.py::TestWriteScoreViz::test_dot_traces_present_for_arxiv \
       tests/test_score_viz.py::TestWriteScoreViz::test_dot_traces_x_positions_wrap_at_20_columns \
       tests/test_score_viz.py::TestWriteScoreViz::test_dot_size_10px_for_small_feed \
       tests/test_score_viz.py::TestWriteScoreViz::test_dot_size_6px_for_large_feed \
       tests/test_score_viz.py::TestWriteScoreViz::test_dot_traces_feed_names_in_legend \
       -v
```

Expected: all 7 PASS.

- [ ] **Step 6: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass. If `test_reading_entries_in_reading_row` or `test_arxiv_entries_in_arxiv_row` fail (they search for text in `t.get("y")` which no longer applies to scatter traces), update them to search `t.get("name")` or `t.get("hovertemplate")` instead:

```python
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
```

- [ ] **Step 7: Commit**

```bash
cd /home/rodgzilla/Documents/rss_gemma_filtering/.worktrees/html-viz-redesign
git add rss_filter/score_viz.py tests/test_score_viz.py
git commit -m "feat: replace bar chart with unit dot grid in per-feed statistics panel"
```

---

## Self-review notes

- Spec requires muted colour = 40% blend toward `#cccccc` — implemented in `_muted_colour(blend=0.4)`. ✓
- Spec requires dot size 10px default, 6px when total > 200 — implemented via `dot_size` local variable. ✓
- Spec requires feed order: kept-count descending — implemented via `sorted(..., key=lambda f: kept_by_feed[f], reverse=True)`. ✓
- Spec requires kept-then-rejected within each feed — implemented in `ordered` list construction. ✓
- Spec requires `showlegend=False` for arXiv row — passed as argument to `_feed_dot_traces`. ✓
- Spec requires both axes `fixedrange: true`, hidden ticks, no grid, y inverted — all in updated layout dict. ✓
- `barmode: "stack"` removed from layout (no longer needed). ✓
- `_COLS = 20` defined at module level inside `write_score_viz` — should be defined at the inner-function closure level to avoid leaking into module scope; it's a local constant to the closure, not a module-level export. Actually it is simplest as a local variable inside `_feed_dot_traces` since that's the only user.
