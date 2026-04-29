# HTML Visualisation Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the generated HTML score visualisation from a 2×3 grid to a 2×2 grid, replacing scatter plots with per-feed bar charts, and adding clickable UMAP points that surface article URLs in the Obsidian note via `postMessage`.

**Architecture:** `score_viz.py` is the sole generator of the HTML file — all chart changes live there. `note_writer.py` generates the Obsidian note that embeds the HTML via iframe — the `postMessage` listener block is added there. Existing tests that assert 2×3-specific behaviour must be updated; new tests cover the bar chart traces and click handler.

**Tech Stack:** Python, Plotly.js (CDN), vanilla JavaScript (`postMessage`), Obsidian DataviewJS.

**Worktree:** `.worktrees/html-viz-redesign` on branch `feature/html-viz-redesign`

---

## Task 1: Update failing tests for removed 2×3 layout assertions

The existing test suite has tests that assert 2×3-specific behaviour that the new design removes (6 axes, scatter on x1/x4, RSS-only UMAP panel). Update those tests to match the new 2×2 design before touching implementation so the test suite correctly defines what "done" looks like.

**Files:**
- Modify: `tests/test_score_viz.py`

- [ ] **Step 1: Identify which existing tests will break under the new design**

The following tests assert 2×3 behaviour that will be removed:

| Test | Why it breaks |
|---|---|
| `test_html_has_six_axis_domains` | New layout has 4 axes (x1/x2/x3/x4 → but only x1,x2,x3,x4 for 2 rows × 2 cols), assert must change to ≥4 |
| `test_html_contains_both_thresholds` | Threshold lines on scatter are removed; threshold values will no longer appear in HTML |
| `test_html_contains_reading_threshold_line` | Scatter threshold line removed |
| `test_html_contains_arxiv_threshold_line` | Orange `#f39c12` threshold line removed |
| `test_rss_only_umap_panel_present` | RSS-only UMAP column removed |
| `test_scatter_marker_opacity_low` | Scatter traces (x1/x4) replaced by bar traces; no scatter on x1/x4 |
| `test_reading_entries_in_reading_row` | Still valid but x3 no longer exists; reading row uses x1, x2 only |
| `test_arxiv_entries_in_arxiv_row` | Still valid but x6 no longer exists; arXiv row uses x3, x4 only |

- [ ] **Step 2: Run the current test suite to confirm baseline**

```bash
cd .worktrees/html-viz-redesign
pytest tests/test_score_viz.py -v 2>&1 | tail -30
```

Expected: all tests pass (green baseline).

- [ ] **Step 3: Rewrite the eight tests to match the new design**

In `tests/test_score_viz.py`, replace the eight tests listed above with the following:

```python
def test_html_has_four_axis_domains(self, tmp_path):
    """Layout must define 4 independent axis pairs for the 2×2 grid."""
    out = self._mixed_call(tmp_path)
    html = out.read_text()
    import json, re

    layout_match = re.search(r"var layout = ({.*?});\s*Plotly", html, re.DOTALL)
    assert layout_match, "Could not find layout JSON in HTML"
    layout = json.loads(layout_match.group(1))
    xaxis_keys = [k for k in layout if k.startswith("xaxis")]
    assert len(xaxis_keys) == 4, f"Expected 4 xaxis keys, got {xaxis_keys}"

def test_no_rss_only_umap_panel(self, tmp_path):
    """RSS-only UMAP column must not appear in the output."""
    out = self._mixed_call(tmp_path, n_entries=4)
    html = out.read_text()
    assert "RSS entries only" not in html

def test_no_scatter_on_x1(self, tmp_path):
    """x1/x3 axes must not carry scatter traces (they are bar chart axes now)."""
    out = self._mixed_call(tmp_path)
    html = out.read_text()
    import json, re

    traces_match = re.search(
        r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
    )
    assert traces_match, "Could not find traces JSON in HTML"
    traces = json.loads(traces_match.group(1))
    scatter_on_bar_axes = [
        t for t in traces
        if t.get("type") == "scatter" and t.get("xaxis") in ("x1", "x3")
    ]
    assert not scatter_on_bar_axes, f"Unexpected scatter on bar axes: {scatter_on_bar_axes}"

def test_bar_traces_present_for_reading(self, tmp_path):
    """Bar traces for the reading row must be present on x1."""
    out = self._mixed_call(tmp_path)
    html = out.read_text()
    import json, re

    traces_match = re.search(
        r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
    )
    traces = json.loads(traces_match.group(1))
    bar_traces = [t for t in traces if t.get("type") == "bar" and t.get("xaxis") == "x1"]
    assert bar_traces, "Expected bar traces on x1 for reading row"

def test_bar_traces_present_for_arxiv(self, tmp_path):
    """Bar traces for the arXiv row must be present on x3."""
    out = self._mixed_call(tmp_path)
    html = out.read_text()
    import json, re

    traces_match = re.search(
        r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
    )
    traces = json.loads(traces_match.group(1))
    bar_traces = [t for t in traces if t.get("type") == "bar" and t.get("xaxis") == "x3"]
    assert bar_traces, "Expected bar traces on x3 for arXiv row"

def test_bar_traces_contain_feed_names(self, tmp_path):
    """Bar chart y-axis values must include the feed name from the entries."""
    out = self._mixed_call(tmp_path)
    html = out.read_text()
    import json, re

    traces_match = re.search(
        r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
    )
    traces = json.loads(traces_match.group(1))
    bar_traces = [t for t in traces if t.get("type") == "bar"]
    all_y = [y for t in bar_traces for y in (t.get("y") or [])]
    # _mixed_call uses feed_name="Test Feed"
    assert "Test Feed" in all_y, f"Expected 'Test Feed' in bar y values, got {all_y}"

def test_bar_kept_colour_green(self, tmp_path):
    """Kept bar segment must use green colour #2ecc71."""
    out = self._mixed_call(tmp_path)
    html = out.read_text()
    import json, re

    traces_match = re.search(
        r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
    )
    traces = json.loads(traces_match.group(1))
    kept_bars = [
        t for t in traces
        if t.get("type") == "bar" and "Kept" in t.get("name", "")
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
    import json, re

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
    import json, re

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
```

Also update `_mixed_call` so it accepts a combined `n_entries` parameter for the basic `_basic_call` helper — the existing helper already works, no change needed there.

- [ ] **Step 4: Run updated tests to confirm they now fail (red)**

```bash
pytest tests/test_score_viz.py -v 2>&1 | tail -40
```

Expected: the 9 new/updated tests FAIL, all others still PASS.

- [ ] **Step 5: Commit the updated tests**

```bash
git add tests/test_score_viz.py
git commit -m "test: update score_viz tests for 2x2 layout (red)"
```

---

## Task 2: Rewrite `write_score_viz` — layout and bar charts

Replace the scatter traces (x1/y1 reading, x4/y4 arXiv) with per-feed horizontal bar chart traces, and drop the RSS-only UMAP column entirely. The vault-fitted UMAP panels move from columns 2/5 to columns 2/4.

**Files:**
- Modify: `rss_filter/score_viz.py`

- [ ] **Step 1: Add a helper that builds per-feed bar traces**

Inside `write_score_viz` in `rss_filter/score_viz.py`, add the following helper function (place it just before the `all_traces` list is built, after the `_umap_scatter` helper):

```python
def _feed_bar_traces(pairs, xax, yax):
    """Build kept + rejected horizontal bar traces for a group of (result, meta) pairs.

    Returns two Plotly bar traces: one for kept counts, one for rejected counts.
    Feeds are sorted descending by kept count.
    """
    if not pairs:
        return []
    from collections import defaultdict
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
        "marker": {"color": "#2ecc71", "opacity": 0.75},
        "xaxis": xax,
        "yaxis": yax,
    }
    rejected_trace = {
        "type": "bar",
        "orientation": "h",
        "name": "Rejected",
        "y": all_feeds,
        "x": rejected_counts,
        "marker": {"color": "#bdc3c7", "opacity": 0.4},
        "xaxis": xax,
        "yaxis": yax,
    }
    return [kept_trace, rejected_trace]
```

- [ ] **Step 2: Replace scatter trace calls with bar trace calls**

In `write_score_viz`, find the two blocks that build reading and arXiv scatter traces and the RSS-only UMAP section, and replace them:

```python
# Remove these blocks entirely:
#   all_traces += _rss_umap_traces(r_matrix, r_kept, r_hovers, "x3", "y3")
#   all_traces += _rss_umap_traces(a_matrix, a_kept, a_hovers, "x6", "y6")
# And the _rss_umap_traces helper function definition.
#
# Replace the reading scatter block (was x1/y1) with:
all_traces += _feed_bar_traces(reading_pairs, "x1", "y1")

# Replace the arXiv scatter block (was x4/y4) with:
all_traces += _feed_bar_traces(arxiv_pairs, "x3", "y3")

# Update vault UMAP axis references:
#   Reading vault UMAP: was "x2", "y2" → stays "x2", "y2"  (no change)
#   arXiv vault UMAP:   was "x5", "y5" → becomes "x4", "y4"
all_traces += _vault_umap_traces(r_matrix, r_kept, r_hovers, "x2", "y2")
all_traces += _vault_umap_traces(a_matrix, a_kept, a_hovers, "x4", "y4")
```

The full updated trace-building block (replacing lines 238–393 in the current file) should be:

```python
all_traces: list[dict] = []

# Row 1 — Reading bar chart (x1/y1)
all_traces += _feed_bar_traces(reading_pairs, "x1", "y1")

# Row 2 — arXiv bar chart (x3/y3)
all_traces += _feed_bar_traces(arxiv_pairs, "x3", "y3")

# Vault-fitted UMAP traces
all_traces += _vault_umap_traces(r_matrix, r_kept, r_hovers, "x2", "y2")
all_traces += _vault_umap_traces(a_matrix, a_kept, a_hovers, "x4", "y4")
```

- [ ] **Step 3: Update the Plotly layout to 2×2 domains**

Replace the layout dict's axis definitions and domain constants. The new domains are:

```python
TOP_Y = [0.55, 1.00]
BOT_Y = [0.00, 0.45]
LEFT_X = [0.00, 0.46]
RIGHT_X = [0.54, 1.00]

layout = {
    "title": {"text": f"RSS Score Analysis — {today} ({n_kept}/{n_total} kept)"},
    "hovermode": "closest",
    "barmode": "stack",
    # Row 1 — Reading
    "xaxis":  {"domain": LEFT_X,  "title": "Entry count", "anchor": "y1"},
    "yaxis":  {"domain": TOP_Y,   "anchor": "x1"},
    "xaxis2": {"domain": RIGHT_X, "title": "UMAP dim 1",  "anchor": "y2"},
    "yaxis2": {"domain": TOP_Y,   "title": "UMAP dim 2",  "anchor": "x2"},
    # Row 2 — arXiv
    "xaxis3": {"domain": LEFT_X,  "title": "Entry count", "anchor": "y3"},
    "yaxis3": {"domain": BOT_Y,   "anchor": "x3"},
    "xaxis4": {"domain": RIGHT_X, "title": "UMAP dim 1",  "anchor": "y4"},
    "yaxis4": {"domain": BOT_Y,   "title": "UMAP dim 2",  "anchor": "x4"},
    "legend": {"orientation": "h", "y": -0.05},
    "paper_bgcolor": "#1e1e2e",
    "plot_bgcolor": "#2a2a3e",
    "font": {"color": "#cdd6f4"},
    "shapes": [],
    "annotations": [
        # Column headings
        {
            "text": "Entries per feed",
            "xref": "paper", "yref": "paper",
            "x": 0.23, "y": 1.04,
            "showarrow": False,
            "font": {"size": 12, "color": "#a6adc8"},
        },
        {
            "text": "UMAP — vault projection",
            "xref": "paper", "yref": "paper",
            "x": 0.77, "y": 1.04,
            "showarrow": False,
            "font": {"size": 12, "color": "#a6adc8"},
        },
        # Row labels
        {
            "text": "<b>Reading</b>",
            "xref": "paper", "yref": "paper",
            "x": -0.01, "y": (TOP_Y[0] + TOP_Y[1]) / 2,
            "showarrow": False, "textangle": -90,
            "font": {"size": 13, "color": "#cdd6f4"},
        },
        {
            "text": "<b>arXiv</b>",
            "xref": "paper", "yref": "paper",
            "x": -0.01, "y": (BOT_Y[0] + BOT_Y[1]) / 2,
            "showarrow": False, "textangle": -90,
            "font": {"size": 13, "color": "#cdd6f4"},
        },
    ],
}
```

Note: `"shapes": []` — threshold lines are removed. The `threshold_reading` and `threshold_arxiv` parameters remain in the function signature unchanged (backwards compatibility) but are no longer used in the layout.

- [ ] **Step 4: Run the tests**

```bash
pytest tests/test_score_viz.py -v 2>&1 | tail -40
```

Expected: all new tests PASS. If `test_html_contains_both_thresholds`, `test_html_contains_reading_threshold_line`, or `test_html_contains_arxiv_threshold_line` still exist and fail, that is expected — they were removed in Task 1. Confirm all Task 1 replacements are in place.

- [ ] **Step 5: Run the full test suite**

```bash
pytest tests/ -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add rss_filter/score_viz.py
git commit -m "feat: replace scatter with per-feed bar chart, drop RSS-only UMAP column"
```

---

## Task 3: Add `customdata` (URLs) to UMAP traces and `plotly_click` postMessage handler

Each UMAP point representing an RSS entry should carry its article URL. On click, a JavaScript handler fires `postMessage` to the parent Obsidian note.

**Files:**
- Modify: `rss_filter/score_viz.py`
- Modify: `tests/test_score_viz.py`

- [ ] **Step 1: Write the failing tests**

Add these tests to `TestWriteScoreViz` in `tests/test_score_viz.py`:

```python
def test_umap_traces_have_customdata(self, tmp_path):
    """UMAP scatter traces for RSS entries must include customdata (URLs)."""
    out = self._mixed_call(tmp_path)
    html = out.read_text()
    import json, re

    traces_match = re.search(
        r"var traces = (\[.*?\]);\s*var layout", html, re.DOTALL
    )
    traces = json.loads(traces_match.group(1))
    # UMAP RSS traces are on x2 (reading) and x4 (arXiv)
    umap_rss_traces = [
        t for t in traces
        if t.get("type") == "scatter" and t.get("xaxis") in ("x2", "x4")
        and t.get("name") in ("Kept", "Rejected")
    ]
    assert umap_rss_traces, "No UMAP RSS traces found on x2/x4"
    for t in umap_rss_traces:
        assert "customdata" in t, f"Trace {t.get('name')} on {t.get('xaxis')} missing customdata"
        assert len(t["customdata"]) == len(t["x"]), "customdata length must match x length"

def test_html_contains_postmessage_click_handler(self, tmp_path):
    """Generated HTML must include a plotly_click postMessage handler."""
    out = self._mixed_call(tmp_path)
    html = out.read_text()
    assert "plotly_click" in html
    assert "postMessage" in html
    assert "rss-viz-click" in html
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_score_viz.py::TestWriteScoreViz::test_umap_traces_have_customdata tests/test_score_viz.py::TestWriteScoreViz::test_html_contains_postmessage_click_handler -v
```

Expected: both FAIL.

- [ ] **Step 3: Add `customdata` to `_umap_scatter` helper**

In `score_viz.py`, update the `_umap_scatter` helper to accept and include a `urls` parameter:

```python
def _umap_scatter(xy, mask, hovers, name, colour, xax, yax, showlegend=False, urls=None):
    idx = np.where(mask)[0].tolist()
    trace = {
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
    if urls is not None:
        trace["customdata"] = [urls[i] for i in idx]
    return trace
```

- [ ] **Step 4: Pass URLs into `_vault_umap_traces`**

Update `_vault_umap_traces` to accept and forward a `urls` list (one URL per RSS entry, in the same order as `hovers`):

```python
def _vault_umap_traces(matrix, kept, hovers, xax, yax, urls=None):
    traces = []
    if vault_2d_bg is not None and len(vault_2d_bg) > 0:
        traces.append(_vault_trace(xax, yax))
    if matrix is not None and umap_reducer is not None:
        try:
            xy = umap_reducer.transform(matrix).astype(np.float32)
            traces.append(
                _umap_scatter(xy, kept, hovers, "Kept", kept_colour, xax, yax,
                              urls=urls)
            )
            traces.append(
                _umap_scatter(xy, ~kept, hovers, "Rejected", rejected_colour, xax, yax,
                              urls=urls)
            )
        except Exception as e:
            print(f"  [WARN] UMAP transform failed: {e}")
    return traces
```

- [ ] **Step 5: Build URL lists and pass them to `_vault_umap_traces`**

After `_group_arrays` is called for reading and arXiv pairs, extract the URL for each entry. Add this just after the `r_scores, r_top1, ...` and `a_scores, a_top1, ...` lines:

```python
r_urls = [r.entry.url for r, _ in reading_pairs] if reading_pairs else []
a_urls = [r.entry.url for r, _ in arxiv_pairs] if arxiv_pairs else []
```

Then update the `_vault_umap_traces` calls:

```python
all_traces += _vault_umap_traces(r_matrix, r_kept, r_hovers, "x2", "y2", urls=r_urls)
all_traces += _vault_umap_traces(a_matrix, a_kept, a_hovers, "x4", "y4", urls=a_urls)
```

- [ ] **Step 6: Add the `plotly_click` postMessage handler to the HTML template**

In the `html = f"""..."""` string at the bottom of `write_score_viz`, add the click handler script after `Plotly.newPlot(...)`:

```python
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
    var div = document.getElementById('chart');
    Plotly.newPlot(div, traces, layout, {{responsive: true}});
    div.on('plotly_click', function(data) {{
      var pt = data.points[0];
      var url = pt.customdata;
      if (!url) return;
      var title = pt.text ? pt.text.replace(/<[^>]+>/g, '').split('\\n')[0] : url;
      window.parent.postMessage({{ type: 'rss-viz-click', url: url, title: title }}, '*');
    }});
  </script>
</body>
</html>
"""
```

- [ ] **Step 7: Run the two new tests**

```bash
pytest tests/test_score_viz.py::TestWriteScoreViz::test_umap_traces_have_customdata tests/test_score_viz.py::TestWriteScoreViz::test_html_contains_postmessage_click_handler -v
```

Expected: both PASS.

- [ ] **Step 8: Run the full test suite**

```bash
pytest tests/ -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add rss_filter/score_viz.py tests/test_score_viz.py
git commit -m "feat: add customdata URLs and plotly_click postMessage handler to UMAP traces"
```

---

## Task 4: Add `postMessage` listener block to `note_writer.py`

The Obsidian note needs a second `dataviewjs` block below the iframe that listens for `rss-viz-click` messages and renders the clicked article as a link in the note.

**Files:**
- Modify: `rss_filter/note_writer.py`
- Modify: `tests/test_note_writer.py`

- [ ] **Step 1: Write the failing tests**

Add these tests to `tests/test_note_writer.py`:

```python
def test_build_note_content_includes_click_listener_when_viz_given():
    """When viz_filename is provided the note must include the postMessage listener block."""
    content = build_note_content(
        [], [], date="2025-01-15", viz_filename="RSS-2025-01-15-scores.html"
    )
    assert "rss-viz-click" in content
    assert "addEventListener" in content

def test_build_note_content_no_click_listener_without_viz():
    """Without a viz_filename, no postMessage listener block should appear."""
    content = build_note_content([], [], date="2025-01-15")
    assert "rss-viz-click" not in content

def test_build_note_content_click_listener_after_viz_block():
    """The postMessage listener block must appear after the iframe viz block."""
    content = build_note_content(
        [], [], date="2025-01-15", viz_filename="RSS-2025-01-15-scores.html"
    )
    viz_pos = content.index("getResourcePath")
    listener_pos = content.index("rss-viz-click")
    assert listener_pos > viz_pos

def test_build_note_content_click_listener_renders_link():
    """The listener block must contain code to render a link element."""
    content = build_note_content(
        [], [], date="2025-01-15", viz_filename="RSS-2025-01-15-scores.html"
    )
    assert "createEl" in content
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_note_writer.py::test_build_note_content_includes_click_listener_when_viz_given tests/test_note_writer.py::test_build_note_content_no_click_listener_without_viz tests/test_note_writer.py::test_build_note_content_click_listener_after_viz_block tests/test_note_writer.py::test_build_note_content_click_listener_renders_link -v
```

Expected: all four FAIL.

- [ ] **Step 3: Add `_render_click_listener_block` to `note_writer.py`**

Add this function after `_render_viz_block` in `rss_filter/note_writer.py`:

```python
def _render_click_listener_block() -> str:
    """Return a DataviewJS block that listens for postMessage click events from the viz iframe.

    When the user clicks a UMAP point in the embedded HTML chart, the iframe fires
    a postMessage with { type: 'rss-viz-click', url, title }.  This block catches
    that message and renders the article title as a clickable link directly in the note.
    """
    js = (
        "const el = this.container.createEl('div', {\n"
        "  attr: { style: 'padding: 4px 0; font-size: 13px; min-height: 20px;' }\n"
        "});\n"
        "el.setText('Click a point in the chart above to open an article.');\n"
        "window.addEventListener('message', (event) => {\n"
        "  if (!event.data || event.data.type !== 'rss-viz-click') return;\n"
        "  const { url, title } = event.data;\n"
        "  el.empty();\n"
        "  el.createEl('span', { text: '\\u2192 ', attr: { style: 'color: #a6adc8;' } });\n"
        "  el.createEl('a', { text: title || url, href: url,\n"
        "    attr: { style: 'color: #89b4fa;' } });\n"
        "});\n"
    )
    return f"```dataviewjs\n{js}```\n"
```

- [ ] **Step 4: Call `_render_click_listener_block` from `build_note_content`**

In `build_note_content`, update the `if viz_filename:` block to also append the listener block:

```python
if viz_filename:
    parts.append(_render_viz_block(viz_filename))
    parts.append(_render_click_listener_block())
```

- [ ] **Step 5: Run the four new tests**

```bash
pytest tests/test_note_writer.py::test_build_note_content_includes_click_listener_when_viz_given tests/test_note_writer.py::test_build_note_content_no_click_listener_without_viz tests/test_note_writer.py::test_build_note_content_click_listener_after_viz_block tests/test_note_writer.py::test_build_note_content_click_listener_renders_link -v
```

Expected: all four PASS.

- [ ] **Step 6: Run the full test suite**

```bash
pytest tests/ -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add rss_filter/note_writer.py tests/test_note_writer.py
git commit -m "feat: add postMessage click listener block to Obsidian note template"
```

---

## Task 5: Clean up prototype files and final verification

Remove the prototype files added during brainstorming and verify the full suite passes cleanly.

**Files:**
- Delete: `click_test.html`
- Delete: `click_test.md`

- [ ] **Step 1: Remove prototype files**

```bash
git rm click_test.html click_test.md
```

- [ ] **Step 2: Run the full test suite one final time**

```bash
pytest tests/ -v 2>&1 | tail -20
```

Expected: all tests pass, no regressions.

- [ ] **Step 3: Final commit**

```bash
git commit -m "chore: remove click_test prototype files"
```
