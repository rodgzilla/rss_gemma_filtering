# arXiv Separate UMAP Space Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the arXiv UMAP panel its own coordinate space fitted on arXiv-tagged vault docs, so its background dots are distinct from the reading panel.

**Architecture:** Extend `build_or_load_umap` with a `source_filter` parameter, call it twice in `main.py` (once without filter for reading, once with `source_filter="arxiv"`), and pass the arXiv reducer + vault arrays to `write_score_viz` as new optional params. Inside `write_score_viz`, the arXiv panel's `_vault_umap_traces` call uses the new arXiv-specific arrays.

**Tech Stack:** Python, umap-learn, joblib, numpy, pytest

---

## File Map

| File | Change |
|------|--------|
| `rss_filter/score_viz.py` | Add `source_filter` to `build_or_load_umap`; add 3 optional arxiv params to `write_score_viz`; split `_vault_umap_traces` into reading/arXiv calls |
| `main.py` | Call `build_or_load_umap` twice; pass arXiv outputs to `write_score_viz` |
| `tests/test_score_viz.py` | Add tests for `source_filter` and arXiv panel isolation |

---

## Task 1: `build_or_load_umap` — add `source_filter` parameter

**Files:**
- Modify: `rss_filter/score_viz.py:92-168`
- Test: `tests/test_score_viz.py`

- [ ] **Step 1: Write failing tests**

Add these tests to `tests/test_score_viz.py`. Import `build_or_load_umap` at the top alongside existing imports:

```python
from rss_filter.score_viz import _muted_colour, _FEED_PALETTE, write_score_viz, build_or_load_umap
```

Then add the test class at the end of the file:

```python
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
        reading_docs = _fake_vault_docs(5, source="reading")
        arxiv_docs = _fake_vault_docs(3, source="arxiv")
        store = _make_mock_store(reading_docs + arxiv_docs)
        model_path = str(tmp_path / "umap_arxiv.joblib")

        reducer, vault_2d, vault_docs = build_or_load_umap(
            store=store,
            model_path=model_path,
            source_filter="arxiv",
        )

        # Only arXiv-tagged docs should be returned
        assert len(vault_docs) == 3
        assert all(d["source_note"] == "arxiv" for d in vault_docs)
        assert vault_2d.shape == (3, 2)

    def test_source_filter_none_returns_all_docs(self, tmp_path):
        """Without source_filter, all vault docs are used (existing behaviour)."""
        reading_docs = _fake_vault_docs(4, source="reading")
        arxiv_docs = _fake_vault_docs(3, source="arxiv")
        store = _make_mock_store(reading_docs + arxiv_docs)
        model_path = str(tmp_path / "umap_all.joblib")

        reducer, vault_2d, vault_docs = build_or_load_umap(
            store=store,
            model_path=model_path,
            source_filter=None,
        )

        assert len(vault_docs) == 7
        assert vault_2d.shape == (7, 2)

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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/rodgzilla/Documents/rss_gemma_filtering/.worktrees/html-viz-redesign
python -m pytest tests/test_score_viz.py::TestBuildOrLoadUmapSourceFilter -v
```

Expected: 3 failures — `build_or_load_umap` not importable or missing `source_filter`.

- [ ] **Step 3: Implement `source_filter` in `build_or_load_umap`**

Change the signature at `score_viz.py:92`:

```python
def build_or_load_umap(
    store: EmbeddingStore,
    model_path: str,
    force_rebuild: bool = False,
    growth_threshold: float = 0.1,
    source_filter: str | None = None,
) -> tuple[umap.UMAP | None, np.ndarray, list[dict]]:
```

Update the docstring to mention `source_filter`:

```python
    """Return a fitted UMAP model, the vault 2-D coordinates, and vault metadata.

    The fitted model and vault coordinates are cached to *model_path* and a
    sidecar ``.meta.json`` file.  Refit is triggered when:
    - ``force_rebuild`` is True, or
    - The cached model file does not exist, or
    - The vault has grown by more than ``growth_threshold`` (fraction) since
      the model was last fitted.

    Args:
        source_filter: If set, only vault docs with ``source_note`` equal to
                       this value are used (e.g. ``"arxiv"``).  Use ``None``
                       (default) to use all vault docs.

    Returns:
        reducer:        fitted umap.UMAP instance (None if < 2 docs)
        vault_2d:       np.ndarray of shape (N_vault, 2)
        vault_docs:     list of dicts {"url", "text", "source_note", "date"}
                        in the same order as vault_2d rows
    """
```

After `vault_docs = store.get_all()` (line 113), add the filter block:

```python
    vault_docs = store.get_all()
    if source_filter is not None:
        vault_docs = [d for d in vault_docs if d["source_note"] == source_filter]
    n_current = len(vault_docs)
```

Then handle the empty case right after `n_current` is set — before the existing cache logic:

```python
    if n_current == 0:
        return None, np.zeros((0, 2), dtype=np.float32), []
```

That's all — no other changes needed; the rest of the function already handles `n_current < 2`.

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_score_viz.py::TestBuildOrLoadUmapSourceFilter -v
```

Expected: 3 passing.

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
python -m pytest tests/test_score_viz.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add rss_filter/score_viz.py tests/test_score_viz.py
git commit -m "feat: add source_filter param to build_or_load_umap"
```

---

## Task 2: `write_score_viz` — accept arXiv vault params and use them in arXiv panel

**Files:**
- Modify: `rss_filter/score_viz.py:205-465`
- Test: `tests/test_score_viz.py`

- [ ] **Step 1: Write failing tests**

Add these tests to `tests/test_score_viz.py`, inside the existing `TestWriteScoreViz` class (or as a separate class at the end — either is fine):

```python
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
            _result(f"arXiv {i}", keep=True, score=0.9, is_arxiv=True)
            for i in range(2)
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
                {"url": "u", "text": "t", "source_note": "reading", "date": "2024-01-01"}
            ] * 3,
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
            _result(f"arXiv {i}", keep=True, score=0.9, is_arxiv=True)
            for i in range(2)
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
                {"url": "u", "text": "t", "source_note": "reading", "date": "2024-01-01"}
            ] * 3,
            umap_reducer=umap_reducer,
            threshold_reading=0.5,
            threshold_arxiv=0.4,
            output_path=out,
            # No arxiv_vault_* args
        )

        assert out.exists()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_score_viz.py::TestWriteScoreVizArxivVault -v
```

Expected: failures — `write_score_viz` doesn't accept `arxiv_vault_*` params yet.

- [ ] **Step 3: Add arXiv params to `write_score_viz` signature**

Change the function signature at `score_viz.py:205`:

```python
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
    arxiv_vault_2d: np.ndarray | None = None,
    arxiv_vault_docs: list[dict] | None = None,
    arxiv_umap_reducer=None,
) -> None:
```

- [ ] **Step 4: Build arXiv vault background arrays inside `write_score_viz`**

After the existing reading vault background block (lines 266-274 in the original — the block that builds `vault_2d_bg` / `vault_docs_bg`), add a parallel block for arXiv:

```python
    # --- arXiv vault background (independent subsample) ---
    if arxiv_vault_2d is not None and len(arxiv_vault_2d) > 0:
        arxiv_vault_2d_bg = arxiv_vault_2d
        arxiv_vault_docs_bg = arxiv_vault_docs or []
        if len(arxiv_vault_2d) > vault_bg_max:
            idx_abg = rng.choice(len(arxiv_vault_2d), vault_bg_max, replace=False)
            idx_abg.sort()
            arxiv_vault_2d_bg = arxiv_vault_2d[idx_abg]
            arxiv_vault_docs_bg = [arxiv_vault_docs_bg[i] for i in idx_abg]
    else:
        arxiv_vault_2d_bg = vault_2d_bg   # fallback: reading vault (backward compat)
        arxiv_vault_docs_bg = vault_docs_bg
```

> Note: `rng` is defined just above the existing vault background block as
> `rng = np.random.default_rng(seed=42)`. It's already there; no new rng needed.

- [ ] **Step 5: Add `_arxiv_vault_trace` inner function**

Right after the existing `_vault_trace` inner function (around line 300), add:

```python
    def _arxiv_vault_trace(xax, yax):
        hover = [
            f"<b>{_truncate(d['text'], 100)}</b><br>Source: {d['source_note']}"
            for d in arxiv_vault_docs_bg
        ]
        return {
            "type": "scatter",
            "mode": "markers",
            "name": "arXiv vault",
            "x": arxiv_vault_2d_bg[:, 0].tolist(),
            "y": arxiv_vault_2d_bg[:, 1].tolist(),
            "marker": {
                "color": "#bbbbbb",
                "size": 4,
                "opacity": 0.4,
            },
            "hovertemplate": "%{text}<extra></extra>",
            "text": hover,
            "showlegend": False,
            "xaxis": xax,
            "yaxis": yax,
        }
```

- [ ] **Step 6: Update `_vault_umap_traces` call for arXiv panel**

Currently both panels are produced by the same `_vault_umap_traces` closure which captures `vault_2d_bg` and `umap_reducer`. We need the arXiv call to use the arXiv arrays. The cleanest way is to add a `use_arxiv: bool = False` parameter to the inner function:

Replace the existing `_vault_umap_traces` definition (around line 422):

```python
    def _vault_umap_traces(matrix, kept, hovers, xax, yax, urls=None, use_arxiv=False):
        traces = []
        if use_arxiv:
            bg_2d = arxiv_vault_2d_bg
            reducer = arxiv_umap_reducer if arxiv_umap_reducer is not None else umap_reducer
            if bg_2d is not None and len(bg_2d) > 0:
                traces.append(_arxiv_vault_trace(xax, yax))
        else:
            bg_2d = vault_2d_bg
            reducer = umap_reducer
            if bg_2d is not None and len(bg_2d) > 0:
                traces.append(_vault_trace(xax, yax))
        if matrix is not None and reducer is not None:
            try:
                xy = reducer.transform(matrix).astype(np.float32)[: len(matrix)]
                traces.append(
                    _umap_scatter(
                        xy, kept, hovers, "Kept", kept_colour, xax, yax, urls=urls
                    )
                )
                traces.append(
                    _umap_scatter(
                        xy,
                        ~kept,
                        hovers,
                        "Rejected",
                        rejected_colour,
                        xax,
                        yax,
                        urls=urls,
                    )
                )
            except Exception as e:
                print(f"  [WARN] UMAP transform failed: {e}")
        return traces
```

Then update the two call sites (around line 459-464):

```python
    # Vault-fitted UMAP traces
    all_traces += _vault_umap_traces(
        r_matrix, r_kept, r_hovers, "x2", "y2", urls=r_urls
    )
    all_traces += _vault_umap_traces(
        a_matrix, a_kept, a_hovers, "x4", "y4", urls=a_urls, use_arxiv=True
    )
```

- [ ] **Step 7: Run tests to confirm they pass**

```bash
python -m pytest tests/test_score_viz.py::TestWriteScoreVizArxivVault -v
```

Expected: 2 passing.

- [ ] **Step 8: Run full test suite to confirm no regressions**

```bash
python -m pytest tests/test_score_viz.py -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add rss_filter/score_viz.py tests/test_score_viz.py
git commit -m "feat: arXiv UMAP panel uses separate vault background"
```

---

## Task 3: `main.py` — call `build_or_load_umap` twice, pass arXiv outputs

**Files:**
- Modify: `main.py:153-155` (config reading) and `main.py:282-301` (viz block)
- Test: manual smoke test (no unit test for `main.py` integration)

- [ ] **Step 1: Add `arxiv_umap_model_path` config reading**

At `main.py:153`, the existing line reads:

```python
    umap_model_path = emb_cfg.get("umap_model_path", "umap_model.joblib")
```

Add immediately after:

```python
    arxiv_umap_model_path = emb_cfg.get("arxiv_umap_model_path", "umap_arxiv_model.joblib")
```

- [ ] **Step 2: Add second `build_or_load_umap` call**

The existing block at `main.py:282-300` reads:

```python
            umap_reducer, vault_2d, vault_docs = build_or_load_umap(
                store=store,
                model_path=umap_model_path,
                force_rebuild=args.rebuild_umap or args.rebuild_embeddings,
                growth_threshold=umap_growth_threshold,
            )
            viz_path = (
                args.vault / vault_cfg["output_folder"] / f"RSS-{today}-scores.html"
            )
            write_score_viz(
                results=all_results,
                entry_metadata=entry_metadata,
                vault_2d=vault_2d,
                vault_docs=vault_docs,
                umap_reducer=umap_reducer,
                threshold_reading=threshold_reading,
                threshold_arxiv=threshold_arxiv,
                output_path=viz_path,
                vault_bg_max=vault_bg_max,
            )
```

Replace it with:

```python
            umap_reducer, vault_2d, vault_docs = build_or_load_umap(
                store=store,
                model_path=umap_model_path,
                force_rebuild=args.rebuild_umap or args.rebuild_embeddings,
                growth_threshold=umap_growth_threshold,
            )
            arxiv_umap_reducer, arxiv_vault_2d, arxiv_vault_docs = build_or_load_umap(
                store=store,
                model_path=arxiv_umap_model_path,
                force_rebuild=args.rebuild_umap or args.rebuild_embeddings,
                growth_threshold=umap_growth_threshold,
                source_filter="arxiv",
            )
            viz_path = (
                args.vault / vault_cfg["output_folder"] / f"RSS-{today}-scores.html"
            )
            write_score_viz(
                results=all_results,
                entry_metadata=entry_metadata,
                vault_2d=vault_2d,
                vault_docs=vault_docs,
                umap_reducer=umap_reducer,
                threshold_reading=threshold_reading,
                threshold_arxiv=threshold_arxiv,
                output_path=viz_path,
                vault_bg_max=vault_bg_max,
                arxiv_vault_2d=arxiv_vault_2d,
                arxiv_vault_docs=arxiv_vault_docs,
                arxiv_umap_reducer=arxiv_umap_reducer,
            )
```

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass (main.py has no unit tests but we verify no import errors).

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: wire arXiv UMAP reducer in main pipeline"
```

---

## Self-Review

**Spec coverage:**
- `build_or_load_umap` `source_filter` → Task 1 ✓
- Empty-filter edge case → Task 1 `test_source_filter_no_matching_docs_returns_empty` ✓
- `write_score_viz` new optional args → Task 2 ✓
- arXiv panel uses arXiv vault → Task 2 step 6 ✓
- Backward compat (no arXiv args) → Task 2 `test_arxiv_vault_none_falls_back_to_reading_vault` ✓
- `main.py` calls twice → Task 3 ✓
- `--rebuild-umap` rebuilds both → Task 3 step 2 (`force_rebuild` passed to both calls) ✓
- Config key `arxiv_umap_model_path` → Task 3 step 1 ✓
- `vault_bg_max` applied independently to arXiv vault → Task 2 step 4 ✓

**No placeholders found.**

**Type consistency:** `arxiv_vault_2d_bg`, `arxiv_vault_docs_bg`, `arxiv_umap_reducer` used consistently across Tasks 2 steps 4/5/6. `source_filter: str | None` matches usage in Task 1 and Task 3.
