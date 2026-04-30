# Spec: Separate arXiv UMAP Space

**Date:** 2026-04-30  
**Status:** Approved

## Problem

Both UMAP panels (Reading and arXiv) currently share the same vault background dots
(`vault_2d_bg`), which is the reading-vault projection. The arXiv panel background is
identical to the reading panel background — same points, same positions — which is
misleading and unhelpful.

## Goal

Give the arXiv UMAP panel its own coordinate space, fitted on arXiv-tagged vault docs
only, with its own background point cloud.

## Approach

Extend `build_or_load_umap` with a `source_filter` parameter. Call it twice in
`main.py` — once for reading (no filter) and once for arXiv (`source_filter="arxiv"`).
Pass both outputs to `write_score_viz` as new optional parameters.

## Data

Vault docs in the embedding store already have `source_note` set to either `"reading"`
or `"arxiv"` (set by `NoteEntry.source` at insert time). No schema changes needed.

## Changes

### 1. `rss_filter/score_viz.py` — `build_or_load_umap`

Add parameter:
```python
source_filter: str | None = None
```

Behaviour:
- After `store.get_all()`, if `source_filter` is set, filter:
  `vault_docs = [d for d in vault_docs if d["source_note"] == source_filter]`
- All existing fit/cache/load logic is unchanged.
- Cache path remains caller-supplied; callers use distinct paths for reading vs arXiv.
- If filtered list is empty, return `(None, np.zeros((0, 2)), [])` — no reducer, no
  background.

### 2. `rss_filter/score_viz.py` — `write_score_viz`

Add optional parameters (all default `None`):
```python
arxiv_vault_2d: np.ndarray | None = None,
arxiv_vault_docs: list[dict] | None = None,
arxiv_umap_reducer = None,
```

Behaviour:
- Existing `vault_2d`, `vault_docs`, `umap_reducer` drive the Reading panel (unchanged).
- When `arxiv_vault_2d` / `arxiv_umap_reducer` are provided, `_vault_umap_traces` for
  the arXiv panel (`x4/y4`) uses them instead of the reading vault arrays.
- If not provided (default `None`), arXiv panel falls back to current behaviour (shared
  reading vault background) for backwards compatibility.
- `vault_bg_max` subsampling is applied independently to each vault array.

### 3. `main.py`

```python
umap_reducer, vault_2d, vault_docs = build_or_load_umap(
    store=store,
    model_path=umap_model_path,
    force_rebuild=...,
    growth_threshold=umap_growth_threshold,
)
arxiv_umap_reducer, arxiv_vault_2d, arxiv_vault_docs = build_or_load_umap(
    store=store,
    model_path=arxiv_umap_model_path,   # new path, e.g. "umap_arxiv_model.joblib"
    force_rebuild=...,
    growth_threshold=umap_growth_threshold,
    source_filter="arxiv",
)
write_score_viz(
    ...,
    arxiv_vault_2d=arxiv_vault_2d,
    arxiv_vault_docs=arxiv_vault_docs,
    arxiv_umap_reducer=arxiv_umap_reducer,
)
```

The arXiv model path should come from config (same `[embeddings]` section, new key
`arxiv_umap_model_path`, defaulting to `"umap_arxiv_model.joblib"`).

## Edge Cases

- **No arXiv vault docs**: `arxiv_umap_reducer` is `None`; arXiv panel shows no
  background dots. No crash.
- **arXiv UMAP cache invalidation**: Uses the same growth-threshold logic as the reading
  UMAP, but evaluated independently against the filtered doc count.
- **`--rebuild-umap` flag**: Should rebuild both reducers (pass `force_rebuild` to both
  calls).

## Tests

- `build_or_load_umap` with `source_filter="arxiv"` only fits on arXiv docs.
- `build_or_load_umap` with `source_filter="arxiv"` when no arXiv docs exist returns
  `(None, zeros array, [])`.
- `write_score_viz` with `arxiv_vault_2d` / `arxiv_umap_reducer` provided uses them for
  the arXiv panel traces (not the reading vault).
- `write_score_viz` without the new args keeps current behaviour (backward compat).

## Out of Scope

- Any change to the reading UMAP panel.
- Changing the vault embedding schema.
- RSS-only UMAP panels (columns 3) — unchanged.
