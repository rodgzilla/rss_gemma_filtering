# NixOS Integration + Embedding Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run rss_gemma_filtering on the T480 from the NixOS flake: EmbeddingGemma-300M served declaratively by llama.cpp, the filter packaged as a Nix flake with a home-manager module, triggered from the i3 `obsidian` mode.

**Architecture:**
- **rss repo** fixes the embedding-input bugs listed below, becomes an installable Python package (`rss-filter` console script), and exports `packages.default` + `homeModules.default` from a new `flake.nix`.
- **nixos repo** (`t480_flake`) gets a NixOS module `embeddingServer` (nixpkgs `services.llama-cpp`, GGUF pinned with `fetchurl`) and a home module `rssFilter` (imports the rss flake's module, sets paths, adds dunst notifications). i3 `obsidian` mode binds `r` to `systemctl --user start --no-block rss-filter`.
- The Python side stays server-agnostic: it still speaks the OpenAI `/v1/embeddings` API, only `base_url` changes (LM Studio `:1234` → llama-server `127.0.0.1:8080`).

**Tech Stack:** Python 3, numpy, umap-learn, openai client, llama.cpp `llama-server` (nixpkgs 0.4.1, build 10964), Nix flakes, flake-parts + import-tree (dendritic layout), home-manager, systemd user units, i3.

---

## Review of the existing implementation

The concept is sound: kNN similarity to previously saved articles, exponential-decay aggregation over the top-K, and per-category quantile thresholds make a reasonable zero-training relevance filter. The cosine, top-K, and decay maths in `embedding_store.query` / `score_filter._exponential_decay_score` are correct.

Findings, verified against a real `llama-server` running the same `embeddinggemma-300m-qat-Q4_0.gguf` LM Studio uses:

| # | Issue | Where | Evidence | Severity |
|---|---|---|---|---|
| 1 | **Oversized inputs crash the run.** llama-server returns HTTP 500 for the *whole* 64-item request if any input exceeds the batch size (`input (5003 tokens) is too large to process`). Nothing catches it, so one long feed summary aborts everything. LM Studio may have been truncating silently. | `score_filter.py:84`, `embedding_store.py:98` | Reproduced | **Blocker for llama.cpp** |
| 2 | **HTML is embedded.** `feedparser`'s `summary` is sanitised HTML (`<p>`, `<a href=…>`, entities); arXiv summaries also start with `arXiv:… Announce Type: new Abstract:`. Tag and URL noise goes into the vector, and vault texts (plain markdown) have no such noise, so the two sides are compared asymmetrically. | `score_filter.py:78` | Code reading | High |
| 3 | **Matryoshka truncation is not re-normalised.** Server vectors are L2-normalised over 768 dims; the first 128 dims have norm ≈ 0.55 and vary per vector, so UMAP (euclidean) mixes vector length with direction. EmbeddingGemma's model card says to re-normalise after truncating. | `score_filter.py:103`, `score_viz.py:146` | Measured norm 0.5532 | Medium (visualisation only) |
| 4 | **The store doesn't record which model or text format produced its vectors.** Switching server, quantisation, or prompt format silently mixes incompatible vectors, and a different dimension crashes `np.stack`. | `embedding_store.py` | Code reading | Medium (hits exactly this migration) |
| 5 | **`query()` re-reads and re-normalises the whole table for every RSS entry.** That's O(entries × vault) SQLite reads. | `embedding_store.py:127` | Code reading | Low (performance) |
| 6 | **EmbeddingGemma task prompts are unused.** The model was trained with prefixes (`title: … \| text: …` for documents). Whether they help here is **unproven**: a 3-text toy check gave a *smaller* related/unrelated gap with the document prompt (0.388 vs 0.426 raw). So the plan makes this configurable and decides with an eval (Task 3), not by assumption. | `score_filter.py:78`, `embedding_store.py:46` | Measured, inconclusive | Evaluate |
| 7 | `args.top_quantile or cfg…` treats an explicit `0.0` as unset (same for decay/vault-bg/timeout). | `main.py:161-176` | Code reading | Low |
| 8 | Relative state paths (`embedding_store.db`, `seen_entries.json`, `umap_*.joblib`) resolve against CWD, which a systemd unit or a Nix-store config can't rely on. | `main.py:158-181` | Code reading | Needed for packaging |
| 9 | Docs say `source_note` is the note filename; the code stores `"reading"`/`"arxiv"` (and UMAP's `source_filter` relies on that). | `EMBEDDING_PIPELINE.md` | Code reading | Docs |

**Why llama.cpp rather than LM Studio:** LM Studio is a closed-source GUI app that can't be configured declaratively. It already uses llama.cpp internally for GGUF models. nixpkgs ships `services.llama-cpp`, a hardened `DynamicUser` systemd service.

Measured on this i7-8550U (4 threads, CPU only):
- ≈ 340 MB resident with the model loaded, ≈ 73 MB once `--sleep-idle-seconds` has unloaded it; the next request reloads it automatically in ~1.1 s
- 64 inputs of ~350 tokens in 21 s

That means a daily batch of ~1 000 entries takes ≈ 5–6 min, and a first vault build scales the same way.

Other measured facts:
- Output is 768-dim and L2-normalised.
- `--pooling mean` works.
- The `model` field in requests is ignored.

---

## Decisions

1. **Public repo.** The flake input is `github:rodgzilla/rss_gemma_filtering`, and the existing `sudo nixos-rebuild` alias keeps working.
2. **Unloading the model after a run.** Use llama-server's own `--sleep-idle-seconds 60`. After 60 s without a request it frees the model weights (346 MB → 73 MB measured), and the next request wakes it transparently (~1.1 s; `/health` still answers while it sleeps). This needs one setting, keeps the server a plain system service, and also works for manual `rss-filter-run` calls.
   - A run may sleep and wake the server once, between the vault build and scoring, while the feeds are being fetched. That costs ~1 s.
   - The alternatives were rejected: a socket-activated proxy, or a user unit bound to the rss-filter run. Each would save the remaining ~73 MB of process overhead but needs more units and readiness handling.
3. **Keybinding:** `$mod+y r` (free in the `obsidian` mode; `y`, `u`, `o` are taken).

---

## File Map

**rss_gemma_filtering**

| File | Change |
|---|---|
| `rss_filter/text_prep.py` | **New.** HTML stripping, arXiv prefix removal, length cap, prompt formatting |
| `rss_filter/embedding_client.py` | Fall back to per-item embedding with halving truncation on server errors |
| `rss_filter/embedding_store.py` | `meta` table + embedding signature → auto-rebuild; cached normalised matrix; use `text_prep` |
| `rss_filter/score_filter.py` | Use `text_prep`; `mrl_truncate()` with re-normalisation |
| `rss_filter/score_viz.py` | Fit UMAP on `mrl_truncate()` vectors |
| `rss_filter/rss_fetcher.py` | Replace `listparser` (not in nixpkgs) with stdlib `xml.etree` |
| `main.py` → `rss_filter/cli.py` | Move CLI into the package; `--state-dir`; `is not None` overrides; `prompt_style`; signature |
| `main.py` | Becomes a 3-line shim (`python main.py …` keeps working) |
| `config.toml` → `rss_filter/config.toml` | Packaged default config; `base_url` → `http://127.0.0.1:8080/v1` |
| `pyproject.toml` | **New.** setuptools, `rss-filter` console script, package data |
| `requirements.txt` | Drop `listparser` |
| `flake.nix`, `flake.lock` | **New.** `packages.default`, `devShells.default`, `homeModules.default` |
| `nix/package.nix`, `nix/hm-module.nix` | **New** |
| `scripts/eval_prompt_style.py` | **New.** Offline eval for finding 6 |
| `tests/test_text_prep.py`, `tests/test_embedding_store.py`, `tests/test_embedding_client.py`, `tests/test_rss_fetcher.py`, `tests/test_score_filter.py` | New/extended tests |
| `README.md`, `USAGE.md`, `EMBEDDING_PIPELINE.md` | llama.cpp, Nix usage, new flags, fix finding 9 |

**t480_flake**

| File | Change |
|---|---|
| `flake.nix` | Add `rss-gemma-filtering` input |
| `modules/features/embedding-server.nix` | **New.** `flake.nixosModules.embeddingServer` |
| `modules/home/rss-filter.nix` | **New.** `flake.homeModules.rssFilter` |
| `modules/hosts/t480/default.nix` | Add `self.nixosModules.embeddingServer` |
| `modules/hosts/t480/home.nix` | Add `rssFilter` to the home imports |
| `modules/home/i3.nix` | `obsidian` mode: `r` → start `rss-filter.service` |

---

## Phase A — Python fixes (rss repo)

Work on a branch: `git switch -c feature/nixos-integration`. Run the tests with `pytest tests/` (use the Phase B dev shell once it exists).

### Task 1: `text_prep` module (findings 1, 2, 6)

**Files:** Create `rss_filter/text_prep.py`, `tests/test_text_prep.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for rss_filter.text_prep."""

from rss_filter.text_prep import MAX_CHARS, clean_summary, format_for_embedding, strip_html


def test_strip_html_removes_tags_and_unescapes():
    assert strip_html("<p>Hello&nbsp;<b>world</b> &amp; co</p>") == "Hello world & co"


def test_strip_html_separates_block_elements():
    assert strip_html("<p>one</p><p>two</p>") == "one two"


def test_strip_html_drops_script_and_style():
    assert strip_html("<style>p{}</style>text<script>x()</script>") == "text"


def test_clean_summary_removes_arxiv_announce_prefix():
    raw = "arXiv:2404.01234v1 Announce Type: new \nAbstract: We study things."
    assert clean_summary(raw) == "We study things."


def test_format_none_style_joins_title_and_body():
    assert format_for_embedding("T", "body", style="none") == "T body"


def test_format_document_style_uses_gemma_prompt():
    assert format_for_embedding("T", "body", style="document") == "title: T | text: body"


def test_format_document_style_missing_title():
    assert format_for_embedding("", "body", style="document") == "title: none | text: body"


def test_format_caps_length():
    out = format_for_embedding("T", "x" * (MAX_CHARS * 2), style="none")
    assert len(out) <= MAX_CHARS
```

- [ ] **Step 2: Run, verify they fail** — `pytest tests/test_text_prep.py -v` → `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
"""Normalise text before it is sent to the embedding server.

Vault notes are plain markdown while feed summaries are HTML, so both sides are
reduced to plain text and formatted identically before embedding. The length cap
keeps every input under llama-server's 2048-token batch: a single oversized input
makes the server reject the whole request.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

# ~4 chars/token for English/French prose → ~1000 tokens, well under 2048.
# EmbeddingClient still falls back to halving on a server error, for token-dense text.
MAX_CHARS = 4000

PROMPT_STYLES = ("none", "document")

_WS_RE = re.compile(r"\s+")
_ARXIV_PREFIX_RE = re.compile(
    r"^arXiv:\S+\s+Announce Type:\s*\S+\s*Abstract:\s*", re.IGNORECASE
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def strip_html(text: str) -> str:
    """Return the visible text of an HTML fragment, whitespace-collapsed."""
    parser = _TextExtractor()
    parser.feed(text)
    parser.close()
    return _WS_RE.sub(" ", " ".join(parser.parts)).strip()


def clean_summary(summary: str) -> str:
    """Plain text of a feed summary, without arXiv's announcement header."""
    return _ARXIV_PREFIX_RE.sub("", strip_html(summary or ""))


def format_for_embedding(title: str, body: str, style: str = "none") -> str:
    """Build the exact string sent to the embedding model."""
    title = _WS_RE.sub(" ", title or "").strip()
    body = _WS_RE.sub(" ", body or "").strip()
    if style == "document":
        text = f"title: {title or 'none'} | text: {body}"
    elif style == "none":
        text = f"{title} {body}".strip()
    else:
        raise ValueError(f"unknown prompt style {style!r}; expected one of {PROMPT_STYLES}")
    return text[:MAX_CHARS]
```

Note: `strip_html("<p>Hello&nbsp;<b>world</b>…")` gives `"Hello world & co"` because `&nbsp;` becomes `\xa0` and `\s` matches it. Check the first test passes as written; adjust the expectation only if Python's `re` disagrees.

- [ ] **Step 4: Run, verify they pass.**
- [ ] **Step 5: Commit** — `feat: add text_prep for HTML stripping, length cap and prompt formatting`

### Task 2: Embedding client survives oversized inputs (finding 1)

**Files:** Modify `rss_filter/embedding_client.py`, `tests/test_embedding_client.py`

- [ ] **Step 1: Failing test.** Add it to the existing test module, reusing its `mock_openai` fixture and `_make_embedding_response`:

```python
import openai


def _status_error():
    return openai.InternalServerError(
        "too large", response=MagicMock(status_code=500), body=None
    )


class TestOversizedFallback:
    def test_batch_error_falls_back_to_per_item_and_halves(self, mock_openai):
        calls = []

        def create(model, input):
            calls.append(input)
            if isinstance(input, list):
                raise _status_error()
            if len(input) > 100:
                raise _status_error()
            return _make_embedding_response([[1.0, 0.0]])

        mock_openai.embeddings.create.side_effect = create
        client = EmbeddingClient(base_url="http://x/v1", model="m")
        out = client.embed_batch(["short", "y" * 400])

        assert len(out) == 2
        assert any(isinstance(c, str) and len(c) == 100 for c in calls)

    def test_gives_up_below_minimum_length(self, mock_openai):
        mock_openai.embeddings.create.side_effect = _status_error()
        client = EmbeddingClient(base_url="http://x/v1", model="m")
        with pytest.raises(openai.APIStatusError):
            client.embed_batch(["tiny"])
```

- [ ] **Step 2: Run → fails** (the exception propagates from the batch call).
- [ ] **Step 3: Implement** in `embedding_client.py`:

```python
import openai

_MIN_FALLBACK_CHARS = 64


class EmbeddingClient:
    """Client for an OpenAI-compatible /v1/embeddings server (llama-server, LM Studio)."""

    def __init__(self, base_url: str, model: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key="unused")
        self._model = model

    # embed() unchanged

    def embed_batch(self, texts: list[str], chunk_size: int = 64) -> list[np.ndarray]:
        """... (existing docstring) ...

        If the server rejects a chunk (llama-server fails the whole request when
        one input is too long), its texts are retried one by one, halving any
        text the server still rejects.
        """
        results: list[np.ndarray] = []
        for start in range(0, len(texts), chunk_size):
            chunk = texts[start : start + chunk_size]
            try:
                response = self._client.embeddings.create(model=self._model, input=chunk)
            except openai.APIStatusError:
                results.extend(self._embed_shrinking(t) for t in chunk)
                continue
            sorted_data = sorted(response.data, key=lambda d: d.index)
            results.extend(
                np.array(item.embedding, dtype=np.float32) for item in sorted_data
            )
        return results

    def _embed_shrinking(self, text: str) -> np.ndarray:
        while True:
            try:
                return self.embed(text)
            except openai.APIStatusError:
                if len(text) <= _MIN_FALLBACK_CHARS:
                    raise
                text = text[: len(text) // 2]
```

- [ ] **Step 4: Run the whole test file → pass.** (If an existing test asserts `api_key="lm-studio"`, update it.)
- [ ] **Step 5: Commit** — `fix: retry oversized embedding inputs one by one instead of aborting`

### Task 3: Apply `text_prep` on both sides + decide the prompt style (findings 2, 6)

**Files:** Modify `rss_filter/embedding_store.py`, `rss_filter/score_filter.py`; create `scripts/eval_prompt_style.py`; tests.

- [ ] **Step 1: Failing tests.**
  - `tests/test_embedding_store.py`: `_embed_text_for_entry(NoteEntry(url="u", context="Title more text", source="reading", date="2024-01-01", title="Title"), style="document") == "title: Title | text: more text"`. A context-less entry gives `"title: Title | text: u"` (the URL fallback is kept as body).
  - `tests/test_score_filter.py`: with a mocked `embed_client`, `score_entries(..., prompt_style="none")` on an entry whose summary is `"<p>Hi</p>"` sends `"T Hi"`.
- [ ] **Step 2: Implement.**
  - `embedding_store._embed_text_for_entry(entry, style)`: `body = context` minus a leading `entry.title` (`notes_parser` puts the title first), falling back to `entry.url` when empty. Then `format_for_embedding(entry.title, body, style)`.
  - `EmbeddingStore.build_or_update(..., prompt_style="none")` passes it through.
  - `score_filter.score_entries(..., prompt_style="none")`: `texts = [format_for_embedding(e.title, clean_summary(e.summary), prompt_style) for e in entries]`.
  - The CLI reads `emb_cfg.get("prompt_style", "none")`. Add `prompt_style = "none"` to the config.
- [ ] **Step 3: Tests pass. Commit** — `fix: embed plain text, formatted identically for vault and feed entries`
- [ ] **Step 4: Offline eval (decides finding 6).** `scripts/eval_prompt_style.py --vault … --feeds … --base-url …`:
  1. Parse the vault. Hold out the most recent 10 % of reading entries as "positives" and use the rest as the store.
  2. Fetch the current feeds as "negatives" (mostly not saved), dropping any URL that is in the vault.
  3. For each `style in ("none", "document")`: embed the store and both sets, score each item with `store.query` + `_exponential_decay_score` (top_k/decay from config), and print ROC-AUC (positives vs negatives, via `numpy` rank sums, no sklearn) and precision@20 %.
  4. Set the winner as the default `prompt_style` in `rss_filter/config.toml`. Record the numbers in the commit message.
  Needs the real vault synced to the T480 and llama-server running (Phase C). **This step may be deferred until then**; `"none"` stays the default meanwhile.

### Task 4: Store signature, auto-rebuild, cached matrix (findings 4, 5)

**Files:** Modify `rss_filter/embedding_store.py`, `tests/test_embedding_store.py`

- [ ] **Step 1: Failing tests** (use `tmp_path` DBs and the existing fake-client helpers):
  - A new store with `signature="a"` inserts N docs; reopening with `signature="a"` embeds 0 new; reopening with `signature="b"` re-embeds all N and `build_or_update` returns `True`.
  - A store created by the old schema (no `meta` table) is rebuilt on first use with a signature.
  - `query()` issues `_SELECT_ALL` once across two consecutive calls, and again after `build_or_update` inserts rows. Spy on `conn.execute`, or count via a wrapper around `_load_matrix`.
- [ ] **Step 2: Implement.**

```python
_CREATE_META = "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"

# __init__: execute _CREATE_META; self._cache = None

def _get_meta(self, key: str) -> str | None: ...
def _set_meta(self, key: str, value: str) -> None:  # INSERT OR REPLACE

def build_or_update(self, entries, client, force_rebuild=False,
                    signature: str = "", prompt_style: str = "none") -> bool:
    """... Returns True if the table was rebuilt from scratch."""
    stored = self._get_meta("signature")
    if signature and stored != signature and self.count() > 0:
        print(f"Embedding store: signature changed ({stored!r} → {signature!r}); rebuilding.")
        force_rebuild = True
    # ... existing body, with self._cache = None after inserting ...
    self._set_meta("signature", signature); self._conn.commit()
    return force_rebuild
```

  - `_load_matrix()` builds `(texts, urls, titles, unit_matrix)` once and stores it in `self._cache`. `query()` uses it; the arithmetic is unchanged.
  - The signature is built in the CLI: `f"{model}|prompt={prompt_style}|prep={text_prep.PREP_VERSION}"`. Add `PREP_VERSION = 1` to `text_prep.py`, and bump it whenever the text cleaning changes.
- [ ] **Step 3: Tests pass. Commit** — `feat: record embedding signature and rebuild on change; cache query matrix`

### Task 5: Re-normalised Matryoshka vectors for UMAP (finding 3)

**Files:** Modify `rss_filter/score_filter.py`, `rss_filter/score_viz.py`, tests.

- [ ] **Step 1: Failing test:** `np.linalg.norm(mrl_truncate(np.r_[np.full(128, 0.1), np.ones(640)])) == pytest.approx(1.0)`, and a zero vector is returned unchanged.
- [ ] **Step 2: Implement** in `score_filter.py`:

```python
def mrl_truncate(emb: np.ndarray, dim: int = MATRYOSHKA_DIM) -> np.ndarray:
    """First *dim* Matryoshka dimensions, re-normalised to unit length."""
    head = emb[:dim]
    norm = np.linalg.norm(head)
    return head / norm if norm else head
```

  Use it for `entry_metadata["embedding_128"]` and in `build_or_load_umap` (`mrl_truncate(d["embedding"])` instead of `d["embedding"][:MATRYOSHKA_DIM]`). Add `"mrl_normalised": True` to the UMAP meta and treat a cache without it as `needs_rebuild`, so stale reducers are refit.
- [ ] **Step 3: Tests pass (`test_score_viz.py` included). Commit** — `fix: re-normalise truncated Matryoshka embeddings before UMAP`

### Task 6: Drop `listparser` (packaging prerequisite)

`listparser` isn't in nixpkgs. OPML is flat enough for the standard library.

- [ ] **Step 1:** Keep the existing `parse_opml` tests. Add one for nested `<outline>` categories and one for an outline with only `text=` (no `title=`).
- [ ] **Step 2: Implement:**

```python
import xml.etree.ElementTree as ET


def parse_opml(opml_path: Path) -> List[Tuple[str, str]]:
    """Parse an OPML file and return list of (title, feed_url) tuples."""
    root = ET.parse(opml_path).getroot()
    feeds = []
    for outline in root.iter("outline"):
        url = outline.get("xmlUrl") or ""
        if url:
            feeds.append((outline.get("title") or outline.get("text") or "", url))
    return feeds
```

- [ ] **Step 3:** Remove `listparser` from the imports and `requirements.txt`. Sanity check: `python -c "from rss_filter.rss_fetcher import parse_opml; print(len(parse_opml(__import__('pathlib').Path('rss_feeds/feeds.opml.xml'))))"` → `69`.
- [ ] **Step 4: Commit** — `refactor: parse OPML with xml.etree, dropping listparser`

### Task 7: Package-ready CLI (findings 7, 8)

**Files:** `git mv main.py rss_filter/cli.py`, `git mv config.toml rss_filter/config.toml`, new `main.py`, `pyproject.toml`

- [ ] **Step 1:** In `cli.py`:
  - `--config` default → `Path(__file__).parent / "config.toml"`.
  - Add `--state-dir PATH` (default: CWD) and resolve every relative config path against it:

    ```python
    def _in_state(p: str) -> str:
        path = Path(p).expanduser()
        return str(path if path.is_absolute() else args.state_dir / path)
    ```

    Apply it to `store_path`, `seen_entries`, `umap_model_path`, `arxiv_umap_model_path`, and `mkdir(parents=True, exist_ok=True)` the directory.
  - Replace every `args.x or cfg.get(...)` with `args.x if args.x is not None else cfg.get(...)`.
  - Pass `signature` / `prompt_style` (Tasks 3–4). Force the UMAP refit when `build_or_update` returns `True`.
- [ ] **Step 2:** New `main.py`:

```python
"""Run the CLI from a checkout: python main.py --vault … --feeds …"""

from rss_filter.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 3:** In `rss_filter/config.toml`, set `base_url = "http://127.0.0.1:8080/v1"`, `model = "embeddinggemma-300m-qat-Q4_0"` and `prompt_style = "none"`.
- [ ] **Step 4:** `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "rss-gemma-filtering"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["openai", "feedparser", "tqdm", "numpy", "umap-learn", "joblib"]

[project.optional-dependencies]
test = ["pytest", "pytest-mock"]

[project.scripts]
rss-filter = "rss_filter.cli:main"

[tool.setuptools.packages.find]
include = ["rss_filter*"]

[tool.setuptools.package-data]
rss_filter = ["config.toml"]
```

- [ ] **Step 5:** Add a CLI test: `main(["--vault", …mock_vault, "--feeds", <tmp OPML with 0 feeds>, "--state-dir", tmp_path])` with `EmbeddingClient` patched creates `tmp_path/"embedding_store.db"`.
- [ ] **Step 6:** `pytest tests/` green. Run `python main.py --help` to check the shim. Commit — `refactor: move CLI into the package, add --state-dir and pyproject`

---

## Phase B — Nix flake in the rss repo

### Task 8: Package + dev shell

**Files:** `flake.nix`, `nix/package.nix`

- [ ] **Step 1:** `nix/package.nix`:

```nix
{
  lib,
  buildPythonApplication,
  setuptools,
  openai,
  feedparser,
  tqdm,
  numpy,
  umap-learn,
  joblib,
  pytestCheckHook,
  pytest-mock,
}:

buildPythonApplication {
  pname = "rss-gemma-filtering";
  version = "0.1.0";
  pyproject = true;

  # Only what the build needs: not the mock vault, feed exports, or docs.
  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../pyproject.toml
      ../rss_filter
      ../tests
    ];
  };

  build-system = [ setuptools ];
  dependencies = [ openai feedparser tqdm numpy umap-learn joblib ];

  nativeCheckInputs = [ pytestCheckHook pytest-mock ];
  # umap imports numba, which wants a writable cache directory.
  preCheck = ''
    export NUMBA_CACHE_DIR=$TMPDIR/numba
  '';

  meta.mainProgram = "rss-filter";
}
```

  Tests that read `mock_vault/` need it in the fileset. Check with `grep -rn mock_vault tests/` and add `../mock_vault` if any do.

- [ ] **Step 2:** `flake.nix`:

```nix
{
  description = "Embedding-based RSS filter that writes daily digests into an Obsidian vault";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      forAllSystems =
        f:
        nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" ] (
          system: f nixpkgs.legacyPackages.${system}
        );
    in
    {
      packages = forAllSystems (pkgs: {
        default = pkgs.python3Packages.callPackage ./nix/package.nix { };
      });

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            (pkgs.python3.withPackages (
              ps: with ps; [ openai feedparser tqdm numpy umap-learn joblib pytest pytest-mock ]
            ))
            pkgs.llama-cpp # for running llama-server by hand on non-NixOS-module machines
          ];
        };
      });

      homeModules.default = import ./nix/hm-module.nix self;
    };
}
```

- [ ] **Step 3:** `git add` the new files (flakes only see tracked files). Then run `nix build .#` and `./result/bin/rss-filter --help`. `nix develop -c pytest tests/` must be green.
- [ ] **Step 4: Commit** (includes `flake.lock`) — `build: add Nix flake packaging rss-filter`

### Task 9: home-manager module

**File:** `nix/hm-module.nix`

- [ ] **Step 1:**

```nix
# home-manager module for rss-filter. Takes the flake's `self` so the default
# package is this flake's own build.
self:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.programs.rss-filter;
  toml = pkgs.formats.toml { };
  configFile = toml.generate "rss-filter.toml" cfg.settings;

  # The command the service runs, also on PATH for manual runs (e.g. `rss-filter-run --dry-run`).
  runner = pkgs.writeShellApplication {
    name = "rss-filter-run";
    runtimeInputs = [ cfg.package ];
    text = ''
      # umap's numba kernels cache next to their module; the store is read-only.
      export NUMBA_CACHE_DIR=${lib.escapeShellArg "${config.xdg.cacheHome}/rss-filter/numba"}
      exec rss-filter \
        --config ${configFile} \
        --state-dir ${lib.escapeShellArg cfg.stateDir} \
        --vault ${lib.escapeShellArg cfg.vault} \
        --feeds ${lib.escapeShellArg cfg.feeds} \
        "$@"
    '';
  };
in
{
  options.programs.rss-filter = {
    enable = lib.mkEnableOption "the embedding-based RSS filter";

    package = lib.mkOption {
      type = lib.types.package;
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
      defaultText = lib.literalMD "this flake's `packages.default`";
    };

    vault = lib.mkOption {
      type = lib.types.str;
      description = "Obsidian vault root (reads `Daily notes/`, writes `Filtered feed/`).";
    };

    feeds = lib.mkOption {
      type = lib.types.str;
      description = "OPML subscriptions file.";
    };

    stateDir = lib.mkOption {
      type = lib.types.str;
      default = "${config.xdg.stateHome}/rss-filter";
      defaultText = lib.literalExpression ''"''${config.xdg.stateHome}/rss-filter"'';
      description = "Embedding store, seen entries and UMAP cache.";
    };

    settings = lib.mkOption {
      type = toml.type;
      default = { };
      description = "config.toml contents; each key overrides the packaged default.";
    };
  };

  config = lib.mkIf cfg.enable {
    # Packaged defaults, each leaf overridable from the host.
    programs.rss-filter.settings = lib.mapAttrsRecursive (_: lib.mkDefault) (
      builtins.fromTOML (builtins.readFile ../rss_filter/config.toml)
    );

    home.packages = [ runner ];

    # Started on demand (i3 keybinding or `systemctl --user start rss-filter`);
    # no timer, no [Install]. A oneshot has no start timeout by default, so a slow
    # first vault build isn't killed.
    systemd.user.services.rss-filter = {
      Unit.Description = "Score new RSS entries against the Obsidian vault";
      Service = {
        Type = "oneshot";
        ExecStart = lib.getExe runner;
      };
    };
  };
}
```

- [ ] **Step 2:** Evaluate it in isolation:

```bash
nix eval --impure --expr '
  let f = builtins.getFlake (toString ./.); in
  builtins.isFunction f.homeModules.default'
```

  The real integration check is Task 12's build.
- [ ] **Step 3:** Docs (finding 9 + Nix usage): a README section "Running on NixOS", with `USAGE.md` and `EMBEDDING_PIPELINE.md` updated for llama-server, `--state-dir` and `prompt_style`. Commit — `feat: home-manager module for rss-filter`. Push the branch and merge to `main` (the nixos flake will lock to it).

---

## Phase C — NixOS configuration (t480_flake)

Branch: `git switch -c rss-filter`.

### Task 10: Embedding server module

**Files:** Create `modules/features/embedding-server.nix`; modify `modules/hosts/t480/default.nix`

- [ ] **Step 1:**

```nix
{ ... }:
{
  # EmbeddingGemma-300M behind llama.cpp's OpenAI-compatible /v1/embeddings,
  # for rss-filter (flake.homeModules.rssFilter). Loopback only.
  #
  # The GGUF is the QAT Q4_0 build LM Studio ships as
  # text-embedding-embeddinggemma-300m-qat, pinned to a Hugging Face commit so
  # the vectors cannot change under the embedding store.
  #
  # ctx/batch/ubatch are all the model's 2048-token window: llama-server embeds
  # an input in a single micro-batch and rejects the whole request if one input
  # is larger. The client caps and retries on its side.
  flake.nixosModules.embeddingServer =
    { pkgs, ... }:
    {
      services.llama-cpp = {
        enable = true;
        settings = {
          host = "127.0.0.1";
          port = 8080;
          model = "${pkgs.fetchurl {
            name = "embeddinggemma-300m-qat-Q4_0.gguf";
            url = "https://huggingface.co/lmstudio-community/embeddinggemma-300m-qat-GGUF/resolve/a81b371598d25d26b714ab9b14948ce8ca375547/embeddinggemma-300m-qat-Q4_0.gguf";
            hash = "sha256-Wp4GRVQbBjZ97D/RTQAZAVuHLP420137Kp11La3gkCA=";
          }}";
          embeddings = true;
          pooling = "mean";
          ctx-size = 2048;
          batch-size = 2048;
          ubatch-size = 2048;
          # Free the model weights once a run is over (Decision 2); the next
          # request reloads them transparently in about a second.
          sleep-idle-seconds = 60;
        };
      };
    };
}
```

- [ ] **Step 2:** Add `self.nixosModules.embeddingServer` to the `modules` list in `modules/hosts/t480/default.nix`. `git add` the new file (import-tree only sees tracked files under flakes).
- [ ] **Step 3: Check the generated command line:**

```bash
nix eval --raw .#nixosConfigurations.t480.config.systemd.services.llama-cpp.serviceConfig.ExecStart
```

  Expect `…/llama-server --batch-size 2048 --ctx-size 2048 --embeddings --host 127.0.0.1 --model /nix/store/…gguf --pooling mean --port 8080 --sleep-idle-seconds 60 --ubatch-size 2048`, with a bare `--embeddings` because `explicitBool = false` in the module.
- [ ] **Step 4: Commit** — `Adding llama.cpp embedding server for EmbeddingGemma`

### Task 11: rss-filter home module + input

**Files:** `flake.nix`, create `modules/home/rss-filter.nix`, modify `modules/hosts/t480/home.nix`

- [ ] **Step 1:** In `flake.nix` inputs (public repo, Decision 1):

```nix
    rss-gemma-filtering = {
      url = "github:rodgzilla/rss_gemma_filtering";
      inputs.nixpkgs.follows = "nixpkgs";
    };
```

- [ ] **Step 2:** `modules/home/rss-filter.nix`:

```nix
{ inputs, ... }:
{
  # Daily RSS digest scored against the vault by embedding similarity. The
  # program, its config and the oneshot user service come from the
  # rss_gemma_filtering flake; this module supplies the paths and the desktop
  # side: dunst notifications, and waiting for the llama.cpp server
  # (flake.nixosModules.embeddingServer) since a user unit cannot order itself
  # after a system one.
  #
  # Started from i3's `obsidian` mode, see flake.homeModules.i3.
  flake.homeModules.rssFilter =
    { config, pkgs, ... }:
    let
      notify = "${pkgs.libnotify}/bin/notify-send --app-name=rss-filter";
      embeddingUrl = "http://127.0.0.1:8080";
    in
    {
      imports = [ inputs.rss-gemma-filtering.homeModules.default ];

      programs.rss-filter = {
        enable = true;
        vault = config.my.obsidianDir;
        feeds = "${config.home.homeDirectory}/Documents/rss_gemma_filtering/rss_feeds/feeds.opml.xml";
        settings.embedding = {
          base_url = "${embeddingUrl}/v1";
          model = "embeddinggemma-300m-qat-Q4_0";
        };
      };

      systemd.user.services.rss-filter = {
        Unit.OnFailure = [ "rss-filter-failed.service" ];
        Service = {
          ExecStartPre = [
            "${notify} 'RSS filter' 'Fetching and scoring feeds…'"
            "${pkgs.curl}/bin/curl -sf --retry 30 --retry-connrefused --retry-delay 2 -o /dev/null ${embeddingUrl}/health"
          ];
          ExecStartPost = "${notify} 'RSS filter' 'Digest written to the vault'";
        };
      };

      systemd.user.services.rss-filter-failed = {
        Unit.Description = "Report a failed rss-filter run";
        Service = {
          Type = "oneshot";
          ExecStart = "${notify} --urgency=critical 'RSS filter failed' 'journalctl --user -u rss-filter'";
        };
      };
    };
}
```

- [ ] **Step 3:** Add `rssFilter` to the home imports in `modules/hosts/t480/home.nix`, under `# Applications`.
- [ ] **Step 4:** `nix flake lock`. Commit — `Adding rss-filter home module`

### Task 12: i3 binding

**File:** `modules/home/i3.nix` (the `obsidian` mode, around line 172)

- [ ] **Step 1:**

```nix
          # Long-running, so it goes to systemd rather than being exec'd here:
          # logs land in the journal and a second press joins the running job.
          r = ''exec "systemctl --user start --no-block rss-filter.service; i3-msg mode 'default'"'';
```

- [ ] **Step 2: Build without switching:** `nixos-rebuild build --flake .#t480`. Then inspect the generated unit (`result/…` or after switch):

```bash
nix eval --raw .#nixosConfigurations.t480.config.home-manager.users.rodgzilla.systemd.user.services.rss-filter.Service.ExecStart
```

- [ ] **Step 3: Commit** — `Binding rss-filter to the obsidian i3 mode`

### Task 13: Switch and verify end to end

- [ ] `update`. Then check:
  - `systemctl status llama-cpp` is active. `journalctl -u llama-cpp` shows no `MemoryDenyWriteExecute`/seccomp failures (it ran fine unconfined in testing; the hardening is the new variable).
  - `curl -s 127.0.0.1:8080/v1/embeddings -H 'Content-Type: application/json' -d '{"input":["hello"]}' | jq '.data[0].embedding | length'` → `768`.
  - `ps -o rss= -C llama-server` ≈ 340 MB right after the request, and ≈ 73 MB a minute later. `journalctl -u llama-cpp` shows `server is entering sleeping state`.
- [ ] Sync the vault to `~/Documents/obsidian_main`. Then `rss-filter-run --dry-run --max-notes 3` prints scored entries, and `~/.local/state/rss-filter/embedding_store.db` exists.
- [ ] Press `$mod+y r`. The notification shows, `journalctl --user -fu rss-filter` shows progress, and `Filtered feed/RSS-<today>.md` + `-scores.html` appear in the vault.
- [ ] Failure path: `sudo systemctl stop llama-cpp`, press `$mod+y r`, and after ~60 s of curl retries the critical notification appears. Restart the server afterwards.
- [ ] Now run the deferred **Task 3 Step 4** eval against the real vault. If `document` wins, change the default in the rss repo, bump the flake input (`nix flake update rss-gemma-filtering`), and the signature change rebuilds the store automatically on the next run.

---

## Later (out of scope)

- **Zero-footprint server:** if the ~73 MB left while sleeping ever matters, move llama-server into a user unit with `StopWhenUnneeded=yes`, which rss-filter `Requires=`, so it exists only during a run.
- **Throughput:** the MX150 is on nouveau, so there's no CUDA; the iGPU could be tried via `llama-cpp.override { vulkanSupport = true; }` if runs get slow.
- **Timer:** add a `systemd.user.timers.rss-filter` (`OnCalendar=daily`, `Persistent=true`) to the HM module behind an option, if manual triggering gets tedious.
- `seen_entries.json` grows without bound. Prune GUIDs older than `max_age_days` × 2.
