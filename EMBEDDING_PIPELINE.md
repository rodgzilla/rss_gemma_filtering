# Embedding-Based RSS Filtering Pipeline

## Overview

This pipeline uses an embedding-based approach to personalise RSS filtering.
Every article the user has previously saved in their Obsidian vault is embedded into a
local vector database. When new RSS entries arrive, they are embedded and scored by
cosine similarity against this database. The top-scoring entries (by quantile threshold)
are written to a daily digest note. No LLM is involved at inference time.

---

## Core Idea

> "Score new articles by how similar they are to articles you've already saved."

A dense vector embedding captures semantic meaning more faithfully than keyword matching
or an LLM-summarised interest profile. By retrieving the most similar past articles and
aggregating their cosine similarities with exponential decay weighting, the pipeline
produces a scalar relevance score for each new entry — no prompt engineering required.

---

## Step 1 — Build the Embedding Database from Vault Notes

### Input
Obsidian daily notes located in the `Daily notes/` folder of the vault. Each note follows
the standard format with `## Reading` and `## [[Arxiv]] monitoring` sections containing
markdown links and optional context paragraphs.

### Process
1. `notes_parser.parse_vault()` scans all `YYYY-MM-DD.md` files and extracts `NoteEntry`
   objects: each holds a URL, a title, a context snippet, the section it came from
   (`source`: `"reading"` or `"arxiv"`), and the date.
2. The text to embed is built by `text_prep.format_for_embedding(title, body, prompt_style)`,
   exactly as for feed entries (the body is the context minus the leading title; the URL is
   used only when there is neither). Texts are capped at 4000 characters so each fits in
   the server's 2048-token batch; if the server still rejects a batch, the client retries
   the inputs one by one, halving any that are too long.
3. Texts are sent to the local OpenAI-compatible embedding API (recommended: `llama-server`
   with EmbeddingGemma 300M QAT Q4_0, default `http://127.0.0.1:8080/v1`).
4. The resulting float32 embedding vector is stored alongside the embedded text, URL,
   section (`source_note` column, holding `"reading"`/`"arxiv"`, not a filename), date and
   title in a SQLite database (`embedding_store.db`, under `--state-dir`).

### Incremental Updates and Rebuilds
The database tracks which `(url, source_note)` pairs have already been embedded. On
subsequent runs only new entries are processed. The `--rebuild-embeddings` CLI flag forces
a full rebuild from scratch.

The store also records a signature, `<model>|prompt=<prompt_style>|prep=<PREP_VERSION>`.
When it differs from the current configuration (other model, prompt style or text
cleaning version) the whole store is re-embedded automatically. New vectors are computed
before the old table is dropped, so a server failure mid-rebuild leaves the old store intact.

---

## Step 2 — Fetch and Deduplicate RSS Entries

- The OPML file is parsed to obtain feed URLs.
- Each feed is fetched via `feedparser`.
- Entries already present in `seen_entries.json` are dropped (deduplication).
- Cross-posted duplicates (same URL from multiple feeds) are removed.
- Entries older than `--max-age-days` (default 7) are dropped.

Every surviving entry is forwarded to the scoring step.

---

## Step 3 — Score Entries by Embedding Similarity

For each RSS entry (`score_filter.score_entries`):

1. The summary is cleaned by `text_prep.clean_summary` (HTML stripped, arXiv's
   `arXiv:… Announce Type: … Abstract:` header removed), then formatted with the title by
   `format_for_embedding` using the same `prompt_style` as the vault, and embedded with the
   same model (in batches of 64).
2. The top-K most similar stored vault articles are retrieved from the database
   (K = `top_k`, default 5; the normalised vault matrix is cached in memory).
3. The K cosine similarities are aggregated using **exponential decay weighting**:

   ```
   weight_i = exp(-lambda * i)   for i = 0 … K-1  (i=0 is the best match)
   score    = Σ(sim_i * weight_i) / Σ(weight_i)
   ```

   At `decay_lambda=1.0` the top match contributes roughly 64 % of the weight (top-5).
   At `decay_lambda=0` this reduces to a plain mean.

### Quantile Thresholds

Two independent quantile thresholds are applied after all scores are computed:

- **Reading entries** (`top_quantile`, default 0.20): keep entries whose score is at or
  above the (1 − 0.20) = 80th percentile of all reading entry scores.
- **arXiv entries** (`top_quantile_arxiv`, default 0.1): keep entries at or above the
  90th percentile of all arXiv entry scores (tighter, since arXiv volume is high).

Both thresholds can be overridden at the command line or in `rss_filter/config.toml`.

---

## Step 4 — Output

Kept entries are written to the Obsidian vault as a new daily note by `note_writer.py`:

```markdown
# RSS Digest — YYYY-MM-DD

## Reading
- [Title](url) *(score: 0.8731)*
  > Nearest matches: "Related article A" (0.91), "Related article B" (0.87)

## Arxiv monitoring
- [Title](url) *(score: 0.6412)*
  > Nearest matches: "Prior paper X" (0.84), "Prior paper Z" (0.79)
```

An interactive **UMAP score visualisation** (`RSS-YYYY-MM-DD-scores.html`) is also
written to the same folder, projecting vault embeddings and incoming RSS entry embeddings
onto a 2-D plane with colour-coded keep/reject status.

The GUIDs of all evaluated entries (kept and rejected) are persisted to
`seen_entries.json` to prevent re-evaluation on future runs.

---

## Data Flow

```
Obsidian vault (Daily notes/*.md)
        │  notes_parser.parse_vault()
        ▼
   [NoteEntry list]
        │  embedding_store.build_or_update()
        │  embedding_client.embed() × new entries
        ▼
   embedding_store.db  (SQLite: url, text, source_note, date, embedding BLOB, title)
        │
        │
RSS feeds (OPML)
        │  rss_fetcher.parse_opml()
        │  rss_fetcher.fetch_feed() × N
        │  rss_fetcher.filter_new_entries()
        │  rss_fetcher.deduplicate_entries()
        │  rss_fetcher.filter_by_age()
        ▼
   [RSSEntry list]
        │  score_filter.score_entries()
        │    └─ embed(format_for_embedding(title, clean_summary(summary)))
        │    └─ store.query(embedding, top_k) per entry
        │    └─ exponential decay aggregation
        │    └─ per-category quantile threshold
        ▼
   [FilterResult list (keep=True/False, score, exemplars)]
        │  note_writer.write_note()
        │  score_viz.write_score_viz()
        ▼
   <vault>/Filtered feed/RSS-YYYY-MM-DD.md
   <vault>/Filtered feed/RSS-YYYY-MM-DD-scores.html
        +
   seen_entries.json (updated)
```

---

## Configuration

All settings live under the `[embedding]` section of `rss_filter/config.toml`:

```toml
[embedding]
model                 = "embeddinggemma-300m-qat-Q4_0"  # label only for llama-server; part of the store signature
base_url              = "http://127.0.0.1:8080/v1"
store_path            = "embedding_store.db"
top_k                 = 5
prompt_style          = "none"   # "none": "title body"; "document": "title: … | text: …"
top_quantile          = 0.2
top_quantile_arxiv    = 0.1
decay_lambda          = 1.0
umap_model_path       = "umap_model.joblib"
umap_growth_threshold = 0.1
vault_bg_max          = 500
```

---

## Module Responsibilities

| Module | Role |
|---|---|
| `rss_filter/embedding_client.py` | OpenAI-compatible embedding client (llama-server); retries oversized inputs one by one |
| `rss_filter/text_prep.py` | HTML stripping, arXiv header removal, `prompt_style` formatting, length cap |
| `rss_filter/embedding_store.py` | SQLite + numpy embedding database: build, update, query |
| `rss_filter/score_filter.py` | Embed entries, retrieve exemplars, aggregate scores, apply quantile thresholds |
| `rss_filter/score_viz.py` | Build and write the interactive UMAP HTML visualisation |
| `rss_filter/notes_parser.py` | Parse vault daily notes into `NoteEntry` objects |
| `rss_filter/rss_fetcher.py` | Fetch and deduplicate RSS entries from OPML feeds |
| `rss_filter/note_writer.py` | Render kept entries as an Obsidian markdown digest note |
| `rss_filter/models.py` | Data models: `NoteEntry`, `RSSEntry`, `FilterResult` |
| `rss_filter/cli.py` | CLI entry point (`main.py` is a thin shim; installed as `rss-filter`) |
