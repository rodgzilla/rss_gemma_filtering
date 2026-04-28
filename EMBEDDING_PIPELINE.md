# Embedding-Based RSS Filtering Pipeline

## Overview

This pipeline replaces the interest-profile approach (iterative LLM summarisation of vault notes)
with an embedding-based approach. Instead of distilling the vault into a single textual profile,
every article the user has previously saved is embedded into a vector database. When new RSS
entries arrive, they are embedded and matched against this database, and the closest past articles
are used as few-shot exemplars in the filtering prompt sent to Gemma-4.

---

## Core Idea

> "Show the model what you liked before, don't tell it what you like."

The key insight is that a dense vector embedding captures semantic meaning more faithfully than a
keyword-extracted or LLM-summarised profile. By retrieving the most similar past articles and
presenting them to Gemma-4 as concrete examples, the model can make a contextualised relevance
judgement without needing a hand-crafted or iteratively-built description of the user's interests.

---

## Step 1 — Build the Embedding Database from Vault Notes

### Input
Obsidian daily notes located in the `Daily notes/` folder of the vault. Each note follows the
format used by the existing pipeline, with `## Reading` and `## [[Arxiv]] monitoring` sections
containing markdown links and optional context paragraphs.

### Process
1. `notes_parser.parse_vault()` scans all `YYYY-MM-DD.md` files and extracts `NoteEntry` objects:
   each entry holds a URL, a context snippet (the surrounding paragraph or abstract), the source
   note filename, and the date.
2. For each `NoteEntry`, the text to embed is constructed as:
   ```
   <context snippet or URL if no context>
   ```
   If a context snippet is present it is used directly; otherwise the URL is used as a fallback.
3. Each text is sent to the local LM Studio embedding API using the model
   `text-embedding-embeddinggemma-300m-qat`.
4. The resulting float32 embedding vector is stored alongside the original text, URL, source note
   name, and date in a SQLite database (`embedding_store.db`).

### Incremental Updates
The database tracks which `(url, source_note)` pairs have already been embedded. On subsequent
runs only new entries are processed. A `--rebuild-embeddings` CLI flag forces a full rebuild from
scratch.

---

## Step 2 — Fetch and Deduplicate RSS Entries

The existing RSS fetching logic is reused without modification:
- OPML file is parsed to obtain feed URLs.
- Each feed is fetched via `feedparser`.
- Entries already present in `seen_entries.json` are dropped (deduplication).
- Entries older than `--max-age-days` are dropped.

No keyword pre-filter is applied. Every new entry is forwarded to the embedding filter.

---

## Step 3 — Embed Incoming RSS Entries and Retrieve Exemplars

For each RSS entry:
1. The text `<title> <summary>` is embedded using the same embedding model.
2. All stored embeddings are loaded from the database into a numpy matrix.
3. Cosine similarity is computed between the entry's embedding and every stored embedding in a
   single vectorised operation.
4. The top-K most similar stored articles are returned as exemplars (K is configurable in
   `config.toml`, default 3).

Cosine similarity is used as the distance metric because it is scale-invariant and standard for
text embeddings.

---

## Step 4 — Single-Pass LLM Filtering with Few-Shot Exemplars

Each RSS entry is filtered by Gemma-4 in a single pass. The prompt is structured as follows:

```
Article title: "<title of the new RSS entry>"

Most similar articles you previously found relevant:
1. "<title or context of exemplar 1>" (similarity: 0.91)
2. "<title or context of exemplar 2>" (similarity: 0.88)
3. "<title or context of exemplar 3>" (similarity: 0.79)

Should this article be included in today's digest?
Answer yes or no with a brief reason.
If the title alone is not sufficient to decide, here is the article summary:
<summary of the new RSS entry>
```

The summary is always included at the end of the prompt as a fallback. Gemma-4 is free to use it
or ignore it depending on whether the title and exemplars are sufficient to make a decision.

Entries are processed in batches to limit the number of API calls. Each batch contains multiple
articles with their respective exemplar sets, formatted as a numbered list, and the model returns
one `yes: <reason>` or `no: <reason>` line per article.

---

## Step 5 — Output

Kept entries are written to the Obsidian vault as a new daily note using the existing
`note_writer.py` module. The output format is identical to the original pipeline:

```markdown
# RSS Digest — YYYY-MM-DD

## Reading
- [Title](url) *(score: 0.9)*
  > LLM one-line reason

## Arxiv monitoring
- [Title](url) *(score: 0.7)*
  > LLM one-line reason
```

Optional re-ranking via the existing `reranker.py` module is supported.

The GUIDs of all evaluated entries (kept and rejected) are persisted to `seen_entries.json` to
prevent re-evaluation on future runs.

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
   embedding_store.db  (SQLite: url, text, source_note, date, embedding BLOB)
        │
        │
RSS feeds (OPML)
        │  rss_fetcher.parse_opml()
        │  rss_fetcher.fetch_feed() × N
        │  rss_fetcher.filter_new_entries()
        │  rss_fetcher.filter_by_age()
        ▼
   [RSSEntry list]
        │  embedding_client.embed(title + summary) per entry
        │  embedding_store.query(embedding, top_k) per entry
        ▼
   [RSSEntry + exemplars list]
        │  embedding_filter.filter_entries_batch()
        │  (single-pass Gemma-4 with few-shot exemplars)
        ▼
   [FilterResult list (keep=True/False)]
        │  (optional) reranker.rerank()
        ▼
   reading_results + arxiv_results
        │  note_writer.write_note()
        ▼
   <vault>/Filtered feed/RSS-YYYY-MM-DD.md
        +
   seen_entries.json (updated)
```

---

## Configuration

All embedding-specific settings live under the `[embedding]` section of `config.toml`:

```toml
[embedding]
model = "text-embedding-embeddinggemma-300m-qat"
base_url = "http://localhost:1234/v1"
store_path = "embedding_store.db"
top_k = 3
```

---

## Module Responsibilities

| Module | Role |
|---|---|
| `rss_filter/embedding_client.py` | Thin wrapper around the LM Studio embedding API |
| `rss_filter/embedding_store.py` | SQLite + numpy embedding database: build, update, query |
| `rss_filter/embedding_filter.py` | Build few-shot prompts, call Gemma-4, parse responses |
| `main_embedding.py` | CLI entry point for the embedding pipeline |
| `rss_filter/notes_parser.py` | Reused as-is to parse vault notes |
| `rss_filter/rss_fetcher.py` | Reused as-is to fetch and deduplicate RSS entries |
| `rss_filter/note_writer.py` | Reused as-is to write the output digest note |
| `rss_filter/reranker.py` | Optionally reused for post-filter re-ranking |
| `rss_filter/models.py` | Reused as-is; `NoteEntry`, `RSSEntry`, `FilterResult` unchanged |
