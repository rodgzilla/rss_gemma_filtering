# Embedding-Based Filtering — Implementation Plan

## Phase 1 — Configuration & Dependencies

### 1.1 Update `config.toml`
Add a new `[embedding]` section:

```toml
[embedding]
model = "text-embedding-embeddinggemma-300m-qat"
base_url = "http://localhost:1234/v1"
store_path = "embedding_store.db"
top_k = 3
```

### 1.2 Update `requirements.txt`
Add `numpy` (if not already present). All other dependencies (`openai`, `feedparser`,
`listparser`, `tqdm`) are reused from the existing pipeline.

---

## Phase 2 — New Modules

### 2.1 `rss_filter/embedding_client.py`

**Responsibility:** Thin wrapper around the LM Studio embedding API.

**Public interface:**
```python
class EmbeddingClient:
    def __init__(self, base_url: str, model: str) -> None: ...
    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]: ...
```

**Implementation notes:**
- Uses `openai.OpenAI(base_url=..., api_key="lm-studio")` (LM Studio accepts any API key).
- `embed()` calls `client.embeddings.create(model=model, input=text)` and returns
  `np.array(response.data[0].embedding, dtype=np.float32)`.
- `embed_batch()` sends all texts in a single API call (LM Studio supports batch input) and
  returns a list of arrays.
- No retry logic in v1; can be added later.

**Tests (`tests/test_embedding_client.py`):**
- Mock `openai.OpenAI` client.
- Verify correct model and input are passed to `embeddings.create`.
- Verify the returned numpy array has the correct dtype and shape.
- Verify `embed_batch` returns a list of arrays of correct length.

---

### 2.2 `rss_filter/embedding_store.py`

**Responsibility:** SQLite-backed embedding database with numpy-powered cosine similarity search.

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT NOT NULL,
    text        TEXT NOT NULL,
    source_note TEXT NOT NULL,
    date        TEXT,
    embedding   BLOB NOT NULL,
    UNIQUE(url, source_note)
);
```

The `embedding` column stores a float32 numpy array serialised with `numpy.tobytes()`.
Deserialisation uses `numpy.frombuffer(blob, dtype=np.float32)`.

**Public interface:**
```python
class EmbeddingStore:
    def __init__(self, db_path: str) -> None: ...
    def build_or_update(
        self,
        entries: list[NoteEntry],
        client: EmbeddingClient,
        force_rebuild: bool = False,
    ) -> None: ...
    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[dict]:  # [{"text": ..., "url": ..., "score": float}, ...]
        ...
    def count(self) -> int: ...
```

**`build_or_update` logic:**
1. If `force_rebuild=True`, drop and recreate the `documents` table.
2. Load all existing `(url, source_note)` pairs from the DB into a set.
3. Filter `entries` to only those not in the set.
4. For new entries, construct embed text: use `entry.context` if non-empty, else `entry.url`.
5. Call `client.embed_batch([text for new entries])` in chunks of 64 to avoid oversized requests.
6. Insert each `(url, text, source_note, date, embedding.tobytes())` row.
7. Print a summary: `N new entries embedded, M already in store`.

**`query` logic:**
1. Load all rows from `documents`: retrieve `text`, `url`, `embedding` BLOB.
2. Deserialise all embeddings into a 2D numpy matrix `E` of shape `(N, D)`.
3. Normalise `E` row-wise and normalise the query `embedding`.
4. Compute `scores = E_norm @ query_norm` (shape `(N,)`).
5. Return top-K rows sorted by descending score as a list of dicts:
   `{"text": ..., "url": ..., "score": float}`.

**Tests (`tests/test_embedding_store.py`):**
- Use a temporary in-memory SQLite DB (`:memory:`) or a `tmp_path` fixture.
- Test that `build_or_update` inserts new entries and skips existing ones.
- Test that `force_rebuild=True` clears and repopulates the table.
- Test `query` returns correct top-K with expected cosine scores using known embeddings.
- Test `count` returns the correct number of stored documents.
- Test empty store returns empty list from `query`.

---

### 2.3 `rss_filter/embedding_filter.py`

**Responsibility:** Embed incoming RSS entries, retrieve exemplars, build few-shot prompts,
call Gemma-4, parse responses into `FilterResult` objects.

**Public interface:**
```python
def filter_entries_batch(
    entries: list[RSSEntry],
    store: EmbeddingStore,
    embed_client: EmbeddingClient,
    llm_client: openai.OpenAI,
    model: str,
    top_k: int = 3,
    batch_size: int = 20,
    temperature: float = 0.1,
) -> list[FilterResult]: ...
```

**Internal helpers:**
```python
def build_filter_prompt(
    entries_with_exemplars: list[dict],  # [{"entry": RSSEntry, "exemplars": list[dict]}, ...]
) -> str: ...

def parse_filter_response(
    response: str,
    entries: list[RSSEntry],
) -> list[FilterResult]: ...
```

**`filter_entries_batch` logic:**
1. Embed all entry texts (`title + " " + summary`) in a single `embed_batch` call.
2. For each entry, call `store.query(embedding, top_k)` to get exemplars.
3. Group entries into batches of `batch_size`.
4. For each batch, call `build_filter_prompt` → LLM → `parse_filter_response`.
5. Return all `FilterResult` objects preserving input order.

**`build_filter_prompt` format:**
```
You are a research assistant helping filter RSS articles based on past reading habits.
For each numbered article below, answer "yes: <reason>" or "no: <reason>" on a single line.

1. Article title: "..."
   Most similar articles previously found relevant:
   - "..." (similarity: 0.91)
   - "..." (similarity: 0.88)
   - "..." (similarity: 0.79)
   If the title alone is not sufficient, here is the article summary:
   <summary>...</summary>

2. Article title: "..."
   ...
```

**`parse_filter_response` logic:**
- Split response by lines, match numbered `yes:`/`no:` patterns.
- Missing lines default to `keep=False, reason="no response"`.
- Returns one `FilterResult(entry=..., keep=bool, reason=str, score=None)` per entry.

**Tests (`tests/test_embedding_filter.py`):**
- Mock `EmbeddingStore`, `EmbeddingClient`, and `openai.OpenAI`.
- Test `build_filter_prompt` produces correct structure for single and multi-entry batches.
- Test `parse_filter_response` correctly handles `yes`/`no` lines, missing lines, and
  malformed output.
- Test `filter_entries_batch` calls embed, query, and LLM the expected number of times.
- Test input order is preserved in the returned `FilterResult` list.

---

### 2.4 `main_embedding.py`

**Responsibility:** CLI entry point for the embedding pipeline. Mirrors `main.py` structure.

**CLI arguments:**

| Flag | Description |
|---|---|
| `--vault PATH` | Path to the Obsidian vault (required) |
| `--feeds PATH` | Path to the OPML subscriptions file (required) |
| `--config PATH` | Path to `config.toml` (default: `config.toml`) |
| `--rebuild-embeddings` | Force full rebuild of the embedding database |
| `--dry-run` | Print digest to stdout instead of writing to vault |
| `--rerank` | Re-rank kept entries by relevance score after filtering |
| `--max-age-days N` | Drop entries older than N days (default: 7) |
| `--batch-size N` | Number of entries per LLM call (default: 20) |
| `--no-seen-filter` | Skip deduplication against seen_entries.json (for testing) |

**Pipeline execution order:**
1. Load `config.toml`.
2. Initialise `EmbeddingClient` (embedding model) and `openai.OpenAI` (LLM).
3. Parse vault with `notes_parser.parse_vault()`.
4. Build/update embedding store with `EmbeddingStore.build_or_update()`.
5. Fetch RSS feeds; filter seen and aged-out entries.
6. Filter entries with `embedding_filter.filter_entries_batch()`.
7. Optionally re-rank with `reranker.rerank()`.
8. Split into `reading_results` and `arxiv_results`.
9. Write output note with `note_writer.write_note()` (or print if `--dry-run`).
10. Persist seen GUIDs with `rss_fetcher.save_seen_guids()`.

---

## Phase 3 — Configuration Updates

### `config.toml` additions
```toml
[embedding]
model = "text-embedding-embeddinggemma-300m-qat"
base_url = "http://localhost:1234/v1"
store_path = "embedding_store.db"
top_k = 3
```

### `requirements.txt` additions
```
numpy
```

---

## File Checklist

- [ ] `rss_filter/embedding_client.py`
- [ ] `rss_filter/embedding_store.py`
- [ ] `rss_filter/embedding_filter.py`
- [ ] `main_embedding.py`
- [ ] `tests/test_embedding_client.py`
- [ ] `tests/test_embedding_store.py`
- [ ] `tests/test_embedding_filter.py`
- [ ] `config.toml` — add `[embedding]` section
- [ ] `requirements.txt` — add `numpy`
- [ ] `EMBEDDING_PIPELINE.md` — pipeline logic documentation
- [ ] `EMBEDDING_PLAN.md` — this file

---

## Out of Scope (v1)

- Retry logic for embedding API failures
- Streaming LLM responses
- Async/parallel embedding of RSS entries
- GUI or web interface
- Automatic vault discovery
