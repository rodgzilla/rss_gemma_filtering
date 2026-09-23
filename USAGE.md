# Usage Guide

## Requirements

- Python 3.11+
- A local embedding server compatible with the OpenAI `/v1/embeddings` API, listening at
  `http://127.0.0.1:8080/v1` (default). Recommended: llama.cpp's `llama-server` with
  EmbeddingGemma 300M QAT Q4_0 (command line in the [README](README.md#embedding-server)).
  Other servers (e.g. LM Studio) work too; set `base_url` accordingly.
- An Obsidian vault with daily notes under `Daily notes/YYYY-MM-DD.md`
- A subscription list: the RSS Dashboard Obsidian plugin's
  `<vault>/.rss-dashboard-data/data.json`, or an OPML export

## Installation

```bash
git clone https://github.com/rodgzilla/rss_gemma_filtering
cd rss_gemma_filtering
pip install .        # or: pip install -e '.[test]' for development
```

This installs the `rss-filter` command; `python main.py` from the checkout is equivalent.
On NixOS, see the README's "Running on NixOS" section.

## Configuration

The defaults live in the packaged `rss_filter/config.toml`; pass `--config` to use your
own copy (a missing key falls back to a built-in default):

```toml
[paths]
seen_entries = "seen_entries.json"

[fetch]
timeout_seconds = 15

[vault]
daily_notes_folder = "Daily notes"
output_folder = "Filtered feed"

[embedding]
model = "embeddinggemma-300m-qat-Q4_0"
base_url = "http://127.0.0.1:8080/v1"
store_path = "embedding_store.db"
top_k = 5
prompt_style = "none"
top_quantile = 0.2
top_quantile_arxiv = 0.1
decay_lambda = 1.0
umap_model_path = "umap_model.joblib"
umap_growth_threshold = 0.1
vault_bg_max = 500
```

Relative paths (`seen_entries`, `store_path`, `umap_model_path`) are resolved against
`--state-dir` (default: the current directory).

Key settings:

| Key | Description |
|---|---|
| `base_url` | Embedding server API endpoint |
| `model` | Model name sent to the server (llama-server ignores it); part of the store signature |
| `store_path` | Path to the SQLite embedding database (created automatically) |
| `prompt_style` | Text format sent for embedding: `none` (`title body`) or `document` (`title: … \| text: …`) |
| `top_k` | Number of nearest-neighbour exemplars retrieved per entry |
| `top_quantile` | Top fraction of reading entries to keep (0.20 → top 20 %) |
| `top_quantile_arxiv` | Top fraction of arXiv entries to keep (separate threshold) |
| `decay_lambda` | Exponential decay weight for score aggregation |
| `timeout_seconds` | Per-feed fetch timeout; unresponsive feeds are skipped |
| `seen_entries` | Path to the deduplication state file (created automatically) |
| `daily_notes_folder` | Subfolder inside your vault containing daily notes |
| `output_folder` | Subfolder inside your vault where digest notes are written |

Changing `model` or `prompt_style` (or an upgrade that changes text cleaning) changes the
store signature, and the embedding store is rebuilt automatically on the next run.

## Basic Usage

```bash
python main.py --vault /path/to/your/vault \
  --feeds /path/to/your/vault/.rss-dashboard-data/data.json
```

On the first run this will:

1. Parse all daily notes and embed every saved article into a local SQLite vector store
2. Fetch every subscribed feed, minus those in a `feeds.exclude_folders` folder
3. Discard entries older than 7 days and entries already seen in a previous run
4. Embed each new entry and score it by cosine similarity against the vault store
5. Keep the top-scoring entries according to the configured quantile thresholds
6. Write the digest to `<vault>/Filtered feed/RSS-YYYY-MM-DD.md`
7. Write an interactive UMAP score visualisation to `<vault>/Filtered feed/RSS-YYYY-MM-DD-scores.html`
8. Save seen entry GUIDs to `seen_entries.json` (in `--state-dir`) so they are skipped next time

## All Flags

| Flag | Default | Description |
|---|---|---|
| `--vault PATH` | required | Path to your Obsidian vault root |
| `--feeds PATH` | required | Subscription list: an RSS Dashboard `data.json`, or an OPML export. `.json` selects the first, anything else the second |
| `--config PATH` | packaged `rss_filter/config.toml` | Path to a custom config file |
| `--state-dir PATH` | current directory | Where the embedding store, seen entries and UMAP models live (relative config paths resolve against it) |
| `--max-notes N` | all notes | Limit embedding build to the N most recent daily notes |
| `--max-age-days N` | `7` | Only evaluate entries published within the last N days |
| `--rebuild-embeddings` | off | Force full rebuild of the embedding database from scratch |
| `--top-quantile Q` | config / `0.25` | Top fraction of reading entries to keep (e.g. `0.20`) |
| `--top-quantile-arxiv Q` | same as `--top-quantile` | Top fraction of arXiv entries to keep |
| `--decay-lambda L` | config / `1.0` | Exponential decay weight for score aggregation |
| `--rebuild-umap` | off | Force refit of the UMAP model on vault embeddings |
| `--vault-bg-max N` | config / `500` | Max vault background points shown in the UMAP panel |
| `--feed-timeout SECONDS` | config / `15` | Per-feed network timeout |
| `--no-seen-filter` | off | Skip deduplication against `seen_entries.json` (testing) |
| `--dry-run` | off | Print filtered entries to stdout; do not write a note |

## Common Workflows

### Normal daily run

```bash
python main.py \
  --vault ~/Documents/MyVault \
  --feeds ~/Documents/subscriptions.opml
```

The embedding store is updated incrementally (only new vault notes are embedded).
Only entries newer than 7 days and not yet seen are scored and evaluated.

### Preview without writing anything

```bash
python main.py \
  --vault ~/Documents/MyVault \
  --feeds ~/Documents/subscriptions.opml \
  --dry-run
```

Prints filtered entries to stdout. Nothing is written to disk and no GUIDs are saved as
seen, so the same entries will appear again on the next run.

### Force a full rebuild of the embedding database

Use this after bulk-importing many old daily notes or if the store seems corrupted:

```bash
python main.py \
  --vault ~/Documents/MyVault \
  --feeds ~/Documents/subscriptions.opml \
  --rebuild-embeddings
```

### Quick smoke test against the mock vault

```bash
python main.py \
  --vault mock_vault \
  --feeds rss_feeds/feeds_2026-04-24.opml.xml \
  --max-notes 3 \
  --dry-run
```

### Fetch entries from the last 14 days

```bash
python main.py \
  --vault ~/Documents/MyVault \
  --feeds ~/Documents/subscriptions.opml \
  --max-age-days 14
```

### Use a custom config file

```bash
python main.py \
  --vault ~/Documents/MyVault \
  --feeds ~/Documents/subscriptions.opml \
  --config ~/dotfiles/rss_filter_config.toml
```

### Override quantile thresholds at the command line

```bash
python main.py \
  --vault ~/Documents/MyVault \
  --feeds ~/Documents/subscriptions.opml \
  --top-quantile 0.30 \
  --top-quantile-arxiv 0.10
```

## Output Format

Filtered entries are written to `<vault>/Filtered feed/RSS-YYYY-MM-DD.md`:

```markdown
# RSS Digest — 2026-04-24

## Reading

- [Some Article Title](https://example.com/article) *(score: 0.8731)*
  > Nearest matches: "Related article A" (0.91), "Related article B" (0.87)

## Arxiv monitoring

- [A Paper on Topic Y](https://arxiv.org/abs/1234.56789) *(score: 0.6412)*
  > Nearest matches: "Prior paper X" (0.84), "Prior paper Z" (0.79)
```

Each entry shows its aggregated similarity score and the top-K nearest vault articles
that contributed to the score.

An interactive HTML UMAP visualisation (`RSS-YYYY-MM-DD-scores.html`) is also written
to the same folder, showing kept and rejected entries projected onto the vault embedding
space.

## Deduplication

Every entry processed in a non-dry-run is recorded in `seen_entries.json` (in `--state-dir`). On the next
run those entries are skipped regardless of their publication date. To reset
deduplication state (e.g. to reprocess all current entries), delete the file:

```bash
rm seen_entries.json
```

## Running the Tests

```bash
pytest tests/
```

All tests use mocked HTTP calls; no embedding server is needed.
