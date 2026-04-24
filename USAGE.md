# Usage Guide

## Requirements

- Python 3.11+
- [LM Studio](https://lmstudio.ai/) running locally with `google/gemma-4-e4b` loaded
  - Set context length to at least **16384** tokens in LM Studio
  - The server must be listening at `http://localhost:1234/v1` (default)
- An Obsidian vault with daily notes under `Daily notes/YYYY-MM-DD.md`
- An OPML export of your RSS subscriptions

## Installation

```bash
git clone https://github.com/rodgzilla/rss_gemma_filtering
cd rss_gemma_filtering
pip install -r requirements.txt
```

## Configuration

Edit `config.toml` before your first run:

```toml
[lmstudio]
base_url = "http://localhost:1234/v1"
model = "google/gemma-4-e4b"
temperature = 0.1

[paths]
seen_entries = "seen_entries.json"
interest_profile = "interest_profile.md"

[vault]
daily_notes_folder = "Daily notes"
output_folder = "Filtered feed"
```

- `base_url` — LM Studio API endpoint (leave as-is unless you changed the port)
- `model` — model identifier as shown in LM Studio
- `seen_entries` — path to the deduplication state file (created automatically)
- `interest_profile` — path to the cached interest profile (created automatically)
- `daily_notes_folder` — subfolder inside your vault that contains daily notes
- `output_folder` — subfolder inside your vault where digest notes are written

## Basic Usage

```bash
python main.py --vault /path/to/your/vault --feeds /path/to/subscriptions.opml
```

On the first run this will:
1. Parse all daily notes to build your interest profile (LLM-intensive, may take 10–30 min)
2. Fetch all RSS feeds from the OPML file
3. Discard entries older than 7 days and entries already seen in a previous run
4. Send each remaining entry to the LLM for yes/no filtering
5. Write the digest to `<vault>/Filtered feed/RSS-YYYY-MM-DD.md`
6. Save seen entry GUIDs to `seen_entries.json` so they are skipped next time

## All Flags

| Flag | Default | Description |
|---|---|---|
| `--vault PATH` | required | Path to your Obsidian vault root |
| `--feeds PATH` | required | Path to your OPML subscriptions file |
| `--config PATH` | `config.toml` | Path to a custom config file |
| `--max-age-days N` | `7` | Only evaluate entries published within the last N days |
| `--max-notes N` | all notes | Limit profiling to the N most recent daily notes |
| `--rebuild-profile` | off | Ignore the cached profile and regenerate it from scratch |
| `--dry-run` | off | Print filtered entries to stdout instead of writing a note |

## Common Workflows

### Normal daily run

```bash
python main.py \
  --vault ~/Documents/MyVault \
  --feeds ~/Documents/subscriptions.opml
```

The profile is loaded from cache (`interest_profile.md`). Only entries newer than 7 days
and not yet seen are evaluated. The digest is written to the vault.

### Preview without writing anything

```bash
python main.py \
  --vault ~/Documents/MyVault \
  --feeds ~/Documents/subscriptions.opml \
  --dry-run
```

Prints the filtered entries to stdout. Nothing is written to disk and no GUIDs are saved
as seen, so the same entries will appear again on the next run.

### Rebuild the interest profile from scratch

Use this after adding many new daily notes or if the cached profile feels stale:

```bash
python main.py \
  --vault ~/Documents/MyVault \
  --feeds ~/Documents/subscriptions.opml \
  --rebuild-profile
```

### Quick prototype / test run

Limit profiling to the 5 most recent notes and preview output without writing:

```bash
python main.py \
  --vault ~/Documents/MyVault \
  --feeds ~/Documents/subscriptions.opml \
  --max-notes 5 \
  --dry-run
```

### Fetch entries from the last 14 days instead of 7

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

### Run against the mock vault (development / smoke test)

```bash
python main.py \
  --vault mock_vault \
  --feeds rss_feeds/feeds_2026-04-24.opml.xml \
  --max-notes 3 \
  --dry-run
```

## Output Format

Filtered entries are written to `<vault>/Filtered feed/RSS-YYYY-MM-DD.md`:

```markdown
# RSS Digest — 2026-04-24

## Reading

- [Some Article Title](https://example.com/article)
  > Directly relevant to your interest in X

## Arxiv monitoring

- [A Paper on Topic Y](https://arxiv.org/abs/1234.56789)
  > Matches your recurring interest in Y and Z
```

## Deduplication

Every entry processed in a non-dry-run is recorded in `seen_entries.json`. On the next run,
those entries are skipped regardless of their publication date. To reset deduplication state
(e.g. to reprocess all current entries), delete `seen_entries.json`:

```bash
rm seen_entries.json
```

## Running the Tests

```bash
pytest tests/
```

All tests use mocked LLM and HTTP calls and complete in under a second.
