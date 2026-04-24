# RSS Gemma Filtering — Implementation Plan

## Overview

A CLI tool that:
1. Reads Obsidian daily notes to build an interest profile using a local LLM
2. Fetches new entries from RSS feeds (imported via OPML)
3. Filters entries using the LLM (yes/no + one-line reason)
4. Writes a filtered digest note into the Obsidian vault

The local LLM is accessed via LM Studio's OpenAI-compatible API (`http://localhost:1234/v1`).
The recommended model is `gemma-4-e4b`.

---

## Project Structure

```
rss_gemma_filtering/
├── rss_filter/
│   ├── __init__.py
│   ├── models.py             # Shared dataclasses
│   ├── notes_parser.py       # Parse Obsidian daily notes → NoteEntry list
│   ├── interest_profiler.py  # LLM call → interest profile text
│   ├── rss_fetcher.py        # Parse OPML, fetch feeds, deduplicate
│   ├── relevance_filter.py   # LLM yes/no + reason per RSS entry
│   └── note_writer.py        # Write filtered results to vault
├── tests/
│   ├── test_notes_parser.py
│   ├── test_interest_profiler.py
│   ├── test_rss_fetcher.py
│   ├── test_relevance_filter.py
│   └── test_note_writer.py
├── mock_vault/                # Sample Obsidian vault used in tests
│   └── Daily notes/
├── main.py                   # CLI entry point
├── config.toml               # User configuration
├── seen_entries.json         # Persistent state (created on first run)
├── interest_profile.md       # Cached interest profile (rebuilt with --rebuild-profile)
├── requirements.txt
├── PLAN.md                   # This file
└── README.md
```

---

## Shared Data Models (`rss_filter/models.py`)

```python
@dataclass
class NoteEntry:
    url: str
    context: str        # Title + surrounding text from the note
    source: str         # "reading" | "arxiv"
    date: str           # YYYY-MM-DD

@dataclass
class RSSEntry:
    title: str
    url: str
    summary: str
    feed_name: str
    is_arxiv: bool
    guid: str           # Unique identifier for deduplication

@dataclass
class FilterResult:
    entry: RSSEntry
    keep: bool
    reason: str
```

---

## Module Descriptions

### `notes_parser.py`

Scans `<vault>/Daily notes/YYYY-MM-DD.md` files and extracts entries from two sections:

- `## Reading` — general web links
- `## [[Arxiv]] monitoring` — research paper links

Each entry is a markdown list item in the format:
```
- [Title](url) -> [[tag1]], [[tag2]]
    Context paragraph...
```

Returns a list of `NoteEntry` objects.

**TDD cycles:**
1. Parse a note with a `Reading` section → returns correct `NoteEntry` list with `source="reading"`
2. Parse a note with an `Arxiv monitoring` section → returns `NoteEntry` list with `source="arxiv"`
3. Parse a note with both sections → returns entries from both
4. Parse a note with neither section → returns empty list
5. Parse a note with only YAML frontmatter → returns empty list
6. Scan a full folder of notes → aggregates entries across all files, tagged with their date

---

### `rss_fetcher.py`

Parses an OPML subscription file, fetches each feed, and returns only new entries.

Deduplication is based on entry GUIDs (falling back to URL) stored in `seen_entries.json`.
Arxiv entries are identified by checking if the feed URL or entry URL contains `arxiv.org`.

**TDD cycles:**
1. Parse OPML → list of `(title, url)` tuples
2. Fetch a mocked feed → list of `RSSEntry` objects
3. Filter out entries whose GUIDs are already in the seen set
4. Classify entry as arxiv vs general from feed/entry URL
5. Load and save `seen_entries.json` correctly
6. Mark fetched GUIDs as seen after a run

---

### `note_writer.py`

Writes the filtered digest to `<vault>/Filtered feed/RSS-YYYY-MM-DD.md`.
Creates the output folder if it does not exist.

Output note format:
```markdown
# RSS Digest — YYYY-MM-DD

## Reading

- [Title](url)
  > LLM reason

## Arxiv monitoring

- [Title](url)
  > LLM reason
```

**TDD cycles:**
1. Generate correct output file path from vault path and date
2. Render the `Reading` section from a list of `FilterResult` objects
3. Render the `Arxiv monitoring` section from a list of `FilterResult` objects
4. Combine both sections into a full note string
5. Write the note to disk (creates parent directory if needed)
6. Skip writing if both filtered lists are empty

---

### `interest_profiler.py`

Builds a prompt from all `NoteEntry` objects and sends it to the LLM.
The LLM returns a plain-text interest profile summarising recurring themes.

The profile is cached in `interest_profile.md`. On subsequent runs it is reused unless
`--rebuild-profile` is passed.

**TDD cycles (LLM mocked with `pytest-mock`):**
1. Build the prompt string correctly from a list of `NoteEntry` objects
2. Parse LLM response into a plain-text profile string
3. Load existing profile from disk when cache file exists
4. Save newly generated profile to disk
5. Force rebuild when `rebuild=True` even if cache exists

---

### `relevance_filter.py`

For each `RSSEntry`, sends a short prompt to the LLM with the interest profile and the
entry's title + summary. Expects a response in the form:

```
yes: <one-line reason>
```
or
```
no: <one-line reason>
```

Returns a `FilterResult`. Defaults to `keep=False` on malformed responses.

**TDD cycles (LLM mocked):**
1. Build the per-entry prompt correctly
2. Parse a `"yes: ..."` response → `FilterResult(keep=True, reason=...)`
3. Parse a `"no: ..."` response → `FilterResult(keep=False, reason=...)`
4. Handle malformed / empty LLM response → `FilterResult(keep=False, reason="parse error")`

---

### `main.py`

CLI entry point using `argparse`.

```
python main.py --vault PATH --feeds subscriptions.opml [--rebuild-profile] [--dry-run]
```

| Flag | Required | Description |
|---|---|---|
| `--vault` | yes | Path to the Obsidian vault |
| `--feeds` | yes | Path to the OPML subscriptions file |
| `--rebuild-profile` | no | Ignore cached profile and regenerate it |
| `--dry-run` | no | Print filtered entries to stdout instead of writing a note |

Execution order:
1. Load or build interest profile
2. Fetch new RSS entries
3. Filter entries via LLM
4. Write output note (or print if `--dry-run`)
5. Persist newly seen GUIDs

---

## Configuration (`config.toml`)

```toml
[lmstudio]
base_url = "http://localhost:1234/v1"
model = "gemma-4-e4b"
temperature = 0.1

[paths]
seen_entries = "seen_entries.json"
interest_profile = "interest_profile.md"

[vault]
daily_notes_folder = "Daily notes"
output_folder = "Filtered feed"
```

---

## Dependencies (`requirements.txt`)

```
openai          # LM Studio OpenAI-compatible client
feedparser      # RSS / Atom feed parsing
listparser      # OPML parsing
pytest          # Test runner
pytest-mock     # Mock LLM and HTTP calls in tests
```

---

## Development Workflow

All modules are developed using **Red / Green / Refactor TDD**:

1. Write a failing test (RED)
2. Write minimal code to pass the test (GREEN)
3. Refactor while keeping tests green

Implementation order (dependency-first):
1. `models.py` — no deps
2. `notes_parser.py` — pure file I/O
3. `rss_fetcher.py` — OPML + HTTP (mocked)
4. `note_writer.py` — pure file I/O
5. `interest_profiler.py` — LLM (mocked)
6. `relevance_filter.py` — LLM (mocked)
7. `main.py` — integration
