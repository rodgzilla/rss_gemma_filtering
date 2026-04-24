# Filtering Pipeline

This document describes how RSS entries are processed from raw feed data to the
final Obsidian digest note.

---

## Overview

```
RSS feeds (OPML)
      │
      ▼
fetch_feed × N feeds
      │  all new entries
      ▼
filter_by_age          ← drop entries older than N days (default: 7)
      │  recent entries
      ▼
prefilter_entries      ← keyword overlap vs. interest profile (no LLM)
      │  entries with at least 1 keyword match
      ▼
filter_entries_batch
  ├─ Pass 1 (titles only, batch 20) → yes / no / ?
  └─ Pass 2 (title + summary, batch 10) → yes / no  [? entries only]
      │  keep=True results only
      ▼
rerank (optional)      ← LLM scores each kept entry 1–10, sorts descending
      │  same entries, reordered by relevance score
      ▼
write_note             ← Obsidian markdown digest
      +
save_seen_guids        ← persist evaluated GUIDs to seen_entries.json
```

The **interest profile** used in the pre-filter and LLM steps is built once from
your Obsidian daily notes and cached to `interest_profile.md`. It is not rebuilt
on every run unless you pass `--rebuild-profile`.

---

## Step 1 — Age filter

**Function:** `rss_fetcher.filter_by_age`
**CLI flag:** `--max-age-days N` (default: `7`)

| | |
|---|---|
| **Input** | All new RSS entries fetched from your feeds (those not already in `seen_entries.json`) |
| **Output** | Entries published within the last N days; entries with no publication date are always kept |

Pure date comparison — no LLM involved. Drops anything older than the configured
window before any further processing.

---

## Step 2 — Keyword pre-filter

**Function:** `prefilter.prefilter_entries`
**CLI flags:** `--prefilter-keywords N` (default: `60`), `--prefilter-min-score N` (default: `1`), `--no-prefilter`

| | |
|---|---|
| **Input** | Age-filtered entries + interest profile text |
| **Output** | Two lists: `passed` (forwarded to the LLM) and `rejected` (silently dropped) |

How it works:

1. `extract_keywords` tokenises the interest profile, removes common stop-words,
   and returns the top N most frequent content words (e.g. `["rust", "transformer",
   "reinforcement", "llm", ...]`).
2. Each entry's title + summary is tokenised the same way.
3. Entries sharing fewer than `--prefilter-min-score` keywords with that list are
   rejected without any LLM call.

This step runs in milliseconds for thousands of entries and is the primary driver
of speed. How many entries survive depends on how broad your keyword list is
relative to your feeds. Disable it with `--no-prefilter` if you want the LLM to
see every entry.

---

## Step 3 — Two-pass batch LLM filtering

**Function:** `relevance_filter.filter_entries_batch`
**CLI flags:** `--batch-size N` (default: `20`), `--pass2-batch-size N` (default: `10`), `--summary-chars N` (default: `300`)

| | |
|---|---|
| **Input** | Entries that passed the keyword pre-filter + interest profile text |
| **Output** | A `FilterResult` per entry: `keep=True/False` + a one-line reason string |

Filtering is split into two passes to keep prompts short while still giving the
model enough signal for borderline entries.

### Pass 1 — title-only (large batches)

1. Entries are grouped into batches of `--batch-size` (default 20).
2. Each batch becomes one LLM call. The prompt contains:
   - The full interest profile (once per batch)
   - Up to N numbered entry **titles only** — no summaries
3. The LLM must reply with exactly one of three verdicts per line:
   ```
   1. yes   — clearly relevant
   2. no    — clearly irrelevant
   3. ?     — title alone is ambiguous
   ```
4. `yes` entries are accepted immediately.
   `no` entries are rejected immediately.
   `?` entries (and any missing/malformed lines) proceed to pass 2.

### Pass 2 — title + summary snippet (smaller batches)

1. Only entries marked `?` in pass 1 are re-evaluated.
2. Batches of `--pass2-batch-size` (default 10) entries per call.
3. Each entry now includes its title plus the first `--summary-chars` characters
   of its summary.
4. The LLM replies `yes: reason` or `no: reason` only — no `?` allowed.
5. Missing or malformed lines default to `keep=False`.

Results are split into `reading_results` and `arxiv_results` based on each
entry's `is_arxiv` flag. Output order matches input order.

---

## Step 4 — Re-ranking (optional)

**Function:** `reranker.rerank`
**CLI flags:** `--rerank` (enable), `--rerank-batch-size N` (default: `20`)

| | |
|---|---|
| **Input** | Kept `FilterResult` objects (reading and arxiv separately) + interest profile |
| **Output** | Same entries sorted descending by relevance score (most relevant first) |

How it works:

1. Kept entries are grouped into batches of `--rerank-batch-size`.
2. Each batch becomes one LLM call. The prompt contains the interest profile and
   each entry's title + first 200 characters of summary.
3. The LLM assigns a score from **1** (weakly relevant) to **10** (highly
   relevant) per entry:
   ```
   1. 8
   2. 3
   3. 10
   ```
4. Scores are normalised to `[0.0, 1.0]` and stored in `FilterResult.score`.
5. Entries with missing or unparseable scores default to `0.5`.
6. The list is sorted descending — the most relevant item appears first in the
   Obsidian note.

Re-ranking is applied separately to reading and arxiv sections so the two lists
remain distinct. Scores are shown inline in the note as `*(score: 0.8)*`.

This step is **opt-in** (`--rerank` flag). Skip it if speed is the priority.

---

## Step 5 — Output

**Function:** `note_writer.write_note`

| | |
|---|---|
| **Input** | `reading_results` and `arxiv_results` (only entries with `keep=True`) |
| **Output** | `<vault>/Filtered feed/RSS-YYYY-MM-DD.md` |

The note is structured as two sections:

```markdown
# RSS Digest — YYYY-MM-DD

## Reading

- [Title](url) *(score: 0.9)*
  > LLM one-line reason

## Arxiv monitoring

- [Title](url) *(score: 0.7)*
  > LLM one-line reason
```

When re-ranking is disabled (default), the `*(score: X.X)*` inline tag is omitted.

In `--dry-run` mode the note is printed to stdout instead and nothing is written
to disk.

---

## Step 6 — Deduplication state

**Function:** `rss_fetcher.save_seen_guids`

| | |
|---|---|
| **Input** | GUIDs of all entries that survived the age filter (whether kept or not) |
| **Output** | Updated `seen_entries.json` |

Every entry that was evaluated is recorded as seen so it is never sent to the LLM
again on future runs, even if it was rejected. Skipped in `--dry-run` mode so the
same entries remain available on the next real run.

---

## Interest profile construction

The interest profile is an input to both the keyword pre-filter and the LLM
filtering step. It is built separately from your Obsidian daily notes using a
map-reduce approach:

**Map phase** — `interest_profiler.chunk_entries` + `build_profile_prompt`
- Note entries are split into batches capped at 8000 characters each
- Each batch gets one LLM call that produces a short 2–3 sentence partial summary

**Reduce phase** — `interest_profiler._recursive_merge`
- Partial summaries are merged in groups of up to 5
- The process repeats until a single unified profile remains

The final profile is saved to `interest_profile.md` and reused on every
subsequent run. Rebuild it with `--rebuild-profile` after adding a significant
number of new daily notes.

---

## Performance characteristics

| Stage | Cost | Typical reduction |
|---|---|---|
| Age filter | Instant (date comparison) | Removes entries older than N days |
| Keyword pre-filter | Instant (string matching) | ~70–90% of entries dropped |
| Batch LLM filtering | ~25 s per batch of 10 | Only runs on pre-filter survivors |
| Note writing | Instant (file I/O) | — |

Expected end-to-end time for ~1200 entries per day with default settings:
**5–15 minutes** (vs. 8–9 hours with single-entry LLM calls and no pre-filter).
