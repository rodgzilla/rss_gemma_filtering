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
filter_entries_batch   ← LLM yes/no, 10 entries per call
      │  keep=True results only
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

## Step 3 — Batch LLM filtering

**Function:** `relevance_filter.filter_entries_batch`
**CLI flag:** `--batch-size N` (default: `10`)

| | |
|---|---|
| **Input** | Entries that passed the keyword pre-filter + interest profile text |
| **Output** | A `FilterResult` per entry: `keep=True/False` + a one-line reason string |

How it works:

1. Entries are grouped into batches of `--batch-size`.
2. Each batch becomes a single LLM call. The prompt contains:
   - The full interest profile (once per batch, not once per entry)
   - Up to N numbered entries, each with title + first 300 characters of summary
3. The LLM is instructed to reply with exactly one numbered line per entry:
   ```
   1. yes: matches your interest in transformers
   2. no: sports news unrelated to tracked topics
   3. yes: relevant to your LLM tooling interest
   ```
4. The response is parsed line by line. Any missing or malformed line defaults
   to `keep=False`.
5. Results are split into `reading_results` and `arxiv_results` based on each
   entry's `is_arxiv` flag.

Sending N entries per call instead of one reduces LLM calls by a factor of N,
while keeping the interest profile token cost amortised across the batch.

---

## Step 4 — Output

**Function:** `note_writer.write_note`

| | |
|---|---|
| **Input** | `reading_results` and `arxiv_results` (only entries with `keep=True`) |
| **Output** | `<vault>/Filtered feed/RSS-YYYY-MM-DD.md` |

The note is structured as two sections:

```markdown
# RSS Digest — YYYY-MM-DD

## Reading

- [Title](url)
  > LLM one-line reason

## Arxiv monitoring

- [Title](url)
  > LLM one-line reason
```

In `--dry-run` mode the note is printed to stdout instead and nothing is written
to disk.

---

## Step 5 — Deduplication state

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
