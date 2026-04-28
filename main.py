"""RSS Gemma Filtering — CLI entry point."""

from __future__ import annotations

import argparse
import sys
import tomllib
from datetime import date
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from rss_filter.interest_profiler import load_or_build_profile
from rss_filter.note_writer import write_note
from rss_filter.notes_parser import parse_vault
from rss_filter.prefilter import prefilter_entries
from rss_filter.relevance_filter import filter_entries_batch
from rss_filter.reranker import rerank
from rss_filter.rss_fetcher import (
    fetch_feed,
    filter_by_age,
    filter_new_entries,
    load_seen_guids,
    parse_opml,
    save_seen_guids,
)


def load_config(config_path: Path) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Filter RSS feeds using a local LLM based on your Obsidian notes."
    )
    parser.add_argument(
        "--vault", required=True, type=Path, help="Path to Obsidian vault"
    )
    parser.add_argument(
        "--feeds", required=True, type=Path, help="Path to OPML subscriptions file"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.toml",
        help="Path to config.toml (default: config.toml next to main.py)",
    )
    parser.add_argument(
        "--max-notes",
        type=int,
        default=None,
        help="Limit the number of daily notes parsed (useful for quick test runs)",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=7,
        help="Only evaluate entries published within this many days (default: 7). "
        "Entries with no publication date are always kept.",
    )
    parser.add_argument(
        "--rebuild-profile",
        action="store_true",
        help="Ignore cached interest profile and regenerate it from notes",
    )
    parser.add_argument(
        "--no-prefilter",
        action="store_true",
        help="Disable keyword pre-filter and send all entries directly to the LLM",
    )
    parser.add_argument(
        "--prefilter-keywords",
        type=int,
        default=30,
        metavar="N",
        help="Number of top keywords to extract from the profile for pre-filtering (default: 30)",
    )
    parser.add_argument(
        "--prefilter-min-score",
        type=int,
        default=2,
        metavar="N",
        help="Minimum keyword matches for an entry to pass the pre-filter (default: 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        metavar="N",
        help="Number of entries per pass-1 (title-only) LLM call (default: 20)",
    )
    parser.add_argument(
        "--pass2-batch-size",
        type=int,
        default=10,
        metavar="N",
        help="Number of entries per pass-2 (title+summary) LLM call (default: 10)",
    )
    parser.add_argument(
        "--summary-chars",
        type=int,
        default=300,
        metavar="N",
        help="Characters of summary shown to the LLM in pass 2 (default: 300)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Re-rank kept entries by relevance score before writing the note",
    )
    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=20,
        metavar="N",
        help="Number of entries per re-ranking LLM call (default: 20)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print filtered entries to stdout instead of writing a note",
    )
    args = parser.parse_args(argv)

    # --- Load config ---
    cfg = load_config(args.config)
    lm_cfg = cfg["lmstudio"]
    path_cfg = cfg["paths"]
    vault_cfg = cfg["vault"]

    client = OpenAI(base_url=lm_cfg["base_url"], api_key="lm-studio")
    model = lm_cfg["model"]
    temperature = lm_cfg.get("temperature", 0.1)

    profile_path = Path(path_cfg["interest_profile"])
    seen_path = Path(path_cfg["seen_entries"])

    # --- Step 1: Build interest profile ---
    vault_daily = args.vault / vault_cfg["daily_notes_folder"]
    print(f"Reading notes from: {vault_daily}")
    entries = parse_vault(vault_daily, max_notes=args.max_notes)
    print(f"  Found {len(entries)} saved entries across all daily notes.")

    if not args.rebuild_profile and profile_path.exists():
        print(f"  Loading cached interest profile from {profile_path}.")
    else:
        print("  Building interest profile (this may take a while)...")

    profile = load_or_build_profile(
        entries=entries,
        profile_path=profile_path,
        client=client,
        model=model,
        rebuild=args.rebuild_profile,
    )
    print("  Interest profile ready.")

    # --- Step 2: Fetch RSS entries ---
    print(f"\nParsing OPML: {args.feeds}")
    feeds = parse_opml(args.feeds)
    print(f"  Found {len(feeds)} subscribed feeds.")

    seen_guids = load_seen_guids(seen_path)
    all_entries = []

    for feed_title, feed_url in tqdm(feeds, desc="Fetching feeds", unit="feed"):
        try:
            fetched = fetch_feed(feed_url, feed_title)
            new = filter_new_entries(fetched, seen_guids)
            all_entries.extend(new)
        except Exception as e:
            tqdm.write(f"  [WARN] Failed to fetch '{feed_title}': {e}")

    print(f"  {len(all_entries)} new entries to evaluate.")

    # Drop entries older than --max-age-days
    before = len(all_entries)
    all_entries = filter_by_age(all_entries, max_age_days=args.max_age_days)
    dropped = before - len(all_entries)
    if dropped:
        print(
            f"  Dropped {dropped} entries older than {args.max_age_days} days → {len(all_entries)} remaining."
        )

    if not all_entries:
        print("No new entries. Nothing to do.")
        return

    # --- Step 3: Pre-filter + LLM filter ---
    reading_results = []
    arxiv_results = []
    new_guids = {e.guid for e in all_entries}

    llm_candidates = all_entries
    if not args.no_prefilter:
        llm_candidates, rejected = prefilter_entries(
            all_entries,
            profile,
            top_n_keywords=args.prefilter_keywords,
            min_score=args.prefilter_min_score,
        )
        print(
            f"  Pre-filter: {len(llm_candidates)} entries passed keyword check, "
            f"{len(rejected)} dropped."
        )

    print(
        f"  Sending {len(llm_candidates)} entries to LLM "
        f"(pass-1 batch {args.batch_size}, pass-2 batch {args.pass2_batch_size})…"
    )

    try:
        all_results = filter_entries_batch(
            llm_candidates,
            profile,
            client,
            model=model,
            temperature=temperature,
            batch_size=args.batch_size,
            pass2_batch_size=args.pass2_batch_size,
            summary_chars=args.summary_chars,
        )
    except Exception as e:
        tqdm.write(f"  [ERROR] Filtering failed: {e}")
        all_results = []

    for result in all_results:
        if result.keep:
            if result.entry.is_arxiv:
                arxiv_results.append(result)
            else:
                reading_results.append(result)

    kept = len(reading_results) + len(arxiv_results)
    print(
        f"  Kept {kept} / {len(all_entries)} entries ({len(llm_candidates)} evaluated by LLM)."
    )

    # --- Step 4: Re-rank (optional) ---
    if args.rerank and kept > 0:
        print(
            f"  Re-ranking {len(reading_results)} reading + {len(arxiv_results)} arxiv entries…"
        )
        if reading_results:
            reading_results = rerank(
                reading_results,
                profile,
                client,
                model=model,
                temperature=temperature,
                batch_size=args.rerank_batch_size,
            )
        if arxiv_results:
            arxiv_results = rerank(
                arxiv_results,
                profile,
                client,
                model=model,
                temperature=temperature,
                batch_size=args.rerank_batch_size,
            )

    # --- Step 5: Output ---
    today = date.today().isoformat()

    if args.dry_run:
        if reading_results:
            print("\n## Reading\n")
            for r in reading_results:
                score_str = f"  [score: {r.score:.1f}]" if r.score is not None else ""
                print(f"- [{r.entry.title}]({r.entry.url}){score_str}")
                print(f"  > {r.reason}")
        if arxiv_results:
            print("\n## Arxiv monitoring\n")
            for r in arxiv_results:
                score_str = f"  [score: {r.score:.1f}]" if r.score is not None else ""
                print(f"- [{r.entry.title}]({r.entry.url}){score_str}")
                print(f"  > {r.reason}")
    else:
        write_note(args.vault, reading_results, arxiv_results, date=today)
        if kept > 0:
            output = args.vault / vault_cfg["output_folder"] / f"RSS-{today}.md"
            print(f"\nNote written to: {output}")
        else:
            print("\nNo relevant entries found. No note written.")

    # --- Step 5: Persist seen GUIDs ---
    if not args.dry_run:
        save_seen_guids(new_guids, seen_path)
        print(f"Saved {len(new_guids)} new GUIDs to {seen_path}.")


if __name__ == "__main__":
    main()
