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
from rss_filter.relevance_filter import filter_entry
from rss_filter.rss_fetcher import (
    fetch_feed,
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
        "--rebuild-profile",
        action="store_true",
        help="Ignore cached interest profile and regenerate it from notes",
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
    entries = parse_vault(vault_daily)
    if args.max_notes is not None:
        entries = entries[: args.max_notes * 10]  # rough cap by entries, not files
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

    if not all_entries:
        print("No new entries. Nothing to do.")
        return

    # --- Step 3: Filter entries ---
    reading_results = []
    arxiv_results = []
    new_guids = set()

    for entry in tqdm(all_entries, desc="Filtering entries", unit="entry"):
        try:
            result = filter_entry(
                entry, profile, client, model=model, temperature=temperature
            )
        except Exception as e:
            tqdm.write(f"  [WARN] Failed to filter '{entry.title[:60]}': {e}")
            result = None

        new_guids.add(entry.guid)

        if result and result.keep:
            if entry.is_arxiv:
                arxiv_results.append(result)
            else:
                reading_results.append(result)

    kept = len(reading_results) + len(arxiv_results)
    print(f"  Kept {kept} / {len(all_entries)} entries.")

    # --- Step 4: Output ---
    today = date.today().isoformat()

    if args.dry_run:
        if reading_results:
            print("\n## Reading\n")
            for r in reading_results:
                print(f"- [{r.entry.title}]({r.entry.url})")
                print(f"  > {r.reason}")
        if arxiv_results:
            print("\n## Arxiv monitoring\n")
            for r in arxiv_results:
                print(f"- [{r.entry.title}]({r.entry.url})")
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
