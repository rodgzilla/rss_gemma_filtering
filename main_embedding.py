"""RSS Gemma Filtering — Embedding-based pipeline CLI entry point."""

from __future__ import annotations

import argparse
import sys
import tomllib
from datetime import date
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from rss_filter.embedding_client import EmbeddingClient
from rss_filter.embedding_filter import filter_entries_batch
from rss_filter.embedding_store import EmbeddingStore
from rss_filter.note_writer import write_note
from rss_filter.notes_parser import parse_vault
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
        description=(
            "Filter RSS feeds using embedding-based few-shot exemplars "
            "and a local LLM (Gemma-4 via LM Studio)."
        )
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
        help="Path to config.toml (default: config.toml next to main_embedding.py)",
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
        help=(
            "Only evaluate entries published within this many days (default: 7). "
            "Entries with no publication date are always kept."
        ),
    )
    parser.add_argument(
        "--rebuild-embeddings",
        action="store_true",
        help="Force a full rebuild of the embedding database from vault notes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        metavar="N",
        help="Number of entries per LLM filtering call (default: 10)",
    )
    parser.add_argument(
        "--summary-chars",
        type=int,
        default=300,
        metavar="N",
        help=(
            "Maximum characters of article summary included in the prompt "
            "(default: 300). Reduce if hitting context length errors."
        ),
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
        "--no-seen-filter",
        action="store_true",
        help="Skip deduplication against seen_entries.json (useful for testing)",
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
    emb_cfg = cfg.get("embedding", {})

    # LLM client (Gemma-4 for filtering and re-ranking)
    llm_client = OpenAI(base_url=lm_cfg["base_url"], api_key="lm-studio")
    llm_model = lm_cfg["model"]
    temperature = lm_cfg.get("temperature", 0.1)

    # Embedding client (small local embedding model)
    emb_base_url = emb_cfg.get("base_url", lm_cfg["base_url"])
    emb_model = emb_cfg.get("model", "text-embedding-embeddinggemma-300m-qat")
    embed_client = EmbeddingClient(base_url=emb_base_url, model=emb_model)

    store_path = emb_cfg.get("store_path", "embedding_store.db")
    top_k = emb_cfg.get("top_k", 3)

    seen_path = Path(path_cfg["seen_entries"])

    # --- Step 1: Build / update embedding database from vault ---
    vault_daily = args.vault / vault_cfg["daily_notes_folder"]
    print(f"Reading notes from: {vault_daily}")
    note_entries = parse_vault(vault_daily, max_notes=args.max_notes)
    print(f"  Found {len(note_entries)} saved entries across all daily notes.")

    print(f"  Building / updating embedding store at: {store_path}")
    store = EmbeddingStore(store_path)
    store.build_or_update(
        note_entries,
        embed_client,
        force_rebuild=args.rebuild_embeddings,
    )
    print(f"  Embedding store contains {store.count()} documents.")

    # --- Step 2: Fetch RSS entries ---
    print(f"\nParsing OPML: {args.feeds}")
    feeds = parse_opml(args.feeds)
    print(f"  Found {len(feeds)} subscribed feeds.")

    seen_guids = load_seen_guids(seen_path) if not args.no_seen_filter else set()
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
            f"  Dropped {dropped} entries older than {args.max_age_days} days "
            f"→ {len(all_entries)} remaining."
        )

    if not all_entries:
        print("No new entries. Nothing to do.")
        store.close()
        return

    # --- Step 3: Embedding filter ---
    new_guids = {e.guid for e in all_entries}

    print(
        f"  Filtering {len(all_entries)} entries with embedding few-shot approach "
        f"(batch size: {args.batch_size}, top_k exemplars: {top_k}, "
        f"summary chars: {args.summary_chars})…"
    )

    try:
        all_results = filter_entries_batch(
            all_entries,
            store=store,
            embed_client=embed_client,
            llm_client=llm_client,
            model=llm_model,
            top_k=top_k,
            batch_size=args.batch_size,
            temperature=temperature,
            summary_chars=args.summary_chars,
        )
    except Exception as e:
        tqdm.write(f"  [ERROR] Filtering failed: {e}")
        all_results = []

    reading_results = []
    arxiv_results = []
    for result in all_results:
        if result.keep:
            if result.entry.is_arxiv:
                arxiv_results.append(result)
            else:
                reading_results.append(result)

    kept = len(reading_results) + len(arxiv_results)
    print(f"  Kept {kept} / {len(all_entries)} entries.")

    # --- Step 4: Re-rank (optional) ---
    if args.rerank and kept > 0:
        # Re-ranking uses the interest profile; we pass a placeholder since reranker
        # expects a profile string. Build a minimal one from exemplar texts if needed.
        # For now we pass an empty string — the reranker prompt is self-contained.
        profile_placeholder = ""
        print(
            f"  Re-ranking {len(reading_results)} reading + "
            f"{len(arxiv_results)} arxiv entries…"
        )
        if reading_results:
            reading_results = rerank(
                reading_results,
                profile_placeholder,
                llm_client,
                model=llm_model,
                temperature=temperature,
                batch_size=args.rerank_batch_size,
            )
        if arxiv_results:
            arxiv_results = rerank(
                arxiv_results,
                profile_placeholder,
                llm_client,
                model=llm_model,
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

    # --- Persist seen GUIDs ---
    if not args.dry_run and not args.no_seen_filter:
        save_seen_guids(new_guids, seen_path)
        print(f"Saved {len(new_guids)} new GUIDs to {seen_path}.")

    store.close()


if __name__ == "__main__":
    main()
