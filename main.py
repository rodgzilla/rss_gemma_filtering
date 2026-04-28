"""RSS Embedding Filter — CLI entry point."""

from __future__ import annotations

import argparse
import tomllib
from datetime import date
from pathlib import Path

from tqdm import tqdm

from rss_filter.embedding_client import EmbeddingClient
from rss_filter.embedding_store import EmbeddingStore
from rss_filter.note_writer import write_note
from rss_filter.notes_parser import parse_vault
from rss_filter.rss_fetcher import (
    deduplicate_entries,
    fetch_feed,
    filter_by_age,
    filter_new_entries,
    load_seen_guids,
    parse_opml,
    save_seen_guids,
)
from rss_filter.score_filter import score_entries
from rss_filter.score_viz import build_or_load_umap, write_score_viz


def load_config(config_path: Path) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter RSS feeds by embedding similarity against your Obsidian vault. "
            "Each entry is scored against previously saved articles; the top-scoring "
            "entries are written to a daily digest note."
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
        "--top-quantile",
        type=float,
        default=None,
        metavar="Q",
        help=(
            "Fraction of top-scoring *reading* entries to keep "
            "(default: from config, fallback 0.25). E.g. 0.25 keeps the top 25%%."
        ),
    )
    parser.add_argument(
        "--top-quantile-arxiv",
        type=float,
        default=None,
        metavar="Q",
        help=(
            "Fraction of top-scoring *arXiv* entries to keep "
            "(default: from config key top_quantile_arxiv, fallback same as "
            "--top-quantile). E.g. 0.50 keeps the top 50%%."
        ),
    )
    parser.add_argument(
        "--decay-lambda",
        type=float,
        default=None,
        metavar="L",
        help=(
            "Exponential decay parameter for score aggregation "
            "(default: from config, fallback 1.0). Higher values give more "
            "weight to the single best exemplar match."
        ),
    )
    parser.add_argument(
        "--rebuild-umap",
        action="store_true",
        help="Force refit of the UMAP model on vault embeddings",
    )
    parser.add_argument(
        "--vault-bg-max",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum number of vault background points shown in the UMAP panel "
            "(default: from config, fallback 500)."
        ),
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
    path_cfg = cfg["paths"]
    vault_cfg = cfg["vault"]
    emb_cfg = cfg.get("embedding", {})

    # Embedding client
    emb_base_url = emb_cfg.get("base_url", "http://localhost:1234/v1")
    emb_model = emb_cfg.get("model", "text-embedding-embeddinggemma-300m-qat")
    embed_client = EmbeddingClient(base_url=emb_base_url, model=emb_model)

    store_path = emb_cfg.get("store_path", "embedding_store.db")
    top_k = emb_cfg.get("top_k", 3)

    # Scoring parameters (CLI overrides config)
    top_quantile = args.top_quantile or emb_cfg.get("top_quantile", 0.25)
    top_quantile_arxiv = args.top_quantile_arxiv or emb_cfg.get(
        "top_quantile_arxiv", None
    )
    decay_lambda = args.decay_lambda or emb_cfg.get("decay_lambda", 1.0)
    umap_model_path = emb_cfg.get("umap_model_path", "umap_model.joblib")
    umap_growth_threshold = emb_cfg.get("umap_growth_threshold", 0.1)
    vault_bg_max = args.vault_bg_max or emb_cfg.get("vault_bg_max", 500)

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

    # Deduplicate cross-posted entries (e.g. same arXiv paper in multiple categories)
    before_dedup = len(all_entries)
    all_entries = deduplicate_entries(all_entries)
    removed = before_dedup - len(all_entries)
    if removed:
        print(f"  Removed {removed} duplicate(s); {len(all_entries)} entries remain.")

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

    new_guids = {e.guid for e in all_entries}

    # --- Step 3: Score entries by embedding similarity ---
    print(
        f"\n  Scoring {len(all_entries)} entries "
        f"(top_quantile={top_quantile}, "
        f"top_quantile_arxiv={top_quantile_arxiv or top_quantile}, "
        f"decay_lambda={decay_lambda}, top_k={top_k})…"
    )
    all_results, entry_metadata, threshold_reading, threshold_arxiv = score_entries(
        all_entries,
        store=store,
        embed_client=embed_client,
        top_k=top_k,
        decay_lambda=decay_lambda,
        top_quantile=top_quantile,
        top_quantile_arxiv=top_quantile_arxiv,
    )

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

    # --- Step 4: Output ---
    today = date.today().isoformat()

    if args.dry_run:
        if reading_results:
            print("\n## Reading\n")
            for r in reading_results:
                score_str = f"  [score: {r.score:.4f}]" if r.score is not None else ""
                print(f"- [{r.entry.title}]({r.entry.url}){score_str}")
                for ex in r.exemplars:
                    print(f"  - {ex['score']:.4f}  {ex['text']}")
        if arxiv_results:
            print("\n## Arxiv monitoring\n")
            for r in arxiv_results:
                score_str = f"  [score: {r.score:.4f}]" if r.score is not None else ""
                print(f"- [{r.entry.title}]({r.entry.url}){score_str}")
                for ex in r.exemplars:
                    print(f"  - {ex['score']:.4f}  {ex['text']}")
    else:
        viz_filename = f"RSS-{today}-scores.html"
        write_note(
            args.vault,
            reading_results,
            arxiv_results,
            date=today,
            viz_filename=viz_filename,
        )
        if kept > 0:
            output = args.vault / vault_cfg["output_folder"] / f"RSS-{today}.md"
            print(f"\nNote written to: {output}")
        else:
            print("\nNo relevant entries found. No note written.")

    # --- Step 5: Score visualisation ---
    if entry_metadata and not args.dry_run:
        print("\n  Building score visualisation…")
        try:
            umap_reducer, vault_2d, vault_docs = build_or_load_umap(
                store=store,
                model_path=umap_model_path,
                force_rebuild=args.rebuild_umap or args.rebuild_embeddings,
                growth_threshold=umap_growth_threshold,
            )
            viz_path = (
                args.vault / vault_cfg["output_folder"] / f"RSS-{today}-scores.html"
            )
            write_score_viz(
                results=all_results,
                entry_metadata=entry_metadata,
                vault_2d=vault_2d,
                vault_docs=vault_docs,
                umap_reducer=umap_reducer,
                threshold_reading=threshold_reading,
                threshold_arxiv=threshold_arxiv,
                output_path=viz_path,
                vault_bg_max=vault_bg_max,
            )
        except Exception as e:
            tqdm.write(f"  [WARN] Visualisation failed: {e}")

    # --- Persist seen GUIDs ---
    if not args.dry_run and not args.no_seen_filter:
        save_seen_guids(new_guids, seen_path)
        print(f"Saved {len(new_guids)} new GUIDs to {seen_path}.")

    store.close()


if __name__ == "__main__":
    main()
