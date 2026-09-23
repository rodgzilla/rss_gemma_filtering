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
    load_feeds,
    save_seen_guids,
)
from rss_filter.score_filter import score_entries
from rss_filter.score_viz import build_or_load_umap, write_score_viz
from rss_filter.text_prep import PREP_VERSION


def load_config(config_path: Path) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _in_state(state_dir: Path, p: str) -> str:
    """Resolve a config path against the state directory unless it is absolute."""
    path = Path(p).expanduser()
    return str(path if path.is_absolute() else state_dir / path)


def _override(cli_value, cfg_value):
    """CLI value wins when given (even if falsy, e.g. 0.0); else the config value."""
    return cli_value if cli_value is not None else cfg_value


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
        "--feeds",
        required=True,
        type=Path,
        help=(
            "Path to the subscription list: an RSS Dashboard data.json "
            "(typically <vault>/.rss-dashboard-data/data.json) or an OPML export"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.toml",
        help="Path to config.toml (default: the one shipped in the rss_filter package)",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path.cwd(),
        help=(
            "Directory for mutable state (embedding store, seen entries, UMAP "
            "models); relative config paths are resolved against it "
            "(default: current directory)"
        ),
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
        "--feed-timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Per-feed network timeout in seconds; unresponsive feeds are "
            "skipped instead of stalling the fetch loop "
            "(default: from config key fetch.timeout_seconds, fallback 15)."
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

    state_dir = args.state_dir.expanduser()
    state_dir.mkdir(parents=True, exist_ok=True)

    store_path = _in_state(state_dir, emb_cfg.get("store_path", "embedding_store.db"))
    top_k = emb_cfg.get("top_k", 3)
    prompt_style = emb_cfg.get("prompt_style", "none")

    # Scoring parameters (CLI overrides config)
    top_quantile = _override(args.top_quantile, emb_cfg.get("top_quantile", 0.25))
    top_quantile_arxiv = _override(
        args.top_quantile_arxiv, emb_cfg.get("top_quantile_arxiv", None)
    )
    decay_lambda = _override(args.decay_lambda, emb_cfg.get("decay_lambda", 1.0))
    umap_model_path = _in_state(
        state_dir, emb_cfg.get("umap_model_path", "umap_model.joblib")
    )
    arxiv_umap_model_path = _in_state(
        state_dir, emb_cfg.get("arxiv_umap_model_path", "umap_arxiv_model.joblib")
    )
    umap_growth_threshold = emb_cfg.get("umap_growth_threshold", 0.1)
    vault_bg_max = _override(args.vault_bg_max, emb_cfg.get("vault_bg_max", 500))

    fetch_cfg = cfg.get("fetch", {})
    feed_timeout = _override(args.feed_timeout, fetch_cfg.get("timeout_seconds", 15))
    exclude_folders = cfg.get("feeds", {}).get("exclude_folders", [])

    seen_path = Path(_in_state(state_dir, path_cfg["seen_entries"]))

    # --- Step 1: Build / update embedding database from vault ---
    vault_daily = args.vault / vault_cfg["daily_notes_folder"]
    print(f"Reading notes from: {vault_daily}")
    note_entries = parse_vault(vault_daily, max_notes=args.max_notes)
    print(f"  Found {len(note_entries)} saved entries across all daily notes.")

    print(f"  Building / updating embedding store at: {store_path}")
    store = EmbeddingStore(store_path)
    embedding_signature = f"{emb_model}|prompt={prompt_style}|prep={PREP_VERSION}"
    rebuilt = store.build_or_update(
        note_entries,
        embed_client,
        force_rebuild=args.rebuild_embeddings,
        signature=embedding_signature,
        prompt_style=prompt_style,
    )
    print(f"  Embedding store contains {store.count()} documents.")

    # --- Step 2: Fetch RSS entries ---
    print(f"\nReading subscriptions: {args.feeds}")
    feeds = load_feeds(args.feeds, exclude_folders=exclude_folders)
    print(f"  Found {len(feeds)} subscribed feeds.")
    if exclude_folders:
        print(f"  Excluded folders: {', '.join(exclude_folders)}")

    seen_guids = load_seen_guids(seen_path) if not args.no_seen_filter else set()
    all_entries = []

    for feed_title, feed_url in tqdm(feeds, desc="Fetching feeds", unit="feed"):
        try:
            fetched = fetch_feed(feed_url, feed_title, timeout=feed_timeout)
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
        f"top_quantile_arxiv={_override(top_quantile_arxiv, top_quantile)}, "
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
        prompt_style=prompt_style,
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
                force_rebuild=args.rebuild_umap or args.rebuild_embeddings or rebuilt,
                growth_threshold=umap_growth_threshold,
            )
            arxiv_umap_reducer, arxiv_vault_2d, arxiv_vault_docs = build_or_load_umap(
                store=store,
                model_path=arxiv_umap_model_path,
                force_rebuild=args.rebuild_umap or args.rebuild_embeddings or rebuilt,
                growth_threshold=umap_growth_threshold,
                source_filter="arxiv",
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
                arxiv_vault_2d=arxiv_vault_2d,
                arxiv_vault_docs=arxiv_vault_docs,
                arxiv_umap_reducer=arxiv_umap_reducer,
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
