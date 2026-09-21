"""Offline evaluation of the embedding prompt style.

Holds out the most recent reading entries of the vault as positives, uses the
current (non-arXiv) feed entries that are not in the vault as negatives, and for
each prompt style reports how well the embedding score separates them:
ROC-AUC and the precision of the top 20 % of items.

Usage:
    python scripts/eval_prompt_style.py --vault ~/vault --feeds feeds.opml \
        --base-url http://localhost:8080/v1
"""

from __future__ import annotations

import argparse
import math
import sys
import tomllib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rss_filter.models import NoteEntry  # noqa: E402

HOLDOUT_FRACTION = 0.1
TOP_FRACTION = 0.2
STYLES = ("none", "document")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def holdout_split(
    entries: list[NoteEntry], fraction: float = HOLDOUT_FRACTION
) -> tuple[list[NoteEntry], list[NoteEntry]]:
    """Split vault entries into (store, holdout).

    The holdout is the most recent *fraction* of reading entries (at least one when
    there are any). The store is every other entry, minus any sharing a URL with the
    holdout, so a positive cannot match itself.
    """
    reading = sorted(
        (e for e in entries if e.source == "reading"),
        key=lambda e: e.date or "",
        reverse=True,
    )
    if not reading:
        return list(entries), []
    n_holdout = max(1, math.ceil(len(reading) * fraction))
    holdout = reading[:n_holdout]
    holdout_ids = {id(e) for e in holdout}
    holdout_urls = {e.url for e in holdout}
    store = [
        e for e in entries if id(e) not in holdout_ids and e.url not in holdout_urls
    ]
    return store, holdout


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks of *values*, ties sharing their average rank."""
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def roc_auc(pos_scores: list[float], neg_scores: list[float]) -> float:
    """ROC-AUC of positives vs negatives (Mann-Whitney U via rank sums)."""
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(np.asarray(list(pos_scores) + list(neg_scores), dtype=float))
    rank_sum_pos = ranks[:n_pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def precision_at_fraction(
    pos_scores: list[float], neg_scores: list[float], fraction: float = TOP_FRACTION
) -> float:
    """Share of positives among the top *fraction* of all items by score."""
    scores = np.asarray(list(pos_scores) + list(neg_scores), dtype=float)
    if len(scores) == 0:
        return float("nan")
    labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    k = max(1, math.ceil(len(scores) * fraction))
    top = np.argsort(-scores, kind="mergesort")[:k]
    return float(labels[top].mean())


# ---------------------------------------------------------------------------
# Evaluation (needs the embedding server)
# ---------------------------------------------------------------------------


def _score_texts(texts, store, client, top_k, decay_lambda) -> list[float]:
    from rss_filter.score_filter import _exponential_decay_score

    if not texts:
        return []
    embeddings = client.embed_batch(texts)
    return [
        _exponential_decay_score(
            [ex["score"] for ex in store.query(emb, top_k=top_k)], decay_lambda
        )
        for emb in embeddings
    ]


def evaluate_style(
    style, store_entries, positives, negatives, client, top_k, decay_lambda
) -> dict:
    from rss_filter.embedding_store import EmbeddingStore, _embed_text_for_entry
    from rss_filter.text_prep import clean_summary, format_for_embedding

    with EmbeddingStore(":memory:") as store:
        store.build_or_update(store_entries, client, prompt_style=style)
        pos_texts = [_embed_text_for_entry(e, style) for e in positives]
        neg_texts = [
            format_for_embedding(e.title, clean_summary(e.summary), style)
            for e in negatives
        ]
        pos = _score_texts(pos_texts, store, client, top_k, decay_lambda)
        neg = _score_texts(neg_texts, store, client, top_k, decay_lambda)
    return {
        "style": style,
        "auc": roc_auc(pos, neg),
        "p_at": precision_at_fraction(pos, neg, TOP_FRACTION),
        "pos_mean": float(np.mean(pos)) if pos else float("nan"),
        "neg_mean": float(np.mean(neg)) if neg else float("nan"),
    }


def _fetch_negatives(feeds_path: Path, vault_urls: set[str], timeout: float):
    from rss_filter.rss_fetcher import deduplicate_entries, fetch_feed, parse_opml

    entries = []
    for name, url in parse_opml(feeds_path):
        try:
            entries.extend(fetch_feed(url, name, timeout=timeout))
        except Exception as e:  # noqa: BLE001 - a dead feed must not stop the eval
            print(f"  [WARN] {name}: {e}", file=sys.stderr)
    entries = deduplicate_entries(entries)
    return [e for e in entries if not e.is_arxiv and e.url not in vault_urls]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vault", type=Path, required=True, help="Obsidian vault root")
    parser.add_argument("--feeds", type=Path, required=True, help="OPML feed list")
    parser.add_argument("--base-url", default=None, help="Embedding server /v1 URL")
    parser.add_argument("--model", default=None, help="Embedding model name")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "rss_filter" / "config.toml",
    )
    args = parser.parse_args(argv)

    from rss_filter.embedding_client import EmbeddingClient
    from rss_filter.notes_parser import parse_vault

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    emb_cfg = cfg.get("embedding", {})
    base_url = args.base_url or emb_cfg.get("base_url", "http://localhost:1234/v1")
    model = args.model or emb_cfg.get("model", "text-embedding-embeddinggemma-300m-qat")
    top_k = emb_cfg.get("top_k", 3)
    decay_lambda = emb_cfg.get("decay_lambda", 1.0)
    timeout = cfg.get("fetch", {}).get("timeout_seconds", 15)
    daily = args.vault / cfg.get("vault", {}).get("daily_notes_folder", "Daily notes")

    vault_entries = parse_vault(daily)
    store_entries, positives = holdout_split(vault_entries)
    negatives = _fetch_negatives(args.feeds, {e.url for e in vault_entries}, timeout)
    print(
        f"store={len(store_entries)}  positives={len(positives)}  "
        f"negatives={len(negatives)}  top_k={top_k}  decay_lambda={decay_lambda}"
    )

    client = EmbeddingClient(base_url=base_url, model=model)
    rows = [
        evaluate_style(s, store_entries, positives, negatives, client, top_k, decay_lambda)
        for s in STYLES
    ]

    print(f"\n{'style':<10} {'ROC-AUC':>8} {'P@20%':>8} {'pos mean':>9} {'neg mean':>9}")
    for r in rows:
        print(
            f"{r['style']:<10} {r['auc']:>8.4f} {r['p_at']:>8.4f} "
            f"{r['pos_mean']:>9.4f} {r['neg_mean']:>9.4f}"
        )
    base_rate = len(positives) / max(1, len(positives) + len(negatives))
    print(f"\n(P@20% of a random ranking = positive base rate = {base_rate:.4f})")


if __name__ == "__main__":
    main()
