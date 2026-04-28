"""Embedding-only filtering: exponential decay score aggregation + quantile threshold."""

from __future__ import annotations

import numpy as np

from rss_filter.embedding_client import EmbeddingClient
from rss_filter.embedding_store import EmbeddingStore
from rss_filter.models import FilterResult, RSSEntry

# Number of Matryoshka dimensions used for embedding (also used in score_viz for UMAP).
MATRYOSHKA_DIM = 128


def _exponential_decay_score(similarities: list[float], decay_lambda: float) -> float:
    """Aggregate a ranked list of cosine similarities using exponential decay weighting.

    weights_i = exp(-lambda * i)  for i = 0..K-1  (i=0 is the best match)
    score = sum(s_i * w_i) / sum(w_i)

    At lambda=0 this reduces to a plain mean.
    At lambda=2 the top match contributes ~86% of the total weight (top-3).
    """
    if not similarities:
        return 0.0
    weights = np.array(
        [np.exp(-decay_lambda * i) for i in range(len(similarities))],
        dtype=np.float64,
    )
    sims = np.array(similarities, dtype=np.float64)
    return float(np.dot(sims, weights) / weights.sum())


def score_entries(
    entries: list[RSSEntry],
    store: EmbeddingStore,
    embed_client: EmbeddingClient,
    top_k: int = 3,
    decay_lambda: float = 1.0,
    top_quantile: float = 0.25,
) -> tuple[list[FilterResult], list[dict], float]:
    """Score RSS entries using embedding similarity and select the top quantile.

    Pipeline:
    1. Embed each entry as ``title + " " + summary`` (summary aids the embedding
       similarity search but is never shown to any LLM).
    2. For each entry retrieve the *top_k* most similar vault documents.
    3. Aggregate the K similarity scores with exponential decay weighting.
    4. Compute the score distribution over all entries and find the threshold
       at the ``(1 - top_quantile)`` percentile.
    5. Mark entries at or above the threshold as kept.

    Returns:
        results:          list[FilterResult] — one per entry, keep/score/reason set
        entry_metadata:   list[dict] — per-entry data for the visualisation:
                          {"embedding": np.ndarray (full-dim),
                           "embedding_128": np.ndarray (Matryoshka 128-dim),
                           "exemplars": [{"text": str, "score": float}],
                           "agg_score": float}
        threshold:        float — the score threshold used
    """
    if not entries:
        return [], [], 0.0

    # --- Step 1: embed all entries ---
    texts = [f"{e.title} {e.summary or ''}".strip() for e in entries]
    print(f"Embedding {len(texts)} RSS entries...")
    embeddings = embed_client.embed_batch(texts)

    # --- Step 2 & 3: retrieve exemplars and compute aggregated scores ---
    agg_scores: list[float] = []
    entry_metadata: list[dict] = []

    for emb in embeddings:
        exemplars = store.query(emb, top_k=top_k)
        sims = [ex["score"] for ex in exemplars]
        agg = _exponential_decay_score(sims, decay_lambda)
        agg_scores.append(agg)
        entry_metadata.append(
            {
                "embedding": emb,
                "embedding_128": emb[:MATRYOSHKA_DIM],
                "exemplars": [
                    {"text": ex["text"], "score": ex["score"]} for ex in exemplars
                ],
                "agg_score": agg,
            }
        )

    # --- Step 4: quantile threshold ---
    scores_arr = np.array(agg_scores)
    if len(scores_arr) == 1:
        # Single entry: always keep it.
        threshold = float(scores_arr[0])
    else:
        threshold = float(np.percentile(scores_arr, (1.0 - top_quantile) * 100))

    # --- Step 5: build FilterResult list ---
    results: list[FilterResult] = []
    for entry, agg, meta in zip(entries, agg_scores, entry_metadata):
        keep = agg >= threshold
        reason = f"embedding score {agg:.4f} (threshold {threshold:.4f})"
        results.append(FilterResult(entry=entry, keep=keep, reason=reason, score=agg))

    kept = sum(1 for r in results if r.keep)
    print(
        f"  Score threshold (top {top_quantile * 100:.0f}%): {threshold:.4f} — "
        f"kept {kept} / {len(entries)} entries."
    )

    return results, entry_metadata, threshold
