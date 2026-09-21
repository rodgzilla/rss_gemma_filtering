"""Embedding-only filtering: exponential decay score aggregation + quantile threshold."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from rss_filter.embedding_client import EmbeddingClient
from rss_filter.embedding_store import EmbeddingStore
from rss_filter.models import FilterResult, RSSEntry
from rss_filter.text_prep import clean_summary, format_for_embedding

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
    top_quantile_arxiv: float | None = None,
    prompt_style: str = "none",
) -> tuple[list[FilterResult], list[dict], float, float]:
    """Score RSS entries using embedding similarity and select the top quantile.

    Two separate quantile thresholds are applied: one for general reading entries
    and one for arXiv entries (``top_quantile_arxiv``).  If ``top_quantile_arxiv``
    is *None* it falls back to the same value as ``top_quantile``.

    Pipeline:
    1. Embed each entry's title and plain-text summary, formatted with
       ``prompt_style`` exactly like the vault entries in the store.
    2. For each entry retrieve the *top_k* most similar vault documents.
    3. Aggregate the K similarity scores with exponential decay weighting.
    4. Compute separate score distributions for reading vs arXiv entries and
       find their respective thresholds at the ``(1 - top_quantile*)`` percentile.
    5. Mark entries at or above their category threshold as kept.

    Returns:
        results:               list[FilterResult] — one per entry, keep/score/reason set
        entry_metadata:        list[dict] — per-entry data for the visualisation:
                               {"embedding": np.ndarray (full-dim),
                                "embedding_128": np.ndarray (Matryoshka 128-dim),
                                "exemplars": [{"text": str, "score": float}],
                                "agg_score": float,
                                "is_arxiv": bool}
        threshold_reading:     float — the score threshold used for reading entries
        threshold_arxiv:       float — the score threshold used for arXiv entries
    """
    if top_quantile_arxiv is None:
        top_quantile_arxiv = top_quantile

    if not entries:
        return [], [], 0.0, 0.0

    # --- Step 1: embed all entries ---
    texts = [
        format_for_embedding(e.title, clean_summary(e.summary), prompt_style)
        for e in entries
    ]
    _SCORE_CHUNK = 64
    embeddings: list[np.ndarray] = []
    chunks = [texts[i : i + _SCORE_CHUNK] for i in range(0, len(texts), _SCORE_CHUNK)]
    with tqdm(total=len(texts), desc="Embedding RSS entries", unit="entry") as pbar:
        for chunk in chunks:
            embeddings.extend(embed_client.embed_batch(chunk))
            pbar.update(len(chunk))

    # --- Step 2 & 3: retrieve exemplars and compute aggregated scores ---
    agg_scores: list[float] = []
    entry_metadata: list[dict] = []

    for entry, emb in tqdm(
        zip(entries, embeddings),
        desc="Scoring entries",
        unit="entry",
        total=len(entries),
    ):
        exemplars = store.query(emb, top_k=top_k)
        sims = [ex["score"] for ex in exemplars]
        agg = _exponential_decay_score(sims, decay_lambda)
        agg_scores.append(agg)
        entry_metadata.append(
            {
                "embedding": emb,
                "embedding_128": emb[:MATRYOSHKA_DIM],
                "exemplars": [
                    {
                        "text": ex["text"],
                        "score": ex["score"],
                        "title": ex.get("title", ""),
                    }
                    for ex in exemplars
                ],
                "agg_score": agg,
                "is_arxiv": entry.is_arxiv,
            }
        )

    # --- Step 4: per-category quantile thresholds ---
    reading_scores = [s for e, s in zip(entries, agg_scores) if not e.is_arxiv]
    arxiv_scores = [s for e, s in zip(entries, agg_scores) if e.is_arxiv]

    def _threshold(scores: list[float], quantile: float) -> float:
        arr = np.array(scores)
        if len(arr) == 0:
            return 0.0
        if len(arr) == 1:
            return float(arr[0])
        return float(np.percentile(arr, (1.0 - quantile) * 100))

    threshold_reading = _threshold(reading_scores, top_quantile)
    threshold_arxiv = _threshold(arxiv_scores, top_quantile_arxiv)

    # --- Step 5: build FilterResult list ---
    results: list[FilterResult] = []
    for entry, agg, meta in zip(entries, agg_scores, entry_metadata):
        if entry.is_arxiv:
            thr = threshold_arxiv
        else:
            thr = threshold_reading
        keep = agg >= thr
        reason = f"embedding score {agg:.4f} (threshold {thr:.4f})"
        results.append(
            FilterResult(
                entry=entry,
                keep=keep,
                reason=reason,
                score=agg,
                exemplars=meta["exemplars"],
            )
        )

    kept_reading = sum(1 for r in results if r.keep and not r.entry.is_arxiv)
    kept_arxiv = sum(1 for r in results if r.keep and r.entry.is_arxiv)
    n_reading = len(reading_scores)
    n_arxiv = len(arxiv_scores)
    print(
        f"  Reading threshold  (top {top_quantile * 100:.0f}%): {threshold_reading:.4f}"
        f" — kept {kept_reading} / {n_reading} entries."
    )
    print(
        f"  Arxiv threshold    (top {top_quantile_arxiv * 100:.0f}%): {threshold_arxiv:.4f}"
        f" — kept {kept_arxiv} / {n_arxiv} entries."
    )

    return results, entry_metadata, threshold_reading, threshold_arxiv
