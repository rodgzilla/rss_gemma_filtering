"""Embedding-based RSS entry filtering using few-shot exemplars and Gemma-4."""

from __future__ import annotations

import re

from openai import OpenAI
from tqdm import tqdm

from rss_filter.embedding_client import EmbeddingClient
from rss_filter.embedding_store import EmbeddingStore
from rss_filter.models import FilterResult, RSSEntry

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a research assistant helping to filter RSS articles based on a "
    "user's past reading habits. For each numbered article, decide whether it "
    "should be included in today's digest.\n"
    "Reply with one line per article using exactly this format:\n"
    "  <number>. yes: <brief reason>\n"
    "  <number>. no: <brief reason>\n"
    "Do not output anything else."
)


_EXEMPLAR_CHARS = 120  # max characters shown per exemplar text


def _format_exemplars(exemplars: list[dict]) -> str:
    """Format retrieved exemplars as a bullet list with similarity scores."""
    if not exemplars:
        return "  (no similar articles found in history)"
    lines = []
    for ex in exemplars:
        score = ex["score"]
        text = ex["text"].strip()
        # Use only the first line, capped at _EXEMPLAR_CHARS characters.
        first_line = text.splitlines()[0] if text else ex["url"]
        if len(first_line) > _EXEMPLAR_CHARS:
            first_line = first_line[:_EXEMPLAR_CHARS] + "…"
        lines.append(f'  - "{first_line}" (similarity: {score:.2f})')
    return "\n".join(lines)


def build_filter_prompt(
    entries_with_exemplars: list[dict],
) -> str:
    """Build a batched few-shot filtering prompt using article titles only.

    *entries_with_exemplars* is a list of dicts:
        {"entry": RSSEntry, "exemplars": list[{"text", "url", "score"}]}
    """
    parts: list[str] = []
    for idx, item in enumerate(entries_with_exemplars, start=1):
        entry: RSSEntry = item["entry"]
        exemplars: list[dict] = item["exemplars"]

        exemplar_block = _format_exemplars(exemplars)

        part = (
            f'{idx}. Article title: "{entry.title}"\n'
            f"   Most similar articles previously found relevant:\n"
            f"{exemplar_block}"
        )
        parts.append(part)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# Matches lines like "1. yes: some reason" or "2. no: another reason"
_RESPONSE_LINE_RE = re.compile(r"^\s*(\d+)\.\s*(yes|no)\s*:\s*(.*)$", re.IGNORECASE)


def parse_filter_response(
    response: str,
    entries: list[RSSEntry],
) -> list[FilterResult]:
    """Parse a batched LLM response into FilterResult objects.

    Lines that cannot be parsed default to keep=False.
    """
    decisions: dict[int, tuple[bool, str]] = {}
    for line in response.splitlines():
        m = _RESPONSE_LINE_RE.match(line)
        if m:
            number = int(m.group(1))
            keep = m.group(2).lower() == "yes"
            reason = m.group(3).strip()
            decisions[number] = (keep, reason)

    results: list[FilterResult] = []
    for idx, entry in enumerate(entries, start=1):
        if idx in decisions:
            keep, reason = decisions[idx]
        else:
            keep, reason = False, "no response from model"
        results.append(FilterResult(entry=entry, keep=keep, reason=reason, score=None))
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def filter_entries_batch(
    entries: list[RSSEntry],
    store: EmbeddingStore,
    embed_client: EmbeddingClient,
    llm_client: OpenAI,
    model: str,
    top_k: int = 3,
    batch_size: int = 20,
    temperature: float = 0.1,
) -> list[FilterResult]:
    """Filter RSS entries using embedding-based few-shot exemplars.

    For each entry:
    1. Embed ``title + " " + summary`` (summary used only for the embedding
       similarity search, not shown to the LLM).
    2. Retrieve the *top_k* most similar stored vault articles as exemplars.
    3. Build a batched prompt (titles + exemplar titles only) and call Gemma-4.
    4. Parse the response into FilterResult objects.

    Batching: *batch_size* articles are grouped into a single LLM API call.
    The prompt lists them numbered 1–N and the model replies with one
    ``yes/no: reason`` line per number. This reduces API round-trips from
    len(entries) down to ceil(len(entries) / batch_size). Title-only prompts
    are short, so a large batch_size (default 50) is safe within any normal
    context window.

    Input order is preserved in the returned list.
    """
    if not entries:
        return []

    # --- Step 1: embed all entries in one go ---
    texts = [f"{e.title} {e.summary or ''}".strip() for e in entries]
    print(f"Embedding {len(texts)} RSS entries...")
    embeddings = embed_client.embed_batch(texts)

    # --- Step 2: retrieve exemplars per entry ---
    entries_with_exemplars = [
        {"entry": entry, "exemplars": store.query(emb, top_k=top_k)}
        for entry, emb in zip(entries, embeddings)
    ]

    # --- Step 3 & 4: batch LLM calls ---
    all_results: list[FilterResult] = []
    batches = [
        entries_with_exemplars[i : i + batch_size]
        for i in range(0, len(entries_with_exemplars), batch_size)
    ]

    for batch in tqdm(batches, desc="Filtering batches"):
        batch_entries = [item["entry"] for item in batch]
        prompt = build_filter_prompt(batch)
        response = llm_client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        batch_results = parse_filter_response(raw, batch_entries)
        all_results.extend(batch_results)

    return all_results
