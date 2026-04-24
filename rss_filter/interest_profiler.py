"""Build and cache an interest profile from Obsidian daily notes."""

from __future__ import annotations

from pathlib import Path
from typing import List

from tqdm import tqdm

from rss_filter.models import NoteEntry

# Default max characters of entry content per LLM batch.
# ~8000 chars ≈ ~2000 tokens of entry content, well within the 16384-token
# context window (prompt wrapper ~200 tokens, reply ~500 tokens reserved).
DEFAULT_MAX_CHARS_PER_BATCH = 8000


def chunk_entries(entries: List[NoteEntry], max_chars: int) -> List[List[NoteEntry]]:
    """Split entries into batches where each batch's total content <= max_chars.

    A single entry that exceeds max_chars is placed alone in its own chunk.
    """
    if not entries:
        return []

    chunks: List[List[NoteEntry]] = []
    current_chunk: List[NoteEntry] = []
    current_size = 0

    for entry in entries:
        entry_size = len(entry.url) + len(entry.context)
        if current_chunk and current_size + entry_size > max_chars:
            chunks.append(current_chunk)
            current_chunk = [entry]
            current_size = entry_size
        else:
            current_chunk.append(entry)
            current_size += entry_size

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def build_profile_prompt(entries: List[NoteEntry]) -> str:
    """Build the LLM prompt used to generate a SHORT interest summary from a batch.

    The model is explicitly asked for 2-3 sentences so that many partial
    summaries can later be merged within the context window.
    """
    lines = [
        "Below is a small collection of articles and research papers that a person "
        "has saved as interesting. Each item includes a URL and its context.",
        "",
        "In 2-3 sentences, summarise the topics and themes that seem most interesting "
        "to this person based on these items. Be concise — this summary will be merged "
        "with others later.",
        "",
        "--- Saved items ---",
        "",
    ]
    for entry in entries:
        lines.append(f"Source: {entry.source}  |  Date: {entry.date}")
        lines.append(f"URL: {entry.url}")
        lines.append(f"Context: {entry.context}")
        lines.append("")
    lines.append("--- End of items ---")
    lines.append("")
    lines.append("Summary (2-3 sentences):")
    return "\n".join(lines)


def build_merge_prompt(partial_profiles: List[str]) -> str:
    """Build a prompt that merges multiple partial interest summaries into one."""
    lines = [
        "Below are several short interest summaries for the same person, each "
        "generated from a different subset of their saved articles.",
        "",
        "Please consolidate and merge them into a single, unified interest profile "
        "(3-5 paragraphs). Remove redundancy, combine related topics, and produce "
        "a coherent summary of this person's interests.",
        "",
        "--- Partial summaries ---",
        "",
    ]
    for i, profile in enumerate(partial_profiles, start=1):
        lines.append(f"[{i}] {profile.strip()}")
        lines.append("")
    lines.append("--- End of partial summaries ---")
    lines.append("")
    lines.append("Merged interest profile:")
    return "\n".join(lines)


def _call_llm(client, model: str, prompt: str, temperature: float = 0.1) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


def save_profile(profile_text: str, profile_path: Path) -> None:
    """Save the profile text to disk."""
    profile_path.write_text(profile_text, encoding="utf-8")


def _recursive_merge(
    partials: List[str],
    client,
    model: str,
    max_per_merge: int = 5,
) -> str:
    """Merge a list of partial profiles down to one via repeated grouped merges."""
    while len(partials) > 1:
        next_round: List[str] = []
        groups = [
            partials[i : i + max_per_merge]
            for i in range(0, len(partials), max_per_merge)
        ]
        desc = f"Merging {len(partials)} → {len(groups)} profiles"
        for group in tqdm(groups, desc=desc, unit="merge", leave=False):
            if len(group) == 1:
                next_round.append(group[0])
            else:
                merge_prompt = build_merge_prompt(group)
                merged = _call_llm(client, model, merge_prompt)
                next_round.append(merged)
        partials = next_round
    return partials[0]


def load_or_build_profile(
    entries: List[NoteEntry],
    profile_path: Path,
    client,
    model: str,
    rebuild: bool = False,
    max_chars_per_batch: int = DEFAULT_MAX_CHARS_PER_BATCH,
) -> str:
    """Return the interest profile, loading from cache or calling the LLM.

    If entries are too large for a single context window, splits them into
    batches, generates a short partial summary per batch, then merges them
    recursively in groups until a single profile remains.
    """
    if not rebuild and profile_path.exists():
        return profile_path.read_text(encoding="utf-8")

    chunks = chunk_entries(entries, max_chars=max_chars_per_batch)

    if len(chunks) <= 1:
        prompt = build_profile_prompt(entries)
        profile_text = _call_llm(client, model, prompt)
    else:
        # Step 1: generate a short partial summary for each chunk
        partial_profiles: List[str] = []
        for chunk in tqdm(
            chunks, desc="Building interest profile (batches)", unit="batch"
        ):
            prompt = build_profile_prompt(chunk)
            partial = _call_llm(client, model, prompt)
            partial_profiles.append(partial)

        # Step 2: merge all partials recursively in groups
        profile_text = _recursive_merge(partial_profiles, client, model)

    save_profile(profile_text, profile_path)
    return profile_text
