"""Filter RSS entries for relevance using the LLM.

Two-pass filtering strategy
----------------------------
Pass 1 — title-only, large batches:
    The LLM sees only entry titles.  For each title it must reply with one of:
      yes  — clearly relevant
      no   — clearly irrelevant
      ?    — uncertain, needs more context

Pass 2 — title + summary snippet, smaller batches:
    Only entries marked '?' in pass 1 are re-evaluated.  Each entry now
    includes the first --summary-chars characters of its summary.  The LLM
    replies yes/no only (no more '?').

This keeps pass-1 prompts very short (many titles per call) while still giving
the model enough signal for borderline entries.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from tqdm import tqdm

from rss_filter.models import FilterResult, RSSEntry

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Single-entry legacy format: "yes: reason" or "no: reason"
_RESPONSE_RE = re.compile(r"^(yes|no)\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL)

# Pass-1 batch line: "1. yes", "2. no", "3. ?"  (reason optional)
_PASS1_LINE_RE = re.compile(r"^(\d+)[.)]\s*(yes|no|\?)\s*(?::\s*(.*))?$", re.IGNORECASE)

# Pass-2 batch line: "1. yes: reason" or "1. no: reason"
_PASS2_LINE_RE = re.compile(r"^(\d+)[.)]\s*(yes|no)\s*:\s*(.+)$", re.IGNORECASE)

# Default number of summary characters shown in pass 2
DEFAULT_SUMMARY_CHARS = 300


# ---------------------------------------------------------------------------
# Legacy single-entry helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------


def build_filter_prompt(entry: RSSEntry, interest_profile: str) -> str:
    """Build the per-entry relevance prompt (single-entry, legacy)."""
    return (
        f"You are a content filter. Below is a user's interest profile followed by a "
        f"new RSS entry. Reply with exactly one line in the format:\n"
        f"  yes: <one-line reason>\n"
        f"or\n"
        f"  no: <one-line reason>\n\n"
        f"--- Interest profile ---\n"
        f"{interest_profile}\n\n"
        f"--- RSS entry ---\n"
        f"Title: {entry.title}\n"
        f"Summary: {entry.summary}\n\n"
        f"Is this entry relevant to the user's interests? (yes/no + reason)"
    )


def parse_llm_response(response: str) -> FilterResult:
    """Parse a single-entry yes/no LLM response (legacy)."""
    match = _RESPONSE_RE.match(response.strip())
    if not match:
        return FilterResult(
            entry=None, keep=False, reason="parse error: unexpected response format"
        )  # type: ignore[arg-type]
    verdict = match.group(1).lower()
    reason = match.group(2).strip()
    keep = verdict == "yes"
    return FilterResult(entry=None, keep=keep, reason=reason)  # type: ignore[arg-type]


def filter_entry(
    entry: RSSEntry,
    interest_profile: str,
    client,
    model: str,
    temperature: float = 0.1,
) -> FilterResult:
    """Ask the LLM whether a single RSS entry is relevant (legacy, one call)."""
    prompt = build_filter_prompt(entry, interest_profile)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    raw = response.choices[0].message.content
    partial = parse_llm_response(raw)
    return FilterResult(entry=entry, keep=partial.keep, reason=partial.reason)


# ---------------------------------------------------------------------------
# Pass 1 — title-only batch
# ---------------------------------------------------------------------------


def build_pass1_prompt(entries: List[RSSEntry], interest_profile: str) -> str:
    """Prompt that shows only titles and asks for yes / no / ? per entry.

    '?' means the title alone is not enough to decide.
    """
    entries_text = "".join(
        f"{i}. {entry.title}\n" for i, entry in enumerate(entries, 1)
    )
    n = len(entries)
    return (
        f"You are a content filter. A user's interest profile is given below, "
        f"followed by {n} article titles.\n\n"
        f"For EACH title reply with ONE of:\n"
        f"  <n>. yes   — clearly relevant to the user's interests\n"
        f"  <n>. no    — clearly outside the user's interests\n"
        f"  <n>. ?     — title alone is ambiguous, more context needed\n\n"
        f"Be selective and conservative. Only mark an entry 'yes' if it is clearly "
        f"and directly relevant to a specific stated interest in the profile. "
        f"When in doubt, mark it 'no'.\n\n"
        f"Output ONLY the {n} numbered lines, nothing else.\n\n"
        f"--- Interest profile ---\n"
        f"{interest_profile}\n\n"
        f"--- Titles ---\n"
        f"{entries_text}"
        f"Evaluate all {n} titles:"
    )


def parse_pass1_response(
    response: str, entries: List[RSSEntry]
) -> Tuple[List[RSSEntry], List[RSSEntry], List[RSSEntry]]:
    """Parse a pass-1 response into (yes_entries, no_entries, unsure_entries).

    Missing or malformed lines are treated as '?' (unsure) so they get a
    second look rather than being silently dropped.
    """
    verdicts: dict[int, str] = {}
    for line in response.strip().splitlines():
        m = _PASS1_LINE_RE.match(line.strip())
        if m:
            idx = int(m.group(1))
            verdicts[idx] = m.group(2).lower()

    yes_entries, no_entries, unsure_entries = [], [], []
    for i, entry in enumerate(entries, 1):
        verdict = verdicts.get(i, "no")
        if verdict == "yes":
            yes_entries.append(entry)
        elif verdict == "no":
            no_entries.append(entry)
        else:
            unsure_entries.append(entry)

    return yes_entries, no_entries, unsure_entries


# ---------------------------------------------------------------------------
# Pass 2 — title + summary snippet batch
# ---------------------------------------------------------------------------


def build_pass2_prompt(
    entries: List[RSSEntry], interest_profile: str, summary_chars: int
) -> str:
    """Prompt that shows title + summary snippet and asks for yes / no per entry."""
    entries_text = ""
    for i, entry in enumerate(entries, 1):
        snippet = entry.summary[:summary_chars].strip()
        entries_text += f"{i}. Title: {entry.title}\n   Summary: {snippet}\n\n"
    n = len(entries)
    return (
        f"You are a content filter. A user's interest profile is given below, "
        f"followed by {n} articles (title + summary excerpt).\n\n"
        f"For EACH article reply with exactly one numbered line:\n"
        f"  <n>. yes: <one-line reason>\n"
        f"or\n"
        f"  <n>. no: <one-line reason>\n\n"
        f"Be selective and conservative. Only mark an entry 'yes' if it is clearly "
        f"and directly relevant to a specific stated interest in the profile. "
        f"When in doubt, mark it 'no'.\n\n"
        f"Output ONLY the {n} numbered lines, nothing else.\n\n"
        f"--- Interest profile ---\n"
        f"{interest_profile}\n\n"
        f"--- Articles ---\n"
        f"{entries_text}"
        f"Evaluate all {n} articles:"
    )


def parse_pass2_response(response: str, entries: List[RSSEntry]) -> List[FilterResult]:
    """Parse a pass-2 numbered yes/no response into FilterResult objects.

    Missing or malformed lines default to keep=False.
    """
    parsed: dict[int, tuple[bool, str]] = {}
    for line in response.strip().splitlines():
        m = _PASS2_LINE_RE.match(line.strip())
        if m:
            idx = int(m.group(1))
            keep = m.group(2).lower() == "yes"
            reason = m.group(3).strip()
            parsed[idx] = (keep, reason)

    results = []
    for i, entry in enumerate(entries, 1):
        if i in parsed:
            keep, reason = parsed[i]
        else:
            keep, reason = False, "parse error: no response for this entry"
        results.append(FilterResult(entry=entry, keep=keep, reason=reason))
    return results


# ---------------------------------------------------------------------------
# Legacy batch helpers (kept so existing tests still pass)
# ---------------------------------------------------------------------------


def build_batch_filter_prompt(entries: List[RSSEntry], interest_profile: str) -> str:
    """Legacy single-pass batch prompt (title + summary)."""
    entries_text = ""
    for i, entry in enumerate(entries, 1):
        entries_text += (
            f"{i}. Title: {entry.title}\n   Summary: {entry.summary[:300]}\n\n"
        )
    return (
        f"You are a content filter. Below is a user's interest profile followed by "
        f"{len(entries)} RSS entries.\n\n"
        f"For EACH entry reply with exactly one numbered line:\n"
        f"  <n>. yes: <one-line reason>\n"
        f"or\n"
        f"  <n>. no: <one-line reason>\n\n"
        f"Output ONLY the {len(entries)} numbered lines, nothing else.\n\n"
        f"--- Interest profile ---\n"
        f"{interest_profile}\n\n"
        f"--- Entries ---\n"
        f"{entries_text}"
        f"Evaluate all {len(entries)} entries:"
    )


def parse_batch_response(response: str, entries: List[RSSEntry]) -> List[FilterResult]:
    """Legacy single-pass batch response parser."""
    _BATCH_LINE_RE = re.compile(r"^(\d+)[.)]\s*(yes|no)\s*:\s*(.+)$", re.IGNORECASE)
    parsed: dict[int, tuple[bool, str]] = {}
    for line in response.strip().splitlines():
        m = _BATCH_LINE_RE.match(line.strip())
        if m:
            idx = int(m.group(1))
            keep = m.group(2).lower() == "yes"
            reason = m.group(3).strip()
            parsed[idx] = (keep, reason)

    results = []
    for i, entry in enumerate(entries, 1):
        if i in parsed:
            keep, reason = parsed[i]
        else:
            keep, reason = False, "parse error: no response for this entry"
        results.append(FilterResult(entry=entry, keep=keep, reason=reason))
    return results


# ---------------------------------------------------------------------------
# Two-pass batch filtering (main entry point)
# ---------------------------------------------------------------------------


def filter_entries_batch(
    entries: List[RSSEntry],
    interest_profile: str,
    client,
    model: str,
    temperature: float = 0.1,
    batch_size: int = 20,
    pass2_batch_size: int = 10,
    summary_chars: int = DEFAULT_SUMMARY_CHARS,
) -> List[FilterResult]:
    """Filter entries using a two-pass strategy.

    Pass 1 — title-only, large batches (batch_size, default 20):
        Each entry is labelled yes / no / ? by the LLM.
        'yes' entries are accepted immediately.
        'no'  entries are rejected immediately.
        '?'   entries proceed to pass 2.

    Pass 2 — title + summary snippet, smaller batches (pass2_batch_size, default 10):
        Only '?' entries from pass 1 are re-evaluated with additional context.
        The LLM replies yes/no only.

    Args:
        entries:          Candidate RSSEntry objects.
        interest_profile: Plain-text profile from the profiler.
        client:           OpenAI-compatible client.
        model:            Model identifier string.
        temperature:      Sampling temperature.
        batch_size:       Entries per pass-1 LLM call.
        pass2_batch_size: Entries per pass-2 LLM call.
        summary_chars:    Characters of summary shown in pass 2.

    Returns:
        List of FilterResult in the same order as the input entries.
    """
    # We need to preserve original order; track results by entry index.
    index_to_result: dict[int, FilterResult] = {}

    # --- Pass 1: title-only ---
    unsure_indices: list[int] = []

    pass1_batches = [
        entries[i : i + batch_size] for i in range(0, len(entries), batch_size)
    ]
    for batch_start, batch in tqdm(
        list(enumerate(pass1_batches)),
        desc="Pass 1 (titles)",
        unit="batch",
    ):
        prompt = build_pass1_prompt(batch, interest_profile)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        raw = response.choices[0].message.content
        yes_batch, _no_batch, unsure_batch = parse_pass1_response(raw, batch)

        yes_set = {id(e) for e in yes_batch}
        unsure_set = {id(e) for e in unsure_batch}

        global_start = batch_start * batch_size
        for local_i, entry in enumerate(batch):
            global_i = global_start + local_i
            if id(entry) in yes_set:
                index_to_result[global_i] = FilterResult(
                    entry=entry, keep=True, reason="relevant (title filter)"
                )
            elif id(entry) in unsure_set:
                unsure_indices.append(global_i)
            else:
                index_to_result[global_i] = FilterResult(
                    entry=entry, keep=False, reason="not relevant (title filter)"
                )

    # --- Pass 2: title + summary snippet for uncertain entries ---
    unsure_entries = [entries[i] for i in unsure_indices]

    if unsure_entries:
        pass2_batches = [
            unsure_entries[i : i + pass2_batch_size]
            for i in range(0, len(unsure_entries), pass2_batch_size)
        ]
        for batch_start, batch in tqdm(
            list(enumerate(pass2_batches)),
            desc="Pass 2 (summaries)",
            unit="batch",
        ):
            prompt = build_pass2_prompt(batch, interest_profile, summary_chars)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            raw = response.choices[0].message.content
            results = parse_pass2_response(raw, batch)

            for local_i, result in enumerate(results):
                global_i = unsure_indices[batch_start * pass2_batch_size + local_i]
                index_to_result[global_i] = result

    # Return results in original input order
    return [index_to_result[i] for i in range(len(entries))]
