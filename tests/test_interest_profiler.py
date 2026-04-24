"""Tests for rss_filter.interest_profiler"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rss_filter.models import NoteEntry
from rss_filter.interest_profiler import (
    build_merge_prompt,
    build_profile_prompt,
    chunk_entries,
    load_or_build_profile,
    save_profile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_entry(url: str, context: str, source: str = "reading") -> NoteEntry:
    return NoteEntry(url=url, context=context, source=source, date="2024-01-01")


SAMPLE_ENTRIES = [
    _make_entry(
        "https://example.com/a",
        "An article about machine learning and neural networks.",
        "reading",
    ),
    _make_entry(
        "https://arxiv.org/abs/1234",
        "A paper on self-supervised learning for vision.",
        "arxiv",
    ),
    _make_entry(
        "https://blog.com/b", "Post about board games and probability.", "reading"
    ),
]


# ---------------------------------------------------------------------------
# build_profile_prompt
# ---------------------------------------------------------------------------


def test_build_profile_prompt_contains_all_urls():
    prompt = build_profile_prompt(SAMPLE_ENTRIES)

    assert "https://example.com/a" in prompt
    assert "https://arxiv.org/abs/1234" in prompt
    assert "https://blog.com/b" in prompt


def test_build_profile_prompt_contains_context_text():
    prompt = build_profile_prompt(SAMPLE_ENTRIES)

    assert "machine learning" in prompt
    assert "board games" in prompt


def test_build_profile_prompt_is_non_empty_string():
    prompt = build_profile_prompt(SAMPLE_ENTRIES)

    assert isinstance(prompt, str)
    assert len(prompt) > 50


def test_build_profile_prompt_instructs_llm_to_summarize():
    prompt = build_profile_prompt(SAMPLE_ENTRIES)

    # Should contain some instruction language
    lower = prompt.lower()
    assert any(word in lower for word in ["interest", "topic", "theme", "summar"])


# ---------------------------------------------------------------------------
# save_profile / load_or_build_profile (cache path)
# ---------------------------------------------------------------------------


def test_save_profile_writes_file(tmp_path):
    profile_path = tmp_path / "interest_profile.md"
    save_profile("I like machine learning and board games.", profile_path)

    assert profile_path.exists()
    assert "machine learning" in profile_path.read_text()


def test_load_or_build_profile_returns_cached_when_exists(tmp_path, mocker):
    profile_path = tmp_path / "interest_profile.md"
    profile_path.write_text("Cached interest profile text.")

    mock_client = MagicMock()
    result = load_or_build_profile(
        entries=SAMPLE_ENTRIES,
        profile_path=profile_path,
        client=mock_client,
        model="gemma-4-e4b",
        rebuild=False,
    )

    assert result == "Cached interest profile text."
    mock_client.chat.completions.create.assert_not_called()


def test_load_or_build_profile_calls_llm_when_no_cache(tmp_path, mocker):
    profile_path = tmp_path / "interest_profile.md"

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Generated interest profile."
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    result = load_or_build_profile(
        entries=SAMPLE_ENTRIES,
        profile_path=profile_path,
        client=mock_client,
        model="gemma-4-e4b",
        rebuild=False,
    )

    assert result == "Generated interest profile."
    mock_client.chat.completions.create.assert_called_once()


def test_load_or_build_profile_saves_generated_profile(tmp_path, mocker):
    profile_path = tmp_path / "interest_profile.md"

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Newly generated profile."
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    load_or_build_profile(
        entries=SAMPLE_ENTRIES,
        profile_path=profile_path,
        client=mock_client,
        model="gemma-4-e4b",
        rebuild=False,
    )

    assert profile_path.exists()
    assert "Newly generated profile." in profile_path.read_text()


def test_load_or_build_profile_rebuilds_when_rebuild_true(tmp_path, mocker):
    profile_path = tmp_path / "interest_profile.md"
    profile_path.write_text("Old cached profile.")

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Fresh profile."
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    result = load_or_build_profile(
        entries=SAMPLE_ENTRIES,
        profile_path=profile_path,
        client=mock_client,
        model="gemma-4-e4b",
        rebuild=True,
    )

    assert result == "Fresh profile."
    mock_client.chat.completions.create.assert_called_once()


# ---------------------------------------------------------------------------
# chunk_entries
# ---------------------------------------------------------------------------


def test_chunk_entries_splits_into_correct_number_of_chunks():
    entries = [_make_entry(f"https://x.com/{i}", f"Context {i}") for i in range(10)]
    chunks = chunk_entries(entries, max_chars=500)
    # All entries must appear across chunks
    all_urls = [e.url for chunk in chunks for e in chunk]
    assert sorted(all_urls) == sorted(e.url for e in entries)


def test_chunk_entries_each_chunk_within_char_limit():
    entries = [_make_entry(f"https://x.com/{i}", "x" * 100) for i in range(20)]
    chunks = chunk_entries(entries, max_chars=500)
    for chunk in chunks:
        total = sum(len(e.url) + len(e.context) for e in chunk)
        assert total <= 500 or len(chunk) == 1  # single oversized entry allowed


def test_chunk_entries_single_entry_always_in_a_chunk():
    entries = [_make_entry("https://x.com/only", "Only entry.")]
    chunks = chunk_entries(entries, max_chars=10)  # smaller than entry
    assert len(chunks) == 1
    assert chunks[0][0].url == "https://x.com/only"


def test_chunk_entries_empty_input_returns_empty():
    assert chunk_entries([], max_chars=1000) == []


# ---------------------------------------------------------------------------
# build_merge_prompt
# ---------------------------------------------------------------------------


def test_build_merge_prompt_contains_all_partial_profiles():
    partials = ["Profile part A about ML.", "Profile part B about games."]
    prompt = build_merge_prompt(partials)

    assert "Profile part A about ML." in prompt
    assert "Profile part B about games." in prompt


def test_build_merge_prompt_instructs_to_merge():
    partials = ["Part A.", "Part B."]
    prompt = build_merge_prompt(partials)

    lower = prompt.lower()
    assert any(word in lower for word in ["merge", "combine", "consolidat", "unified"])


# ---------------------------------------------------------------------------
# load_or_build_profile with batching
# ---------------------------------------------------------------------------


def test_load_or_build_profile_makes_multiple_llm_calls_for_large_input(tmp_path):
    """When entries exceed max_chars_per_batch, multiple LLM calls are made."""
    profile_path = tmp_path / "interest_profile.md"

    # 10 entries each with 200 chars of context → total ~2000 chars
    # Set max_chars_per_batch=400 to force ~5 batches + 1 merge call
    large_entries = [_make_entry(f"https://x.com/{i}", "a" * 200) for i in range(10)]

    call_count = 0

    def fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        mock_response = MagicMock()
        mock_response.choices[0].message.content = f"Partial profile {call_count}."
        return mock_response

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = fake_create

    load_or_build_profile(
        entries=large_entries,
        profile_path=profile_path,
        client=mock_client,
        model="test-model",
        rebuild=True,
        max_chars_per_batch=400,
    )

    # Should have been called more than once (batches + merge)
    assert mock_client.chat.completions.create.call_count > 1


def test_load_or_build_profile_single_batch_makes_one_llm_call(tmp_path):
    """When all entries fit in one batch, exactly one LLM call is made."""
    profile_path = tmp_path / "interest_profile.md"

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Single batch profile."
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    load_or_build_profile(
        entries=SAMPLE_ENTRIES,
        profile_path=profile_path,
        client=mock_client,
        model="test-model",
        rebuild=True,
        max_chars_per_batch=100_000,
    )

    mock_client.chat.completions.create.assert_called_once()
