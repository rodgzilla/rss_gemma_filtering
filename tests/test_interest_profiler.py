"""Tests for rss_filter.interest_profiler"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rss_filter.models import NoteEntry
from rss_filter.interest_profiler import (
    build_profile_prompt,
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
