"""Tests for the packaged CLI entry point."""

from pathlib import Path

import numpy as np
import pytest

from rss_filter import cli

REPO_ROOT = Path(__file__).resolve().parent.parent
MOCK_VAULT = REPO_ROOT / "mock_vault"

EMPTY_OPML = """<?xml version="1.0" encoding="UTF-8"?>
<opml version="1.0"><head><title>empty</title></head><body></body></opml>
"""


class FakeEmbeddingClient:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url
        self.model = model

    def embed(self, text: str) -> np.ndarray:
        v = np.ones(8, dtype=np.float32)
        return v / np.linalg.norm(v)

    def embed_batch(self, texts, chunk_size: int = 64):
        return [self.embed(t) for t in texts]


@pytest.fixture
def empty_opml(tmp_path: Path) -> Path:
    p = tmp_path / "feeds.opml"
    p.write_text(EMPTY_OPML)
    return p


def test_packaged_config_is_default():
    assert (Path(cli.__file__).parent / "config.toml").is_file()


def test_state_dir_holds_store(tmp_path, empty_opml, monkeypatch):
    monkeypatch.setattr(cli, "EmbeddingClient", FakeEmbeddingClient)
    state = tmp_path / "state"
    cli.main(
        [
            "--vault", str(MOCK_VAULT),
            "--feeds", str(empty_opml),
            "--state-dir", str(state),
            "--max-notes", "2",
        ]
    )
    assert (state / "embedding_store.db").is_file()


def test_in_state_resolves_relative_only(tmp_path):
    assert cli._in_state(tmp_path, "a.db") == str(tmp_path / "a.db")
    assert cli._in_state(tmp_path, "/abs/a.db") == "/abs/a.db"


def test_override_uses_is_not_none():
    assert cli._override(0.0, 0.25) == 0.0
    assert cli._override(None, 0.25) == 0.25
