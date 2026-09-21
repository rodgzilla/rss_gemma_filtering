"""Tests for rss_filter.embedding_client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import openai
import pytest

from rss_filter.embedding_client import EmbeddingClient


def _make_embedding_response(vectors: list[list[float]]) -> MagicMock:
    """Build a fake openai embeddings response."""
    response = MagicMock()
    response.data = [MagicMock(index=i, embedding=vec) for i, vec in enumerate(vectors)]
    return response


def _status_error() -> openai.InternalServerError:
    """Build the error the openai client raises when the server answers HTTP 500."""
    request = httpx.Request("POST", "http://x/v1/embeddings")
    response = httpx.Response(500, request=request)
    return openai.InternalServerError("too large", response=response, body=None)


@pytest.fixture()
def mock_openai(monkeypatch):
    """Patch openai.OpenAI so no real HTTP calls are made."""
    with patch("rss_filter.embedding_client.OpenAI") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


class TestEmbedSingle:
    def test_returns_float32_numpy_array(self, mock_openai):
        vec = [0.1, 0.2, 0.3]
        mock_openai.embeddings.create.return_value = _make_embedding_response([vec])

        client = EmbeddingClient(base_url="http://localhost:1234/v1", model="my-model")
        result = client.embed("hello world")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, np.array(vec, dtype=np.float32))

    def test_passes_correct_model_and_input(self, mock_openai):
        mock_openai.embeddings.create.return_value = _make_embedding_response([[0.0]])

        client = EmbeddingClient(base_url="http://localhost:1234/v1", model="my-model")
        client.embed("test text")

        mock_openai.embeddings.create.assert_called_once_with(
            model="my-model", input="test text"
        )


class TestEmbedBatch:
    def test_returns_list_of_arrays(self, mock_openai):
        vecs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_openai.embeddings.create.return_value = _make_embedding_response(vecs)

        client = EmbeddingClient(base_url="http://localhost:1234/v1", model="my-model")
        results = client.embed_batch(["a", "b", "c"])

        assert len(results) == 3
        for r, v in zip(results, vecs):
            assert isinstance(r, np.ndarray)
            assert r.dtype == np.float32
            np.testing.assert_array_almost_equal(r, np.array(v, dtype=np.float32))

    def test_chunking_splits_requests(self, mock_openai):
        # With chunk_size=2 and 5 texts, we expect 3 API calls.
        def side_effect(model, input):
            return _make_embedding_response([[float(i)] for i in range(len(input))])

        mock_openai.embeddings.create.side_effect = side_effect

        client = EmbeddingClient(base_url="http://localhost:1234/v1", model="my-model")
        results = client.embed_batch(["a", "b", "c", "d", "e"], chunk_size=2)

        assert mock_openai.embeddings.create.call_count == 3
        assert len(results) == 5

    def test_empty_batch_returns_empty_list(self, mock_openai):
        client = EmbeddingClient(base_url="http://localhost:1234/v1", model="my-model")
        results = client.embed_batch([])

        mock_openai.embeddings.create.assert_not_called()
        assert results == []


class TestClientConstruction:
    def test_uses_placeholder_key_and_few_retries(self):
        with patch("rss_filter.embedding_client.OpenAI") as mock_cls:
            EmbeddingClient(base_url="http://x/v1", model="m")
        mock_cls.assert_called_once_with(
            base_url="http://x/v1", api_key="unused", max_retries=1
        )


class TestOversizedFallback:
    def test_batch_error_falls_back_to_per_item_and_halves(self, mock_openai):
        calls = []

        def create(model, input):
            calls.append(input)
            if isinstance(input, list):
                raise _status_error()
            if len(input) > 100:
                raise _status_error()
            return _make_embedding_response([[1.0, 0.0]])

        mock_openai.embeddings.create.side_effect = create
        client = EmbeddingClient(base_url="http://x/v1", model="m")
        out = client.embed_batch(["short", "y" * 400])

        assert len(out) == 2
        assert any(isinstance(c, str) and len(c) == 100 for c in calls)

    def test_gives_up_below_minimum_length(self, mock_openai):
        mock_openai.embeddings.create.side_effect = _status_error()
        client = EmbeddingClient(base_url="http://x/v1", model="m")
        with pytest.raises(openai.APIStatusError):
            client.embed_batch(["tiny"])
