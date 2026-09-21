"""Thin wrapper around an OpenAI-compatible embedding API (llama-server, LM Studio)."""

from __future__ import annotations

import numpy as np
import openai
from openai import OpenAI

# Below this length an input the server still rejects is not shrunk further.
_MIN_FALLBACK_CHARS = 64

# One retry covers a transient server hiccup; an oversized input never succeeds on
# retry, so a larger budget would only add backoff delay before the fallback kicks in.
_MAX_RETRIES = 1


class EmbeddingClient:
    """Client for an OpenAI-compatible /v1/embeddings server (llama-server, LM Studio)."""

    def __init__(self, base_url: str, model: str) -> None:
        self._client = OpenAI(
            base_url=base_url, api_key="unused", max_retries=_MAX_RETRIES
        )
        self._model = model

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns a float32 numpy array."""
        response = self._client.embeddings.create(model=self._model, input=text)
        return np.array(response.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: list[str], chunk_size: int = 64) -> list[np.ndarray]:
        """Embed a list of texts. Sends requests in chunks to avoid oversized payloads.

        Returns a list of float32 numpy arrays in the same order as the input.

        If the server rejects a chunk (llama-server fails the whole request when
        one input is too long), its texts are retried one by one, halving any
        text the server still rejects.
        """
        results: list[np.ndarray] = []
        for start in range(0, len(texts), chunk_size):
            chunk = texts[start : start + chunk_size]
            try:
                response = self._client.embeddings.create(model=self._model, input=chunk)
            except openai.APIStatusError:
                results.extend(self._embed_shrinking(t) for t in chunk)
                continue
            # The API returns embeddings in the same order as the input.
            sorted_data = sorted(response.data, key=lambda d: d.index)
            results.extend(
                np.array(item.embedding, dtype=np.float32) for item in sorted_data
            )
        return results

    def _embed_shrinking(self, text: str) -> np.ndarray:
        """Embed ``text``, halving it while the server keeps rejecting it."""
        while True:
            try:
                return self.embed(text)
            except openai.APIStatusError:
                if len(text) <= _MIN_FALLBACK_CHARS:
                    raise
                text = text[: len(text) // 2]
