"""Thin wrapper around the LM Studio embedding API."""

from __future__ import annotations

import numpy as np
from openai import OpenAI


class EmbeddingClient:
    """Client for computing text embeddings via the LM Studio embedding API."""

    def __init__(self, base_url: str, model: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key="lm-studio")
        self._model = model

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns a float32 numpy array."""
        response = self._client.embeddings.create(model=self._model, input=text)
        return np.array(response.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: list[str], chunk_size: int = 64) -> list[np.ndarray]:
        """Embed a list of texts. Sends requests in chunks to avoid oversized payloads.

        Returns a list of float32 numpy arrays in the same order as the input.
        """
        results: list[np.ndarray] = []
        for start in range(0, len(texts), chunk_size):
            chunk = texts[start : start + chunk_size]
            response = self._client.embeddings.create(model=self._model, input=chunk)
            # The API returns embeddings in the same order as the input.
            sorted_data = sorted(response.data, key=lambda d: d.index)
            results.extend(
                np.array(item.embedding, dtype=np.float32) for item in sorted_data
            )
        return results
