"""SQLite-backed embedding database with numpy cosine similarity search."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
from tqdm import tqdm

from rss_filter.embedding_client import EmbeddingClient
from rss_filter.models import NoteEntry

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT NOT NULL,
    text        TEXT NOT NULL,
    source_note TEXT NOT NULL,
    date        TEXT,
    embedding   BLOB NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    UNIQUE(url, source_note)
);
"""

_DROP_TABLE = "DROP TABLE IF EXISTS documents;"

_INSERT_DOC = """
INSERT OR IGNORE INTO documents (url, text, source_note, date, embedding, title)
VALUES (?, ?, ?, ?, ?, ?);
"""

_SELECT_EXISTING_KEYS = "SELECT url, source_note FROM documents;"

_SELECT_ALL = "SELECT text, url, embedding, title FROM documents;"
_SELECT_ALL_WITH_META = "SELECT url, text, source_note, date, embedding FROM documents;"

_COUNT = "SELECT COUNT(*) FROM documents;"

# Maximum number of texts sent to the embedding API in one chunk.
_EMBED_CHUNK_SIZE = 64


def _embed_text_for_entry(entry: NoteEntry) -> str:
    """Return the text to embed for a vault note entry."""
    return (
        entry.context.strip() if entry.context and entry.context.strip() else entry.url
    )


class EmbeddingStore:
    """Manages a persistent embedding database backed by SQLite."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(_CREATE_TABLE)
        # Migration: add title column if it doesn't exist (existing databases)
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(documents);")}
        if "title" not in cols:
            self._conn.execute(
                "ALTER TABLE documents ADD COLUMN title TEXT NOT NULL DEFAULT '';"
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Building / updating
    # ------------------------------------------------------------------

    def build_or_update(
        self,
        entries: list[NoteEntry],
        client: EmbeddingClient,
        force_rebuild: bool = False,
    ) -> None:
        """Embed and store vault note entries.

        If *force_rebuild* is True the entire table is dropped and rebuilt.
        Otherwise only entries whose (url, source_note) pair is not yet in
        the database are embedded and inserted.
        """
        if force_rebuild:
            self._conn.execute(_DROP_TABLE)
            self._conn.execute(_CREATE_TABLE)
            self._conn.commit()
            existing: set[tuple[str, str]] = set()
        else:
            rows = self._conn.execute(_SELECT_EXISTING_KEYS).fetchall()
            existing = {(row[0], row[1]) for row in rows}

        new_entries = [e for e in entries if (e.url, e.source) not in existing]

        if not new_entries:
            print(f"Embedding store: 0 new entries, {len(existing)} already stored.")
            return

        texts = [_embed_text_for_entry(e) for e in new_entries]
        embeddings = _embed_in_chunks(
            client, texts, _EMBED_CHUNK_SIZE, desc="Embedding vault"
        )

        rows_to_insert = [
            (
                entry.url,
                text,
                entry.source,
                str(entry.date) if entry.date else None,
                emb.tobytes(),
                entry.title,
            )
            for entry, text, emb in zip(new_entries, texts, embeddings)
        ]
        self._conn.executemany(_INSERT_DOC, rows_to_insert)
        self._conn.commit()

        print(
            f"Embedding store: {len(new_entries)} new entries embedded, "
            f"{len(existing)} already stored."
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        """Return the *top_k* most similar stored documents.

        Each result is a dict with keys ``text``, ``url``, and ``score``
        (cosine similarity, float in [-1, 1]).
        """
        rows = self._conn.execute(_SELECT_ALL).fetchall()
        if not rows:
            return []

        texts = [row[0] for row in rows]
        urls = [row[1] for row in rows]
        matrix = np.stack(
            [np.frombuffer(row[2], dtype=np.float32) for row in rows], axis=0
        )  # shape (N, D)
        titles = [row[3] for row in rows]

        # Normalise stored embeddings and query vector.
        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Avoid division by zero for any zero vectors.
        matrix_norms = np.where(matrix_norms == 0, 1.0, matrix_norms)
        matrix_norm = matrix / matrix_norms

        query_norm_val = np.linalg.norm(embedding)
        if query_norm_val == 0:
            query_unit = embedding
        else:
            query_unit = embedding / query_norm_val

        scores = matrix_norm @ query_unit  # shape (N,)

        k = min(top_k, len(rows))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            {
                "text": texts[i],
                "url": urls[i],
                "score": float(scores[i]),
                "title": titles[i],
            }
            for i in top_indices
        ]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of documents in the store."""
        row = self._conn.execute(_COUNT).fetchone()
        return row[0] if row else 0

    def get_all(self) -> list[dict]:
        """Return all stored documents with their embeddings.

        Each entry is a dict with keys:
        ``url``, ``text``, ``source_note``, ``date``, ``embedding`` (np.ndarray float32).
        """
        rows = self._conn.execute(_SELECT_ALL_WITH_META).fetchall()
        return [
            {
                "url": row[0],
                "text": row[1],
                "source_note": row[2],
                "date": row[3],
                "embedding": np.frombuffer(row[4], dtype=np.float32).copy(),
            }
            for row in rows
        ]

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __enter__(self) -> "EmbeddingStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _embed_in_chunks(
    client: EmbeddingClient,
    texts: list[str],
    chunk_size: int,
    desc: str = "Embedding",
) -> list[np.ndarray]:
    """Embed *texts* using *client*, sending at most *chunk_size* per request."""
    results: list[np.ndarray] = []
    chunks = [
        texts[start : start + chunk_size] for start in range(0, len(texts), chunk_size)
    ]
    with tqdm(total=len(texts), desc=desc, unit="doc") as pbar:
        for chunk in chunks:
            results.extend(client.embed_batch(chunk))
            pbar.update(len(chunk))
    return results
