"""SQLite-backed embedding database with numpy cosine similarity search."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
from tqdm import tqdm

from rss_filter.embedding_client import EmbeddingClient
from rss_filter.models import NoteEntry
from rss_filter.text_prep import format_for_embedding

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

_CREATE_META = (
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
)
_GET_META = "SELECT value FROM meta WHERE key = ?;"
_SET_META = "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?);"

# Maximum number of texts sent to the embedding API in one chunk.
_EMBED_CHUNK_SIZE = 64


def _embed_text_for_entry(entry: NoteEntry, style: str = "none") -> str:
    """Return the text to embed for a vault note entry.

    ``notes_parser`` puts the title first in ``context``; it is stripped off so the
    body can be formatted exactly like a feed entry's (title, body). The URL is
    only used when there is neither a title nor a body, as before: appending it to
    every link-only entry would add URL noise to the vector.
    """
    title = (entry.title or "").strip()
    body = (entry.context or "").strip()
    if title and body.startswith(title):
        body = body[len(title) :].strip()
    if not title and not body:
        body = entry.url
    return format_for_embedding(title, body, style)


class EmbeddingStore:
    """Manages a persistent embedding database backed by SQLite."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_META)
        # Cached (texts, urls, titles, unit_matrix) for query(); None when stale.
        self._cache: tuple[list[str], list[str], list[str], np.ndarray] | None = None
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
        signature: str = "",
        prompt_style: str = "none",
    ) -> bool:
        """Embed and store vault note entries.

        If *force_rebuild* is True the entire table is dropped and rebuilt.
        Otherwise only entries whose (url, source_note) pair is not yet in
        the database are embedded and inserted.

        *signature* identifies how the stored vectors were produced (model,
        prompt style, text preparation version). When it differs from the one
        recorded in the database and the store is not empty, the table is
        rebuilt. An empty *signature* never triggers a rebuild and leaves the
        recorded one untouched.

        *prompt_style* selects the embedding text format (see
        ``text_prep.format_for_embedding``); feed entries must use the same one.

        Returns True if the table was rebuilt from scratch.
        """
        stored = self._get_meta("signature")
        if signature and stored != signature and self.count() > 0:
            print(
                f"Embedding store: signature changed ({stored!r} → {signature!r}); "
                "rebuilding."
            )
            force_rebuild = True

        if force_rebuild:
            self._conn.execute(_DROP_TABLE)
            self._conn.execute(_CREATE_TABLE)
            self._conn.commit()
            self._cache = None
            existing: set[tuple[str, str]] = set()
        else:
            rows = self._conn.execute(_SELECT_EXISTING_KEYS).fetchall()
            existing = {(row[0], row[1]) for row in rows}

        new_entries = [e for e in entries if (e.url, e.source) not in existing]

        if signature:
            self._set_meta("signature", signature)
            self._conn.commit()

        if not new_entries:
            print(f"Embedding store: 0 new entries, {len(existing)} already stored.")
            return force_rebuild

        texts = [_embed_text_for_entry(e, prompt_style) for e in new_entries]
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
        self._cache = None

        print(
            f"Embedding store: {len(new_entries)} new entries embedded, "
            f"{len(existing)} already stored."
        )
        return force_rebuild

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        """Return the *top_k* most similar stored documents.

        Each result is a dict with keys ``text``, ``url``, and ``score``
        (cosine similarity, float in [-1, 1]).
        """
        if self._cache is None:
            self._cache = self._load_matrix()
        texts, urls, titles, matrix_norm = self._cache
        if not texts:
            return []

        query_norm_val = np.linalg.norm(embedding)
        if query_norm_val == 0:
            query_unit = embedding
        else:
            query_unit = embedding / query_norm_val

        scores = matrix_norm @ query_unit  # shape (N,)

        k = min(top_k, len(texts))
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

    def _load_matrix(
        self,
    ) -> tuple[list[str], list[str], list[str], np.ndarray]:
        """Read all documents and return ``(texts, urls, titles, unit_matrix)``.

        *unit_matrix* holds the stored embeddings normalised to unit length
        (zero vectors are left as-is); it is empty when the store is.
        """
        rows = self._conn.execute(_SELECT_ALL).fetchall()
        if not rows:
            return [], [], [], np.empty((0, 0), dtype=np.float32)

        texts = [row[0] for row in rows]
        urls = [row[1] for row in rows]
        matrix = np.stack(
            [np.frombuffer(row[2], dtype=np.float32) for row in rows], axis=0
        )  # shape (N, D)
        titles = [row[3] for row in rows]

        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Avoid division by zero for any zero vectors.
        matrix_norms = np.where(matrix_norms == 0, 1.0, matrix_norms)
        return texts, urls, titles, matrix / matrix_norms

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _get_meta(self, key: str) -> str | None:
        """Return the metadata value stored under *key*, or None."""
        row = self._conn.execute(_GET_META, (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        """Store *value* under *key* (caller commits)."""
        self._conn.execute(_SET_META, (key, value))

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
