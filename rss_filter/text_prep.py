"""Normalise text before it is sent to the embedding server.

Vault notes are plain markdown while feed summaries are HTML, so both sides are
reduced to plain text and formatted identically before embedding. The length cap
keeps every input under llama-server's 2048-token batch: a single oversized input
makes the server reject the whole request.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

# ~4 chars/token for English/French prose → ~1000 tokens, well under 2048.
# EmbeddingClient still falls back to halving on a server error, for token-dense text.
MAX_CHARS = 4000

# Bump whenever the text cleaning changes: it is part of the embedding store signature.
PREP_VERSION = 1

PROMPT_STYLES = ("none", "document")

_WS_RE = re.compile(r"\s+")
_ARXIV_PREFIX_RE = re.compile(
    r"^arXiv:\S+\s+Announce Type:\s*\S+\s*Abstract:\s*", re.IGNORECASE
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def strip_html(text: str) -> str:
    """Return the visible text of an HTML fragment, whitespace-collapsed."""
    parser = _TextExtractor()
    parser.feed(text)
    parser.close()
    return _WS_RE.sub(" ", " ".join(parser.parts)).strip()


def clean_summary(summary: str) -> str:
    """Plain text of a feed summary, without arXiv's announcement header."""
    return _ARXIV_PREFIX_RE.sub("", strip_html(summary or ""))


def format_for_embedding(title: str, body: str, style: str = "none") -> str:
    """Build the exact string sent to the embedding model."""
    title = _WS_RE.sub(" ", title or "").strip()
    body = _WS_RE.sub(" ", body or "").strip()
    if style == "document":
        text = f"title: {title or 'none'} | text: {body}"
    elif style == "none":
        text = f"{title} {body}".strip()
    else:
        raise ValueError(f"unknown prompt style {style!r}; expected one of {PROMPT_STYLES}")
    return text[:MAX_CHARS]
