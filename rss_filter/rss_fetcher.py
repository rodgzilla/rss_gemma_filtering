"""Fetch RSS feeds from an OPML file and manage seen-entry state."""

from __future__ import annotations

import json
import re
import socket
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional, Set, Tuple

import feedparser

from rss_filter.models import RSSEntry

DEFAULT_FEED_TIMEOUT = 15  # seconds


@contextmanager
def _socket_timeout(seconds: float):
    """Temporarily set the process-wide default socket timeout.

    feedparser has no built-in per-request timeout. Under the hood it uses
    urllib, which falls back to the global socket default (normally "block
    forever") whenever no explicit timeout is given. Setting that default
    here bounds every connection feedparser makes for the duration of the
    call, then restores the previous value.
    """
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous)


def _is_timeout_exception(exc: Optional[BaseException]) -> bool:
    """Return True if `exc` is (or wraps) a socket/connection timeout."""
    if exc is None:
        return False
    if isinstance(exc, TimeoutError):  # socket.timeout is an alias of this
        return True
    return isinstance(getattr(exc, "reason", None), TimeoutError)


def parse_opml(opml_path: Path) -> List[Tuple[str, str]]:
    """Parse an OPML file and return list of (title, feed_url) tuples."""
    root = ET.parse(opml_path).getroot()
    feeds = []
    for outline in root.iter("outline"):
        url = outline.get("xmlUrl") or ""
        if url:
            feeds.append((outline.get("title") or outline.get("text") or "", url))
    return feeds


def classify_is_arxiv(feed_url: str, entry_url: str) -> bool:
    """Return True if either URL is from arxiv.org."""
    return "arxiv.org" in feed_url or "arxiv.org" in entry_url


def _parse_published(item) -> Optional[str]:
    """Extract a YYYY-MM-DD string from a feedparser entry, or return None."""
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        t = getattr(item, attr, None)
        if t is not None:
            try:
                return date(*t[:3]).isoformat()
            except (TypeError, ValueError):
                pass
    return None


def fetch_feed(
    feed_url: str, feed_name: str, timeout: float = DEFAULT_FEED_TIMEOUT
) -> List[RSSEntry]:
    """Fetch a single RSS/Atom feed and return its entries as RSSEntry objects.

    Raises TimeoutError if the feed does not respond within `timeout` seconds,
    so an unresponsive server doesn't stall the whole fetch loop.
    """
    with _socket_timeout(timeout):
        parsed = feedparser.parse(feed_url)

    if getattr(parsed, "bozo", False) and _is_timeout_exception(
        getattr(parsed, "bozo_exception", None)
    ):
        raise TimeoutError(
            f"Timed out after {timeout}s fetching feed '{feed_name}' ({feed_url})"
        )

    entries = []
    for item in parsed.entries:
        link = getattr(item, "link", "") or ""
        guid = item.get("id", "") or link
        title = getattr(item, "title", "") or ""
        summary = getattr(item, "summary", "") or ""
        is_arxiv = classify_is_arxiv(feed_url, link)
        published = _parse_published(item)
        entries.append(
            RSSEntry(
                title=title,
                url=link,
                summary=summary,
                feed_name=feed_name,
                is_arxiv=is_arxiv,
                guid=guid,
                published=published,
            )
        )
    return entries


_ARXIV_ID_RE = re.compile(r"arxiv\.org/abs/(.+?)(?:v\d+)?$")


def _arxiv_canonical_id(url: str) -> Optional[str]:
    """Extract the canonical arXiv paper ID from a URL, or return None."""
    m = _ARXIV_ID_RE.search(url)
    return m.group(1).rstrip("/") if m else None


def deduplicate_entries(entries: List[RSSEntry]) -> List[RSSEntry]:
    """Remove duplicate entries, keeping the first occurrence.

    arXiv entries are deduplicated by canonical paper ID extracted from the URL.
    Non-arXiv entries are deduplicated by URL.
    """
    seen_keys: Set[str] = set()
    result: List[RSSEntry] = []
    for entry in entries:
        if entry.is_arxiv:
            key = _arxiv_canonical_id(entry.url) or entry.url
        else:
            key = entry.url
        if key not in seen_keys:
            seen_keys.add(key)
            result.append(entry)
    return result


def filter_new_entries(entries: List[RSSEntry], seen: Set[str]) -> List[RSSEntry]:
    """Return only entries whose GUID is not in the seen set."""
    return [e for e in entries if e.guid not in seen]


def filter_by_age(
    entries: List[RSSEntry], max_age_days: int, reference_date: Optional[date] = None
) -> List[RSSEntry]:
    """Return entries published within max_age_days of reference_date.

    Entries with no published date are kept (we cannot determine age).
    """
    ref = reference_date or date.today()
    result = []
    for e in entries:
        if e.published is None:
            result.append(e)
        else:
            try:
                pub = date.fromisoformat(e.published)
                if (ref - pub).days <= max_age_days:
                    result.append(e)
            except ValueError:
                result.append(e)
    return result


def load_seen_guids(state_path: Path) -> Set[str]:
    """Load the set of already-seen GUIDs from a JSON file."""
    if not state_path.exists():
        return set()
    data = json.loads(state_path.read_text(encoding="utf-8"))
    return set(data)


def save_seen_guids(new_guids: Set[str], state_path: Path) -> None:
    """Merge new_guids with existing state and persist to disk."""
    existing = load_seen_guids(state_path)
    merged = existing | new_guids
    state_path.write_text(json.dumps(sorted(merged), indent=2), encoding="utf-8")
