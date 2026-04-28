"""Fetch RSS feeds from an OPML file and manage seen-entry state."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional, Set, Tuple

import feedparser
import listparser

from rss_filter.models import RSSEntry


def parse_opml(opml_path: Path) -> List[Tuple[str, str]]:
    """Parse an OPML file and return list of (title, feed_url) tuples."""
    result = listparser.parse(opml_path.read_text(encoding="utf-8"))
    feeds = []
    for feed in result.feeds:
        title = getattr(feed, "title", "") or ""
        url = getattr(feed, "url", "") or ""
        if url:
            feeds.append((title, url))
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


def fetch_feed(feed_url: str, feed_name: str) -> List[RSSEntry]:
    """Fetch a single RSS/Atom feed and return its entries as RSSEntry objects."""
    parsed = feedparser.parse(feed_url)
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
