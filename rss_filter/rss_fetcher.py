"""Fetch RSS feeds from an OPML file and manage seen-entry state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Set, Tuple

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
        entries.append(
            RSSEntry(
                title=title,
                url=link,
                summary=summary,
                feed_name=feed_name,
                is_arxiv=is_arxiv,
                guid=guid,
            )
        )
    return entries


def filter_new_entries(entries: List[RSSEntry], seen: Set[str]) -> List[RSSEntry]:
    """Return only entries whose GUID is not in the seen set."""
    return [e for e in entries if e.guid not in seen]


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
