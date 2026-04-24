from dataclasses import dataclass


@dataclass
class NoteEntry:
    url: str
    context: str  # Title + surrounding text from the note
    source: str  # "reading" | "arxiv"
    date: str  # YYYY-MM-DD


@dataclass
class RSSEntry:
    title: str
    url: str
    summary: str
    feed_name: str
    is_arxiv: bool
    guid: str  # Unique identifier for deduplication
    published: str | None = None  # ISO date string YYYY-MM-DD, or None if unknown


@dataclass
class FilterResult:
    entry: RSSEntry
    keep: bool
    reason: str
