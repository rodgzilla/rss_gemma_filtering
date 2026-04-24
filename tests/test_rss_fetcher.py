"""Tests for rss_filter.rss_fetcher"""

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rss_filter.models import RSSEntry
from rss_filter.rss_fetcher import (
    classify_is_arxiv,
    fetch_feed,
    filter_new_entries,
    load_seen_guids,
    parse_opml,
    save_seen_guids,
)

# ---------------------------------------------------------------------------
# Sample OPML
# ---------------------------------------------------------------------------

SAMPLE_OPML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <opml version="1.0">
      <head><title>My Feeds</title></head>
      <body>
        <outline text="Ars Technica" title="Ars Technica" type="rss"
                 xmlUrl="https://feeds.arstechnica.com/arstechnica/index"
                 htmlUrl="https://arstechnica.com"/>
        <outline text="ArXiv CS.LG" title="ArXiv CS.LG" type="rss"
                 xmlUrl="https://export.arxiv.org/rss/cs.LG"
                 htmlUrl="https://arxiv.org/list/cs.LG/recent"/>
      </body>
    </opml>
""")

# ---------------------------------------------------------------------------
# parse_opml
# ---------------------------------------------------------------------------


def test_parse_opml_returns_list_of_tuples(tmp_path):
    opml_file = tmp_path / "subs.opml"
    opml_file.write_text(SAMPLE_OPML)

    feeds = parse_opml(opml_file)

    assert len(feeds) == 2


def test_parse_opml_extracts_title_and_url(tmp_path):
    opml_file = tmp_path / "subs.opml"
    opml_file.write_text(SAMPLE_OPML)

    feeds = parse_opml(opml_file)
    titles = [f[0] for f in feeds]
    urls = [f[1] for f in feeds]

    assert "Ars Technica" in titles
    assert "ArXiv CS.LG" in titles
    assert "https://feeds.arstechnica.com/arstechnica/index" in urls
    assert "https://export.arxiv.org/rss/cs.LG" in urls


# ---------------------------------------------------------------------------
# classify_is_arxiv
# ---------------------------------------------------------------------------


def test_classify_is_arxiv_true_for_arxiv_feed_url():
    assert classify_is_arxiv("https://export.arxiv.org/rss/cs.LG", "") is True


def test_classify_is_arxiv_true_for_arxiv_entry_url():
    assert (
        classify_is_arxiv(
            "https://feeds.example.com/feed", "https://arxiv.org/abs/1234.56789"
        )
        is True
    )


def test_classify_is_arxiv_false_for_general_feed():
    assert (
        classify_is_arxiv(
            "https://feeds.arstechnica.com/arstechnica/index",
            "https://arstechnica.com/article",
        )
        is False
    )


# ---------------------------------------------------------------------------
# fetch_feed (feedparser mocked)
# ---------------------------------------------------------------------------


def _make_mock_feed(entries):
    mock_feed = MagicMock()
    mock_feed.entries = entries
    return mock_feed


def _make_entry(title, link, summary, id_=None):
    e = MagicMock()
    e.title = title
    e.link = link
    e.summary = summary
    e.get = lambda key, default="": id_ if key == "id" else default
    return e


def test_fetch_feed_returns_rss_entries(mocker):
    mock_entry = _make_entry(
        title="Cool Article",
        link="https://example.com/cool",
        summary="A cool article about things.",
        id_="https://example.com/cool",
    )
    mocker.patch("feedparser.parse", return_value=_make_mock_feed([mock_entry]))

    results = fetch_feed(
        feed_url="https://example.com/feed",
        feed_name="Example",
    )

    assert len(results) == 1
    assert isinstance(results[0], RSSEntry)
    assert results[0].title == "Cool Article"
    assert results[0].url == "https://example.com/cool"
    assert results[0].feed_name == "Example"


def test_fetch_feed_sets_is_arxiv_false_for_general(mocker):
    mock_entry = _make_entry("Title", "https://example.com/a", "Summary", "guid-1")
    mocker.patch("feedparser.parse", return_value=_make_mock_feed([mock_entry]))

    results = fetch_feed("https://example.com/feed", "Example")

    assert results[0].is_arxiv is False


def test_fetch_feed_sets_is_arxiv_true_for_arxiv(mocker):
    mock_entry = _make_entry(
        "A Paper",
        "https://arxiv.org/abs/1234.5678",
        "Abstract text.",
        "arxiv:1234.5678",
    )
    mocker.patch("feedparser.parse", return_value=_make_mock_feed([mock_entry]))

    results = fetch_feed("https://export.arxiv.org/rss/cs.LG", "ArXiv CS.LG")

    assert results[0].is_arxiv is True


def test_fetch_feed_uses_link_as_guid_fallback(mocker):
    mock_entry = _make_entry("Title", "https://example.com/b", "Summary", id_=None)
    # Make .get always return "" so id falls back to link
    mock_entry.get = lambda key, default="": default
    mocker.patch("feedparser.parse", return_value=_make_mock_feed([mock_entry]))

    results = fetch_feed("https://example.com/feed", "Example")

    assert results[0].guid == "https://example.com/b"


# ---------------------------------------------------------------------------
# filter_new_entries
# ---------------------------------------------------------------------------


def _make_rss_entry(guid: str) -> RSSEntry:
    return RSSEntry(
        title="T",
        url="https://x.com",
        summary="S",
        feed_name="F",
        is_arxiv=False,
        guid=guid,
    )


def test_filter_new_entries_removes_seen_guids():
    seen = {"guid-old", "guid-also-old"}
    entries = [_make_rss_entry("guid-old"), _make_rss_entry("guid-new")]

    result = filter_new_entries(entries, seen)

    assert len(result) == 1
    assert result[0].guid == "guid-new"


def test_filter_new_entries_returns_all_when_none_seen():
    entries = [_make_rss_entry("a"), _make_rss_entry("b")]

    result = filter_new_entries(entries, set())

    assert len(result) == 2


def test_filter_new_entries_returns_empty_when_all_seen():
    seen = {"a", "b"}
    entries = [_make_rss_entry("a"), _make_rss_entry("b")]

    result = filter_new_entries(entries, seen)

    assert result == []


# ---------------------------------------------------------------------------
# load_seen_guids / save_seen_guids
# ---------------------------------------------------------------------------


def test_load_seen_guids_returns_empty_set_when_file_missing(tmp_path):
    result = load_seen_guids(tmp_path / "nonexistent.json")
    assert result == set()


def test_load_seen_guids_returns_correct_set(tmp_path):
    state_file = tmp_path / "seen.json"
    state_file.write_text(json.dumps(["guid-1", "guid-2"]))

    result = load_seen_guids(state_file)

    assert result == {"guid-1", "guid-2"}


def test_save_seen_guids_creates_file(tmp_path):
    state_file = tmp_path / "seen.json"
    save_seen_guids({"guid-a", "guid-b"}, state_file)

    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert set(data) == {"guid-a", "guid-b"}


def test_save_seen_guids_merges_with_existing(tmp_path):
    state_file = tmp_path / "seen.json"
    state_file.write_text(json.dumps(["old-guid"]))

    save_seen_guids({"new-guid"}, state_file)

    data = set(json.loads(state_file.read_text()))
    assert "old-guid" in data
    assert "new-guid" in data
