"""Tests for rss_filter.text_prep."""

from rss_filter.text_prep import MAX_CHARS, clean_summary, format_for_embedding, strip_html


def test_strip_html_removes_tags_and_unescapes():
    assert strip_html("<p>Hello&nbsp;<b>world</b> &amp; co</p>") == "Hello world & co"


def test_strip_html_separates_block_elements():
    assert strip_html("<p>one</p><p>two</p>") == "one two"


def test_strip_html_drops_script_and_style():
    assert strip_html("<style>p{}</style>text<script>x()</script>") == "text"


def test_clean_summary_removes_arxiv_announce_prefix():
    raw = "arXiv:2404.01234v1 Announce Type: new \nAbstract: We study things."
    assert clean_summary(raw) == "We study things."


def test_format_none_style_joins_title_and_body():
    assert format_for_embedding("T", "body", style="none") == "T body"


def test_format_document_style_uses_gemma_prompt():
    assert format_for_embedding("T", "body", style="document") == "title: T | text: body"


def test_format_document_style_missing_title():
    assert format_for_embedding("", "body", style="document") == "title: none | text: body"


def test_format_caps_length():
    out = format_for_embedding("T", "x" * (MAX_CHARS * 2), style="none")
    assert len(out) <= MAX_CHARS
