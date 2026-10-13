"""Unit tests for NewsboatTool (baton/tools/newsboat.py)."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from baton.tools.newsboat import NewsboatTool, clean_html


@pytest.fixture
def mock_newsboat_db(tmp_path: Path) -> Path:
    """Creates a temporary mock Newsboat SQLite cache.db with sample data."""
    db_file = tmp_path / "mock_cache.db"
    conn = sqlite3.connect(db_file)
    conn.executescript("""
        CREATE TABLE rss_feed (
            rssurl TEXT PRIMARY KEY,
            url TEXT,
            title TEXT,
            lastmodified INTEGER,
            is_rtl INTEGER,
            etag TEXT
        );

        CREATE TABLE rss_item (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            guid TEXT,
            title TEXT,
            author TEXT,
            url TEXT,
            feedurl TEXT,
            pubDate INTEGER,
            content TEXT,
            unread INTEGER,
            enclosure_url TEXT,
            enclosure_type TEXT,
            enqueued INTEGER,
            flags TEXT,
            deleted INTEGER,
            base TEXT,
            content_mime_type TEXT,
            enclosure_description TEXT,
            enclosure_description_mime_type TEXT
        );

        INSERT INTO rss_feed (rssurl, url, title) VALUES
        ('https://feeds.arstechnica.com/arstechnica/index', 'https://arstechnica.com', 'Ars Technica'),
        ('https://www.nature.com/nature.rss', 'https://www.nature.com', 'Nature');

        INSERT INTO rss_item (guid, title, author, url, feedurl, pubDate, content, unread) VALUES
        ('guid-1', 'Breakthrough in Solid-State Lithium Batteries', 'Alice Scientist', 'https://arstechnica.com/battery', 'https://feeds.arstechnica.com/arstechnica/index', 1700000000, '<p>Researchers have developed a <b>solid-state</b> electrolyte with <a href="https://example.com">ultra-high</a> conductivity.</p>', 1),
        ('guid-2', 'New Exoplanet Discovered in Habitable Zone', 'Bob Astronomer', 'https://nature.com/exoplanet', 'https://www.nature.com/nature.rss', 1700001000, '<div>Spectroscopy reveals water vapor and nitrogen atmospheres.</div>', 0);
    """)
    conn.close()
    return db_file


def test_clean_html():
    raw = '<p>Test <b>bold</b> text with <a href="https://link.com">a link</a> and &amp; entities.</p>'
    cleaned = clean_html(raw)
    assert "Test bold text" in cleaned
    assert "[a link](https://link.com)" in cleaned
    assert "&" in cleaned
    assert "<p>" not in cleaned


def test_newsboat_tool_list_feeds(mock_newsboat_db: Path):
    nb = NewsboatTool(db_path=mock_newsboat_db)
    feeds = nb.list_feeds()
    assert len(feeds) == 2
    feed_titles = {f["title"] for f in feeds}
    assert "Ars Technica" in feed_titles
    assert "Nature" in feed_titles

    ars = next(f for f in feeds if f["title"] == "Ars Technica")
    assert ars["total_articles"] == 1
    assert ars["unread_articles"] == 1


def test_newsboat_tool_get_latest_articles(mock_newsboat_db: Path):
    nb = NewsboatTool(db_path=mock_newsboat_db)
    latest = nb.get_latest_articles(limit=10)
    assert len(latest) == 2
    # Exoplanet has higher pubDate (1700001000 vs 1700000000)
    assert latest[0]["title"] == "New Exoplanet Discovered in Habitable Zone"
    assert latest[0]["feed"] == "Nature"

    unread_only = nb.get_latest_articles(unread_only=True)
    assert len(unread_only) == 1
    assert unread_only[0]["title"] == "Breakthrough in Solid-State Lithium Batteries"


def test_newsboat_tool_search_articles(mock_newsboat_db: Path):
    nb = NewsboatTool(db_path=mock_newsboat_db)
    res_battery = nb.search_articles("Lithium")
    assert len(res_battery) == 1
    assert res_battery[0]["title"] == "Breakthrough in Solid-State Lithium Batteries"

    res_content = nb.search_articles("water vapor")
    assert len(res_content) == 1
    assert res_content[0]["title"] == "New Exoplanet Discovered in Habitable Zone"

    res_empty = nb.search_articles("nonexistent topic")
    assert len(res_empty) == 0


def test_newsboat_tool_read_article(mock_newsboat_db: Path):
    nb = NewsboatTool(db_path=mock_newsboat_db)
    art = nb.read_article(item_id=1)
    assert art["id"] == 1
    assert art["title"] == "Breakthrough in Solid-State Lithium Batteries"
    assert "solid-state" in art["content"]
    assert art["is_truncated"] is False

    err = nb.read_article(item_id=999)
    assert "error" in err


def test_newsboat_tool_execute_dispatcher(mock_newsboat_db: Path):
    nb = NewsboatTool(db_path=mock_newsboat_db)

    feed_str = nb.execute("rss_list_feeds", {})
    assert "Subscribed RSS Feeds" in feed_str
    assert "Ars Technica" in feed_str

    latest_str = nb.execute("rss_latest", {"limit": 5})
    assert "Latest Articles" in latest_str
    assert "Exoplanet" in latest_str

    search_str = nb.execute("rss_search", {"query": "Lithium"})
    assert "Search Results" in search_str
    assert "Batteries" in search_str

    read_str = nb.execute("rss_read_article", {"id": 1})
    assert "Breakthrough in Solid-State" in read_str
    assert "Alice Scientist" in read_str
