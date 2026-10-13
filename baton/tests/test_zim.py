"""Unit tests for Baton Kiwix ZIM Wikipedia reader and tool integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from baton.tools.zim import ZimTool, find_default_zim
from baton.tools.zim_reader import sanitize_article_title


def test_sanitize_valid_titles():
    assert sanitize_article_title("../A/Kevin_Bacon") == "Kevin Bacon"
    assert sanitize_article_title("./Apollo_13_(film)") == "Apollo 13 (film)"
    assert sanitize_article_title("/A/Tom_Hanks#Personal_life") == "Tom Hanks"
    assert sanitize_article_title("Oliver_Stone?action=view") == "Oliver Stone"


def test_filter_namespaces():
    assert sanitize_article_title("../A/File:Kevin_Bacon.jpg") is None
    assert sanitize_article_title("./Category:Living_people") is None
    assert sanitize_article_title("/A/Help:Contents") is None
    assert sanitize_article_title("Wikipedia:About") is None
    assert sanitize_article_title("Template:Actor_infobox") is None


def test_filter_dates_and_calendar():
    assert sanitize_article_title("1958") is None
    assert sanitize_article_title("July_8") is None
    assert sanitize_article_title("1980s") is None
    assert sanitize_article_title("20th_century") is None


def test_find_default_zim_with_env(monkeypatch, tmp_path):
    fake_zim = tmp_path / "custom.zim"
    fake_zim.write_bytes(b"\x00" * (1024 * 1024 + 10))
    monkeypatch.setenv("BATON_ZIM_PATH", str(fake_zim))

    resolved = find_default_zim()
    assert resolved == fake_zim


def test_zim_tool_dispatcher():
    mock_reader = MagicMock()
    mock_reader.search.return_value = [{"title": "Apollo 11", "snippet": "Moon landing"}]
    mock_reader.read_lead.return_value = "Apollo 11 was the spaceflight..."
    mock_reader.read_section.return_value = "Mission details..."
    mock_reader.get_infobox_markdown.return_value = "| Operator | NASA |"
    mock_reader.get_toc.return_value = [{"heading": "Launch", "level": 2}]

    tool = ZimTool(zim_path="dummy.zim")
    tool._reader = mock_reader

    # Test search
    res = tool.execute("wiki_search", {"query": "Apollo 11"})
    assert "Found 1 matches" in res
    assert "Apollo 11" in res

    # Test lead
    res = tool.execute("wiki_read_lead", {"title": "Apollo 11"})
    assert "# Apollo 11 (Lead Section)" in res

    # Test section
    res = tool.execute("wiki_read_section", {"title": "Apollo 11", "section": "Launch"})
    assert "# Apollo 11 - Launch" in res

    # Test infobox
    res = tool.execute("wiki_infobox", {"title": "Apollo 11"})
    assert "# Apollo 11 (Infobox)" in res

    # Test toc
    res = tool.execute("wiki_toc", {"title": "Apollo 11"})
    assert "# Apollo 11 (Table of Contents)" in res
    assert "- Launch" in res

    # Unknown tool
    res = tool.execute("unknown_tool", {})
    assert "Error: Unknown Zim tool" in res
