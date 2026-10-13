"""Newsboat RSS offline knowledge tool for Baton System 1."""

from __future__ import annotations

import html
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from bs4 import BeautifulSoup


def find_default_newsboat_db() -> Optional[Path]:
    """Locate the default Newsboat cache.db database."""
    env_db = os.getenv("BATON_NEWSBOAT_DB")
    if env_db:
        p = Path(env_db)
        if p.exists() and p.stat().st_size > 0:
            return p

    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "rss" / "cache.db",
        Path("baton/data/rss/cache.db"),
        Path.home() / ".local/share/newsboat/cache.db",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def clean_html(raw_html: str) -> str:
    """Converts HTML markup from RSS items into clean text/markdown."""
    if not raw_html:
        return ""
    try:
        soup = BeautifulSoup(raw_html, "html.parser")
        # Remove script and style elements
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Format links to markdown if present
        for a in soup.find_all("a", href=True):
            text = a.get_text().strip()
            href = a["href"].strip()
            if text and href and text != href:
                a.replace_with(f"[{text}]({href})")

        # Replace block tags with newlines and break tags
        for br in soup.find_all("br"):
            br.replace_with("\n")
        for block in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"]):
            block.append("\n\n")

        text = soup.get_text()
        text = html.unescape(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" ?\n ?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text
    except Exception:
        # Fallback regex strip
        clean = re.sub(r"<[^>]+>", " ", raw_html)
        return html.unescape(re.sub(r"\s+", " ", clean).strip())


class NewsboatTool:
    """Provides high-throughput offline RSS news lookups to System 1."""

    def __init__(self, db_path: Optional[Path | str] = None):
        if db_path is None:
            resolved = find_default_newsboat_db()
            if resolved is None:
                raise FileNotFoundError(
                    "No valid Newsboat cache.db found in baton/data/rss/. "
                    "Run '.\\scripts\\reload_newsboat.bat' to fetch feeds first."
                )
            self.db_path = resolved
        else:
            self.db_path = Path(db_path)

    def _get_connection(self) -> sqlite3.Connection:
        """Connects in read-only immutable URI mode to prevent locking conflicts with Newsboat in WSL."""
        abs_path = self.db_path.resolve().as_posix()
        if abs_path.startswith("//"):
            uri = f"file:{abs_path}?mode=ro&immutable=1"
        else:
            uri = f"file:///{abs_path.lstrip('/')}?mode=ro&immutable=1"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 5000;")
            return conn
        except sqlite3.OperationalError:
            fallback_uri = f"file:{abs_path}?mode=ro"
            conn = sqlite3.connect(fallback_uri, uri=True, timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 5000;")
            return conn

    def list_feeds(self) -> list[dict[str, Any]]:
        """Lists all registered RSS feeds with total and unread article counts."""
        query = """
            SELECT
                f.title,
                f.rssurl,
                f.url,
                COUNT(i.id) AS total_count,
                SUM(CASE WHEN i.unread = 1 THEN 1 ELSE 0 END) AS unread_count
            FROM rss_feed f
            LEFT JOIN rss_item i ON f.rssurl = i.feedurl
            GROUP BY f.rssurl
            ORDER BY f.title COLLATE NOCASE ASC
        """
        with self._get_connection() as conn:
            rows = conn.execute(query).fetchall()
            return [
                {
                    "title": r["title"] or "Untitled Feed",
                    "feed_url": r["rssurl"],
                    "site_url": r["url"] or "",
                    "total_articles": r["total_count"] or 0,
                    "unread_articles": r["unread_count"] or 0,
                }
                for r in rows
            ]

    def get_latest_articles(
        self,
        feed_query: Optional[str] = None,
        unread_only: bool = False,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieves the newest articles, optionally filtered by feed name or unread state."""
        where_clauses = ["1=1"]
        params: list[Any] = []

        if feed_query:
            where_clauses.append("(f.title LIKE ? OR f.rssurl LIKE ?)")
            params.extend([f"%{feed_query}%", f"%{feed_query}%"])

        if unread_only:
            where_clauses.append("i.unread = 1")

        where_sql = " AND ".join(where_clauses)
        query = f"""
            SELECT
                i.id,
                i.title,
                f.title AS feed_title,
                i.author,
                i.pubDate,
                i.url,
                i.unread
            FROM rss_item i
            LEFT JOIN rss_feed f ON i.feedurl = f.rssurl
            WHERE {where_sql}
            ORDER BY i.pubDate DESC
            LIMIT ?
        """
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for r in rows:
                pub_ts = r["pubDate"]
                pub_str = datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d %H:%M") if pub_ts else "Unknown"
                results.append({
                    "id": r["id"],
                    "title": r["title"] or "Untitled",
                    "feed": r["feed_title"] or "Unknown Feed",
                    "author": r["author"] or "",
                    "published": pub_str,
                    "url": r["url"] or "",
                    "unread": bool(r["unread"]),
                })
            return results

    def search_articles(
        self,
        query: str,
        feed_query: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Full-text search across article titles and content bodies."""
        where_clauses = ["(i.title LIKE ? OR i.content LIKE ?)"]
        params: list[Any] = [f"%{query}%", f"%{query}%"]

        if feed_query:
            where_clauses.append("(f.title LIKE ? OR f.rssurl LIKE ?)")
            params.extend([f"%{feed_query}%", f"%{feed_query}%"])

        where_sql = " AND ".join(where_clauses)
        sql = f"""
            SELECT
                i.id,
                i.title,
                f.title AS feed_title,
                i.author,
                i.pubDate,
                i.url,
                i.content
            FROM rss_item i
            LEFT JOIN rss_feed f ON i.feedurl = f.rssurl
            WHERE {where_sql}
            ORDER BY i.pubDate DESC
            LIMIT ?
        """
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            results = []
            for r in rows:
                pub_ts = r["pubDate"]
                pub_str = datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d %H:%M") if pub_ts else "Unknown"
                cleaned = clean_html(r["content"] or "")
                snippet = cleaned[:250].replace("\n", " ").strip()
                results.append({
                    "id": r["id"],
                    "title": r["title"] or "Untitled",
                    "feed": r["feed_title"] or "Unknown Feed",
                    "author": r["author"] or "",
                    "published": pub_str,
                    "url": r["url"] or "",
                    "snippet": snippet + ("..." if len(cleaned) > 250 else ""),
                })
            return results

    def read_article(self, item_id: int, max_chars: int = 3500) -> dict[str, Any]:
        """Fetches and clean-formats the full text content of an article by its numeric ID."""
        query = """
            SELECT
                i.id,
                i.title,
                f.title AS feed_title,
                i.author,
                i.pubDate,
                i.url,
                i.content
            FROM rss_item i
            LEFT JOIN rss_feed f ON i.feedurl = f.rssurl
            WHERE i.id = ?
        """
        with self._get_connection() as conn:
            row = conn.execute(query, (item_id,)).fetchone()
            if not row:
                return {"error": f"Article with ID {item_id} not found."}

            pub_ts = row["pubDate"]
            pub_str = datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d %H:%M") if pub_ts else "Unknown"
            cleaned_text = clean_html(row["content"] or "")

            is_truncated = len(cleaned_text) > max_chars
            body = cleaned_text[:max_chars]

            return {
                "id": row["id"],
                "title": row["title"] or "Untitled",
                "feed": row["feed_title"] or "Unknown Feed",
                "author": row["author"] or "",
                "published": pub_str,
                "url": row["url"] or "",
                "content": body,
                "is_truncated": is_truncated,
                "total_chars": len(cleaned_text),
            }

    # -------------------------------------------------------------------------
    # String Formatting Dispatcher for System 1 Prompts
    # -------------------------------------------------------------------------

    def execute(self, tool_name: str, params: dict[str, Any]) -> str:
        """Deterministic dispatcher for named tool calls, returning clean Markdown."""
        name = tool_name.lower().strip()

        if name in ("rss_list_feeds", "list_feeds"):
            feeds = self.list_feeds()
            if not feeds:
                return "No RSS feeds found in Newsboat cache."
            lines = [f"# Subscribed RSS Feeds ({len(feeds)} total)"]
            for f in feeds:
                lines.append(
                    f"- **{f['title']}**: {f['total_articles']} articles ({f['unread_articles']} unread) | `{f['feed_url']}`"
                )
            return "\n".join(lines)

        elif name in ("rss_latest", "get_latest"):
            feed_q = params.get("feed")
            limit = int(params.get("limit", 10))
            unread = bool(params.get("unread_only", False))
            articles = self.get_latest_articles(feed_query=feed_q, unread_only=unread, limit=limit)
            if not articles:
                return "No articles found matching criteria."
            lines = [f"# Latest Articles (Top {len(articles)})"]
            for a in articles:
                lines.append(
                    f"- **[ID {a['id']}] {a['title']}**\n"
                    f"  *Feed: {a['feed']} | Date: {a['published']} | Author: {a['author'] or 'N/A'}*\n"
                    f"  Link: {a['url']}"
                )
            return "\n\n".join(lines)

        elif name in ("rss_search", "search_articles"):
            query = params.get("query", "")
            if not query:
                return "Error: Search query cannot be empty."
            feed_q = params.get("feed")
            limit = int(params.get("limit", 5))
            results = self.search_articles(query, feed_query=feed_q, limit=limit)
            if not results:
                return f"No articles found matching query '{query}'."
            lines = [f"# RSS Search Results for '{query}' ({len(results)} matches)"]
            for r in results:
                lines.append(
                    f"### [ID {r['id']}] {r['title']}\n"
                    f"*Feed: {r['feed']} | Date: {r['published']}*\n"
                    f"{r['snippet']}\n"
                    f"URL: {r['url']}"
                )
            return "\n\n".join(lines)

        elif name in ("rss_read_article", "read_article"):
            item_id = params.get("id") or params.get("item_id")
            if item_id is None:
                return "Error: Please specify the article ID (e.g. params={'id': 42})."
            try:
                item_id = int(item_id)
            except ValueError:
                return f"Error: Invalid article ID '{item_id}' (must be an integer)."

            art = self.read_article(item_id, max_chars=params.get("max_chars", 3500))
            if "error" in art:
                return f"Error: {art['error']}"

            trunc_msg = f"\n\n*(Truncated at {len(art['content'])} of {art['total_chars']} chars)*" if art["is_truncated"] else ""
            return (
                f"# {art['title']}\n"
                f"**Feed:** {art['feed']} | **Published:** {art['published']} | **Author:** {art['author'] or 'N/A'}\n"
                f"**Source URL:** {art['url']}\n\n"
                f"{art['content']}{trunc_msg}"
            )

        else:
            return f"Error: Unknown Newsboat tool '{tool_name}'."
