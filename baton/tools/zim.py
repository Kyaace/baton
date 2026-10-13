"""Kiwix ZIM Wikipedia offline knowledge tool for Baton System 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from smart_search.zim_reader import ZimReader


def find_default_zim() -> Optional[Path]:
    """Locate the largest available Wikipedia ZIM archive."""
    candidates = [
        Path("smart_search/wikipedia/wikipedia_en_all_nopic_2026-06.zim"),
        Path("smart_search/wikipedia/wikipedia_en_movies_nopic_2026-07.zim"),
        Path("E:/offline_data/zim/wikipedia_en_all_maxi.zim"),
        Path("D:/offline_data/zim/wikipedia_en_all_maxi.zim"),
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 1024 * 1024:
            return p

    # Fallback: scan smart_search/wikipedia
    wiki_dir = Path("smart_search/wikipedia")
    if wiki_dir.exists():
        zims = sorted(
            [f for f in wiki_dir.glob("*.zim") if f.stat().st_size > 1024 * 1024],
            key=lambda x: x.stat().st_size,
            reverse=True,
        )
        if zims:
            return zims[0]
    return None


class ZimTool:
    """Provides high-throughput offline knowledge lookups to System 1."""

    def __init__(self, zim_path: Optional[Path | str] = None):
        if zim_path is None:
            resolved = find_default_zim()
            if resolved is None:
                raise FileNotFoundError("No valid .zim file found in repository.")
            self.zim_path = resolved
        else:
            self.zim_path = Path(zim_path)

        self._reader: Optional[ZimReader] = None

    @property
    def reader(self) -> ZimReader:
        if self._reader is None:
            self._reader = ZimReader(self.zim_path)
        return self._reader

    def search(self, query: str, limit: int = 5) -> str:
        """Full-text search in Wikipedia archive."""
        results = self.reader.search(query, limit=limit)
        if not results:
            # Try title suggestions
            suggestions = self.reader.suggest_titles(query, limit=limit)
            if suggestions:
                return "No exact full-text matches. Title suggestions:\n" + "\n".join(
                    f"- {s}" for s in suggestions
                )
            return f"No results found for query: '{query}'"

        lines = [f"Found {len(results)} matches for '{query}':"]
        for r in results:
            snippet = r.get("snippet", "").strip().replace("\n", " ")
            lines.append(f"### {r['title']}\nSnippet: {snippet[:200]}...")
        return "\n\n".join(lines)

    def read_lead(self, title: str) -> str:
        """Read lead introductory section of an article."""
        lead = self.reader.read_lead(title)
        if not lead:
            return f"Article '{title}' has no lead text or was not found."
        return f"# {title} (Lead Section)\n\n{lead}"

    def read_section(self, title: str, section: str) -> str:
        """Extract a specific named section from an article."""
        content = self.reader.read_section(title, section)
        if not content:
            return f"Section '{section}' in article '{title}' was empty or not found."
        return f"# {title} - {section}\n\n{content}"

    def get_infobox(self, title: str) -> str:
        """Extract structured infobox key-value constants."""
        info = self.reader.get_infobox_markdown(title)
        if not info:
            return f"No infobox found for '{title}'."
        return f"# {title} (Infobox)\n\n{info}"

    def get_toc(self, title: str) -> str:
        """Get table of contents / sections for an article."""
        toc = self.reader.get_toc(title)
        if not toc:
            return f"No table of contents available for '{title}'."
        lines = [f"# {title} (Table of Contents)"]
        for item in toc:
            indent = "  " * (item.get("level", 2) - 2)
            h_text = item.get("heading") or item.get("title", "")
            lines.append(f"{indent}- {h_text}")
        return "\n".join(lines)

    def execute(self, tool_name: str, params: dict[str, Any]) -> str:
        """Deterministic dispatcher for named tool calls."""
        name = tool_name.lower().strip()
        if name in ("wiki_search", "search"):
            return self.search(query=params.get("query", ""), limit=params.get("limit", 5))
        elif name in ("wiki_read_lead", "read_lead"):
            return self.read_lead(title=params.get("title", ""))
        elif name in ("wiki_read_section", "read_section"):
            return self.read_section(
                title=params.get("title", ""), section=params.get("section", "")
            )
        elif name in ("wiki_infobox", "infobox"):
            return self.get_infobox(title=params.get("title", ""))
        elif name in ("wiki_toc", "toc"):
            return self.get_toc(title=params.get("title", ""))
        else:
            return f"Error: Unknown Zim tool '{tool_name}'."
