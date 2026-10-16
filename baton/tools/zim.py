"""Kiwix ZIM Multi-Corpus offline knowledge tool for Baton System 1."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

try:
    from baton.tools.zim_reader import ZimReader
except ImportError:
    from smart_search.zim_reader import ZimReader  # type: ignore

logger = logging.getLogger("baton.tools.zim")


def derive_corpus_alias(path: Path) -> str:
    """Derives a normalized corpus alias from a ZIM filename."""
    stem = path.stem.lower()

    # Specific domains & subtopics first
    if "medicine" in stem or "wikimed" in stem:
        return "wikimed"
    if "python" in stem:
        return "python"
    if "movies" in stem:
        return "movies"
    if "stackoverflow" in stem or "stackexchange" in stem:
        return "stackoverflow"
    if "wikivoyage" in stem or "voyage" in stem:
        return "wikivoyage"
    if "wikibooks" in stem:
        return "wikibooks"
    if "gutenberg" in stem:
        return "gutenberg"
    if "archlinux" in stem or "archwiki" in stem:
        return "archwiki"
    if "physics" in stem:
        return "physics"
    if "chemistry" in stem:
        return "chemistry"
    if "math" in stem:
        return "math"

    # General Wikipedia (e.g. wikipedia_en_all_*)
    if "wikipedia" in stem:
        m = re.match(r"wikipedia_[a-z]{2}_([a-z0-9]+)_", stem)
        if m and m.group(1) not in ("all", "maxi", "mini", "nopic"):
            return m.group(1)
        return "wikipedia"

    # Clean date or packaging stamps
    clean = re.sub(r"_\d{4}-\d{2}[a-z]?", "", stem)
    clean = re.sub(r"_(?:en|all|maxi|nopic|mini)", "", clean)
    clean = clean.replace("docs.", "").replace(".org", "")
    return clean or stem


def find_available_zims() -> dict[str, Path]:
    """Scans filesystem locations for all available .zim archives."""
    found: dict[str, Path] = {}

    # 1. Explicit env variables
    env_zim = os.getenv("BATON_ZIM_PATH")
    if env_zim:
        p = Path(env_zim)
        if p.exists() and p.stat().st_size > 1024 * 1024:
            found[derive_corpus_alias(p)] = p

    env_dir = os.getenv("BATON_ZIM_DIR")
    search_dirs: list[Path] = []
    if env_dir:
        search_dirs.append(Path(env_dir))

    pkg_root = Path(__file__).resolve().parent.parent
    search_dirs.extend([
        pkg_root / "data",
        pkg_root / "data" / "zim",
        pkg_root / "data" / "wikipedia",
        Path("baton/data"),
        Path("baton/data/zim"),
        Path("baton/data/wikipedia"),
        Path("data"),
        Path("data/zim"),
        Path("data/wikipedia"),
        Path("smart_search/zim"),
        Path("smart_search/wikipedia"),
        Path("E:/offline_data/zim"),
        Path("D:/offline_data/zim"),
        Path("E:/dev/Antigravity/local_llm/smart_search/wikipedia"),
    ])

    seen_files: set[Path] = set()
    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        try:
            # Recursively scan for any .zim files
            for f in s_dir.rglob("*.zim"):
                if f.is_file() and f not in seen_files and f.stat().st_size > 1024 * 1024:
                    seen_files.add(f)
                    alias = derive_corpus_alias(f)
                    if alias not in found or f.stat().st_size > found[alias].stat().st_size:
                        found[alias] = f
        except (OSError, PermissionError):
            continue

    return found


def find_default_zim() -> Optional[Path]:
    """Locate the largest available default Wikipedia ZIM archive (backwards compatibility)."""
    env_path = os.getenv("BATON_ZIM_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists() and p.stat().st_size > 1024 * 1024:
            return p

    available = find_available_zims()
    if not available:
        return None
    if "wikipedia" in available:
        return available["wikipedia"]
    return max(available.values(), key=lambda p: p.stat().st_size)


class ZimTool:
    """Provides high-throughput offline multi-corpus knowledge lookups to System 1."""

    def __init__(self, zim_path: Optional[Path | str] = None):
        self._corpora: dict[str, Path] = find_available_zims()
        self._readers: dict[str, ZimReader] = {}
        self._reader: Optional[ZimReader] = None

        if zim_path is not None:
            p = Path(zim_path)
            alias = derive_corpus_alias(p)
            self._corpora[alias] = p
            self.default_corpus = alias
            self.zim_path = p
        else:
            default_p = find_default_zim()
            if default_p is None and not self._corpora:
                raise FileNotFoundError("No valid .zim file found in repository.")
            self.default_corpus = derive_corpus_alias(default_p) if default_p else next(iter(self._corpora))
            self.zim_path = default_p or self._corpora[self.default_corpus]

    @property
    def reader(self) -> ZimReader:
        """Default reader for backwards compatibility."""
        return self.get_reader(self.default_corpus)

    @reader.setter
    def reader(self, val: ZimReader) -> None:
        self._reader = val
        if hasattr(self, "default_corpus") and self.default_corpus:
            self._readers[self.default_corpus] = val

    def resolve_corpus(self, corpus: Optional[str]) -> str:
        """Normalizes and resolves corpus names to an active archive key."""
        if not corpus:
            return self.default_corpus
        c = corpus.lower().strip()
        aliases = {
            "wiki": "wikipedia",
            "enwiki": "wikipedia",
            "so": "stackoverflow",
            "stack": "stackoverflow",
            "med": "wikimed",
            "medicine": "wikimed",
            "medical": "wikimed",
            "py": "python",
            "pydocs": "python",
            "python": "python",
            "voyage": "wikivoyage",
            "books": "wikibooks",
            "arch": "archwiki",
        }
        c = aliases.get(c, c)
        if c in self._corpora:
            return c
        for k in self._corpora:
            if c in k:
                return k
        return c

    def get_reader(self, corpus: Optional[str] = None) -> ZimReader:
        """Retrieves or lazily instantiates a ZimReader for the requested corpus."""
        target_corpus = self.resolve_corpus(corpus)
        if self._reader is not None and target_corpus == self.default_corpus:
            return self._reader

        if target_corpus not in self._readers:
            if target_corpus not in self._corpora:
                available = ", ".join(sorted(self._corpora.keys())) or "none"
                raise KeyError(f"Corpus '{corpus}' not found. Available corpora: {available}")
            self._readers[target_corpus] = ZimReader(self._corpora[target_corpus])
        return self._readers[target_corpus]

    def list_corpora(self) -> list[dict[str, Any]]:
        """Lists all mounted and discovered offline ZIM corpora."""
        res = []
        for alias, path in sorted(self._corpora.items()):
            size_mb = round(path.stat().st_size / (1024 * 1024), 1) if path.exists() else 0.0
            res.append({
                "corpus": alias,
                "filename": path.name,
                "path": str(path),
                "size_mb": size_mb,
                "is_default": (alias == self.default_corpus),
            })
        return res

    def format_corpora_markdown(self) -> str:
        """Formats the list of available corpora as Markdown."""
        corpora = self.list_corpora()
        if not corpora:
            return "No offline ZIM corpora currently detected."
        lines = [
            f"### Mounted Offline ZIM Corpora ({len(corpora)} available)",
            "| Corpus | Filename | Size (MB) | Status |",
            "| :--- | :--- | :--- | :--- |",
        ]
        for c in corpora:
            status = "[Default]" if c["is_default"] else "Mounted"
            lines.append(f"| `{c['corpus']}` | `{c['filename']}` | {c['size_mb']} MB | {status} |")
        return "\n".join(lines)

    def search(self, query: str, limit: int = 5, corpus: Optional[str] = None) -> str:
        """Full-text search in specified corpus or across all corpora."""
        if corpus == "all":
            all_matches: list[str] = []
            for c_alias in sorted(self._corpora.keys()):
                try:
                    r = self.get_reader(c_alias)
                    matches = r.search(query, limit=limit)
                    for m in matches:
                        snip = m.get("snippet", "").strip().replace("\n", " ")
                        all_matches.append(f"### [{c_alias}] {m['title']}\nSnippet: {snip[:200]}...")
                except Exception as e:
                    logger.debug("Error searching corpus '%s': %s", c_alias, e)
            if not all_matches:
                return f"No results found across any corpus for query: '{query}'"
            return f"Found {len(all_matches)} matches across corpora for '{query}':\n\n" + "\n\n".join(all_matches[: limit * 2])

        reader = self.get_reader(corpus)
        target_name = self.resolve_corpus(corpus)
        results = reader.search(query, limit=limit)
        if not results:
            suggestions = reader.suggest_titles(query, limit=limit)
            if suggestions:
                return f"[{target_name}] No exact full-text matches. Title suggestions:\n" + "\n".join(
                    f"- {s}" for s in suggestions
                )
            return f"No results found in [{target_name}] for query: '{query}'"

        if target_name == "wikipedia":
            lines = [f"Found {len(results)} matches for '{query}':"]
        else:
            lines = [f"Found {len(results)} matches in [{target_name}] for '{query}':"]
        for r in results:
            snippet = r.get("snippet", "").strip().replace("\n", " ")
            lines.append(f"### {r['title']}\nSnippet: {snippet[:200]}...")
        return "\n\n".join(lines)

    def read_lead(self, title: str, corpus: Optional[str] = None) -> str:
        """Read lead introductory section of an article from the target corpus."""
        reader = self.get_reader(corpus)
        c_name = self.resolve_corpus(corpus)
        lead = reader.read_lead(title)
        if not lead:
            return f"Article '{title}' has no lead text or was not found in [{c_name}]."
        corpus_note = f"\n*Corpus: {c_name}*" if c_name != "wikipedia" else ""
        return f"# {title} (Lead Section){corpus_note}\n\n{lead}"

    def read_section(self, title: str, section: str, corpus: Optional[str] = None) -> str:
        """Extract a specific named section from an article in the target corpus."""
        reader = self.get_reader(corpus)
        c_name = self.resolve_corpus(corpus)
        content = reader.read_section(title, section)
        if not content:
            return f"Section '{section}' in article '{title}' was empty or not found in [{c_name}]."
        corpus_note = f"\n*Corpus: {c_name}*" if c_name != "wikipedia" else ""
        return f"# {title} - {section}{corpus_note}\n\n{content}"

    def get_infobox(self, title: str, corpus: Optional[str] = None) -> str:
        """Extract structured infobox key-value constants from the target corpus."""
        reader = self.get_reader(corpus)
        c_name = self.resolve_corpus(corpus)
        info = reader.get_infobox_markdown(title)
        if not info:
            return f"No infobox found for '{title}' in [{c_name}]."
        corpus_note = f"\n*Corpus: {c_name}*" if c_name != "wikipedia" else ""
        return f"# {title} (Infobox){corpus_note}\n\n{info}"

    def get_toc(self, title: str, corpus: Optional[str] = None) -> str:
        """Get table of contents / sections for an article from the target corpus."""
        reader = self.get_reader(corpus)
        c_name = self.resolve_corpus(corpus)
        toc = reader.get_toc(title)
        if not toc:
            return f"No table of contents available for '{title}' in [{c_name}]."
        corpus_note = f"\n*Corpus: {c_name}*" if c_name != "wikipedia" else ""
        lines = [f"# {title} (Table of Contents){corpus_note}"]
        for item in toc:
            indent = "  " * (item.get("level", 2) - 2)
            h_text = item.get("heading") or item.get("title", "")
            lines.append(f"{indent}- {h_text}")
        return "\n".join(lines)

    def execute(self, tool_name: str, params: dict[str, Any]) -> str:
        """Deterministic dispatcher for named tool calls."""
        name = tool_name.lower().strip()
        corpus = params.get("corpus")
        title = (
            params.get("title")
            or params.get("page_title")
            or params.get("article")
            or params.get("query")
            or ""
        )
        section = (
            params.get("section")
            or params.get("heading")
            or params.get("section_title")
            or ""
        )
        if name in ("zim_list_corpora", "list_corpora", "corpora", "zim_corpora"):
            return self.format_corpora_markdown()
        elif name in ("wiki_search", "zim_search", "search"):
            return self.search(
                query=params.get("query") or params.get("search") or title,
                limit=params.get("limit", 5),
                corpus=corpus,
            )
        elif name in ("wiki_read_lead", "zim_read_lead", "read_lead"):
            return self.read_lead(title=title, corpus=corpus)
        elif name in ("wiki_read_section", "zim_read_section", "read_section"):
            if not section:
                return self.read_lead(title=title, corpus=corpus)
            res = self.read_section(
                title=title,
                section=section,
                corpus=corpus,
            )
            if "was empty or not found" in res:
                lead = self.read_lead(title=title, corpus=corpus)
                if lead and "not found" not in lead:
                    return f"{res}\n\nFalling back to lead section:\n\n{lead}"
            return res
        elif name in ("wiki_infobox", "zim_infobox", "infobox"):
            return self.get_infobox(title=title, corpus=corpus)
        elif name in ("wiki_toc", "zim_toc", "toc"):
            return self.get_toc(title=title, corpus=corpus)
        else:
            return f"Error: Unknown Zim tool '{tool_name}'."
