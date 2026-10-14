"""Offline API & Technical Documentation tool (Dash / Zeal docsets) for Baton System 1."""

from __future__ import annotations

import html
import logging
import os
import re
import sqlite3
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("baton.tools.docs")


class HTMLToMarkdownParser(HTMLParser):
    """Token-efficient converter from Docset HTML to clean Markdown."""

    def __init__(self) -> None:
        super().__init__()
        self.output: list[str] = []
        self._in_pre = False
        self._in_code = False
        self._ignore_stack: list[str] = []
        self._current_tag = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag_lower = tag.lower()
        self._current_tag = tag_lower
        if tag_lower in ("script", "style", "nav", "footer", "header", "svg", "noscript"):
            self._ignore_stack.append(tag_lower)
            return

        if self._ignore_stack:
            return

        if tag_lower in ("pre",):
            self._in_pre = True
            self.output.append("\n```python\n")
        elif tag_lower == "code" and not self._in_pre:
            self._in_code = True
            self.output.append("`")
        elif tag_lower in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag_lower[1])
            self.output.append(f"\n\n{'#' * min(level, 4)} ")
        elif tag_lower in ("p", "div", "section", "article"):
            self.output.append("\n\n")
        elif tag_lower == "li":
            self.output.append("\n- ")
        elif tag_lower == "dt":
            self.output.append("\n\n**")
        elif tag_lower == "dd":
            self.output.append(":** ")
        elif tag_lower == "br":
            self.output.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if self._ignore_stack:
            if tag_lower == self._ignore_stack[-1]:
                self._ignore_stack.pop()
            return

        if tag_lower == "pre":
            self._in_pre = False
            self.output.append("\n```\n\n")
        elif tag_lower == "code" and not self._in_pre:
            self._in_code = False
            self.output.append("`")
        elif tag_lower == "dt":
            self.output.append("**")
        elif tag_lower in ("p", "div"):
            self.output.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignore_stack:
            return
        if self._in_pre:
            self.output.append(data)
        else:
            # Compress multiple spaces into a single space
            clean = re.sub(r"[ \t]+", " ", data)
            self.output.append(clean)

    def get_markdown(self) -> str:
        raw = "".join(self.output)
        # Clean up excessive newlines
        clean = re.sub(r"\n{3,}", "\n\n", raw)
        return html.unescape(clean).strip()


def html_to_markdown(html_content: str) -> str:
    """Converts HTML documentation into token-dense clean Markdown."""
    parser = HTMLToMarkdownParser()
    try:
        parser.feed(html_content)
        parser.close()
        return parser.get_markdown()
    except Exception as e:
        logger.debug("HTML parsing fallback: %s", e)
        # Fallback regex stripper
        clean = re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r"<style[^>]*>.*?</style>", "", clean, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r"<[^>]+>", " ", clean)
        clean = re.sub(r"\s+", " ", clean)
        return html.unescape(clean).strip()


class DocsTool:
    """Offline API & Technical Documentation tool backed by Dash/Zeal docsets."""

    def __init__(self, docsets_dir: Optional[Path | str] = None):
        self.docsets_dir = Path(docsets_dir) if docsets_dir else None
        self._docsets: dict[str, Path] = {}
        self.refresh_docsets()

    def refresh_docsets(self) -> dict[str, Path]:
        """Discovers all available .docset folders across configured directories."""
        self._docsets.clear()
        search_dirs: list[Path] = []

        if self.docsets_dir:
            search_dirs.append(self.docsets_dir)

        env_dir = os.getenv("BATON_DOCSETS_DIR")
        if env_dir:
            search_dirs.append(Path(env_dir))

        pkg_root = Path(__file__).resolve().parent.parent
        search_dirs.extend([
            pkg_root / "data" / "docsets",
            Path("baton/data/docsets"),
            Path("data/docsets"),
            Path("E:/offline_data/docsets"),
            Path("D:/offline_data/docsets"),
        ])

        for s_dir in search_dirs:
            if not s_dir.exists():
                continue
            try:
                for item in s_dir.iterdir():
                    if item.is_dir() and item.name.lower().endswith(".docset"):
                        # Extract library name (e.g. 'python.docset' -> 'python', 'NumPy.docset' -> 'numpy')
                        lib_name = item.name[:-7].lower()
                        # Ensure it has docSet.dsidx
                        idx_file = item / "Contents" / "Resources" / "docSet.dsidx"
                        if idx_file.exists():
                            self._docsets[lib_name] = item
            except (OSError, PermissionError):
                continue

        return self._docsets

    def register_docset(self, name: str, path: Path | str) -> None:
        """Manually registers a .docset directory."""
        p = Path(path)
        self._docsets[name.lower().strip()] = p

    def list_libraries(self) -> list[dict[str, Any]]:
        """Lists available offline docsets with total indexed symbol counts."""
        self.refresh_docsets()
        res: list[dict[str, Any]] = []

        for name, d_path in sorted(self._docsets.items()):
            idx_file = d_path / "Contents" / "Resources" / "docSet.dsidx"
            count = 0
            if idx_file.exists():
                try:
                    con = sqlite3.connect(f"file:{idx_file.as_posix()}?mode=ro", uri=True, timeout=3.0)
                    cur = con.cursor()
                    cur.execute("SELECT count(*) FROM searchIndex;")
                    row = cur.fetchone()
                    count = row[0] if row else 0
                    con.close()
                except Exception as e:
                    logger.debug("Could not count symbols in %s: %s", idx_file, e)

            res.append({
                "library": name,
                "name": d_path.name,
                "path": str(d_path),
                "symbols_count": count,
            })
        return res

    def format_libraries_markdown(self) -> str:
        """Formats available docsets list as Markdown."""
        libs = self.list_libraries()
        if not libs:
            return (
                "### Offline Documentation Libraries (0 available)\n"
                "No `.docset` bundles detected. Drop Dash/Zeal docsets into `baton/data/docsets/` "
                "or set `BATON_DOCSETS_DIR`."
            )
        lines = [
            f"### Available Offline API Docsets ({len(libs)} libraries)",
            "| Library | Docset Folder | Indexed Symbols |",
            "| :--- | :--- | :--- |",
        ]
        for lib in libs:
            lines.append(f"| `{lib['library']}` | `{lib['name']}` | {lib['symbols_count']:,} |")
        return "\n".join(lines)

    def search(
        self,
        library: str,
        query: str,
        symbol_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Searches the SQLite index of the specified docset for matching symbols."""
        lib_key = library.lower().strip()
        if lib_key not in self._docsets:
            self.refresh_docsets()
        if lib_key not in self._docsets:
            available = ", ".join(sorted(self._docsets.keys())) or "none"
            raise KeyError(f"Docset library '{library}' not found. Available libraries: {available}")

        idx_file = self._docsets[lib_key] / "Contents" / "Resources" / "docSet.dsidx"
        if not idx_file.exists():
            raise FileNotFoundError(f"Missing docSet.dsidx in {self._docsets[lib_key]}")

        con = sqlite3.connect(f"file:{idx_file.as_posix()}?mode=ro", uri=True, timeout=5.0)
        cur = con.cursor()

        sql = "SELECT name, type, path FROM searchIndex WHERE name LIKE ?"
        params: list[Any] = [f"%{query}%"]
        if symbol_type:
            sql += " AND lower(type) = lower(?)"
            params.append(symbol_type)

        # Prioritize exact prefix match, then shorter length
        sql += " ORDER BY CASE WHEN lower(name) = lower(?) THEN 0 WHEN lower(name) LIKE ? THEN 1 ELSE 2 END, length(name) ASC, name ASC LIMIT ?"
        params.extend([query, f"{query}%", limit])

        cur.execute(sql, params)
        rows = cur.fetchall()
        con.close()

        results = []
        for name, stype, path in rows:
            results.append({
                "name": name,
                "type": stype,
                "path": path,
                "library": lib_key,
            })
        return results

    def format_search_markdown(
        self,
        library: str,
        query: str,
        symbol_type: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """Runs symbol search and formats results into Markdown."""
        try:
            results = self.search(library=library, query=query, symbol_type=symbol_type, limit=limit)
        except Exception as e:
            return f"Error searching docset '{library}': {e}"

        if not results:
            return f"No symbols found in `{library}` matching query '{query}'."

        lines = [
            f"### Symbols in `{library}` matching '{query}' ({len(results)} found):",
            "| Symbol | Type | Document Path |",
            "| :--- | :--- | :--- |",
        ]
        for r in results:
            lines.append(f"| `{r['name']}` | `{r['type']}` | `{r['path']}` |")
        lines.append(f"\n*Use `docs_read_symbol(library='{library}', symbol_or_path='...')` to read full documentation.*")
        return "\n".join(lines)

    def read_symbol(
        self,
        library: str,
        symbol_or_path: str,
        max_chars: int = 3500,
    ) -> str:
        """Extracts clean Markdown documentation for a symbol or relative document path."""
        lib_key = library.lower().strip()
        if lib_key not in self._docsets:
            self.refresh_docsets()
        if lib_key not in self._docsets:
            return f"Error: Docset library '{library}' not found."

        docset_dir = self._docsets[lib_key]
        docs_dir = docset_dir / "Contents" / "Resources" / "Documents"

        rel_path = symbol_or_path
        symbol_name = symbol_or_path
        symbol_type = ""

        # If symbol_or_path does not look like an html file, look it up in searchIndex
        is_direct_doc_file = (
            symbol_or_path.lower().endswith(".html")
            or symbol_or_path.lower().endswith(".htm")
            or ("/" in symbol_or_path and ".htm" in symbol_or_path.lower())
        )
        if not is_direct_doc_file:
            matches = self.search(library=lib_key, query=symbol_or_path, limit=1)
            if matches:
                rel_path = matches[0]["path"]
                symbol_name = matches[0]["name"]
                symbol_type = matches[0]["type"]
            else:
                return f"Symbol '{symbol_or_path}' not found in docset '{library}'."

        # Parse out any URL anchor (e.g. index.html#symbol_anchor)
        parsed = urllib.parse.urlparse(rel_path)
        file_path_part = urllib.parse.unquote(parsed.path)
        anchor = parsed.fragment

        target_file = (docs_dir / file_path_part).resolve()
        if not target_file.exists():
            # Try finding case-insensitively or searching Documents recursively
            candidates = list(docs_dir.glob(f"**/{Path(file_path_part).name}"))
            if candidates:
                target_file = candidates[0]
            else:
                return f"Documentation file `{file_path_part}` not found inside `{library}.docset`."

        try:
            raw_html = target_file.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Could not read documentation file `{target_file.name}`: {e}"

        # If an anchor was provided, extract content around the anchor if possible
        if anchor:
            anchor_pattern = re.compile(rf'(?:id|name)=["\']?{re.escape(anchor)}["\']?', re.IGNORECASE)
            m = anchor_pattern.search(raw_html)
            if m:
                # Slice from anchor onwards (up to 20,000 chars before markdown conversion)
                start_idx = max(0, m.start() - 200)
                raw_html = raw_html[start_idx : start_idx + 25000]

        md = html_to_markdown(raw_html)
        if len(md) > max_chars:
            md = md[:max_chars].rstrip() + f"\n\n... [Truncated at {max_chars} chars. Consult `{file_path_part}` for full reference]"

        type_header = f" (`{symbol_type}`)" if symbol_type else ""
        return (
            f"## Documentation: `{symbol_name}`{type_header} [{library}]\n"
            f"*Source: `{file_path_part}`*\n\n"
            f"{md}"
        )

    def get_signature(self, library: str, symbol_name: str) -> str:
        """Extracts the immediate signature and docstring summary for a symbol."""
        full_doc = self.read_symbol(library=library, symbol_or_path=symbol_name, max_chars=1800)
        return full_doc

    def execute(self, tool_name: str, params: dict[str, Any]) -> str:
        """Deterministic dispatcher for named tool calls."""
        name = tool_name.lower().strip()
        if name in ("docs_list_libraries", "list_libraries", "docs_libraries", "libraries"):
            return self.format_libraries_markdown()
        elif name in ("docs_search", "search_docs", "doc_search"):
            return self.format_search_markdown(
                library=params.get("library", ""),
                query=params.get("query", ""),
                symbol_type=params.get("symbol_type") or params.get("type"),
                limit=params.get("limit", 10),
            )
        elif name in ("docs_read_symbol", "read_symbol", "docs_read", "read_docs"):
            return self.read_symbol(
                library=params.get("library", ""),
                symbol_or_path=params.get("symbol_or_path") or params.get("symbol") or params.get("path", ""),
                max_chars=params.get("max_chars", 3500),
            )
        elif name in ("docs_get_signature", "get_signature", "docs_signature"):
            return self.get_signature(
                library=params.get("library", ""),
                symbol_name=params.get("symbol_name") or params.get("symbol", ""),
            )
        else:
            return f"Error: Unknown Docs tool '{tool_name}'."
