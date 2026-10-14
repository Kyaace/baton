"""Offline API & Technical Documentation tool for Baton System 1.

Supports:
1. Dash / Zeal .docset folders (SQLite docSet.dsidx + HTML documents).
2. Sphinx documentation .zip archives (objects.inv + in-place HTML documents).
"""

from __future__ import annotations

import html
import logging
import os
import re
import sqlite3
import urllib.parse
import zipfile
import zlib
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("baton.tools.docs")

DOCS_ALIASES: dict[str, str] = {
    "np": "numpy",
    "numpy": "numpy",
    "torch": "torch",
    "pytorch": "torch",
    "torchvision": "torchvision",
    "sklearn": "scikit-learn",
    "scikit-learn": "scikit-learn",
    "scikitlearn": "scikit-learn",
    "mpl": "matplotlib",
    "matplotlib": "matplotlib",
    "mathplotlib": "matplotlib",
}

LIB_NAME_FALLBACKS: dict[str, str] = {
    "mathplotlib.zip": "matplotlib",
    "numpy-html.zip": "numpy",
    "scikit-learn.zip": "scikit-learn",
    "torch.zip": "torch",
    "torchvision.zip": "torchvision",
}


class HTMLToMarkdownParser(HTMLParser):
    """Token-efficient converter from Docset / Sphinx HTML to clean Markdown."""

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
        if tag_lower in ("script", "style", "nav", "footer", "header", "svg", "noscript", "aside"):
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
            clean = re.sub(r"[ \t]+", " ", data)
            self.output.append(clean)

    def get_markdown(self) -> str:
        raw = "".join(self.output)
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
        clean = re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r"<style[^>]*>.*?</style>", "", clean, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r"<[^>]+>", " ", clean)
        clean = re.sub(r"\s+", " ", clean)
        return html.unescape(clean).strip()


def extract_main_html(raw_html: str, anchor: str = "") -> str:
    """Extracts the primary documentation container and slices around anchor if present."""
    if anchor:
        pattern = re.compile(rf'(?:id|name)=["\']?{re.escape(anchor)}["\']?', re.IGNORECASE)
        m = pattern.search(raw_html)
        if m:
            start = max(0, m.start() - 200)
            return raw_html[start : start + 30000]

    # Look for standard Sphinx article/main containers
    m_art = re.search(r'<article[^>]*class=["\'][^"\']*bd-article[^"\']*["\'][^>]*>(.*?)</article>', raw_html, re.DOTALL | re.IGNORECASE)
    if m_art:
        return m_art.group(1)

    m_main = re.search(r'(?:<article[^>]*>|<div[^>]*role=["\']main["\'][^>]*>|<main[^>]*>)(.*?)(?:</article>|</div>\s*</div>|</main>)', raw_html, re.DOTALL | re.IGNORECASE)
    if m_main:
        return m_main.group(1)

    return raw_html


SPHINX_LINE_RE = re.compile(r"^(.*?)\s+([a-zA-Z0-9_]+:[a-zA-Z0-9_]+)\s+(-?\d+)\s+([^\s]+)\s*(.*?)$")


class DocsTool:
    """Offline API & Technical Documentation tool backed by Dash/Zeal docsets and Sphinx .zip archives."""

    def __init__(self, docsets_dir: Optional[Path | str] = None, docs_dir: Optional[Path | str] = None):
        self.docsets_dir = Path(docsets_dir) if docsets_dir else None
        self.docs_dir = Path(docs_dir) if docs_dir else None
        self._docsets: dict[str, Path] = {}
        self._sphinx_zips: dict[str, dict[str, Any]] = {}
        self._zip_handles: dict[str, zipfile.ZipFile] = {}
        self._sqlite_cache_path: Optional[Path] = None
        self.refresh_docsets()

    def close(self) -> None:
        """Closes all cached ZipFile handles."""
        for z in self._zip_handles.values():
            try:
                z.close()
            except Exception:
                pass
        self._zip_handles.clear()

    def __del__(self) -> None:
        self.close()

    def _get_zip(self, zip_path: str) -> zipfile.ZipFile:
        """Returns a cached ZipFile handle for fast consecutive reads."""
        if zip_path not in self._zip_handles:
            self._zip_handles[zip_path] = zipfile.ZipFile(zip_path, "r")
        return self._zip_handles[zip_path]

    def _get_cache_db(self) -> sqlite3.Connection:
        """Returns a connection to the persistent SQLite index for Sphinx archives."""
        if self._sqlite_cache_path is None:
            candidate_dirs = []
            if self.docs_dir:
                candidate_dirs.append(self.docs_dir)
            pkg_root = Path(__file__).resolve().parent.parent
            candidate_dirs.extend([
                pkg_root / "data" / "docs",
                Path("baton/data/docs"),
                Path("data/docs"),
                Path.home() / ".cache" / "baton",
            ])
            for c in candidate_dirs:
                try:
                    c.mkdir(parents=True, exist_ok=True)
                    self._sqlite_cache_path = c / ".sphinx_index.sqlite"
                    break
                except Exception:
                    continue
            if self._sqlite_cache_path is None:
                self._sqlite_cache_path = Path(":memory:")

        con = sqlite3.connect(str(self._sqlite_cache_path), timeout=10.0)
        with con:
            con.execute("""
            CREATE TABLE IF NOT EXISTS sphinx_archives (
                zip_path TEXT PRIMARY KEY,
                library TEXT,
                title TEXT,
                version TEXT,
                mtime REAL,
                symbols_count INTEGER
            );
            """)
            con.execute("""
            CREATE TABLE IF NOT EXISTS sphinx_symbols (
                library TEXT,
                name TEXT,
                type TEXT,
                zip_path TEXT,
                file_path TEXT,
                anchor TEXT
            );
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_sphinx_lib_name ON sphinx_symbols (library, name);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_sphinx_name ON sphinx_symbols (name);")
        return con

    def refresh_docsets(self) -> dict[str, Any]:
        """Discovers all available .docset folders and Sphinx .zip documentation archives."""
        self._docsets.clear()
        self._sphinx_zips.clear()

        search_dirs: list[Path] = []
        if self.docsets_dir:
            search_dirs.append(self.docsets_dir)
        if self.docs_dir:
            search_dirs.append(self.docs_dir)

        env_dir = os.getenv("BATON_DOCSETS_DIR")
        if env_dir:
            search_dirs.append(Path(env_dir))
        env_docs_dir = os.getenv("BATON_DOCS_DIR")
        if env_docs_dir:
            search_dirs.append(Path(env_docs_dir))

        pkg_root = Path(__file__).resolve().parent.parent
        search_dirs.extend([
            pkg_root / "data" / "docs",
            pkg_root / "data" / "docsets",
            Path("baton/data/docs"),
            Path("baton/data/docsets"),
            Path("data/docs"),
            Path("data/docsets"),
            Path("E:/offline_data/docs"),
            Path("E:/offline_data/docsets"),
            Path("D:/offline_data/docs"),
            Path("D:/offline_data/docsets"),
        ])

        seen_zips: set[Path] = set()

        for s_dir in search_dirs:
            if not s_dir.exists():
                continue
            try:
                for item in s_dir.iterdir():
                    # 1. Dash / Zeal Docsets (.docset folders)
                    if item.is_dir() and item.name.lower().endswith(".docset"):
                        lib_name = item.name[:-7].lower()
                        if lib_name not in self._docsets:
                            idx_file = item / "Contents" / "Resources" / "docSet.dsidx"
                            if idx_file.exists():
                                self._docsets[lib_name] = item

                    # 2. Sphinx documentation zip files (.zip archives)
                    elif item.is_file() and item.name.lower().endswith(".zip"):
                        if item.resolve() in seen_zips:
                            continue
                        seen_zips.add(item.resolve())
                        self._index_sphinx_zip(item)
            except (OSError, PermissionError):
                continue

        return {"docsets": self._docsets, "sphinx_zips": self._sphinx_zips}

    def _index_sphinx_zip(self, zip_path: Path) -> None:
        """Checks if a zip contains Sphinx objects.inv and indexes its symbols incrementally."""
        try:
            cur_mtime = zip_path.stat().st_mtime
            con = self._get_cache_db()
            cur = con.cursor()
            cur.execute(
                "SELECT library, title, version, symbols_count, mtime FROM sphinx_archives WHERE zip_path = ?",
                (str(zip_path),)
            )
            row = cur.fetchone()

            if row and row[4] == cur_mtime:
                lib_name, title, version, count, _ = row
                if lib_name not in self._sphinx_zips:
                    self._sphinx_zips[lib_name] = {
                        "path": zip_path,
                        "title": title,
                        "version": version,
                        "symbols_count": count,
                    }
                con.close()
                return

            # Read objects.inv from zip
            with zipfile.ZipFile(zip_path, "r") as z:
                inv_list = [n for n in z.namelist() if n.endswith("objects.inv")]
                if not inv_list:
                    con.close()
                    return

                inv_path = inv_list[0]
                base_dir = inv_path[:-len("objects.inv")]
                raw = z.read(inv_path)
                lines = raw.split(b"\n", 4)
                title = lines[1].decode("utf-8", errors="ignore").replace("# Project:", "").strip()
                version = lines[2].decode("utf-8", errors="ignore").replace("# Version:", "").strip()
                decomp = zlib.decompress(lines[4]).decode("utf-8", errors="ignore")

                default_name = LIB_NAME_FALLBACKS.get(zip_path.name.lower(), title.lower().split()[0])
                lib_name = DOCS_ALIASES.get(default_name, default_name)

                cur.execute("DELETE FROM sphinx_symbols WHERE zip_path = ?", (str(zip_path),))

                batch = []
                for line in decomp.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    m = SPHINX_LINE_RE.match(line)
                    if not m:
                        continue
                    name, type_str, prio, uri, _ = m.groups()
                    if not uri or uri == "-1" or (".htm" not in uri.lower()):
                        continue
                    if uri.endswith("$"):
                        uri = uri[:-1] + name
                    file_part, _, anchor = uri.partition("#")
                    full_file = base_dir + file_part
                    batch.append((lib_name, name, type_str, str(zip_path), full_file, anchor))

                cur.executemany("INSERT INTO sphinx_symbols VALUES (?, ?, ?, ?, ?, ?)", batch)
                cur.execute(
                    "INSERT OR REPLACE INTO sphinx_archives VALUES (?, ?, ?, ?, ?, ?)",
                    (str(zip_path), lib_name, title, version, cur_mtime, len(batch)),
                )
                con.commit()
                con.close()

                self._sphinx_zips[lib_name] = {
                    "path": zip_path,
                    "title": title,
                    "version": version,
                    "symbols_count": len(batch),
                }
        except Exception as e:
            logger.debug("Failed indexing sphinx zip %s: %s", zip_path, e)

    def register_docset(self, name: str, path: Path | str) -> None:
        """Manually registers a .docset directory."""
        p = Path(path)
        self._docsets[name.lower().strip()] = p

    def list_libraries(self) -> list[dict[str, Any]]:
        """Lists available offline docsets and Sphinx archives with total indexed symbol counts."""
        self.refresh_docsets()
        res: list[dict[str, Any]] = []

        # 1. Dash/Zeal Docsets
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
                "format": "docset",
                "symbols_count": count,
            })

        # 2. Sphinx Zip Archives
        for name, info in sorted(self._sphinx_zips.items()):
            res.append({
                "library": name,
                "name": info["path"].name,
                "title": info["title"],
                "version": info["version"],
                "path": str(info["path"]),
                "format": "sphinx_zip",
                "symbols_count": info["symbols_count"],
            })

        return res

    def format_libraries_markdown(self) -> str:
        """Formats available docsets and Sphinx documentation archives as Markdown."""
        libs = self.list_libraries()
        if not libs:
            return (
                "### Offline Documentation Libraries (0 available)\n"
                "No documentation archives detected. Drop Dash/Zeal `.docset` folders or Sphinx `.zip` documentation archives into `baton/data/docs/`."
            )
        lines = [
            f"### Available Offline API Documentation ({len(libs)} libraries)",
            "| Library | Format | Archive / Folder | Indexed Symbols |",
            "| :--- | :--- | :--- | :--- |",
        ]
        for lib in libs:
            fmt = "Docset" if lib.get("format") == "docset" else f"Sphinx Zip ({lib.get('version', '')})"
            lines.append(f"| `{lib['library']}` | {fmt} | `{lib['name']}` | {lib['symbols_count']:,} |")
        return "\n".join(lines)

    def search(
        self,
        library: str,
        query: str,
        symbol_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Searches symbols across the specified library or across all libraries if library='all'."""
        raw_key = library.lower().strip()
        lib_key = DOCS_ALIASES.get(raw_key, raw_key)

        if not self._docsets and not self._sphinx_zips:
            self.refresh_docsets()

        # Handle cross-library search
        if lib_key in ("all", "*", ""):
            results = []
            for known_lib in sorted(list(self._docsets.keys()) + list(self._sphinx_zips.keys())):
                try:
                    sub_res = self.search(library=known_lib, query=query, symbol_type=symbol_type, limit=limit)
                    results.extend(sub_res)
                except Exception:
                    continue
            results.sort(key=lambda r: (0 if r["name"].lower() == query.lower() else 1, len(r["name"])))
            return results[:limit]

        # 1. Search Dash/Zeal Docset
        if lib_key in self._docsets:
            idx_file = self._docsets[lib_key] / "Contents" / "Resources" / "docSet.dsidx"
            if not idx_file.exists():
                raise FileNotFoundError(f"Missing docSet.dsidx in {self._docsets[lib_key]}")

            con = sqlite3.connect(f"file:{idx_file.as_posix()}?mode=ro", uri=True, timeout=5.0)
            cur = con.cursor()
            sql = "SELECT name, type, path FROM searchIndex WHERE name LIKE ?"
            params = [f"%{query}%"]
            if symbol_type:
                sql += " AND lower(type) = lower(?)"
                params.append(symbol_type)

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
                    "format": "docset",
                })
            return results

        # 2. Search Sphinx Zip archive
        if lib_key in self._sphinx_zips:
            con = self._get_cache_db()
            cur = con.cursor()
            sql = """
            SELECT name, type, file_path, anchor, library
            FROM sphinx_symbols
            WHERE library = ? AND name LIKE ?
            """
            params: list[Any] = [lib_key, f"%{query}%"]
            if symbol_type:
                sql += " AND lower(type) = lower(?)"
                params.append(symbol_type)

            sql += """
            ORDER BY
                CASE
                    WHEN lower(name) = lower(?) THEN 0
                    WHEN lower(name) LIKE lower(?) AND type LIKE 'py:%' THEN 1
                    WHEN lower(name) LIKE lower(?) AND type LIKE 'py:%' THEN 2
                    WHEN lower(name) LIKE lower(?) AND type LIKE 'py:%' THEN 3
                    WHEN type LIKE 'py:%' THEN 4
                    WHEN type LIKE 'c:%' THEN 5
                    ELSE 6
                END,
                length(name) ASC,
                name ASC
            LIMIT ?
            """
            p0 = query.lower()
            p1 = f"%.{p0}"
            p2 = f"{p0}.%"
            p3 = f"%{p0}%"
            params.extend([p0, p1, p2, p3, limit])
            cur.execute(sql, params)
            rows = cur.fetchall()
            con.close()

            results = []
            for name, stype, file_path, anchor, l_name in rows:
                path_str = f"{file_path}#{anchor}" if anchor else file_path
                results.append({
                    "name": name,
                    "type": stype,
                    "path": path_str,
                    "file_path": file_path,
                    "anchor": anchor,
                    "library": l_name,
                    "format": "sphinx_zip",
                })
            return results

        # Unknown library
        available = ", ".join(sorted(list(self._docsets.keys()) + list(self._sphinx_zips.keys()))) or "none"
        raise KeyError(f"Documentation library '{library}' not found. Available libraries: {available}")

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
            return f"Error searching documentation for '{library}': {e}"

        if not results:
            target = f"`{library}`" if library else "offline docs"
            return f"No symbols found in {target} matching query '{query}'."

        lines = [
            f"### Documentation symbols matching '{query}' ({len(results)} found):",
            "| Symbol | Library | Type | Document Path |",
            "| :--- | :--- | :--- | :--- |",
        ]
        for r in results:
            lines.append(f"| `{r['name']}` | `{r['library']}` | `{r['type']}` | `{r['path']}` |")
        lines.append(f"\n*Use `docs_read_symbol(library='<lib>', symbol_or_path='...')` to read full documentation.*")
        return "\n".join(lines)

    def read_symbol(
        self,
        library: str,
        symbol_or_path: str,
        max_chars: int = 3500,
    ) -> str:
        """Extracts clean Markdown documentation for a symbol or relative document path."""
        raw_key = library.lower().strip()
        lib_key = DOCS_ALIASES.get(raw_key, raw_key)

        if lib_key not in self._docsets and lib_key not in self._sphinx_zips:
            self.refresh_docsets()

        if lib_key not in self._docsets and lib_key not in self._sphinx_zips:
            return f"Error: Documentation library '{library}' not found."

        # Case A: Sphinx Zip Archive
        if lib_key not in self._docsets and lib_key in self._sphinx_zips:
            zip_info = self._sphinx_zips[lib_key]
            zip_path = str(zip_info["path"])

            # Resolve symbol metadata
            file_path = symbol_or_path
            anchor = ""
            symbol_name = symbol_or_path
            symbol_type = ""

            is_direct_doc_file = (
                symbol_or_path.lower().endswith(".html")
                or symbol_or_path.lower().endswith(".htm")
                or ("/" in symbol_or_path and ".htm" in symbol_or_path.lower())
            )
            if not is_direct_doc_file:
                matches = self.search(library=lib_key, query=symbol_or_path, limit=1)
                if matches:
                    file_path = matches[0].get("file_path", matches[0]["path"].split("#")[0])
                    anchor = matches[0].get("anchor", matches[0]["path"].split("#")[-1] if "#" in matches[0]["path"] else "")
                    symbol_name = matches[0]["name"]
                    symbol_type = matches[0]["type"]
                else:
                    return f"Symbol '{symbol_or_path}' not found in `{lib_key}`."
            else:
                if "#" in file_path:
                    file_path, _, anchor = file_path.partition("#")

            try:
                z = self._get_zip(zip_path)
                # Check file existence in zip
                target_file = file_path
                if target_file not in z.namelist():
                    # Case-insensitive or substring search inside zip
                    base_name = Path(file_path).name
                    candidates = [n for n in z.namelist() if n.endswith(base_name)]
                    if candidates:
                        target_file = candidates[0]
                    else:
                        return f"Documentation file `{file_path}` not found inside `{zip_info['path'].name}`."

                raw_bytes = z.read(target_file)
                raw_html = raw_bytes.decode("utf-8", errors="replace")
            except Exception as e:
                return f"Could not read `{file_path}` from `{zip_info['path'].name}`: {e}"

            # Extract main content and convert to markdown
            clean_html = extract_main_html(raw_html, anchor=anchor)
            md = html_to_markdown(clean_html)
            if len(md) > max_chars:
                md = md[:max_chars].rstrip() + f"\n\n... [Truncated at {max_chars} chars. Consult `{file_path}` for full reference]"

            type_header = f" (`{symbol_type}`)" if symbol_type else ""
            return (
                f"## Documentation: `{symbol_name}`{type_header} [{lib_key}]\n"
                f"*Source: `{file_path}`*\n\n"
                f"{md}"
            )

        # Case B: Dash/Zeal Docset
        docset_dir = self._docsets[lib_key]
        docs_dir = docset_dir / "Contents" / "Resources" / "Documents"

        rel_path = symbol_or_path
        symbol_name = symbol_or_path
        symbol_type = ""

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

        parsed = urllib.parse.urlparse(rel_path)
        file_path_part = urllib.parse.unquote(parsed.path)
        anchor = parsed.fragment

        target_file = (docs_dir / file_path_part).resolve()
        if not target_file.exists():
            candidates = list(docs_dir.glob(f"**/{Path(file_path_part).name}"))
            if candidates:
                target_file = candidates[0]
            else:
                return f"Documentation file `{file_path_part}` not found inside `{library}.docset`."

        try:
            raw_html = target_file.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Could not read documentation file `{target_file.name}`: {e}"

        clean_html = extract_main_html(raw_html, anchor=anchor)
        md = html_to_markdown(clean_html)
        if len(md) > max_chars:
            md = md[:max_chars].rstrip() + f"\n\n... [Truncated at {max_chars} chars. Consult `{file_path_part}` for full reference]"

        type_header = f" (`{symbol_type}`)" if symbol_type else ""
        return (
            f"## Documentation: `{symbol_name}`{type_header} [{lib_key}]\n"
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
