"""Personal Local Corpus & Notes Search tool (SQLite FTS5 + BM25) for Baton System 1."""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("baton.tools.vault")


def sanitize_fts5_query(query: str) -> str:
    """Sanitizes user queries to avoid FTS5 syntax errors with special symbols."""
    # Strip dangerous characters that cause SQLite syntax errors
    clean = re.sub(r'[^\w\s"*\-]', " ", query)
    tokens = clean.split()
    if not tokens:
        return '""'

    # If query already has boolean operators (AND, OR, NOT), preserve
    has_operator = any(t in ("AND", "OR", "NOT") for t in tokens)
    if has_operator:
        return clean.strip()

    # Default to conjunction of terms or exact phrases
    sanitized_tokens = []
    for t in tokens:
        if t.startswith("-") and len(t) > 1:
            sanitized_tokens.append(f"NOT {t[1:]}")
        elif "*" in t:
            sanitized_tokens.append(t)
        else:
            sanitized_tokens.append(f'"{t}"')
    return " AND ".join(sanitized_tokens)


class VaultTool:
    """High-speed local document and personal notes search engine powered by SQLite FTS5 with BM25."""

    def __init__(self, db_path: Optional[Path | str] = None):
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            env_path = os.getenv("BATON_VAULT_PATH")
            if env_path:
                self.db_path = Path(env_path)
            else:
                pkg_data = Path(__file__).resolve().parent.parent / "data" / "vault"
                pkg_data.mkdir(parents=True, exist_ok=True)
                self.db_path = pkg_data / "vault.db"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path.as_posix(), timeout=5.0)
        con.execute("PRAGMA journal_mode = WAL;")
        con.execute("PRAGMA busy_timeout = 5000;")
        con.execute("PRAGMA synchronous = NORMAL;")
        return con

    def _init_db(self) -> None:
        """Initializes relational metadata and FTS5 full-text index tables."""
        con = self._get_connection()
        try:
            with con:
                con.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vault_documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        path TEXT UNIQUE NOT NULL,
                        filename TEXT NOT NULL,
                        title TEXT NOT NULL,
                        extension TEXT NOT NULL,
                        category TEXT NOT NULL,
                        word_count INTEGER NOT NULL,
                        mtime REAL NOT NULL,
                        content TEXT NOT NULL,
                        indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                con.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_vault_category ON vault_documents(category);
                    """
                )
                con.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS vault_fts USING fts5(
                        title,
                        content,
                        content='vault_documents',
                        content_rowid='id',
                        tokenize='porter unicode61'
                    );
                    """
                )
                con.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS vault_ai AFTER INSERT ON vault_documents BEGIN
                        INSERT INTO vault_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
                    END;
                    """
                )
                con.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS vault_ad AFTER DELETE ON vault_documents BEGIN
                        INSERT INTO vault_fts(vault_fts, rowid, title, content) VALUES('delete', old.id, old.title, old.content);
                    END;
                    """
                )
                con.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS vault_au AFTER UPDATE ON vault_documents BEGIN
                        INSERT INTO vault_fts(vault_fts, rowid, title, content) VALUES('delete', old.id, old.title, old.content);
                        INSERT INTO vault_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
                    END;
                    """
                )
        finally:
            con.close()

    def index_path(
        self,
        path: Path | str,
        glob_pattern: str = "*.md",
        category: Optional[str] = None,
        recursive: bool = True,
    ) -> dict[str, int]:
        """Indexes all text/markdown files under a directory or single file."""
        target = Path(path).resolve()
        if not target.exists():
            return {"error": f"Path '{path}' does not exist.", "scanned": 0, "indexed": 0, "updated": 0, "skipped": 0, "errors": 0}

        files_to_index: list[Path] = []
        if target.is_file():
            files_to_index.append(target)
        elif target.is_dir():
            pattern = f"**/{glob_pattern}" if recursive else glob_pattern
            files_to_index = [f for f in target.glob(pattern) if f.is_file()]

        scanned = len(files_to_index)
        indexed = 0
        updated = 0
        skipped = 0
        errors = 0

        con = self._get_connection()
        try:
            cur = con.cursor()
            for f in files_to_index:
                try:
                    stat = f.stat()
                    mtime = stat.st_mtime
                    path_str = str(f)

                    # Check if already up-to-date
                    cur.execute("SELECT id, mtime FROM vault_documents WHERE path = ?;", (path_str,))
                    row = cur.fetchone()
                    if row and abs(row[1] - mtime) < 0.001:
                        skipped += 1
                        continue

                    # Read document text
                    raw_content = f.read_text(encoding="utf-8", errors="replace")
                    word_count = len(raw_content.split())
                    title = self._extract_title(f.name, raw_content)
                    ext = f.suffix.lower()
                    doc_category = category or self._infer_category(f)

                    if row:
                        doc_id = row[0]
                        with con:
                            con.execute(
                                """
                                UPDATE vault_documents
                                SET filename = ?, title = ?, extension = ?, category = ?,
                                    word_count = ?, mtime = ?, content = ?, indexed_at = CURRENT_TIMESTAMP
                                WHERE id = ?;
                                """,
                                (f.name, title, ext, doc_category, word_count, mtime, raw_content, doc_id),
                            )
                        updated += 1
                    else:
                        with con:
                            con.execute(
                                """
                                INSERT INTO vault_documents (path, filename, title, extension, category, word_count, mtime, content)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                                """,
                                (path_str, f.name, title, ext, doc_category, word_count, mtime, raw_content),
                            )
                        indexed += 1
                except Exception as e:
                    logger.warning("Error indexing file %s: %s", f, e)
                    errors += 1
        finally:
            con.close()

        return {
            "scanned": scanned,
            "indexed": indexed,
            "updated": updated,
            "skipped": skipped,
            "errors": errors,
        }

    def index_text(
        self,
        title: str,
        content: str,
        category: str = "notes",
        path: Optional[str] = None,
    ) -> int:
        """Indexes an in-memory document or past synthesis artifact."""
        con = self._get_connection()
        try:
            doc_path = path or f"vault://{category}/{title.lower().replace(' ', '_')}.md"
            word_count = len(content.split())
            with con:
                cur = con.cursor()
                cur.execute(
                    """
                    INSERT INTO vault_documents (path, filename, title, extension, category, word_count, mtime, content)
                    VALUES (?, ?, ?, '.md', ?, ?, 0.0, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        title = excluded.title,
                        word_count = excluded.word_count,
                        content = excluded.content,
                        indexed_at = CURRENT_TIMESTAMP;
                    """,
                    (doc_path, f"{title}.md", title, category, word_count, content),
                )
                return cur.lastrowid or 0
        finally:
            con.close()

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Executes a BM25-ranked full-text search across the vault using FTS5."""
        clean_q = sanitize_fts5_query(query)
        con = self._get_connection()
        try:
            cur = con.cursor()
            sql = """
                SELECT d.id, d.path, d.filename, d.title, d.category, d.word_count,
                       snippet(vault_fts, 1, '**', '**', '...', 25) AS snippet,
                       bm25(vault_fts) AS rank
                FROM vault_fts f
                JOIN vault_documents d ON f.rowid = d.id
                WHERE vault_fts MATCH ?
            """
            params: list[Any] = [clean_q]
            if category:
                sql += " AND lower(d.category) = lower(?)"
                params.append(category)

            sql += " ORDER BY rank ASC LIMIT ?;"
            params.append(limit)

            try:
                cur.execute(sql, params)
                rows = cur.fetchall()
            except sqlite3.OperationalError:
                # If query syntax failed, fallback to plain word search
                fallback_q = ' '.join(f'"{w}"' for w in query.split() if w.isalnum())
                params[0] = fallback_q
                cur.execute(sql, params)
                rows = cur.fetchall()

            results = []
            for r in rows:
                results.append({
                    "id": r[0],
                    "path": r[1],
                    "filename": r[2],
                    "title": r[3],
                    "category": r[4],
                    "word_count": r[5],
                    "snippet": r[6],
                    "bm25_score": round(float(r[7]), 3),
                })
            return results
        finally:
            con.close()

    def format_search_markdown(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5,
    ) -> str:
        """Runs search and returns clean Markdown results with snippets and highlights."""
        results = self.search(query=query, category=category, limit=limit)
        if not results:
            cat_note = f" in category `{category}`" if category else ""
            return f"No documents in local vault matched query: '{query}'{cat_note}."

        lines = [f"### Vault Search Results for '{query}' ({len(results)} matches):"]
        for r in results:
            clean_snip = r["snippet"].replace("\n", " ").strip()
            lines.append(
                f"#### [{r['id']}] {r['title']} (`{r['filename']}` | {r['category']} | {r['word_count']} words)\n"
                f"- **Source:** `{r['path']}` (Score: {r['bm25_score']})\n"
                f"- **Snippet:** {clean_snip}\n"
            )
        lines.append(f"*Use `vault_read_document(doc_id_or_path='{results[0]['id']}')` to view full text.*")
        return "\n".join(lines)

    def read_document(self, doc_id_or_path: str, max_chars: int = 3500) -> str:
        """Retrieves document content by numeric ID or path."""
        con = self._get_connection()
        try:
            cur = con.cursor()
            if str(doc_id_or_path).isdigit():
                cur.execute("SELECT id, title, path, category, content FROM vault_documents WHERE id = ?;", (int(doc_id_or_path),))
            else:
                cur.execute(
                    """
                    SELECT id, title, path, category, content FROM vault_documents
                    WHERE lower(title) = lower(?)
                       OR lower(filename) = lower(?)
                       OR path LIKE ?
                       OR title LIKE ?
                       OR filename LIKE ?
                    LIMIT 1;
                    """,
                    (doc_id_or_path, doc_id_or_path, f"%{doc_id_or_path}%", f"%{doc_id_or_path}%", f"%{doc_id_or_path}%"),
                )
            row = cur.fetchone()
            if not row:
                return f"Document '{doc_id_or_path}' not found in vault database."

            doc_id, title, path, cat, content = row
            if len(content) > max_chars:
                content = content[:max_chars].rstrip() + f"\n\n... [Truncated at {max_chars} chars. Consult `{path}` for complete document]"

            return f"## [{doc_id}] {title} (Category: {cat})\n*Path: `{path}`*\n\n{content}"
        finally:
            con.close()

    def get_status(self) -> dict[str, Any]:
        """Returns statistics on the local vault database."""
        con = self._get_connection()
        try:
            cur = con.cursor()
            cur.execute("SELECT count(*), coalesce(sum(word_count), 0) FROM vault_documents;")
            doc_count, word_count = cur.fetchone() or (0, 0)

            cur.execute("SELECT category, count(*) FROM vault_documents GROUP BY category ORDER BY count(*) DESC;")
            categories = dict(cur.fetchall())

            db_size_kb = round(self.db_path.stat().st_size / 1024, 1) if self.db_path.exists() else 0.0

            return {
                "db_path": str(self.db_path),
                "documents_count": doc_count,
                "words_indexed": word_count,
                "categories": categories,
                "db_size_kb": db_size_kb,
            }
        finally:
            con.close()

    def format_status_markdown(self) -> str:
        """Formats vault statistics as Markdown."""
        st = self.get_status()
        lines = [
            "### Local Personal Knowledge Vault (SQLite FTS5)",
            f"- **Database Path:** `{st['db_path']}` ({st['db_size_kb']} KB)",
            f"- **Total Documents:** {st['documents_count']:,}",
            f"- **Total Words Indexed:** {st['words_indexed']:,}",
            "",
            "#### Categories Breakdown:",
        ]
        if st["categories"]:
            for cat, count in st["categories"].items():
                lines.append(f"- **`{cat}`:** {count} document(s)")
        else:
            lines.append("- *(No documents indexed yet)*")
        return "\n".join(lines)

    def execute(self, tool_name: str, params: dict[str, Any]) -> str:
        """Deterministic dispatcher for named tool calls."""
        name = tool_name.lower().strip()
        if name in ("vault_search", "search_vault", "vault"):
            return self.format_search_markdown(
                query=params.get("query", ""),
                category=params.get("category"),
                limit=params.get("limit", 5),
            )
        elif name in ("vault_read_document", "read_document", "vault_read"):
            return self.read_document(
                doc_id_or_path=params.get("doc_id_or_path") or params.get("id") or params.get("path", ""),
                max_chars=params.get("max_chars", 3500),
            )
        elif name in ("vault_index_path", "index_path", "vault_index"):
            res = self.index_path(
                path=params.get("path", "."),
                glob_pattern=params.get("glob_pattern", "*.md"),
                category=params.get("category"),
                recursive=params.get("recursive", True),
            )
            return (
                f"✅ **Vault Indexing Complete:**\n"
                f"- Scanned: {res['scanned']} | Indexed: {res['indexed']} | Updated: {res['updated']} | Skipped: {res['skipped']} | Errors: {res['errors']}"
            )
        elif name in ("vault_status", "status_vault"):
            return self.format_status_markdown()
        else:
            return f"Error: Unknown Vault tool '{tool_name}'."

    @staticmethod
    def _extract_title(filename: str, content: str) -> str:
        # Check first level 1 heading
        m = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
        if m:
            return m.group(1).strip()
        # Fallback to filename stem
        return Path(filename).stem.replace("_", " ").replace("-", " ").title()

    @staticmethod
    def _infer_category(path: Path) -> str:
        ext = path.suffix.lower()
        if ext in (".py", ".js", ".ts", ".rs", ".cpp", ".c", ".go"):
            return "code"
        if ext in (".md", ".txt", ".rst"):
            parent = path.parent.name.lower()
            if parent in ("docs", "doc", "notes", "archive", "specs"):
                return parent
            return "notes"
        if ext == ".json":
            return "data"
        if ext == ".pdf":
            return "papers"
        return "general"
