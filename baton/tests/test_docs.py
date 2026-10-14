"""Unit tests for DocsTool (Dash/Zeal offline API docsets)."""

import sqlite3
import zlib
from pathlib import Path

import pytest
from baton.tools.docs_tool import DocsTool, html_to_markdown


@pytest.fixture
def mock_docset(tmp_path: Path) -> Path:
    """Creates a temporary mock Python.docset for testing."""
    docset_dir = tmp_path / "python.docset"
    res_dir = docset_dir / "Contents" / "Resources"
    docs_dir = res_dir / "Documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize SQLite index
    db_file = res_dir / "docSet.dsidx"
    con = sqlite3.connect(db_file)
    with con:
        con.execute("CREATE TABLE searchIndex(id INTEGER PRIMARY KEY, name TEXT, type TEXT, path TEXT);")
        con.execute("CREATE UNIQUE INDEX anchor ON searchIndex (name, type, path);")
        con.execute(
            "INSERT INTO searchIndex (name, type, path) VALUES (?, ?, ?);",
            ("os.listdir", "Function", "os.html#os.listdir"),
        )
        con.execute(
            "INSERT INTO searchIndex (name, type, path) VALUES (?, ?, ?);",
            ("os.path.join", "Function", "os.path.html#os.path.join"),
        )
        con.execute(
            "INSERT INTO searchIndex (name, type, path) VALUES (?, ?, ?);",
            ("Path", "Class", "pathlib.html#pathlib.Path"),
        )
    con.close()

    # 2. Write HTML doc file
    html_content = """
    <!DOCTYPE html>
    <html>
    <head><title>os module</title></head>
    <body>
        <h1>os — Miscellaneous operating system interfaces</h1>
        <p>This module provides a portable way of using operating system dependent functionality.</p>
        <dl>
            <dt id="os.listdir"><code>os.listdir(path='.')</code></dt>
            <dd>Return a list containing the names of the entries in the directory given by path.</dd>
        </dl>
        <pre><code>def example():
    return os.listdir('.')
</code></pre>
    </body>
    </html>
    """
    (docs_dir / "os.html").write_text(html_content, encoding="utf-8")
    return docset_dir


def test_html_to_markdown_converter():
    raw_html = "<h1>Header</h1><p>Test paragraph with <code>code_var</code>.</p><pre>result = 42</pre>"
    md = html_to_markdown(raw_html)
    assert "# Header" in md
    assert "`code_var`" in md
    assert "result = 42" in md


def test_docs_tool_discovery_and_search(mock_docset: Path, tmp_path: Path):
    tool = DocsTool(docsets_dir=tmp_path)
    libs = tool.list_libraries()
    assert len(libs) >= 1
    python_lib = next(lib for lib in libs if lib["library"] == "python")
    assert python_lib["symbols_count"] == 3

    # Search for listdir
    matches = tool.search("python", "listdir")
    assert len(matches) == 1
    assert matches[0]["name"] == "os.listdir"
    assert matches[0]["type"] == "Function"
    assert "os.html" in matches[0]["path"]


def test_docs_tool_read_symbol(mock_docset: Path, tmp_path: Path):
    tool = DocsTool(docsets_dir=tmp_path)
    content = tool.read_symbol("python", "os.listdir")
    assert "Documentation: `os.listdir`" in content
    assert "os.listdir(path='.')" in content
    assert "Return a list containing the names of the entries" in content


def test_docs_tool_dispatcher(mock_docset: Path, tmp_path: Path):
    tool = DocsTool(docsets_dir=tmp_path)

    # 1. List libraries
    res = tool.execute("docs_list_libraries", {})
    assert "Available Offline API Documentation" in res or "Available Offline API Docsets" in res
    assert "python" in res

    # 2. Search
    res = tool.execute("docs_search", {"library": "python", "query": "path"})
    assert "Symbols in `python` matching 'path'" in res or "symbols matching 'path'" in res
    assert "os.path.join" in res

    # 3. Read symbol
    res = tool.execute("docs_read_symbol", {"library": "python", "symbol_or_path": "os.listdir"})
    assert "os.listdir" in res

    # 4. Unknown tool
    res = tool.execute("unknown_doc_tool", {})
    assert "Error: Unknown Docs tool" in res


@pytest.fixture
def mock_sphinx_zip(tmp_path: Path) -> Path:
    """Creates a temporary mock NumPy Sphinx documentation zip for testing."""
    zip_path = tmp_path / "numpy-html.zip"

    # 1. Build objects.inv payload
    inv_header = b"# Sphinx inventory version 2\n# Project: NumPy\n# Version: 1.26\n# The remainder of this file is compressed using zlib.\n"
    entries = [
        "numpy.zeros py:function 1 reference/generated/numpy.zeros.html#numpy.zeros -",
        "numpy.ones py:function 1 reference/generated/numpy.ones.html#numpy.ones -",
        "numpy.ndarray py:class 1 reference/generated/numpy.ndarray.html#numpy.ndarray -",
    ]
    decomp_body = "\n".join(entries).encode("utf-8")
    compressed_body = zlib.compress(decomp_body)
    objects_inv_bytes = inv_header + compressed_body

    html_zeros = """
    <!DOCTYPE html>
    <html>
    <head><title>numpy.zeros</title></head>
    <body>
        <article class="bd-article">
            <h1>numpy.zeros</h1>
            <p><strong>numpy.zeros(shape, dtype=float, order='C')</strong></p>
            <p>Return a new array of given shape and type, filled with zeros.</p>
        </article>
    </body>
    </html>
    """

    import zipfile
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("objects.inv", objects_inv_bytes)
        z.writestr("reference/generated/numpy.zeros.html", html_zeros.encode("utf-8"))

    return zip_path


def test_sphinx_zip_discovery_and_search(mock_sphinx_zip: Path, tmp_path: Path):
    tool = DocsTool(docs_dir=tmp_path)
    libs = tool.list_libraries()
    assert any(l["library"] == "numpy" and l["format"] == "sphinx_zip" for l in libs)

    # Search via canonical name
    matches = tool.search("numpy", "zeros")
    assert len(matches) >= 1
    assert matches[0]["name"] == "numpy.zeros"
    assert matches[0]["type"] == "py:function"
    assert "numpy.zeros.html" in matches[0]["path"]

    # Search via alias 'np'
    matches_alias = tool.search("np", "zeros")
    assert len(matches_alias) >= 1
    assert matches_alias[0]["name"] == "numpy.zeros"
    tool.close()


def test_sphinx_zip_read_symbol(mock_sphinx_zip: Path, tmp_path: Path):
    tool = DocsTool(docs_dir=tmp_path)
    content = tool.read_symbol("numpy", "zeros")
    assert "Documentation: `numpy.zeros`" in content
    assert "numpy.zeros" in content
    assert "filled with zeros" in content
    tool.close()
