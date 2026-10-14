"""Unit tests for VaultTool (SQLite FTS5 + BM25 personal corpus search)."""

from pathlib import Path

import pytest
from baton.tools.vault import VaultTool, sanitize_fts5_query


def test_sanitize_fts5_query():
    assert sanitize_fts5_query("orbital mechanics") == '"orbital" AND "mechanics"'
    assert sanitize_fts5_query("python AND pydantic") == "python AND pydantic"
    assert sanitize_fts5_query("sdr -telemetry") == '"sdr" AND NOT telemetry'


def test_vault_indexing_and_bm25_search(tmp_path: Path):
    db_file = tmp_path / "vault.db"
    vault = VaultTool(db_path=db_file)

    # 1. Create mock notes
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()

    doc1 = notes_dir / "satellites.md"
    doc1.write_text(
        "# NOAA Weather Satellite Reception\n"
        "To receive NOAA 15, 18, and 19 APT transmissions on 137 MHz, use an RTL-SDR dongle "
        "and a horizontal V-dipole antenna cut to 53.4 cm per element.\n",
        encoding="utf-8",
    )

    doc2 = notes_dir / "power.md"
    doc2.write_text(
        "# Off-Grid Solar Power Budget\n"
        "Calculate daily watt-hour consumption and battery capacity. Use 100W mono solar panels "
        "with MPPT charge controllers.\n",
        encoding="utf-8",
    )

    # 2. Index folder
    res = vault.index_path(notes_dir, glob_pattern="*.md")
    assert res["scanned"] == 2
    assert res["indexed"] == 2
    assert res["errors"] == 0

    # 3. Test search
    matches = vault.search("NOAA satellite")
    assert len(matches) == 1
    assert matches[0]["filename"] == "satellites.md"
    assert "NOAA Weather Satellite Reception" in matches[0]["title"]
    assert "**satellite**" in matches[0]["snippet"].lower() or "noaa" in matches[0]["snippet"].lower()

    # 4. Test category search
    solar_matches = vault.search("solar MPPT")
    assert len(solar_matches) == 1
    assert solar_matches[0]["filename"] == "power.md"


def test_vault_direct_text_and_read(tmp_path: Path):
    db_file = tmp_path / "vault.db"
    vault = VaultTool(db_path=db_file)

    doc_id = vault.index_text(
        title="Antenna Impedance Notes",
        content="Coaxial feedline impedance is typically 50 ohms for RF transceivers and 75 ohms for TV cables.",
        category="rf",
    )
    assert doc_id > 0

    # Read document by ID
    content = vault.read_document(str(doc_id))
    assert "Antenna Impedance Notes" in content
    assert "50 ohms" in content


def test_vault_dispatcher(tmp_path: Path):
    db_file = tmp_path / "vault.db"
    vault = VaultTool(db_path=db_file)

    vault.index_text(
        title="Baton Design Specs",
        content="System 1 handles agile recon with Qwen while System 2 runs overnight synthesis with DeepSeek.",
        category="specs",
    )

    # Status
    st_res = vault.execute("vault_status", {})
    assert "Local Personal Knowledge Vault" in st_res
    assert "**Total Documents:** 1" in st_res

    # Search
    search_res = vault.execute("vault_search", {"query": "DeepSeek synthesis"})
    assert "Vault Search Results" in search_res
    assert "Baton Design Specs" in search_res

    # Read
    read_res = vault.execute("vault_read_document", {"doc_id_or_path": "Baton Design Specs"})
    assert "System 1 handles agile recon" in read_res
