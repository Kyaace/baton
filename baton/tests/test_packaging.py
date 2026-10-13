"""Unit tests for packaging, CLI entrypoints, and environment resolution."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from baton.cli import main
from baton.tools.sandbox import get_default_python_executable


def test_cli_entrypoint_callable():
    """Verify baton.cli.main is importable and callable."""
    assert callable(main)


def test_pyproject_toml_metadata():
    """Verify pyproject.toml defines script entrypoints and pydantic dependency."""
    pyproject_path = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml must exist in repo root"

    content = pyproject_path.read_text(encoding="utf-8")
    assert 'baton = "baton.cli:main"' in content, "baton CLI script entrypoint missing"
    assert "pydantic>=2.0.0" in content, "pydantic dependency missing from pyproject.toml"
    assert 'name = "baton"' in content, "project name must be baton"


def test_baton_cmd_portability():
    """Verify baton.cmd has multi-tiered Python resolution."""
    baton_cmd = Path(__file__).resolve().parent.parent.parent / "baton.cmd"
    assert baton_cmd.exists(), "baton.cmd must exist in repo root"

    content = baton_cmd.read_text(encoding="utf-8")
    assert "VIRTUAL_ENV" in content
    assert ".venv" in content
    assert "CONDA_PREFIX" in content
    assert "where python" in content
    assert ":check_python" in content


def test_get_default_python_executable_prefers_sys_executable():
    """Verify get_default_python_executable returns current Python when valid."""
    exe = get_default_python_executable()
    assert Path(exe).exists()
    assert exe == sys.executable


def test_get_default_python_executable_virtual_env_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify virtual environment resolution when sys.executable is empty."""
    fake_venv = tmp_path / "fake_venv"
    fake_scripts = fake_venv / ("Scripts" if os.name == "nt" else "bin")
    fake_scripts.mkdir(parents=True)
    fake_python = fake_scripts / ("python.exe" if os.name == "nt" else "python")
    fake_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(sys, "executable", "")
    monkeypatch.setenv("VIRTUAL_ENV", str(fake_venv))

    resolved = get_default_python_executable()
    assert resolved == str(fake_python)
