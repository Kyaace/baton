"""Unit tests for Baton EngineManager lifecycle and decoupling."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from baton.engine_manager import EngineManager
from baton.models import EngineStatus, Tier


def test_engine_manager_default_config():
    """Verify default configurations when no env vars are set."""
    with patch.dict(os.environ, {}, clear=True):
        em = EngineManager()
        assert em.ollama_url == "http://127.0.0.1:11434"
        assert em.colibri_url == "http://127.0.0.1:8000"
        assert em.ollama_start_cmd is None
        assert em.ollama_stop_cmd is None
        assert em.colibri_start_cmd is None
        assert em.colibri_stop_cmd is None
        assert em.active_tier is None


def test_engine_manager_env_var_overrides():
    """Verify environment variables properly configure endpoints and hooks."""
    env = {
        "BATON_OLLAMA_URL": "http://192.168.1.50:11434/",
        "BATON_COLIBRI_URL": "http://192.168.1.60:8000/",
        "BATON_OLLAMA_START_CMD": "systemctl start ollama",
        "BATON_OLLAMA_STOP_CMD": "systemctl stop ollama",
        "BATON_COLIBRI_START_CMD": "docker start colibri",
        "BATON_COLIBRI_STOP_CMD": "docker stop colibri",
    }
    with patch.dict(os.environ, env, clear=True):
        em = EngineManager()
        assert em.ollama_url == "http://192.168.1.50:11434"
        assert em.colibri_url == "http://192.168.1.60:8000"
        assert em.ollama_start_cmd == "systemctl start ollama"
        assert em.ollama_stop_cmd == "systemctl stop ollama"
        assert em.colibri_start_cmd == "docker start colibri"
        assert em.colibri_stop_cmd == "docker stop colibri"


def test_engine_manager_explicit_params_precedence():
    """Verify explicit constructor arguments override environment variables."""
    env = {
        "BATON_OLLAMA_URL": "http://env-ollama:11434",
        "BATON_COLIBRI_URL": "http://env-colibri:8000",
    }
    with patch.dict(os.environ, env, clear=True):
        em = EngineManager(
            ollama_url="http://custom-ollama:11434",
            colibri_url="http://custom-colibri:8000",
            ollama_start_cmd="custom_start_ollama",
            colibri_stop_cmd="custom_stop_colibri",
        )
        assert em.ollama_url == "http://custom-ollama:11434"
        assert em.colibri_url == "http://custom-colibri:8000"
        assert em.ollama_start_cmd == "custom_start_ollama"
        assert em.colibri_stop_cmd == "custom_stop_colibri"


@patch("requests.get")
def test_engine_manager_health_probes(mock_get):
    """Verify health probes return expected booleans based on HTTP responses."""
    em = EngineManager()

    # Both up
    mock_get.return_value = MagicMock(status_code=200)
    assert em.is_ollama_online() is True
    assert em.is_colibri_online() is True

    # Request exception (offline)
    mock_get.side_effect = requests.RequestException("Connection refused")
    assert em.is_ollama_online() is False
    assert em.is_colibri_online() is False


@patch("requests.get")
def test_engine_manager_get_status(mock_get):
    """Verify get_status builds a dictionary reflecting engine health."""
    mock_get.return_value = MagicMock(status_code=200)
    em = EngineManager()
    status = em.get_status()
    assert status["active_tier"] == "idle"
    assert status["ollama"]["online"] is True
    assert status["ollama"]["status"] == EngineStatus.READY.value
    assert status["colibri"]["online"] is True
    assert status["colibri"]["status"] == EngineStatus.READY.value


@patch("subprocess.run")
@patch.object(EngineManager, "is_ollama_online")
def test_engine_manager_ollama_lifecycle_with_custom_commands(mock_online, mock_subproc):
    """Verify configured commands are called when starting/stopping Ollama."""
    mock_online.side_effect = [False, True, False]
    em = EngineManager(
        ollama_start_cmd="echo starting ollama",
        ollama_stop_cmd="echo stopping ollama",
    )

    # Start
    res_start = em.start_ollama(timeout_sec=2)
    assert res_start is True
    mock_subproc.assert_called_with("echo starting ollama", cwd=str(em.repo_root), shell=True, check=False)

    # Stop
    res_stop = em.stop_ollama()
    assert res_stop is True
    mock_subproc.assert_called_with("echo stopping ollama", cwd=str(em.repo_root), shell=True, check=False)


@patch("subprocess.run")
@patch.object(EngineManager, "is_colibri_online")
def test_engine_manager_colibri_lifecycle_with_custom_commands(mock_online, mock_subproc):
    """Verify configured commands are called when starting/stopping Colibrì."""
    mock_online.side_effect = [False, True, False]
    em = EngineManager(
        colibri_start_cmd="echo starting colibri",
        colibri_stop_cmd="echo stopping colibri",
    )

    # Start
    res_start = em.start_colibri(timeout_sec=2)
    assert res_start is True
    mock_subproc.assert_called_with("echo starting colibri", cwd=str(em.repo_root), shell=True, check=False)

    # Stop
    res_stop = em.stop_colibri()
    assert res_stop is True
    mock_subproc.assert_called_with("echo stopping colibri", cwd=str(em.repo_root), shell=True, check=False)


@patch.object(EngineManager, "start_ollama", return_value=True)
@patch.object(EngineManager, "stop_ollama", return_value=True)
@patch.object(EngineManager, "start_colibri", return_value=True)
@patch.object(EngineManager, "stop_colibri", return_value=True)
def test_engine_manager_mode_a_mutual_exclusion(
    mock_stop_coli, mock_start_coli, mock_stop_ollama, mock_start_ollama
):
    """Verify Mode A switches between System 1 and System 2 mutually exclusively."""
    em = EngineManager(use_router=False)

    # Initial state: switch to System 1 while Colibrì is online
    with patch.object(em, "is_colibri_online", return_value=True), patch.object(
        em, "is_ollama_online", return_value=False
    ), patch.object(em, "is_router_online", return_value=False):
        ok = em.switch_to_tier(Tier.SYSTEM_1)
        assert ok is True
        assert em.active_tier == Tier.SYSTEM_1
        mock_stop_coli.assert_called_once()
        mock_start_ollama.assert_called_once()

    mock_stop_coli.reset_mock()
    mock_start_ollama.reset_mock()

    # Switch to System 2 while Ollama is online
    with patch.object(em, "is_colibri_online", return_value=False), patch.object(
        em, "is_ollama_online", return_value=True
    ), patch.object(em, "is_router_online", return_value=False):
        ok = em.switch_to_tier(Tier.SYSTEM_2)
        assert ok is True
        assert em.active_tier == Tier.SYSTEM_2
        mock_stop_ollama.assert_called_once()
        mock_start_coli.assert_called_once()

    # Stop all
    with patch.object(em, "is_router_online", return_value=False):
        em.stop_all()
    assert em.active_tier is None


@patch.object(EngineManager, "start_router", return_value=True)
def test_engine_manager_router_mode_tier_switching(mock_start_router):
    """Verify router mode provides zero-downtime tier switching without process restart."""
    em = EngineManager(use_router=True)

    with patch.object(em, "is_router_online", return_value=True):
        # Switching between tiers should not stop any server
        ok1 = em.switch_to_tier(Tier.SYSTEM_1)
        assert ok1 is True
        assert em.active_tier == Tier.SYSTEM_1
        mock_start_router.assert_not_called()

        ok2 = em.switch_to_tier(Tier.SYSTEM_2)
        assert ok2 is True
        assert em.active_tier == Tier.SYSTEM_2
        mock_start_router.assert_not_called()

    # If router is offline, switch_to_tier starts it
    with patch.object(em, "is_router_online", side_effect=[False, True]):
        ok3 = em.switch_to_tier(Tier.SYSTEM_1)
        assert ok3 is True
        mock_start_router.assert_called_once()


@patch("subprocess.run")
@patch.object(EngineManager, "is_colibri_online")
def test_engine_manager_prioritizes_ornith_script(mock_online, mock_subproc, tmp_path: Path):
    """Verify that EngineManager prioritizes run_ornith.bat as the default System 2 engine."""
    mock_online.side_effect = [False, True, False]
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    ornith_bat = scripts_dir / "run_ornith.bat"
    ornith_bat.write_text("@echo off", encoding="utf-8")
    stop_bat = scripts_dir / "stop_ornith.bat"
    stop_bat.write_text("@echo off", encoding="utf-8")

    em = EngineManager(repo_root=tmp_path)
    res_start = em.start_colibri(timeout_sec=2)
    assert res_start is True

    # Check that subprocess.run was called with run_ornith.bat
    called_cmd = mock_subproc.call_args_list[0][0][0]
    assert "run_ornith.bat" in str(called_cmd)

    # Check stop
    res_stop = em.stop_colibri()
    assert res_stop is True
    stop_cmd = mock_subproc.call_args_list[1][0][0]
    assert "stop_ornith.bat" in str(stop_cmd)

