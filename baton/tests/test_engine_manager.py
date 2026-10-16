"""Unit tests for Baton EngineManager lifecycle and router decoupling."""

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
        assert em.router_url == "http://127.0.0.1:8080"
        assert em.router_start_cmd is None
        assert em.router_stop_cmd is None
        assert em.active_tier is None
        assert em.is_router_mode() is True


def test_engine_manager_env_var_overrides():
    """Verify environment variables properly configure router endpoint and hooks."""
    env = {
        "BATON_ROUTER_URL": "http://192.168.1.80:8080/",
        "BATON_ROUTER_START_CMD": "systemctl start llama-router",
        "BATON_ROUTER_STOP_CMD": "systemctl stop llama-router",
    }
    with patch.dict(os.environ, env, clear=True):
        em = EngineManager()
        assert em.router_url == "http://192.168.1.80:8080"
        assert em.router_start_cmd == "systemctl start llama-router"
        assert em.router_stop_cmd == "systemctl stop llama-router"


def test_engine_manager_explicit_params_precedence():
    """Verify explicit constructor arguments override environment variables."""
    env = {
        "BATON_ROUTER_URL": "http://env-router:8080",
    }
    with patch.dict(os.environ, env, clear=True):
        em = EngineManager(
            router_url="http://custom-router:8080",
            router_start_cmd="custom_start_router",
            router_stop_cmd="custom_stop_router",
        )
        assert em.router_url == "http://custom-router:8080"
        assert em.router_start_cmd == "custom_start_router"
        assert em.router_stop_cmd == "custom_stop_router"


@patch("requests.get")
def test_engine_manager_health_probes(mock_get):
    """Verify health probes return expected booleans based on HTTP responses."""
    em = EngineManager()

    # Router up
    mock_get.return_value = MagicMock(status_code=200)
    assert em.is_router_online() is True
    assert em.is_online() is True
    assert em.is_ollama_online() is True
    assert em.is_colibri_online() is True

    # Offline
    mock_get.side_effect = requests.RequestException("Connection refused")
    assert em.is_router_online() is False
    assert em.is_online() is False
    assert em.is_ollama_online() is False
    assert em.is_colibri_online() is False


@patch("requests.get")
def test_engine_manager_get_status(mock_get):
    """Verify get_status builds a dictionary reflecting engine health."""
    mock_get.return_value = MagicMock(status_code=200)
    em = EngineManager()
    status = em.get_status()
    assert status["active_tier"] == "idle"
    assert status["router_mode"] is True
    assert status["router"]["online"] is True
    assert status["router"]["status"] == EngineStatus.READY.value
    assert status["ollama"]["online"] is True
    assert status["colibri"]["online"] is True


@patch("subprocess.run")
@patch.object(EngineManager, "is_router_online")
def test_engine_manager_router_lifecycle_with_custom_commands(mock_online, mock_subproc):
    """Verify configured commands are called when starting/stopping router."""
    mock_online.side_effect = [False, True, False]
    em = EngineManager(
        router_start_cmd="echo starting router",
        router_stop_cmd="echo stopping router",
    )

    # Start
    res_start = em.start_router(timeout_sec=2)
    assert res_start is True
    mock_subproc.assert_called_with("echo starting router", cwd=str(em.repo_root), shell=True, check=False)

    # Stop
    res_stop = em.stop_router()
    assert res_stop is True
    mock_subproc.assert_called_with("echo stopping router", cwd=str(em.repo_root), shell=True, check=False)


@patch("subprocess.run")
@patch.object(EngineManager, "is_router_online")
def test_engine_manager_prioritizes_router_script(mock_online, mock_subproc, tmp_path: Path):
    """Verify that EngineManager locates and prioritizes run_router.bat as default script."""
    mock_online.side_effect = [False, True, False]
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    router_bat = scripts_dir / "run_router.bat"
    router_bat.write_text("@echo off", encoding="utf-8")
    stop_bat = scripts_dir / "stop_router.bat"
    stop_bat.write_text("@echo off", encoding="utf-8")

    em = EngineManager(repo_root=tmp_path)
    res_start = em.start_router(timeout_sec=2)
    assert res_start is True

    # Check start command
    called_cmd = mock_subproc.call_args_list[0][0][0]
    assert "run_router.bat" in str(called_cmd)

    # Check stop command
    res_stop = em.stop_router()
    assert res_stop is True
    stop_cmd = mock_subproc.call_args_list[1][0][0]
    assert "stop_router.bat" in str(stop_cmd)


@patch.object(EngineManager, "start_router", return_value=True)
def test_engine_manager_router_mode_tier_switching(mock_start_router):
    """Verify router mode provides zero-downtime tier switching without process restart."""
    em = EngineManager()

    with patch.object(em, "is_router_online", return_value=True):
        ok1 = em.switch_to_tier(Tier.SYSTEM_1)
        assert ok1 is True
        assert em.active_tier == Tier.SYSTEM_1
        mock_start_router.assert_not_called()

        ok2 = em.switch_to_tier(Tier.SYSTEM_2)
        assert ok2 is True
        assert em.active_tier == Tier.SYSTEM_2
        mock_start_router.assert_not_called()

    # If router is offline, switch_to_tier boots it up
    with patch.object(em, "is_router_online", side_effect=[False, True]):
        ok3 = em.switch_to_tier(Tier.SYSTEM_1)
        assert ok3 is True
        mock_start_router.assert_called_once()


def test_engine_manager_backwards_compatible_aliases():
    """Verify legacy method aliases delegate to the router without error."""
    em = EngineManager()
    with patch.object(em, "start_router", return_value=True) as mock_start, \
         patch.object(em, "stop_router", return_value=True) as mock_stop:
        assert em.start_ollama() is True
        assert em.start_colibri() is True
        assert mock_start.call_count == 2

        assert em.stop_ollama() is True
        assert em.stop_colibri() is True
        assert mock_stop.call_count == 2

        em.active_tier = Tier.SYSTEM_1
        em.stop_all()
        assert em.active_tier is None
