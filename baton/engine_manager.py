"""Lifecycle supervisor for the Baton multi-model inference router (llama-server).

Manages the unified llama-server process serving both System 1 (Qwen3:14B)
and System 2 (Ornith-1.5-35B-A3B) via on-demand VRAM/RAM model swapping.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import requests

from baton.models import EngineStatus, Tier

logger = logging.getLogger("baton.engine_manager")


class EngineManager:
    """Manages the llama-server multi-model router process and tier routing."""

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        router_url: Optional[str] = None,
        router_start_cmd: Optional[str] = None,
        router_stop_cmd: Optional[str] = None,
        ollama_url: Optional[str] = None,
        colibri_url: Optional[str] = None,
        ollama_start_cmd: Optional[str] = None,
        ollama_stop_cmd: Optional[str] = None,
        colibri_start_cmd: Optional[str] = None,
        colibri_stop_cmd: Optional[str] = None,
        use_router: Optional[bool] = None,
    ):
        self.repo_root = repo_root or Path(__file__).resolve().parent.parent

        # Resolve primary router URL (with backwards-compatible env fallbacks)
        resolved_router = (
            router_url
            or os.getenv("BATON_ROUTER_URL")
            or os.getenv("BATON_API_URL")
            or "http://127.0.0.1:8080"
        ).rstrip("/")
        self.router_url = resolved_router

        # Backwards compatibility attributes
        self.ollama_url = (
            ollama_url
            or os.getenv("BATON_OLLAMA_URL")
            or resolved_router
        ).rstrip("/")
        self.colibri_url = (
            colibri_url
            or os.getenv("BATON_COLIBRI_URL")
            or os.getenv("BATON_SYSTEM2_URL")
            or resolved_router
        ).rstrip("/")

        # Command hooks (env vars > explicit args > automatic resolution)
        self.router_start_cmd = (
            router_start_cmd
            or os.getenv("BATON_ROUTER_START_CMD")
            or ollama_start_cmd
            or os.getenv("BATON_OLLAMA_START_CMD")
            or colibri_start_cmd
            or os.getenv("BATON_COLIBRI_START_CMD")
        )
        self.router_stop_cmd = (
            router_stop_cmd
            or os.getenv("BATON_ROUTER_STOP_CMD")
            or ollama_stop_cmd
            or os.getenv("BATON_OLLAMA_STOP_CMD")
            or colibri_stop_cmd
            or os.getenv("BATON_COLIBRI_STOP_CMD")
        )

        # Backwards compatibility aliases for hooks
        self.ollama_start_cmd = self.router_start_cmd
        self.ollama_stop_cmd = self.router_stop_cmd
        self.colibri_start_cmd = self.router_start_cmd
        self.colibri_stop_cmd = self.router_stop_cmd

        self._use_router = True if use_router is None else use_router
        self.active_tier: Optional[Tier] = None

    # -------------------------------------------------------------------------
    # Health Probes
    # -------------------------------------------------------------------------

    def is_router_online(self) -> bool:
        """Probe llama-server multi-model router on port 8080 (or configured router_url)."""
        try:
            res = requests.get(f"{self.router_url}/v1/models", timeout=1.5)
            return res.status_code == 200
        except Exception:
            return False

    def is_online(self) -> bool:
        """Alias for is_router_online."""
        return self.is_router_online()

    def is_router_mode(self) -> bool:
        """Returns True: Baton uses the unified multi-model router architecture."""
        return self._use_router

    # Backwards compatibility probe aliases
    def is_ollama_online(self) -> bool:
        """Probe engine health (delegates to router or fallback URL)."""
        if self.is_router_online():
            return True
        if self.ollama_url and self.ollama_url != self.router_url:
            try:
                res = requests.get(f"{self.ollama_url}/api/version", timeout=1.5)
                return res.status_code == 200
            except Exception:
                return False
        return False

    def is_colibri_online(self) -> bool:
        """Probe engine health (delegates to router or fallback URL)."""
        if self.is_router_online():
            return True
        if self.colibri_url and self.colibri_url != self.router_url:
            try:
                res = requests.get(f"{self.colibri_url}/v1/models", timeout=1.5)
                return res.status_code == 200
            except Exception:
                return False
        return False

    def get_status(self) -> dict[str, Any]:
        """Returns live status of the router engine."""
        router_up = self.is_router_online()

        return {
            "active_tier": self.active_tier.value if self.active_tier else "idle",
            "router_mode": True,
            "router": {
                "online": router_up,
                "url": self.router_url,
                "status": EngineStatus.READY.value if router_up else EngineStatus.STOPPED.value,
            },
            # Backwards-compatible aliases for status consumers
            "ollama": {
                "online": router_up,
                "url": self.router_url,
                "status": EngineStatus.READY.value if router_up else EngineStatus.STOPPED.value,
            },
            "colibri": {
                "online": router_up,
                "url": self.router_url,
                "status": EngineStatus.READY.value if router_up else EngineStatus.STOPPED.value,
            },
        }

    # -------------------------------------------------------------------------
    # llama-server Multi-Model Router Lifecycle
    # -------------------------------------------------------------------------

    def start_router(self, timeout_sec: int = 60) -> bool:
        """Starts the llama-server multi-model router."""
        if self.is_router_online():
            logger.info("llama-server multi-model router is already online.")
            return True

        if self.router_start_cmd:
            subprocess.run(self.router_start_cmd, cwd=str(self.repo_root), shell=True, check=False)
        else:
            script_candidates = [
                self.repo_root / "scripts" / "run_router.bat",
                self.repo_root / "scripts" / "run_router.sh",
                Path("E:/dev/Antigravity/local_llm/scripts/run_router.bat"),
                Path("E:/dev/Antigravity/local_llm/scripts/run_router.sh"),
            ]
            found_script = next((s for s in script_candidates if s.exists()), None)
            if found_script:
                if os.name == "nt" and found_script.suffix == ".bat":
                    cmd = f'start "llama-server Router" /min cmd /c "{found_script} serve > router_server.log 2>&1"'
                    subprocess.run(cmd, cwd=str(self.repo_root), shell=True, check=False)
                else:
                    subprocess.Popen([str(found_script), "serve"], cwd=str(self.repo_root), shell=True)
            else:
                logger.info(
                    "No router start script found. Waiting for external router at %s...",
                    self.router_url,
                )

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.is_router_online():
                logger.info("llama-server router initialized successfully.")
                return True
            time.sleep(1.0)

        logger.warning(
            "llama-server router start timed out (endpoint: %s).",
            self.router_url,
        )
        return False

    def stop_router(self) -> bool:
        """Stops llama-server router and frees GPU VRAM and RAM."""
        if self.router_stop_cmd:
            subprocess.run(self.router_stop_cmd, cwd=str(self.repo_root), shell=True, check=False)
        else:
            script_candidates = [
                self.repo_root / "scripts" / "stop_router.bat",
                self.repo_root / "scripts" / "stop_router.sh",
            ]
            found_script = next((s for s in script_candidates if s.exists()), None)
            if found_script:
                subprocess.run([str(found_script)], cwd=str(self.repo_root), shell=True, check=False)

            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/IM", "llama-server.exe"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.run(
                    ["pkill", "-f", "llama-server"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        time.sleep(1.0)
        return not self.is_router_online()

    # Backwards-compatible lifecycle aliases
    def start_ollama(self, timeout_sec: int = 15) -> bool:
        return self.start_router(timeout_sec=timeout_sec)

    def stop_ollama(self) -> bool:
        return self.stop_router()

    def start_colibri(self, timeout_sec: int = 60) -> bool:
        return self.start_router(timeout_sec=timeout_sec)

    def stop_colibri(self) -> bool:
        return self.stop_router()

    # -------------------------------------------------------------------------
    # Tier Routing & Switching
    # -------------------------------------------------------------------------

    def switch_to_tier(self, target_tier: Tier) -> bool:
        """Switches active tier. In router mode, llama-server pages models on-demand."""
        if not self.is_router_online():
            logger.info("Starting unified llama-server router on port 8080...")
            ok = self.start_router()
            if not ok:
                return False

        self.active_tier = target_tier
        logger.info(
            "Active tier set to %s (llama-server router handles VRAM paging on demand)",
            target_tier.value,
        )
        return True

    def stop_all(self) -> None:
        """Clean shutdown of the router engine."""
        self.stop_router()
        self.active_tier = None
