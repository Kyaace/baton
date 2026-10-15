"""Subprocess and lifecycle supervisor for System 1 (Ollama) and System 2 (Colibrì).

Enforces Mode A interleaved execution to ensure exclusive GPU VRAM allocation.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import requests

from baton.models import EngineStatus, Tier

logger = logging.getLogger("baton.engine_manager")


class EngineManager:
    """Manages Ollama and Colibrì/System 2 processes, enforcing mutual exclusion."""

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        ollama_url: Optional[str] = None,
        colibri_url: Optional[str] = None,
        ollama_start_cmd: Optional[str] = None,
        ollama_stop_cmd: Optional[str] = None,
        colibri_start_cmd: Optional[str] = None,
        colibri_stop_cmd: Optional[str] = None,
    ):
        self.repo_root = repo_root or Path(__file__).resolve().parent.parent
        self.ollama_url = (
            ollama_url
            or os.getenv("BATON_OLLAMA_URL")
            or "http://127.0.0.1:11434"
        ).rstrip("/")
        self.colibri_url = (
            colibri_url
            or os.getenv("BATON_COLIBRI_URL")
            or os.getenv("BATON_SYSTEM2_URL")
            or "http://127.0.0.1:8000"
        ).rstrip("/")

        # Command hooks (env vars > explicit args > automatic resolution)
        self.ollama_start_cmd = ollama_start_cmd or os.getenv("BATON_OLLAMA_START_CMD")
        self.ollama_stop_cmd = ollama_stop_cmd or os.getenv("BATON_OLLAMA_STOP_CMD")
        self.colibri_start_cmd = (
            colibri_start_cmd
            or os.getenv("BATON_COLIBRI_START_CMD")
            or os.getenv("BATON_SYSTEM2_START_CMD")
        )
        self.colibri_stop_cmd = (
            colibri_stop_cmd
            or os.getenv("BATON_COLIBRI_STOP_CMD")
            or os.getenv("BATON_SYSTEM2_STOP_CMD")
        )

        self.active_tier: Optional[Tier] = None

    # -------------------------------------------------------------------------
    # Health Probes
    # -------------------------------------------------------------------------

    def is_ollama_online(self) -> bool:
        """Probe Ollama version / tags endpoint."""
        try:
            res = requests.get(f"{self.ollama_url}/api/version", timeout=1.5)
            return res.status_code == 200
        except Exception:
            return False

    def is_colibri_online(self) -> bool:
        """Probe Colibrì / System 2 health or OpenAPI spec endpoint."""
        try:
            res = requests.get(f"{self.colibri_url}/health", timeout=1.5)
            if res.status_code == 200:
                return True
        except Exception:
            pass
        try:
            res = requests.get(f"{self.colibri_url}/v1/models", timeout=1.5)
            return res.status_code == 200
        except Exception:
            return False

    def get_status(self) -> dict[str, Any]:
        """Returns live status of both engines."""
        ollama_up = self.is_ollama_online()
        colibri_up = self.is_colibri_online()
        return {
            "active_tier": self.active_tier.value if self.active_tier else "idle",
            "ollama": {
                "online": ollama_up,
                "url": self.ollama_url,
                "status": EngineStatus.READY.value if ollama_up else EngineStatus.STOPPED.value,
            },
            "colibri": {
                "online": colibri_up,
                "url": self.colibri_url,
                "status": EngineStatus.READY.value if colibri_up else EngineStatus.STOPPED.value,
            },
        }

    # -------------------------------------------------------------------------
    # Ollama Management
    # -------------------------------------------------------------------------

    def start_ollama(self, timeout_sec: int = 15) -> bool:
        """Starts Ollama using configured hook, repository script, or system binary."""
        if self.is_ollama_online():
            logger.info("Ollama is already online.")
            return True

        if self.ollama_start_cmd:
            subprocess.run(self.ollama_start_cmd, cwd=str(self.repo_root), shell=True, check=False)
        else:
            # Check for local helper scripts
            script_candidates = [
                self.repo_root / "scripts" / "start_ollama.bat",
                self.repo_root / "scripts" / "start_ollama.sh",
            ]
            found_script = next((s for s in script_candidates if s.exists()), None)
            if found_script:
                subprocess.run([str(found_script)], cwd=str(self.repo_root), shell=True, check=False)
            else:
                # Direct system binary fallback (cross-platform)
                try:
                    subprocess.Popen(["ollama", "serve"], shell=(os.name == "nt"))
                except Exception as e:
                    logger.warning("Failed to launch 'ollama serve': %s", e)

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.is_ollama_online():
                logger.info("Ollama initialized successfully.")
                return True
            time.sleep(1.0)

        logger.warning("Ollama start timed out.")
        return False

    def stop_ollama(self) -> bool:
        """Stops Ollama and terminates running runners to free GPU VRAM."""
        if self.ollama_stop_cmd:
            subprocess.run(self.ollama_stop_cmd, cwd=str(self.repo_root), shell=True, check=False)
        else:
            script_candidates = [
                self.repo_root / "scripts" / "stop_ollama.bat",
                self.repo_root / "scripts" / "stop_ollama.sh",
            ]
            found_script = next((s for s in script_candidates if s.exists()), None)
            if found_script:
                subprocess.run([str(found_script)], cwd=str(self.repo_root), shell=True, check=False)
            elif os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/IM", "ollama.exe"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                subprocess.run(
                    ["taskkill", "/F", "/T", "/IM", "llama-server.exe"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.run(
                    ["pkill", "-f", "ollama"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                subprocess.run(
                    ["pkill", "-f", "llama-server"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        time.sleep(1.5)
        return not self.is_ollama_online()

    # -------------------------------------------------------------------------
    # Colibrì / System 2 Management
    # -------------------------------------------------------------------------

    def start_colibri(self, timeout_sec: int = 60) -> bool:
        """Starts System 2 / Colibrì server using configured hook, script, or endpoint check."""
        if self.is_colibri_online():
            logger.info("Colibrì server is already online.")
            return True

        if self.colibri_start_cmd:
            subprocess.run(self.colibri_start_cmd, cwd=str(self.repo_root), shell=True, check=False)
        else:
            # Check local repository candidates (Ornith / llama-server preferred, then DeepSeek-V4 fallback)
            script_candidates = [
                self.repo_root / "scripts" / "run_ornith.bat",
                self.repo_root / "scripts" / "run_ornith.sh",
                self.repo_root / "scripts" / "run_deepseek_v4_gpu.bat",
                self.repo_root / "scripts" / "start_coli_server.bat",
                self.repo_root / "scripts" / "run_deepseek_v4_gpu.sh",
                self.repo_root / "scripts" / "start_coli_server.sh",
                Path("E:/dev/Antigravity/local_llm/scripts/run_ornith.bat"),
                Path("E:/dev/Antigravity/local_llm/scripts/run_deepseek_v4_gpu.bat"),
                Path("E:/dev/Antigravity/local_llm/scripts/start_coli_server.bat"),
            ]
            found_script = next((s for s in script_candidates if s.exists()), None)
            if found_script:
                model_label = "Ornith (llama-server)" if "ornith" in found_script.name.lower() else "System 2 LLM"
                if os.name == "nt" and found_script.suffix == ".bat":
                    cmd = f'start "{model_label}" /min cmd /c "{found_script} serve > system2_server.log 2>&1"'
                    subprocess.run(cmd, cwd=str(self.repo_root), shell=True, check=False)
                else:
                    subprocess.Popen([str(found_script), "serve"], cwd=str(self.repo_root), shell=True)
            else:
                logger.info(
                    "No System 2 start script found. Waiting for external endpoint at %s...",
                    self.colibri_url,
                )

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.is_colibri_online():
                logger.info("System 2 (Ornith / llama-server) initialized successfully.")
                return True
            time.sleep(2.0)

        logger.warning(
            "System 2 server start timed out (endpoint: %s). Configure BATON_COLIBRI_START_CMD if customized.",
            self.colibri_url,
        )
        return False

    def stop_colibri(self) -> bool:
        """Stops System 2 / Colibrì / llama-server and frees RAM and VRAM."""
        if self.colibri_stop_cmd:
            subprocess.run(self.colibri_stop_cmd, cwd=str(self.repo_root), shell=True, check=False)
        else:
            # Check for local helper scripts
            script_candidates = [
                self.repo_root / "scripts" / "stop_ornith.bat",
                self.repo_root / "scripts" / "stop_ornith.sh",
                self.repo_root / "scripts" / "stop_coli_server.bat",
                self.repo_root / "scripts" / "stop_coli_server.sh",
                Path("E:/dev/Antigravity/local_llm/scripts/stop_ornith.bat"),
                Path("E:/dev/Antigravity/local_llm/scripts/stop_coli_server.bat"),
                Path("E:/dev/Antigravity/local_llm/scripts/stop_coli_server.sh"),
            ]
            found_script = next((s for s in script_candidates if s.exists()), None)
            if found_script:
                subprocess.run([str(found_script)], cwd=str(self.repo_root), shell=True, check=False)

            # Ensure llama-server process is terminated on Windows
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

            # Check for local coli CLI wrapper if present
            coli_cmd = self.repo_root / "coli.cmd"
            if not coli_cmd.exists():
                coli_cmd = Path("E:/dev/Antigravity/local_llm/coli.cmd")
            if coli_cmd.exists():
                subprocess.run(
                    [str(coli_cmd), "stop"],
                    cwd=str(self.repo_root),
                    shell=True,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            # Ensure WSL Colibri process is stopped
            if os.name == "nt":
                subprocess.run(
                    ["wsl", "-e", "/mnt/e/dev/Antigravity/local_llm/colibri/c/coli", "stop"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            # Check for coli in system PATH or WSL without hardcoded dev path
            try:
                subprocess.run(["coli", "stop"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

            if os.name == "nt":
                try:
                    subprocess.run(
                        ["wsl", "-e", "bash", "-c", "which coli >/dev/null && coli stop"],
                        timeout=5,
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    pass

        time.sleep(2.0)
        return not self.is_colibri_online()

    # -------------------------------------------------------------------------
    # Mode A Mutual Exclusion Switching
    # -------------------------------------------------------------------------

    def switch_to_tier(self, target_tier: Tier) -> bool:
        """Guarantees the target tier's engine is online while stopping the other."""
        if target_tier == Tier.SYSTEM_1:
            if self.is_colibri_online():
                logger.info("Mode A: Stopping System 2 (Colibrì) to release GPU VRAM & RAM...")
                self.stop_colibri()

            if not self.is_ollama_online():
                logger.info("Mode A: Starting System 1 (Ollama)...")
                ok = self.start_ollama()
                if not ok:
                    return False
            self.active_tier = Tier.SYSTEM_1
            return True

        elif target_tier == Tier.SYSTEM_2:
            if self.is_ollama_online():
                logger.info("Mode A: Stopping System 1 (Ollama) to release RTX 3060 VRAM...")
                self.stop_ollama()

            if not self.is_colibri_online():
                logger.info("Mode A: Starting System 2 (Colibrì DeepSeek-V4)...")
                ok = self.start_colibri()
                if not ok:
                    return False
            self.active_tier = Tier.SYSTEM_2
            return True

        return False

    def stop_all(self) -> None:
        """Clean shutdown of all engines."""
        self.stop_ollama()
        self.stop_colibri()
        self.active_tier = None
