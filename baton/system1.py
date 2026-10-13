"""System 1 Agent Runner: Fast Recon Scout & Dossier Compiler.

Powered by Qwen3:14B via Ollama and equipped with offline knowledge tools (ZimReader).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

import requests

from baton.models import ArtifactType, Task, TaskCancelledError, TaskProgress, Tier
from baton.tools.zim import ZimTool

logger = logging.getLogger("baton.system1")


class System1Runner:
    """System 1 Recon Worker: Executes offline lookups and builds condensed briefing dossiers."""

    def __init__(
        self,
        ollama_url: Optional[str] = None,
        model_name: Optional[str] = None,
        artifacts_dir: Optional[Path] = None,
        zim_tool: Optional[ZimTool] = None,
        newsboat_tool: Optional[Any] = None,
        sandbox_tool: Optional[Any] = None,
        sdr_tool: Optional[Any] = None,
    ):
        self.ollama_url = (
            ollama_url
            or os.getenv("BATON_OLLAMA_URL")
            or "http://127.0.0.1:11434"
        ).rstrip("/")
        self.model_name = (
            model_name
            or os.getenv("BATON_SYSTEM1_MODEL")
            or "qwen3:14b"
        )
        self.artifacts_dir = artifacts_dir or Path("baton/artifacts/dossiers")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._zim_tool = zim_tool
        self._newsboat_tool = newsboat_tool
        self._sandbox_tool = sandbox_tool
        self._sdr_tool = sdr_tool
        self._cancel_requested = False
        self._active_response: Optional[requests.Response] = None

    def cancel_active(self) -> None:
        """Interrupts and closes any active HTTP generation stream."""
        self._cancel_requested = True
        if self._active_response is not None:
            try:
                self._active_response.close()
            except Exception:
                pass

    @property
    def zim_tool(self) -> ZimTool:
        if self._zim_tool is None:
            self._zim_tool = ZimTool()
        return self._zim_tool

    @property
    def newsboat_tool(self) -> Any:
        if self._newsboat_tool is None:
            from baton.tools.newsboat import NewsboatTool
            self._newsboat_tool = NewsboatTool()
        return self._newsboat_tool

    @property
    def sandbox_tool(self) -> Any:
        if self._sandbox_tool is None:
            from baton.tools.sandbox import SandboxTool
            self._sandbox_tool = SandboxTool()
        return self._sandbox_tool

    @property
    def sdr_tool(self) -> Any:
        if self._sdr_tool is None:
            from baton.tools.sdr import SdrTool
            self._sdr_tool = SdrTool()
        return self._sdr_tool

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: float | tuple[float, float] | None = (30.0, 180.0),
        stream: bool = True,
        progress: Optional[TaskProgress] = None,
        progress_callback: Optional[Callable[[TaskProgress], None]] = None,
    ) -> dict[str, Any]:
        """Calls Ollama generate API with optional streaming."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": 0.2,
                "num_ctx": 4096,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        t0 = time.time()
        self._cancel_requested = False

        if not stream:
            if self._cancel_requested:
                raise TaskCancelledError("System 1 generation stopped by user.")
            res = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=timeout)
            res.raise_for_status()
            data = res.json()
            elapsed = time.time() - t0
            return {
                "response": data.get("response", "").strip(),
                "tokens_generated": data.get("eval_count", 0),
                "elapsed_seconds": elapsed,
            }

        try:
            res = requests.post(f"{self.ollama_url}/api/generate", json=payload, stream=True, timeout=timeout)
            res.raise_for_status()
            self._active_response = res

            content_parts: list[str] = []
            tokens_count = 0
            last_callback_time = 0.0

            try:
                for raw_line in res.iter_lines(decode_unicode=True):
                    if self._cancel_requested:
                        raise TaskCancelledError("System 1 generation stopped by user.")
                    if not raw_line:
                        continue
                    if isinstance(raw_line, bytes):
                        line_str = raw_line.decode("utf-8", errors="replace").strip()
                    else:
                        line_str = str(raw_line).strip()
                    try:
                        data = json.loads(line_str)
                    except Exception:
                        continue

                    chunk = data.get("response", "")
                    if chunk:
                        content_parts.append(chunk)
                        tokens_count += 1

                    if data.get("done") and "eval_count" in data:
                        tokens_count = data["eval_count"]

                    now = time.time()
                    if progress and chunk:
                        progress.content_text = "".join(content_parts)
                        progress.tokens_generated = tokens_count
                        progress.tokens_per_second = tokens_count / max(now - progress.started_at, 0.001)
                        progress.last_update = now

                        if progress_callback and (now - last_callback_time >= 0.05):
                            last_callback_time = now
                            progress_callback(progress)
            except (requests.exceptions.RequestException, OSError):
                if self._cancel_requested:
                    raise TaskCancelledError("System 1 generation stopped by user.")
                raise
        finally:
            self._active_response = None

        elapsed = time.time() - t0
        final_text = "".join(content_parts).strip()

        if progress:
            progress.content_text = final_text
            progress.tokens_generated = tokens_count
            progress.tokens_per_second = tokens_count / max(elapsed, 0.001)
            progress.last_update = time.time()
            progress.is_completed = True
            if progress_callback:
                progress_callback(progress)

        return {
            "response": final_text,
            "tokens_generated": tokens_count,
            "elapsed_seconds": elapsed,
        }

    def execute_task(
        self,
        task: Task,
        upstream_dossiers: Optional[list[dict[str, Any]]] = None,
        progress_callback: Optional[Callable[[TaskProgress], None]] = None,
    ) -> tuple[str, Optional[str], int, float]:
        """
        Executes a System 1 task (recon or verification).
        Returns: (result_summary, artifact_path, tokens_generated, elapsed_seconds)
        """
        t0 = time.time()
        raw_tool_data = ""

        is_verify = (
            (task.task_type and task.task_type.value == "fast_verify")
            or (
                task.tool_name
                and any(
                    task.tool_name.lower().startswith(p)
                    for p in ("python_", "sandbox_", "pytest_")
                )
            )
        )

        is_sdr = bool(
            task.tool_name
            and any(
                task.tool_name.lower().startswith(p)
                for p in ("sdr_", "rtl_")
            )
        )

        # Step 1: Execute tool if specified, or auto-verify upstream artifacts
        if task.tool_name:
            logger.info("System 1 executing tool '%s' with params: %s", task.tool_name, task.tool_params)
            params = dict(task.tool_params or {})
            tool_lower = task.tool_name.lower()
            if tool_lower.startswith("rss_") or tool_lower.startswith("newsboat"):
                raw_tool_data = self.newsboat_tool.execute(task.tool_name, params)
            elif (
                tool_lower.startswith("python_")
                or tool_lower.startswith("sandbox_")
                or tool_lower.startswith("pytest_")
                or tool_lower in ("syntax_check", "eval_code", "run_pytest")
            ):
                # If testing an upstream artifact and filename is not set, resolve from upstream
                if ("code" not in params and "filename" not in params) and upstream_dossiers:
                    for u in upstream_dossiers:
                        art_p = u.get("artifact_path")
                        if art_p and Path(art_p).exists():
                            art_file = Path(art_p)
                            if art_file.suffix == ".py":
                                code_str = art_file.read_text(encoding="utf-8", errors="replace")
                                self.sandbox_tool.write_file(art_file.name, code_str)
                                if "filename" not in params:
                                    params["filename"] = art_file.name
                                if "target" not in params and "pytest" in tool_lower:
                                    params["target"] = art_file.name
                                break
                raw_tool_data = self.sandbox_tool.execute(task.tool_name, params)
            elif tool_lower.startswith("sdr_") or tool_lower.startswith("rtl_"):
                raw_tool_data = self.sdr_tool.execute(task.tool_name, params)
            else:
                raw_tool_data = self.zim_tool.execute(task.tool_name, params)
        elif is_verify and upstream_dossiers:
            # Auto-verify upstream python artifacts
            verification_findings = []
            for u in upstream_dossiers:
                art_p = u.get("artifact_path")
                if art_p and Path(art_p).exists():
                    art_file = Path(art_p)
                    if art_file.suffix == ".py":
                        code_str = art_file.read_text(encoding="utf-8", errors="replace")
                        self.sandbox_tool.write_file(art_file.name, code_str)
                        syn_res = self.sandbox_tool.execute("python_syntax_check", {"code": code_str})
                        verification_findings.append(f"### Syntax Verification (`{art_file.name}`):\n{syn_res}")
                        if "def test_" in code_str or "assert " in code_str:
                            py_res = self.sandbox_tool.execute("pytest_run", {"target": art_file.name})
                            verification_findings.append(f"### Test Suite Execution (`{art_file.name}`):\n{py_res}")
            if verification_findings:
                raw_tool_data = "\n\n".join(verification_findings)

        # Step 2: Build distillation prompt for Qwen3:14B
        if is_verify:
            system_prompt = (
                "You are Baton System 1: an autonomous, rigorous Code Verifier.\n"
                "Your objective is to analyze verification, syntax, and test execution results and compile a concise\n"
                "Markdown verification dossier (maximum 300 words).\n"
                "Rules:\n"
                "1. Report exact test metrics (passed, failed, syntax status, execution errors).\n"
                "2. Identify any broken assumptions, missing imports, or runtime exceptions.\n"
                "3. Conclude with a clear verdict: VERIFIED or FAILED, with actionable remediation points."
            )
        elif is_sdr:
            system_prompt = (
                "You are Baton System 1: an autonomous RF Telemetry & Signal Intelligence Scout.\n"
                "Your objective is to analyze radio frequency sweeps, carrier peaks, demodulations, and telemetry packets,\n"
                "and compile a concise, structured RF Briefing Dossier (maximum 300 words) for System 2 overnight synthesis.\n"
                "Rules:\n"
                "1. Summarize detected carriers, frequencies (MHz), peak power (dBm), and Signal-to-Noise Ratio (SNR).\n"
                "2. State band allocations, identified transmissions (e.g. APRS, FM, Aviation, ISM sensors), and active transmitters.\n"
                "3. Conclude with clear mathematical recommendations or channel parameters for downstream processing."
            )
        else:
            system_prompt = (
                "You are Baton System 1: an autonomous, high-throughput Recon Scout.\n"
                "Your objective is to extract, filter, and distill facts into a dense, mathematically precise\n"
                "Markdown briefing dossier (maximum 300 words) for System 2 (Deep Synthesizer).\n"
                "Rules:\n"
                "1. Preserve exact equations, physical constants, parameters, dates, and variable names.\n"
                "2. Eliminate conversational filler, redundant prose, and navigational boilerplate.\n"
                "3. Format with clean Markdown headers and bullet points."
            )

        user_prompt_parts = [
            f"# Mission Objective: {task.title}",
            f"Description: {task.prompt_template or ''}",
        ]

        if raw_tool_data:
            # Budget raw tool data so it fits comfortably in prompt
            truncated_data = raw_tool_data[:4000]
            data_header = "Verification & Test Data" if is_verify else "Raw Tool Reconnaissance Data"
            user_prompt_parts.append(f"## {data_header}:\n{truncated_data}")

        if upstream_dossiers:
            user_prompt_parts.append("## Prior Upstream Context:")
            for d in upstream_dossiers:
                user_prompt_parts.append(f"- **{d['title']}**: {d['result_payload'][:300]}")

        action_phrase = (
            "\nSynthesize the above verification results into a structured verification dossier now:"
            if is_verify
            else "\nSynthesize the above data into a dense, structured briefing dossier now:"
        )
        user_prompt_parts.append(action_phrase)
        full_prompt = "\n\n".join(user_prompt_parts)

        # Step 3: Call Qwen3:14B
        progress = TaskProgress(
            task_id=task.task_id,
            tier=Tier.SYSTEM_1,
            started_at=t0,
        )
        gen_result = self.generate(
            full_prompt,
            system_prompt=system_prompt,
            stream=True,
            progress=progress,
            progress_callback=progress_callback,
        )
        dossier_content = gen_result["response"]
        tokens = gen_result["tokens_generated"]
        elapsed = time.time() - t0

        # Step 4: Write Dossier Artifact
        artifact_filename = f"{task.task_id}_dossier.md"
        artifact_path = self.artifacts_dir / artifact_filename
        header_title = "Verification Dossier" if is_verify else "Briefing Dossier"
        worker_label = "Baton System 1 (Verifier)" if is_verify else "Baton System 1 (Qwen3:14B)"
        header = f"# {header_title}: {task.title}\n*{worker_label} on {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        artifact_path.write_text(header + dossier_content, encoding="utf-8")

        summary_snippet = dossier_content.split("\n\n")[0] if "\n\n" in dossier_content else dossier_content[:200]
        return summary_snippet, str(artifact_path), tokens, elapsed
