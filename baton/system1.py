"""System 1 Agent Runner: Fast Recon Scout & Dossier Compiler.

Powered by Qwen3:14B via Ollama and equipped with offline knowledge tools (ZimReader).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

import requests

from baton.models import (
    ArtifactType,
    BriefingDossier,
    RFDossier,
    Task,
    TaskCancelledError,
    TaskProgress,
    Tier,
    VerificationDossier,
    VerificationFailedError,
)
from baton.tools.zim import ZimTool

logger = logging.getLogger("baton.system1")


def strip_conversational_fluff(text: str) -> str:
    """Strips informal conversational preambles and pleasantries from LLM output."""
    cleaned = text.strip()
    # Remove leading conversational preamble lines
    cleaned = re.sub(
        r"^(?:Sure(?: thing)?[!,.]?|Here is[^\n]*|Certainly[!,.]?|Below is[^\n]*|I have[^\n]*|Please find[^\n]*|As requested[^\n]*)[^\n]*\n+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Remove trailing pleasantries
    cleaned = re.sub(
        r"\n+(?:I hope this helps[^\n]*|Let me know if[^\n]*|Feel free to[^\n]*|Hope this helps[^\n]*|Cheers[!,.]?|Best regards[^\n]*)$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def parse_structured_dossier(
    raw_text: str, is_verify: bool = False, is_sdr: bool = False
) -> tuple[str, Optional[Any]]:
    """
    Parses structured JSON emitted by System 1 Qwen into a validated Pydantic model
    and renders it into ultra-compact, token-dense Markdown.
    Falls back gracefully to cleaned text if JSON parsing fails.
    """
    cleaned = raw_text.strip()
    # Strip markdown code fence if wrapped in ```json ... ```
    match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if match:
        json_candidate = match.group(1).strip()
    else:
        obj_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        json_candidate = obj_match.group(1).strip() if obj_match else cleaned

    try:
        data = json.loads(json_candidate)
        if isinstance(data, dict):
            if is_verify:
                if "verdict" in data or "tests_passed" in data or "key_findings" in data:
                    model = VerificationDossier.model_validate(data)
                    return model.to_compact_markdown(), model
            elif is_sdr:
                if "band_name" in data or "carriers_detected" in data:
                    model = RFDossier.model_validate(data)
                    return model.to_compact_markdown(), model
            else:
                if "summary" in data or "core_facts" in data or "equations_or_apis" in data:
                    model = BriefingDossier.model_validate(data)
                    return model.to_compact_markdown(), model
    except Exception as e:
        logger.debug("Structured dossier JSON parsing skipped/failed: %s", e)

    # Fallback: clean fluff and return
    return strip_conversational_fluff(raw_text), None


def extract_code_block(text: str) -> str:
    """Extracts code enclosed in triple backtick markdown fences (```python ... ``` or ``` ... ```)."""
    match = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match2 = re.search(r"```(?:python)?\s*\n(.*)", text, re.DOTALL | re.IGNORECASE)
    if match2:
        return match2.group(1).rstrip("`").strip()
    return text.strip()


def compress_traceback(raw_trace: str, max_chars: int = 1500) -> str:
    """Extracts high-signal traceback lines and failures from raw execution output."""
    if not raw_trace or len(raw_trace) <= max_chars:
        return raw_trace or ""
    lines = raw_trace.strip().splitlines()
    high_signal = []
    capture = False
    for line in lines:
        l_lower = line.lower()
        if any(k in l_lower for k in ("traceback", "syntaxerror", "assertionerror", "nameerror", "importerror", "typeerror", "valueerror", "failed", "error:")):
            high_signal.append(line)
            capture = True
        elif capture and (line.startswith(" ") or line.startswith(">")):
            high_signal.append(line)
        elif "====" in line:
            high_signal.append(line)
            capture = False
    compressed = "\n".join(high_signal)
    if len(compressed) > max_chars:
        compressed = compressed[:max_chars] + "\n... [truncated]"
    return compressed or raw_trace[-max_chars:]


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
        max_repair_retries: int = 2,
        structured_output: bool = True,
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
        self.max_repair_retries = max_repair_retries
        self.structured_output = structured_output
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
        format: Optional[str | dict[str, Any]] = None,
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
        if format is not None:
            payload["format"] = format
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
        has_verification_error = False
        target_code_str = ""

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
                        target_code_str = code_str
                        self.sandbox_tool.write_file(art_file.name, code_str)
                        syn_info = self.sandbox_tool.syntax_check(code_str)
                        syn_res = self.sandbox_tool.execute("python_syntax_check", {"code": code_str})
                        verification_findings.append(f"### Syntax Verification (`{art_file.name}`):\n{syn_res}")
                        item_error = not syn_info.get("valid", True)
                        last_err = syn_info.get("message", "") if item_error else ""

                        if not item_error:
                            if "def test_" in code_str or "assert " in code_str:
                                py_res_dict = self.sandbox_tool.run_pytest(target=art_file.name)
                                py_res_md = self.sandbox_tool.execute("pytest_run", {"target": art_file.name})
                                verification_findings.append(f"### Test Suite Execution (`{art_file.name}`):\n{py_res_md}")
                                metrics = py_res_dict.get("metrics", {})
                                if not py_res_dict.get("success", False) or metrics.get("failed", 0) > 0 or metrics.get("errors", 0) > 0:
                                    item_error = True
                                    last_err = py_res_dict.get("output", "") or py_res_dict.get("error", "Pytest suite failed")
                            else:
                                eval_res = self.sandbox_tool.eval_code(code_str, filename=art_file.name)
                                if not eval_res.get("success"):
                                    item_error = True
                                    last_err = eval_res.get("error", "Execution failed")
                                    verification_findings.append(f"### Execution (`{art_file.name}`): FAILED\n{last_err}")

                        # System 1 Local Repair Loop if error detected
                        if item_error:
                            logger.info("System 1 entering local repair loop for '%s'...", art_file.name)
                            for retry in range(1, self.max_repair_retries + 1):
                                comp_trace = compress_traceback(last_err)
                                repair_prompt = (
                                    f"The Python file `{art_file.name}` failed automated verification.\n\n"
                                    f"### Error Diagnostics & Traceback:\n```text\n{comp_trace}\n```\n\n"
                                    f"### Failing Code:\n```python\n{code_str}\n```\n\n"
                                    "Fix the syntax error, missing import, or assertion failure.\n"
                                    "Return ONLY the complete, corrected Python code inside a ```python ``` block."
                                )
                                repair_sys_prompt = (
                                    "You are Baton System 1: Fast Code Repair Specialist.\n"
                                    "Fix the error and emit clean, working Python. Zero conversational filler."
                                )
                                patch_res = self.generate(repair_prompt, system_prompt=repair_sys_prompt)
                                patch_code = extract_code_block(patch_res.get("response", ""))
                                if not patch_code or patch_code == code_str:
                                    continue

                                # Stage patched code in sandbox
                                self.sandbox_tool.write_file(art_file.name, patch_code)
                                syn_check = self.sandbox_tool.syntax_check(patch_code)
                                if not syn_check.get("valid", True):
                                    last_err = syn_check.get("message", "Syntax error")
                                    code_str = patch_code
                                    target_code_str = patch_code
                                    verification_findings.append(f"### Local Patch (Attempt {retry}/{self.max_repair_retries}): FAILED (Syntax)\n{last_err}")
                                    continue

                                # If syntax passed, test again
                                if "def test_" in patch_code or "assert " in patch_code:
                                    py_re_dict = self.sandbox_tool.run_pytest(target=art_file.name)
                                    metrics_re = py_re_dict.get("metrics", {})
                                    if not py_re_dict.get("success", False) or metrics_re.get("failed", 0) > 0 or metrics_re.get("errors", 0) > 0:
                                        last_err = py_re_dict.get("output", "") or "Tests failed"
                                        code_str = patch_code
                                        target_code_str = patch_code
                                        verification_findings.append(f"### Local Patch (Attempt {retry}/{self.max_repair_retries}): FAILED (Tests)\n{last_err[:300]}")
                                        continue
                                else:
                                    eval_re = self.sandbox_tool.eval_code(patch_code, filename=art_file.name)
                                    if not eval_re.get("success"):
                                        last_err = eval_re.get("error", "Execution failed")
                                        code_str = patch_code
                                        target_code_str = patch_code
                                        verification_findings.append(f"### Local Patch (Attempt {retry}/{self.max_repair_retries}): FAILED (Runtime)\n{last_err}")
                                        continue

                                # Patch SUCCEEDED!
                                code_str = patch_code
                                target_code_str = patch_code
                                item_error = False
                                art_file.write_text(patch_code, encoding="utf-8")
                                verification_findings.append(
                                    f"### Local Patch (Attempt {retry}/{self.max_repair_retries}): SUCCESS\n"
                                    f"System 1 successfully resolved all errors in `{art_file.name}` and updated artifact on disk."
                                )
                                logger.info("System 1 successfully patched '%s' on attempt %d.", art_file.name, retry)
                                break

                        if item_error:
                            has_verification_error = True

            if verification_findings:
                raw_tool_data = "\n\n".join(verification_findings)

        # Step 2: Build distillation prompt for Qwen3:14B
        if is_verify:
            system_prompt = (
                "You are Baton System 1: an autonomous, rigorous Code Verifier.\n"
                "Analyze the verification and test data and emit a structured JSON dossier with zero conversational fluff.\n"
                "Output JSON matching this exact schema:\n"
                "{\n"
                '  "verdict": "VERIFIED" or "FAILED",\n'
                '  "syntax_valid": true,\n'
                '  "tests_passed": 0,\n'
                '  "tests_failed": 0,\n'
                '  "test_errors": 0,\n'
                '  "key_findings": ["exact finding or error trace line"],\n'
                '  "remediation_points": ["actionable remediation step"]\n'
                "}\n"
                "Output ONLY valid JSON. Zero pleasantries."
            )
        elif is_sdr:
            system_prompt = (
                "You are Baton System 1: an autonomous RF Telemetry & Signal Intelligence Scout.\n"
                "Analyze radio frequency sweeps, carrier peaks, and packets, and emit a structured JSON dossier.\n"
                "Output JSON matching this exact schema:\n"
                "{\n"
                '  "band_name": "band name or range",\n'
                '  "noise_floor_db": -48.2,\n'
                '  "carriers_detected": ["carrier frequency, power, and SNR"],\n'
                '  "demodulated_transmissions": ["demodulated packet details"],\n'
                '  "recommendations": ["channel parameter or downstream processing recommendations"]\n'
                "}\n"
                "Output ONLY valid JSON. Zero pleasantries."
            )
        else:
            system_prompt = (
                "You are Baton System 1: an autonomous, high-throughput Recon Scout.\n"
                "Extract, filter, and distill facts into a structured JSON briefing dossier for System 2.\n"
                "Output JSON matching this exact schema:\n"
                "{\n"
                '  "summary": "dense one-sentence executive summary",\n'
                '  "core_facts": ["verified constant, parameter, or factual statement"],\n'
                '  "equations_or_apis": ["exact equation, formula, or API signature"],\n'
                '  "open_questions": ["unresolved ambiguity or edge case"],\n'
                '  "source_citations": ["source title, file, or reference"]\n'
                "}\n"
                "Output ONLY valid JSON. Zero pleasantries."
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
            format="json" if self.structured_output else None,
        )
        raw_response = gen_result["response"]
        dossier_content, parsed_model = parse_structured_dossier(
            raw_response, is_verify=is_verify, is_sdr=is_sdr
        )
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

        if is_verify and has_verification_error:
            logger.warning("Verification failed and local repairs exhausted for '%s'.", task.task_id)
            raise VerificationFailedError(
                message=summary_snippet,
                artifact_path=str(artifact_path),
                failure_trace=raw_tool_data,
                target_code=target_code_str,
            )

        return summary_snippet, str(artifact_path), tokens, elapsed
