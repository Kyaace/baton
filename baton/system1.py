"""System 1 Agent Runner: Fast Recon Scout & Dossier Compiler.

Powered by Qwen3:14B via Ollama and equipped with offline knowledge tools (ZimReader).
"""

from __future__ import annotations

import ast
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
    DAGPlan,
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


def validate_delimiters(text: str) -> tuple[bool, str]:
    """
    Deterministically checks that brackets (), [], {} and markdown code fences ```
    are properly balanced and closed.
    Returns: (is_valid, reason)
    """
    # 1. Check code fences
    fence_matches = re.findall(r"^```|\n```", text)
    if len(fence_matches) % 2 != 0:
        return False, "Unclosed code fence (```)"

    # 2. Check brackets outside of code fences
    non_code = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    openers = set(pairs.values())

    for char in non_code:
        if char in openers:
            stack.append(char)
        elif char in pairs:
            if not stack or stack[-1] != pairs[char]:
                return False, f"Mismatched delimiter '{char}'"
            stack.pop()

    if stack:
        return False, f"Unclosed delimiter '{stack[-1]}'"

    return True, ""


def validate_embedded_code_syntax(text: str) -> tuple[bool, list[str]]:
    """
    Extracts all ```python ... ``` code snippets and verifies AST syntax validity.
    Returns: (all_valid, list_of_syntax_errors)
    """
    snippets = re.findall(r"```(?:python)?\s*\n(.*?)\n```", text, flags=re.DOTALL | re.IGNORECASE)
    errors: list[str] = []
    for i, snippet in enumerate(snippets, 1):
        if not snippet.strip():
            continue
        try:
            ast.parse(snippet)
        except SyntaxError as e:
            errors.append(f"Snippet {i} syntax error on line {e.lineno}: {e.msg}")

    return (len(errors) == 0, errors)


def generate_diagnostic_dossier(failure_trace: str, code: str = "", max_words: int = 150) -> str:
    """
    Deterministically distills an ultra-compact (<= 150 words) failure diagnostic
    focusing purely on root causes, failing assertions, and key error lines.
    """
    if not failure_trace:
        return "No diagnostic trace available."

    lines = failure_trace.strip().splitlines()
    high_signal = []
    for line in lines:
        l_lower = line.lower()
        if any(k in l_lower for k in ("assertionerror", "syntaxerror", "nameerror", "importerror", "typeerror", "valueerror", "failed", "error:")):
            high_signal.append(line.strip())
        elif line.strip().startswith(">") or line.strip().startswith("E "):
            high_signal.append(line.strip())

    if not high_signal:
        high_signal = [lines[-1].strip()] if lines else ["Execution failed."]

    body = "\n".join(high_signal[:8])
    words = body.split()
    if len(words) > max_words:
        body = " ".join(words[:max_words]) + " ... [diagnostic truncated]"

    return f"Root Cause Analysis:\n{body}"


def repair_dag_plan_json(
    runner: System1Runner,
    raw_text: str,
    error_msg: str,
) -> tuple[Optional[DAGPlan], str]:
    """
    One-shot System 1 fast repair for malformed or cyclic DAGPlan JSON output
    from System 2. Prevents costly System 2 wakeups.
    """
    system_prompt = (
        "You are Baton System 1: an autonomous JSON Syntactic Repair Worker.\n"
        "Your task is to fix syntax errors, missing fields, or cyclic dependencies in a DAG plan JSON.\n"
        "Output ONLY valid JSON matching this schema:\n"
        "{\n"
        '  "project": "...",\n'
        '  "goal": "...",\n'
        '  "tasks": [\n'
        '    {\n'
        '      "id": "task_id",\n'
        '      "title": "title",\n'
        '      "type": "fast_recon" or "deep_synthesis" or "fast_verify",\n'
        '      "tier": "system_1" or "system_2",\n'
        '      "depends_on": ["upstream_id"],\n'
        '      "prompt": "instructions",\n'
        '      "priority": 1\n'
        '    }\n'
        '  ]\n'
        "}\n"
        "Ensure all task IDs are unique and the graph is strictly acyclic.\n"
        "Output ONLY the corrected JSON. Zero pleasantries."
    )
    user_prompt = (
        f"Validation Error:\n{error_msg}\n\n"
        f"Raw Malformed Output from System 2:\n{raw_text[:3000]}\n\n"
        f"Fix the JSON syntax and dependencies now:"
    )
    try:
        res = runner.generate(user_prompt, system_prompt=system_prompt, stream=False, format="json")
        repaired_text = res["response"]
        clean = re.sub(r"^```(?:json)?\s*", "", repaired_text, flags=re.MULTILINE)
        clean = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE).strip()
        data = json.loads(clean)
        plan = DAGPlan.model_validate(data)
        plan.validate_dag()
        return plan, "System 1 successfully repaired System 2 DAG plan"
    except Exception as e:
        logger.warning("System 1 DAG repair failed: %s", e)
        return None, str(e)


class System1Runner:
    """System 1 Recon Worker: Executes offline lookups and builds condensed briefing dossiers."""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        model_name: Optional[str] = None,
        artifacts_dir: Optional[Path] = None,
        zim_tool: Optional[ZimTool] = None,
        newsboat_tool: Optional[Any] = None,
        sandbox_tool: Optional[Any] = None,
        sdr_tool: Optional[Any] = None,
        max_repair_retries: int = 3,
        structured_output: bool = True,
        ollama_url: Optional[str] = None,
    ):
        resolved_url = (
            endpoint_url
            or ollama_url
            or os.getenv("BATON_ROUTER_URL")
            or os.getenv("BATON_API_URL")
            or os.getenv("BATON_OLLAMA_URL")
            or "http://127.0.0.1:8080"
        ).rstrip("/")
        self.endpoint_url = resolved_url
        self.ollama_url = resolved_url  # Backwards compatibility alias
        self.model_name = (
            model_name
            or os.getenv("BATON_SYSTEM1_MODEL")
            or "qwen3"
        )
        self.artifacts_dir = artifacts_dir or Path("baton/artifacts/dossiers")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._zim_tool = zim_tool
        self._newsboat_tool = newsboat_tool
        self._sandbox_tool = sandbox_tool
        self._sdr_tool = sdr_tool
        self._time_tool: Any = None
        self._docs_tool: Any = None
        self._vault_tool: Any = None
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

    @property
    def time_tool(self) -> Any:
        if self._time_tool is None:
            from baton.tools.time_tool import TimeTool
            self._time_tool = TimeTool()
        return self._time_tool

    @property
    def docs_tool(self) -> Any:
        if self._docs_tool is None:
            from baton.tools.docs_tool import DocsTool
            self._docs_tool = DocsTool()
        return self._docs_tool

    @property
    def vault_tool(self) -> Any:
        if self._vault_tool is None:
            from baton.tools.vault import VaultTool
            self._vault_tool = VaultTool()
        return self._vault_tool

    @property
    def is_openai_backend(self) -> bool:
        """Determines if the endpoint is an OpenAI-compatible / llama-server router endpoint."""
        url = self.endpoint_url.lower()
        if ":11434" in url and os.getenv("BATON_USE_ROUTER") not in ("1", "true", "yes"):
            return False
        return True

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
        """Calls System 1 generation API (Ollama or OpenAI-compatible llama-server router)."""
        t0 = time.time()
        self._cancel_requested = False

        # ---------------------------------------------------------------------
        # OpenAI / llama-server router mode
        # ---------------------------------------------------------------------
        if self.is_openai_backend:
            endpoint = (
                f"{self.endpoint_url}/chat/completions"
                if self.endpoint_url.endswith("/v1")
                else f"{self.endpoint_url}/v1/chat/completions"
            )
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Normalise model name for router: accept qwen3, qwen3:14b, etc.
            model_target = self.model_name
            if model_target.lower() in ("qwen3:14b", "qwen3-14b"):
                model_target = "qwen3"

            payload: dict[str, Any] = {
                "model": model_target,
                "messages": messages,
                "temperature": 0.2,
                "stream": stream,
            }
            if format == "json" or format == {"type": "json_object"}:
                payload["response_format"] = {"type": "json_object"}
            if stream:
                payload["stream_options"] = {"include_usage": True}

            if not stream:
                if self._cancel_requested:
                    raise TaskCancelledError("System 1 generation stopped by user.")
                res = requests.post(endpoint, json=payload, timeout=timeout)
                res.raise_for_status()
                data = res.json()
                choice = data.get("choices", [{}])[0]
                msg = choice.get("message", {})
                content = msg.get("content", "").strip()
                if not content and msg.get("reasoning_content"):
                    content = msg["reasoning_content"].strip()
                tokens = data.get("usage", {}).get("completion_tokens", 0)
                elapsed = time.time() - t0
                return {
                    "response": content,
                    "tokens_generated": tokens,
                    "elapsed_seconds": elapsed,
                }

            # Streaming mode (SSE)
            try:
                res = requests.post(endpoint, json=payload, stream=True, timeout=timeout)
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
                        if not line_str.startswith("data:"):
                            continue
                        data_str = line_str[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                        except Exception:
                            continue

                        usage = data.get("usage")
                        if usage and "completion_tokens" in usage:
                            tokens_count = usage["completion_tokens"]

                        choices = data.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        chunk = delta.get("content", "")
                        if chunk:
                            content_parts.append(chunk)
                            if not usage:
                                tokens_count += 1

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

        # ---------------------------------------------------------------------
        # Classic Ollama /api/generate mode
        # ---------------------------------------------------------------------
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

        if not stream:
            if self._cancel_requested:
                raise TaskCancelledError("System 1 generation stopped by user.")
            res = requests.post(f"{self.endpoint_url}/api/generate", json=payload, timeout=timeout)
            res.raise_for_status()
            data = res.json()
            elapsed = time.time() - t0
            return {
                "response": data.get("response", "").strip(),
                "tokens_generated": data.get("eval_count", 0),
                "elapsed_seconds": elapsed,
            }

        try:
            res = requests.post(f"{self.endpoint_url}/api/generate", json=payload, stream=True, timeout=timeout)
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

    def _dispatch_tool(self, tool_name: str, params: dict[str, Any]) -> str:
        """Dispatches execution to the appropriate tool bundle backend."""
        tool_lower = tool_name.lower()
        if tool_lower.startswith("rss_") or tool_lower.startswith("newsboat"):
            return self.newsboat_tool.execute(tool_name, params)
        elif (
            tool_lower.startswith("python_")
            or tool_lower.startswith("sandbox_")
            or tool_lower.startswith("pytest_")
            or tool_lower in ("syntax_check", "eval_code", "run_pytest")
        ):
            return self.sandbox_tool.execute(tool_name, params)
        elif tool_lower.startswith("sdr_") or tool_lower.startswith("rtl_"):
            return self.sdr_tool.execute(tool_name, params)
        elif (
            tool_lower.startswith("time_")
            or tool_lower.startswith("note_")
            or tool_lower in (
                "get_current_time",
                "schedule_task_alarm",
                "create_scheduled_task",
                "list_alarms",
                "cancel_alarm",
                "save_note",
                "append_note",
                "read_note",
                "list_notes",
                "dismiss_note",
                "keepalive_note",
            )
        ):
            return self.time_tool.execute(tool_name, params)
        elif tool_lower.startswith("doc_") or tool_lower.startswith("docs_"):
            return self.docs_tool.execute(tool_name, params)
        elif tool_lower.startswith("vault_") or tool_lower in ("search_vault", "vault"):
            return self.vault_tool.execute(tool_name, params)
        else:
            return self.zim_tool.execute(tool_name, params)

    @staticmethod
    def _is_recon_miss(text: str) -> bool:
        """Checks if a tool result indicates a lookup miss, empty content, or error."""
        if not text or not text.strip():
            return True
        t_low = text.lower()
        return any(
            phrase in t_low
            for phrase in (
                "was empty or not found",
                "has no lead text or was not found",
                "no infobox found",
                "no results found",
                "not found in [",
                "error:",
                "no article found",
                "unknown tool",
                "article not found",
            )
        )

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
            if (
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
                                if "code" not in params:
                                    params["code"] = code_str
                                if "target" not in params and "pytest" in tool_lower:
                                    params["target"] = art_file.name
                                break
            raw_tool_data = self._dispatch_tool(task.tool_name, params)

            # Autonomous Recon Self-Recovery Loop (on lookup miss / typo)
            if not is_verify and not is_sdr and self._is_recon_miss(raw_tool_data):
                logger.info(
                    "System 1 Recon tool '%s' returned a miss/empty result. Entering autonomous recovery loop...",
                    task.tool_name,
                )
                suggestions_text = ""
                if self.zim_tool:
                    query_term = params.get("page_title") or params.get("title") or params.get("query") or ""
                    if query_term:
                        try:
                            suggs = self.zim_tool.reader.suggest_titles(str(query_term), limit=5)
                            if suggs:
                                suggestions_text = f"Title suggestions from index: {', '.join(suggs)}\n"
                            if not suggs:
                                search_res = self.zim_tool.search(str(query_term), limit=3)
                                if search_res and "no results found" not in search_res.lower():
                                    suggestions_text = f"Search matches:\n{search_res[:400]}\n"
                        except Exception as err:
                            logger.debug("Error querying ZIM suggestions during recovery: %s", err)

                recovery_sys_prompt = (
                    "You are Baton System 1: Autonomous Recon Recovery Scout.\n"
                    "A knowledge retrieval tool returned a miss, typo, or empty result.\n"
                    "Analyze the mission objective and error, and output a JSON repair action to re-query knowledge:\n"
                    "{\n"
                    '  "diagnosis": "brief explanation of typo, bad title, or wrong tool",\n'
                    '  "tool": "wiki_read_lead" | "wiki_read_section" | "wiki_search" | "wiki_infobox" | "rss_search" | "doc_search",\n'
                    '  "params": {"title": "...", "page_title": "...", "section": "..."}\n'
                    "}\n"
                    "Output ONLY valid JSON matching this schema."
                )

                recovery_user_prompt = (
                    f"### Recon Mission: {task.title}\n"
                    f"Description: {task.prompt_template or ''}\n\n"
                    f"### Failed Tool Call:\n"
                    f"Tool: {task.tool_name}\n"
                    f"Parameters: {json.dumps(params)}\n"
                    f"Result: {raw_tool_data}\n\n"
                    f"{suggestions_text}"
                    "Diagnose the miss and provide the corrected tool call in JSON:"
                )

                try:
                    repair_res = self.generate(recovery_user_prompt, system_prompt=recovery_sys_prompt, format="json")
                    repair_text = repair_res.get("response", "").strip()
                    clean_repair = re.sub(r"^```(?:json)?\s*", "", repair_text, flags=re.MULTILINE)
                    clean_repair = re.sub(r"\s*```$", "", clean_repair, flags=re.MULTILINE).strip()
                    repair_data = json.loads(clean_repair)
                    new_tool = repair_data.get("tool")
                    new_params = repair_data.get("params", {})
                    diagnosis = repair_data.get("diagnosis", "")

                    if new_tool and isinstance(new_params, dict):
                        logger.info(
                            "Recon Recovery diagnosis: '%s'. Retrying with '%s'(%s)...",
                            diagnosis,
                            new_tool,
                            new_params,
                        )
                        retry_data = self._dispatch_tool(new_tool, new_params)
                        if not self._is_recon_miss(retry_data):
                            logger.info("System 1 Recon Recovery succeeded! Recovered %d bytes.", len(retry_data))
                            raw_tool_data = (
                                f"[Recon Recovery: Resolved '{task.tool_name}' miss via {new_tool}({json.dumps(new_params)}) - {diagnosis}]\n\n"
                                + retry_data
                            )
                except Exception as e:
                    logger.debug("Recon Recovery parse or dispatch failed: %s", e)
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

                        if not item_error and not syn_info.get("has_substance", True):
                            item_error = True
                            last_err = syn_info.get("message", "Artifact contains only imports or docstrings with no substantive functions, classes, or logic.")
                            verification_findings.append(f"### Substance Verification (`{art_file.name}`): FAILED\n{last_err}")

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
                                    last_err = eval_res.get("error") or eval_res.get("output") or "Execution failed"
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

                                if not syn_check.get("has_substance", True):
                                    last_err = syn_check.get("message", "Patch contains only imports/docstrings without substantive code.")
                                    code_str = patch_code
                                    target_code_str = patch_code
                                    verification_findings.append(f"### Local Patch (Attempt {retry}/{self.max_repair_retries}): FAILED (Substance)\n{last_err}")
                                    continue

                                # If syntax and substance passed, test again
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
                                        last_err = eval_re.get("error") or eval_re.get("output") or "Execution failed"
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

        if task.note_id:
            linked_note = self.db.get_note(task.note_id, apply_mercy_hour=True)
            if linked_note and not linked_note.dismissed:
                user_prompt_parts.append(
                    f"## 📝 Attached Scratchpad Note (`#{linked_note.id}`: {linked_note.title}):\n{linked_note.content}"
                )

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

        # Inbound Pre-Flight Airlock Gate (for System 2 inbound briefing dossiers)
        if not is_verify and not is_sdr:
            word_count = len(dossier_content.split())
            if word_count > 300:
                logger.info("Dossier exceeds 300-word density budget (%d words). Triggering single-pass recompression.", word_count)
                recomp_prompt = (
                    f"The following briefing dossier exceeds the 300-word density budget (current: {word_count} words).\n"
                    f"Condense it strictly into fewer than 250 words without losing core facts, equations, or API signatures.\n"
                    f"Current Dossier:\n{dossier_content}\n\n"
                    f"Output ONLY valid JSON matching the schema."
                )
                try:
                    recomp_res = self.generate(recomp_prompt, system_prompt=system_prompt, stream=False, format="json")
                    recomp_content, recomp_model = parse_structured_dossier(recomp_res["response"], is_verify=False, is_sdr=False)
                    if recomp_model and recomp_model.word_count <= 350:
                        dossier_content = recomp_content
                        parsed_model = recomp_model
                    elif parsed_model and hasattr(parsed_model, "compact_truncate"):
                        dossier_content = parsed_model.compact_truncate(300)
                except Exception as e:
                    logger.debug("Recompression pass failed: %s", e)
                    if parsed_model and hasattr(parsed_model, "compact_truncate"):
                        dossier_content = parsed_model.compact_truncate(300)

            # Delimiter integrity check
            valid_delims, delim_err = validate_delimiters(dossier_content)
            if not valid_delims:
                logger.warning("Dossier delimiter check flagged: %s. Auto-sanitizing unclosed fences.", delim_err)
                if delim_err.startswith("Unclosed code fence"):
                    dossier_content += "\n```"

            # Embedded code syntax validation
            valid_code, code_errs = validate_embedded_code_syntax(dossier_content)
            if not valid_code:
                logger.warning("Dossier contains invalid Python code snippet: %s", code_errs)
                dossier_content += f"\n\n> [!WARNING]\n> Sanitizer Warning: Embedded code snippet has syntax errors: {', '.join(code_errs)}"

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
            diag = generate_diagnostic_dossier(raw_tool_data, target_code_str, max_words=150)
            raise VerificationFailedError(
                message=f"{summary_snippet}\n\n{diag}",
                artifact_path=str(artifact_path),
                failure_trace=raw_tool_data,
                target_code=target_code_str,
            )

        return summary_snippet, str(artifact_path), tokens, elapsed
