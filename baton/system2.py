"""System 2 Runner: Deep Macro-Planner & Nocturnal Synthesizer.

Powered by DeepSeek-V4-Flash via Colibrì (WSL2 CUDA). Tool-blind; consumes
only curated briefing dossiers and emits structured DAG plans or production artifacts.
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
    DAGPlan,
    Task,
    TaskCancelledError,
    TaskPlan,
    TaskProgress,
    TaskType,
    Tier,
)

logger = logging.getLogger("baton.system2")


def check_degenerate_repetition(text: str, min_chars: int = 100) -> bool:
    """Detects degenerate token loops (e.g. repeated single characters '0000...' or repeating cycles)."""
    if len(text) < min_chars:
        return False
    tail = text[-min_chars:]
    # 1. Single character repeating
    if len(set(tail)) == 1:
        return True
    # 2. Short repeating cycles of period 2 to 8
    for period in range(2, 9):
        pattern = tail[-period:]
        repeated = pattern * (min_chars // period + 1)
        if tail == repeated[-min_chars:]:
            return True
    return False


def sanitize_continuation_seam(prior: str, new_chunk: str, max_overlap: int = 80) -> str:
    """Stitches new_chunk onto prior content across a model continuation boundary.

    Resolves:
    1. Token stutter / overlapping prefix/suffix: If new_chunk repeats the tail of prior,
       the duplicate prefix in new_chunk is stripped.
    2. Missing newline between distinct statement boundaries: E.g. if prior ends with
       a complete statement and new_chunk begins with a top-level or indented statement
       without a newline.
    """
    if not prior:
        return new_chunk
    if not new_chunk:
        return ""

    cleaned_new = new_chunk

    # 1. Check for overlapping prefix/suffix (longest match first down to 1)
    max_search = min(len(prior), len(cleaned_new), max_overlap)
    best_overlap = 0
    for k in range(max_search, 0, -1):
        suffix = prior[-k:]
        prefix = cleaned_new[:k]
        if suffix == prefix:
            best_overlap = k
            break

    if best_overlap > 0:
        cleaned_new = cleaned_new[best_overlap:]

    # 2. Check for missing newline between statements
    # If prior ends without newline and cleaned_new starts without newline,
    # and prior ends with statement terminus (colon, closing paren, bracket, quote, semicolon, return/pass/break/continue)
    # and cleaned_new starts with a new statement keyword or assignment
    if prior and cleaned_new:
        prior_tail = prior.rstrip(" \t")
        if (
            not prior.endswith("\n")
            and not cleaned_new.startswith("\n")
            and prior_tail
            and (
                prior_tail[-1] in (":", ")", "]", "}", '"', "'", ";")
                or re.search(r"\b(return|pass|break|continue)(\s+[^\n]+)?$", prior_tail)
            )
            and re.match(
                r"^\s*(def |class |return |import |from |if |elif |else:|for |while |try:|except |finally:|with |@|[a-zA-Z_][a-zA-Z0-9_]*\s*=)",
                cleaned_new,
            )
        ):
            cleaned_new = "\n" + cleaned_new

    return cleaned_new



STATIC_SYNTHESIS_SYSTEM_PROMPT = (
    "You are Baton System 2: Deep Synthesizer & Principal Engineer.\n"
    "You write complete, production-grade, mathematically rigorous implementations and specifications.\n"
    "Reasoning & Execution Directives:\n"
    "- Think decisively: plan the essential architecture, formulas, and edge cases concisely in <think>.\n"
    "- Do not deliberate indefinitely or loop over minor stylistic conventions.\n"
    "- Once decided, immediately emit </think> and output the complete production code wrapped in ```python ... ```.\n"
    "- Include zero placeholders, ellipses (...), or pseudo-code.\n"
    "- Every function, formula, and class must be fully written and tested."
)


class System2Runner:
    """System 2 Deep Worker: Macro-Planning (Task DAGs) and Deep Nocturnal Synthesis."""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        model_name: Optional[str] = None,
        artifacts_dir: Optional[Path] = None,
        max_prompt_tokens: int = 1200,
        auto_continue: bool = True,
        max_continuations: int = 3,
        cache_prompt: bool = True,
        colibri_url: Optional[str] = None,
    ):
        resolved_url = (
            endpoint_url
            or colibri_url
            or os.getenv("BATON_ROUTER_URL")
            or os.getenv("BATON_API_URL")
            or os.getenv("BATON_SYSTEM2_URL")
            or os.getenv("BATON_COLIBRI_URL")
            or "http://127.0.0.1:8080"
        ).rstrip("/")
        self.endpoint_url = resolved_url
        self.colibri_url = resolved_url  # Backwards compatibility alias
        self.model_name = (
            model_name
            or os.getenv("BATON_SYSTEM2_MODEL")
            or "ornith"
        )
        self._discovered_model: Optional[str] = None
        self.artifacts_dir = artifacts_dir or Path("baton/artifacts/synthesis")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.max_prompt_tokens = max_prompt_tokens
        self.auto_continue = auto_continue
        self.max_continuations = max_continuations
        self.cache_prompt = cache_prompt
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

    def get_model_id(self) -> str:
        """Dynamically discovers the model ID currently registered with Colibrì / llama-server."""
        if self._discovered_model:
            return self._discovered_model
        if self.model_name:
            return self.model_name
        try:
            res = requests.get(f"{self.endpoint_url}/v1/models", timeout=2.0)
            if res.status_code == 200:
                data = res.json()
                models = data.get("data", [])
                # In router mode where multiple models are registered, select Ornith
                for m in models:
                    m_id = m.get("id", "")
                    aliases = [a.lower() for a in m.get("aliases", [])]
                    if "ornith" in m_id.lower() or any("ornith" in a for a in aliases):
                        self._discovered_model = "ornith"
                        logger.info("Discovered System 2 Ornith model: %s", self._discovered_model)
                        return self._discovered_model

                if models and "id" in models[0]:
                    self._discovered_model = models[0]["id"]
                    logger.info("Discovered System 2 model ID: %s", self._discovered_model)
                    return self._discovered_model
        except Exception as e:
            logger.debug("Failed to query System 2 models: %s", e)
        return "ornith"

    def _post_with_retry(
        self,
        endpoint: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        stream: bool,
        timeout: float | tuple[float, float] | None,
        max_retries: int = 3,
        backoff_delays: tuple[float, ...] = (3.0, 6.0, 12.0),
    ) -> requests.Response:
        """Sends POST request to Colibrì with exponential backoff on HTTP 429."""
        for attempt in range(max_retries + 1):
            if self._cancel_requested:
                raise TaskCancelledError("System 2 generation stopped by user.")
            res = requests.post(endpoint, json=payload, headers=headers, stream=stream, timeout=timeout)
            if res.status_code == 429:
                if attempt < max_retries:
                    delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                    logger.warning(
                        "Colibrì returned HTTP 429 (Too Many Requests). Backing off for %.1fs (attempt %d/%d)...",
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    sleep_intervals = int(delay / 0.1)
                    for _ in range(sleep_intervals):
                        if self._cancel_requested:
                            res.close()
                            raise TaskCancelledError("System 2 generation stopped by user.")
                        time.sleep(0.1)
                    continue
            res.raise_for_status()
            return res
        res.raise_for_status()
        return res

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        timeout: float | tuple[float, float] | None = (120.0, 600.0),
        stream: bool = True,
        progress: Optional[TaskProgress] = None,
        progress_callback: Optional[Callable[[TaskProgress], None]] = None,
        initial_tokens: int = 0,
        initial_content: str = "",
        initial_reasoning: str = "",
        cache_prompt: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Calls OpenAI-compatible chat completion endpoint with optional SSE streaming."""
        endpoint = f"{self.endpoint_url}/v1/chat/completions"
        active_model = self.get_model_id()
        use_cache = self.cache_prompt if cache_prompt is None else cache_prompt

        payload: dict[str, Any] = {
            "model": active_model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if use_cache:
            # Tell llama.cpp / Colibri to pin and reuse prompt KV cache
            payload["cache_prompt"] = True
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stream:
            payload["stream_options"] = {"include_usage": True}

        headers = {
            "Content-Type": "application/json",
            "X-Baton-Cache-Prefix": "pinned" if use_cache else "bypass",
        }

        t0 = time.time()
        self._cancel_requested = False

        if not stream:
            if self._cancel_requested:
                raise TaskCancelledError("System 2 generation stopped by user.")
            res = self._post_with_retry(endpoint, payload, headers, stream=False, timeout=timeout)
            data = res.json()
            elapsed = time.time() - t0

            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "").strip()
            finish_reason = choice.get("finish_reason")
            usage = data.get("usage", {})
            tokens = usage.get("completion_tokens", 0)

            return {
                "content": content,
                "reasoning_content": "",
                "tokens_generated": tokens,
                "elapsed_seconds": elapsed,
                "finish_reason": finish_reason,
                "is_truncated": (finish_reason == "length"),
            }

        # Streaming mode (SSE)
        try:
            res = self._post_with_retry(endpoint, payload, headers, stream=True, timeout=timeout)
            self._active_response = res

            content_chunks: list[str] = []
            reasoning_chunks: list[str] = []
            tokens_count = 0
            finish_reason: Optional[str] = None
            last_callback_time = 0.0

            try:
                for raw_line in res.iter_lines(decode_unicode=True):
                    if self._cancel_requested:
                        raise TaskCancelledError("System 2 generation stopped by user.")
                    if not raw_line:
                        continue
                    if isinstance(raw_line, bytes):
                        line = raw_line.decode("utf-8", errors="replace").strip()
                    else:
                        line = str(raw_line).strip()
                    if not line.startswith("data:"):
                        continue

                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except Exception:
                        continue

                    choices = chunk.get("choices", [])
                    has_delta = False
                    if choices:
                        choice_obj = choices[0]
                        fr = choice_obj.get("finish_reason")
                        if fr:
                            finish_reason = fr
                        delta = choice_obj.get("delta", {})

                        # Check for DeepSeek-V4-Flash thinking tokens
                        r_delta = delta.get("reasoning_content", "")
                        if r_delta:
                            reasoning_chunks.append(r_delta)
                            tokens_count += 1
                            has_delta = True

                        # Check for drafting output tokens
                        c_delta = delta.get("content", "")
                        if c_delta:
                            content_chunks.append(c_delta)
                            tokens_count += 1
                            has_delta = True

                    usage = chunk.get("usage")
                    if usage and isinstance(usage, dict):
                        comp = usage.get("completion_tokens")
                        if comp:
                            tokens_count = comp

                    # Check for degenerate repetition loops (e.g. infinite 00000... or repeating cycles)
                    if content_chunks:
                        tail_content = "".join(content_chunks[-120:])
                        if check_degenerate_repetition(tail_content, min_chars=100):
                            logger.warning(
                                "Degenerate repetition detected in System 2 content! Breaking stream to preserve valid code."
                            )
                            finish_reason = "repetition_aborted"
                            break

                    if reasoning_chunks:
                        tail_reasoning = "".join(reasoning_chunks[-120:])
                        if check_degenerate_repetition(tail_reasoning, min_chars=100):
                            logger.warning(
                                "Degenerate repetition detected in System 2 reasoning! Breaking stream."
                            )
                            finish_reason = "repetition_aborted"
                            break

                    now = time.time()
                    if has_delta and progress:
                        cum_tokens = initial_tokens + tokens_count
                        progress.reasoning_text = initial_reasoning + "".join(reasoning_chunks)
                        progress.content_text = initial_content + "".join(content_chunks)
                        progress.tokens_generated = cum_tokens
                        progress.tokens_per_second = cum_tokens / max(now - progress.started_at, 0.001)
                        progress.last_update = now
                        progress.finish_reason = finish_reason
                        progress.is_truncated = (finish_reason == "length")

                        # Throttle progress callbacks slightly to 50ms intervals
                        if progress_callback and (now - last_callback_time >= 0.05):
                            last_callback_time = now
                            progress_callback(progress)

            except (requests.exceptions.RequestException, OSError):
                if self._cancel_requested:
                    raise TaskCancelledError("System 2 generation stopped by user.")
                raise
        finally:
            self._active_response = None

        elapsed = time.time() - t0
        final_content = "".join(content_chunks)
        final_reasoning = "".join(reasoning_chunks)
        total_tokens = initial_tokens + tokens_count

        if progress:
            progress.reasoning_text = initial_reasoning + final_reasoning
            progress.content_text = initial_content + final_content
            progress.tokens_generated = total_tokens
            progress.tokens_per_second = total_tokens / max(elapsed, 0.001)
            progress.last_update = time.time()
            progress.finish_reason = finish_reason
            progress.is_truncated = (finish_reason == "length")
            progress.is_completed = True
            if progress_callback:
                progress_callback(progress)

        return {
            "content": final_content,
            "reasoning_content": final_reasoning,
            "tokens_generated": tokens_count,
            "elapsed_seconds": elapsed,
            "finish_reason": finish_reason,
            "is_truncated": (finish_reason == "length"),
        }

    # -------------------------------------------------------------------------
    # Macro-Planning: Decomposes User Goal into a Structured DAG
    # -------------------------------------------------------------------------

    def plan_project(
        self,
        project_name: str,
        goal: str,
        system1_repair_runner: Optional[Any] = None,
    ) -> DAGPlan:
        """Prompts System 2 to decompose the goal into a valid typed Task DAG with outbound airlock validation."""
        system_prompt = (
            "You are Baton System 2: Chief Architect & Macro-Planner.\n"
            "Your role is to decompose high-level projects into a Directed Acyclic Graph (DAG) of discrete tasks.\n"
            "Task Types & Tiers:\n"
            "- 'fast_recon' (tier: 'system_1'): Offline knowledge extraction (tools: wiki_read_section, wiki_search, rss_search).\n"
            "- 'deep_synthesis' (tier: 'system_2'): Pure mathematical derivations, multi-module architecture, or complete production code.\n"
            "- 'fast_verify' (tier: 'system_1'): Deterministic code verification and automated test execution (tools: pytest_run, python_syntax_check, python_eval) depending on upstream synthesis.\n\n"
            "You must output ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "project": "...",\n'
            '  "goal": "...",\n'
            '  "tasks": [\n'
            '    {\n'
            '      "id": "recon_xyz",\n'
            '      "title": "Short title",\n'
            '      "type": "fast_recon",\n'
            '      "tier": "system_1",\n'
            '      "tool": "wiki_read_section",\n'
            '      "params": {"title": "Article", "section": "Section"},\n'
            '      "depends_on": [],\n'
            '      "description": "What to extract",\n'
            '      "priority": 10\n'
            '    },\n'
            '    {\n'
            '      "id": "synth_abc",\n'
            '      "title": "Synthesis title",\n'
            '      "type": "deep_synthesis",\n'
            '      "tier": "system_2",\n'
            '      "depends_on": ["recon_xyz"],\n'
            '      "prompt": "Detailed instructions",\n'
            '      "output_artifact": "filename.py",\n'
            '      "priority": 1\n'
            '    },\n'
            '    {\n'
            '      "id": "verify_abc",\n'
            '      "title": "Verify Synthesis",\n'
            '      "type": "fast_verify",\n'
            '      "tier": "system_1",\n'
            '      "depends_on": ["synth_abc"],\n'
            '      "tool": "pytest_run",\n'
            '      "params": {"target": "filename.py"},\n'
            '      "description": "Run pytest and validate syntax on synthesized module",\n'
            '      "priority": 1\n'
            '    }\n'
            '  ]\n'
            "}"
        )

        user_prompt = f"Project: {project_name}\nGoal: {goal}\n\nDecompose this project into a valid DAG now:"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info("System 2 Macro-Planner decomposing project '%s'...", project_name)
        result = self.chat_completion(messages, temperature=0.1, timeout=600)
        raw_json = result["content"]

        # Clean markdown code fences if model enclosed in ```json ... ```
        clean_json = re.sub(r"^```(?:json)?\s*", "", raw_json, flags=re.MULTILINE)
        clean_json = re.sub(r"\s*```$", "", clean_json, flags=re.MULTILINE).strip()

        try:
            data = json.loads(clean_json)
            plan = DAGPlan.model_validate(data)
            plan.validate_dag()
            return plan
        except Exception as e:
            logger.warning("Outbound DAG Airlock: System 2 output validation failed: %s", e)
            if system1_repair_runner is not None:
                from baton.system1 import repair_dag_plan_json
                logger.info("Engaging System 1 outbound airlock repair for System 2 DAGPlan...")
                repaired_plan, repair_msg = repair_dag_plan_json(
                    system1_repair_runner, raw_json, str(e)
                )
                if repaired_plan is not None:
                    logger.info("Outbound DAG Airlock: System 1 successfully repaired plan: %s", repair_msg)
                    return repaired_plan
                logger.warning("System 1 repair also failed: %s", repair_msg)
            raise

    @staticmethod
    def sort_upstream_dossiers(
        upstream_dossiers: Optional[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Sorts upstream dossiers deterministically by completed_at, task_id, and title."""
        if not upstream_dossiers:
            return []
        return sorted(
            upstream_dossiers,
            key=lambda d: (
                str(d.get("completed_at") or ""),
                str(d.get("task_id") or ""),
                str(d.get("title") or ""),
            ),
        )

    def build_canonical_prompt(
        self,
        task: Task,
        upstream_dossiers: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, str]]:
        """
        Builds canonical, prefix-pinned messages for System 2 synthesis.
        Enforces strict ordering to maximize KV-cache reuse in Colibrì:
        1. Static System Prompt (invariant, zero timestamps or dynamic tokens)
        2. User Message:
           a. Immutable Upstream Briefing Dossiers (sorted deterministically)
           b. Task Objective and Instructions
           c. Action Directive
        """
        prompt_parts: list[str] = []

        # 1. Immutable Upstream Context/Dossiers (sorted deterministically)
        sorted_dossiers = self.sort_upstream_dossiers(upstream_dossiers)
        if sorted_dossiers:
            prompt_parts.append("## Briefing Dossiers from System 1 Reconnaissance:")
            for d in sorted_dossiers:
                payload = d.get("result_payload", "").strip()
                max_chars = int((self.max_prompt_tokens / max(len(sorted_dossiers), 1)) * 3.5)
                budgeted_payload = payload[:max_chars]
                title = d.get("title") or "Dossier"
                prompt_parts.append(f"### Upstream Finding: {title}\n{budgeted_payload}\n")

        # 2. Task Goal / Instructions
        prompt_parts.append(f"# Objective: {task.title}")
        prompt_parts.append(f"Instructions:\n{task.prompt_template or ''}")

        # 3. Action Directive
        prompt_parts.append(
            "\nReasoning & Output Protocol:\n"
            "1. Plan your solution concisely in <think>. Keep reasoning focused on essential architecture and equations (under 500-1000 tokens).\n"
            "2. Conclude reasoning, close the thinking block (</think>), and write the code.\n"
            "3. Enclose all production code inside a standard markdown code block (```python ... ```) with self-contained unit tests.\n\n"
            "Generate the complete, robust implementation now:"
        )

        full_user_prompt = "\n".join(prompt_parts)

        return [
            {"role": "system", "content": STATIC_SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": full_user_prompt},
        ]

    # -------------------------------------------------------------------------
    # Deep Synthesis: Nocturnal Code & Specification Generation
    # -------------------------------------------------------------------------

    def synthesize(
        self,
        task: Task,
        upstream_dossiers: Optional[list[dict[str, Any]]] = None,
        progress_callback: Optional[Callable[[TaskProgress], None]] = None,
        auto_continue: Optional[bool] = None,
        max_continuations: Optional[int] = None,
    ) -> tuple[str, Optional[str], int, float]:
        """
        Executes a deep synthesis task by combining task instructions with upstream dossiers.
        Enforces strict context budgeting (<= max_prompt_tokens).
        Returns: (result_summary, artifact_path, tokens_generated, elapsed_seconds)
        """
        t0 = time.time()
        messages = self.build_canonical_prompt(task, upstream_dossiers)

        logger.info("System 2 executing deep synthesis for task '%s'...", task.task_id)

        tool_params = task.tool_params or {}
        max_tokens = tool_params.get("max_tokens")
        max_conts = max_continuations if max_continuations is not None else tool_params.get("max_continuations", self.max_continuations)
        auto_cont = auto_continue if auto_continue is not None else tool_params.get("auto_continue", self.auto_continue)

        progress = TaskProgress(
            task_id=task.task_id,
            tier=Tier.SYSTEM_2,
            started_at=t0,
        )

        res = self.chat_completion(
            messages,
            temperature=0.2,
            max_tokens=max_tokens,
            timeout=(120.0, 600.0),
            stream=True,
            progress=progress,
            progress_callback=progress_callback,
        )
        output_code = res["content"]
        total_tokens = res["tokens_generated"]
        reasoning = res.get("reasoning_content", "")
        finish_reason = res.get("finish_reason")
        continuations_performed = 0

        # Seamless turn continuation if generation hit token ceiling and auto_continue is enabled
        while finish_reason == "length" and auto_cont and (continuations_performed < max_conts):
            if self._cancel_requested:
                break
            continuations_performed += 1
            logger.info(
                "System 2 output reached token ceiling (finish_reason='length') after %d tokens. Continuing generation (continuation turn %d/%d)...",
                total_tokens,
                continuations_performed,
                max_conts,
            )

            # Colibrì native continuation contract:
            # If output_code is empty, the model hit token limit while still inside reasoning (<think>)
            if not output_code:
                continue_messages = list(messages) + [
                    {
                        "role": "user",
                        "content": (
                            "You reached the token ceiling during reasoning. "
                            "Conclude your thinking immediately with </think> and output the complete production code "
                            "now inside a ```python ``` code block."
                        ),
                    }
                ]
            else:
                continue_messages = list(messages) + [
                    {"role": "assistant", "content": output_code}
                ]

            res = self.chat_completion(
                continue_messages,
                temperature=0.2,
                max_tokens=max_tokens,
                timeout=(120.0, 600.0),
                stream=True,
                progress=progress,
                progress_callback=progress_callback,
                initial_tokens=total_tokens,
                initial_content=output_code,
                initial_reasoning=reasoning,
            )
            new_content = res["content"]
            new_tokens = res["tokens_generated"]
            new_reasoning = res.get("reasoning_content", "")
            finish_reason = res.get("finish_reason")

            if not new_content and new_tokens == 0:
                logger.warning("Continuation turn produced no additional tokens. Ending continuation loop.")
                break

            sanitized_new = sanitize_continuation_seam(output_code, new_content)
            output_code += sanitized_new
            total_tokens += new_tokens
            if progress:
                progress.content_text = output_code
            if new_reasoning:
                reasoning = (reasoning + "\n" + new_reasoning).strip() if reasoning else new_reasoning

        elapsed = time.time() - t0

        # Salvage code from reasoning monologue if output_code is empty but code was drafted inside <think>
        if not output_code.strip() and reasoning:
            code_match = re.search(r"```(?:python)?\s*\n(.*?)\n```", reasoning, re.DOTALL | re.IGNORECASE)
            if not code_match:
                code_match = re.search(r"```(?:python)?\s*\n(.*)", reasoning, re.DOTALL | re.IGNORECASE)
            if code_match:
                salvaged = code_match.group(1).rstrip("`").strip()
                if salvaged:
                    logger.warning(
                        "System 2 model generated code inside reasoning (<think>) block without closing </think>. "
                        "Successfully salvaged %d characters of production code from reasoning monologue.",
                        len(salvaged),
                    )
                    output_code = salvaged
            else:
                tail_code_match = re.search(
                    r"\n((?:from\s+\S+\s+import|import\s+\S+|def\s+\w+|class\s+\w+)[\s\S]{50,})$",
                    reasoning,
                )
                if tail_code_match:
                    candidate = tail_code_match.group(1).strip()
                    logger.warning(
                        "System 2 model drafted raw Python code in reasoning tail without code fences. "
                        "Salvaged %d characters of code.",
                        len(candidate),
                    )
                    output_code = candidate

        if progress:
            progress.is_completed = True
            progress.finish_reason = finish_reason
            progress.is_truncated = (finish_reason == "length")
            if progress_callback:
                progress_callback(progress)

        # Determine output artifact filename
        filename = f"{task.task_id}_output.txt"
        if task.tool_params and "output_artifact" in task.tool_params:
            filename = task.tool_params["output_artifact"]
        elif task.prompt_template and ".py" in task.prompt_template:
            # Guess filename if mentioned
            match = re.search(r"([a-zA-Z0-9_\-]+\.py)", task.prompt_template)
            if match:
                filename = match.group(1)
        elif (
            task.task_type == TaskType.DEEP_SYNTHESIS
            or any(
                kw in (task.title + " " + (task.prompt_template or "")).lower()
                for kw in ("python", "function", "class", "module", "script", "def ")
            )
            or output_code.startswith("def ")
            or output_code.startswith("import ")
            or output_code.startswith("from ")
            or "```python" in reasoning
        ):
            filename = f"{task.task_id}.py"

        artifact_path = self.artifacts_dir / filename
        artifact_path.write_text(output_code.strip(), encoding="utf-8")

        # If thinking/reasoning was captured, save companion thinking log
        if reasoning:
            thinking_path = self.artifacts_dir / f"{task.task_id}_thinking.md"
            thinking_header = f"# Reasoning Monologue: {task.title}\n*Generated by DeepSeek-V4-Flash on {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
            thinking_path.write_text(thinking_header + reasoning, encoding="utf-8")

        if finish_reason == "length":
            trunc_note = f" [TRUNCATED: hit context budget ceiling after {continuations_performed} continuation(s)]"
            summary = f"Synthesized {total_tokens} tokens into {artifact_path.name} in {elapsed:.1f}s{trunc_note}"
        else:
            cont_note = f" (across {continuations_performed + 1} turns)" if continuations_performed > 0 else ""
            summary = f"Synthesized {total_tokens} tokens into {artifact_path.name} in {elapsed:.1f}s{cont_note}"
        return summary, str(artifact_path), total_tokens, elapsed
