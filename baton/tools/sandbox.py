"""Sandboxed Python REPL & Verification Tool for Baton System 1.

Provides safe Python code execution, AST syntax verification, file staging,
and pytest test suite execution with Windows process-tree termination guards.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("baton.tools.sandbox")


def get_default_python_executable() -> str:
    """
    Finds the preferred Python executable for sandboxed execution.
    Prioritizes the active Python interpreter (sys.executable) or active virtualenv/conda,
    falling back to Anaconda3 or system PATH python.
    """
    if sys.executable and Path(sys.executable).exists():
        return sys.executable
    if "VIRTUAL_ENV" in os.environ:
        venv_py = Path(os.environ["VIRTUAL_ENV"]) / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
        if venv_py.exists():
            return str(venv_py)
    if "CONDA_PREFIX" in os.environ:
        conda_py = Path(os.environ["CONDA_PREFIX"]) / ("python.exe" if os.name == "nt" else "bin/python")
        if conda_py.exists():
            return str(conda_py)
    conda_exe = Path("C:/ProgramData/Anaconda3/python.exe")
    if conda_exe.exists():
        return str(conda_exe)
    return "python"


def windows_safe_run(
    cmd: list[str] | str,
    cwd: Path | str,
    env: dict[str, str],
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    """
    Executes a command with strict timeout and process-tree termination.
    On Windows, uses 'taskkill /F /T /PID' to guarantee no orphaned child processes.
    """
    is_win = os.name == "nt"
    is_shell = isinstance(cmd, str)

    proc = subprocess.Popen(
        cmd,
        shell=is_shell,
        text=True,
        cwd=str(cwd),
        env=env,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=not is_win,
    )

    try:
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if is_win:
            try:
                subprocess.run(
                    f"taskkill /F /T /PID {proc.pid}",
                    shell=True,
                    capture_output=True,
                    check=False,
                )
            except Exception as e:
                logger.warning("Failed to taskkill PID %s: %s", proc.pid, e)
            proc.kill()
        else:
            try:
                import signal
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()

        try:
            stdout, _ = proc.communicate(timeout=3)
        except Exception:
            stdout = ""
        raise subprocess.TimeoutExpired(cmd, timeout, output=stdout)

    return subprocess.CompletedProcess(cmd, proc.returncode, stdout=stdout)


def slice_text_by_lines(
    text: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    max_chars: Optional[int] = 3500,
) -> dict[str, Any]:
    """
    Slices text by 1-indexed line numbers and/or max characters budget.
    Returns preview metadata and the sliced content without wordy conversational distractors.
    """
    if not text:
        return {
            "content": "",
            "preview": "0 lines (0 chars)",
            "start_line": 0,
            "end_line": 0,
            "total_lines": 0,
            "total_chars": 0,
            "is_truncated": False,
        }

    lines = text.splitlines(keepends=True)
    total_lines = len(lines)
    total_chars = len(text)

    req_start = max(1, int(start_line)) if start_line is not None else 1
    req_end = min(total_lines, max(req_start, int(end_line))) if end_line is not None else total_lines

    if req_start > total_lines:
        return {
            "content": "",
            "preview": f"Lines {req_start}-{req_end} of {total_lines} (0 of {total_chars} chars)",
            "start_line": req_start,
            "end_line": req_end,
            "total_lines": total_lines,
            "total_chars": total_chars,
            "is_truncated": True,
        }

    sub_lines = lines[req_start - 1 : req_end]
    char_limit = max_chars if (max_chars is not None and max_chars > 0) else float("inf")

    acc_chars = 0
    included_count = 0
    for line in sub_lines:
        if acc_chars + len(line) > char_limit and included_count > 0:
            break
        acc_chars += len(line)
        included_count += 1

    if included_count == 0 and sub_lines:
        content = sub_lines[0][: int(char_limit)]
        actual_start = req_start
        actual_end = req_start
    else:
        content = "".join(sub_lines[:included_count])
        actual_start = req_start
        actual_end = req_start + included_count - 1

    is_truncated = (
        len(content) < total_chars
        or actual_start > 1
        or actual_end < total_lines
    )

    line_range_str = (
        f"Line {actual_start}"
        if actual_start == actual_end
        else f"Lines {actual_start}-{actual_end}"
    )
    preview = f"{line_range_str} of {total_lines} ({len(content)} of {total_chars} chars)"

    return {
        "content": content,
        "preview": preview,
        "start_line": actual_start,
        "end_line": actual_end,
        "total_lines": total_lines,
        "total_chars": total_chars,
        "is_truncated": is_truncated,
    }


class SandboxTool:
    """
    Isolated execution environment for System 1 code verification.
    """

    def __init__(
        self,
        sandbox_dir: Optional[Path | str] = None,
        python_exe: Optional[str] = None,
        default_timeout: int = 30,
    ):
        self.sandbox_dir = Path(sandbox_dir) if sandbox_dir else Path("baton/artifacts/sandbox")
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.python_exe = python_exe or get_default_python_executable()
        self.default_timeout = default_timeout

    def _get_env(self) -> dict[str, str]:
        """Builds clean execution environment with python and project in paths."""
        env = dict(os.environ)
        py_dir = str(Path(self.python_exe).parent)
        scripts_dir = str(Path(py_dir) / "Scripts")

        sep = ";" if os.name == "nt" else ":"
        existing_path = env.get("PATH", "")
        env["PATH"] = f"{py_dir}{sep}{scripts_dir}{sep}{existing_path}"

        # Injects sandbox dir and project root into PYTHONPATH
        project_root = str(Path(__file__).resolve().parent.parent.parent)
        sandbox_path = str(self.sandbox_dir.resolve())
        existing_pypath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{sandbox_path}{sep}{project_root}{sep}{existing_pypath}"
            if existing_pypath
            else f"{sandbox_path}{sep}{project_root}"
        )
        return env

    def _resolve_safe_path(self, filename: str) -> Path:
        """Resolves path within sandbox directory and blocks directory traversal."""
        safe_sandbox = self.sandbox_dir.resolve()
        target = (safe_sandbox / filename).resolve()
        try:
            target.relative_to(safe_sandbox)
        except ValueError:
            raise PermissionError(f"Access denied: '{filename}' escapes sandbox directory.")
        return target

    # -------------------------------------------------------------------------
    # 1. AST Syntax Checking (<1ms)
    # -------------------------------------------------------------------------

    def syntax_check(self, code: str) -> dict[str, Any]:
        """
        Validates Python syntax via native AST parsing without executing code.
        Returns line numbers and error diagnostics if invalid, and verifies
        that the code contains substantive statements (not just comments or imports).
        """
        try:
            tree = ast.parse(code)
            # Identify substantive logic: functions, classes, assignments, control flow, calls
            substantive_nodes = [
                node for node in tree.body
                if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
                and not isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass))
            ]
            has_substance = len(substantive_nodes) > 0
            msg = (
                "Syntax valid. No parsing errors detected."
                if has_substance
                else "Syntax valid, but code contains only imports, comments, or docstrings without substantive functions, classes, or logic."
            )
            return {
                "valid": True,
                "has_substance": has_substance,
                "message": msg,
            }
        except SyntaxError as e:
            return {
                "valid": False,
                "has_substance": False,
                "error": e.msg,
                "line": e.lineno,
                "offset": e.offset,
                "text": e.text.strip() if e.text else "",
                "message": f"SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}",
            }
        except Exception as e:
            return {
                "valid": False,
                "has_substance": False,
                "error": str(e),
                "message": f"Parsing error: {e}",
            }

    # -------------------------------------------------------------------------
    # 2. File Staging & Reading
    # -------------------------------------------------------------------------

    def write_file(self, filename: str, content: str) -> dict[str, Any]:
        """Writes source code or test file into the sandbox."""
        try:
            target = self._resolve_safe_path(filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return {
                "success": True,
                "path": str(target),
                "filename": filename,
                "size_bytes": len(content.encode("utf-8")),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def read_file(
        self,
        filename: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        max_chars: Optional[int] = 3500,
    ) -> dict[str, Any]:
        """Reads a file from the sandbox directory with optional line range slicing."""
        try:
            target = self._resolve_safe_path(filename)
            if not target.exists() or not target.is_file():
                return {"success": False, "error": f"File '{filename}' does not exist in sandbox."}
            content = target.read_text(encoding="utf-8", errors="replace")
            sliced = slice_text_by_lines(
                content,
                start_line=start_line,
                end_line=end_line,
                max_chars=max_chars,
            )
            return {
                "success": True,
                "filename": filename,
                "preview": sliced["preview"],
                "content": sliced["content"],
                "start_line": sliced["start_line"],
                "end_line": sliced["end_line"],
                "total_lines": sliced["total_lines"],
                "total_chars": sliced["total_chars"],
                "is_truncated": sliced["is_truncated"],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_files(self) -> list[dict[str, Any]]:
        """Lists all files resident in the sandbox."""
        results = []
        safe_sandbox = self.sandbox_dir.resolve()
        for p in safe_sandbox.rglob("*"):
            if p.is_file():
                rel = p.relative_to(safe_sandbox).as_posix()
                results.append({"filename": rel, "size_bytes": p.stat().st_size})
        return results

    # -------------------------------------------------------------------------
    # 3. Python Code Execution (`python_eval`)
    # -------------------------------------------------------------------------

    def eval_code(
        self,
        code: str,
        filename: str = "_snippet.py",
        timeout: Optional[int] = None,
        max_chars: int = 3500,
    ) -> dict[str, Any]:
        """
        Executes a Python code snippet or script in the isolated sandbox.
        Captures stdout, stderr, return code, and execution time.
        """
        t0 = time.time()
        timeout_sec = timeout if timeout is not None else self.default_timeout

        # First, run instant syntax check
        check = self.syntax_check(code)
        if not check["valid"]:
            return {
                "success": False,
                "returncode": 1,
                "stdout": "",
                "stderr": check["message"],
                "error": check["message"],
                "elapsed_seconds": 0.0,
                "syntax_error": check,
            }

        # Stage snippet into sandbox
        write_res = self.write_file(filename, code)
        if not write_res.get("success"):
            err_msg = f"Failed to stage script: {write_res.get('error')}"
            return {
                "success": False,
                "returncode": 1,
                "stdout": "",
                "stderr": err_msg,
                "error": err_msg,
                "elapsed_seconds": 0.0,
            }

        script_target = self._resolve_safe_path(filename)
        cmd = [self.python_exe, str(script_target)]
        env = self._get_env()

        try:
            res = windows_safe_run(cmd, cwd=self.sandbox_dir, env=env, timeout=timeout_sec)
            elapsed = time.time() - t0
            raw_output = res.stdout or ""
            is_truncated = len(raw_output) > max_chars
            has_error = (res.returncode != 0)
            err_msg = raw_output[:max_chars] if has_error else None
            return {
                "success": not has_error,
                "returncode": res.returncode,
                "output": raw_output[:max_chars],
                "error": err_msg,
                "is_truncated": is_truncated,
                "total_chars": len(raw_output),
                "elapsed_seconds": round(elapsed, 3),
            }
        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - t0
            out = e.output or ""
            return {
                "success": False,
                "returncode": 124,
                "output": out[:max_chars],
                "error": f"Execution timed out after {timeout_sec}s (process tree terminated).",
                "timed_out": True,
                "elapsed_seconds": round(elapsed, 3),
            }
        except Exception as e:
            elapsed = time.time() - t0
            return {
                "success": False,
                "returncode": -1,
                "output": "",
                "error": str(e),
                "elapsed_seconds": round(elapsed, 3),
            }

    # -------------------------------------------------------------------------
    # 4. Pytest Execution (`pytest_run`)
    # -------------------------------------------------------------------------

    def run_pytest(
        self,
        target: str = ".",
        timeout: Optional[int] = None,
        max_chars: int = 3500,
    ) -> dict[str, Any]:
        """
        Executes pytest in the sandbox and parses structured pass/fail metrics.
        """
        t0 = time.time()
        timeout_sec = timeout if timeout is not None else (self.default_timeout * 2)

        cmd = [self.python_exe, "-m", "pytest", target, "-v"]
        env = self._get_env()

        try:
            res = windows_safe_run(cmd, cwd=self.sandbox_dir, env=env, timeout=timeout_sec)
            elapsed = time.time() - t0
            raw_out = res.stdout or ""

            # Parse structured summary from pytest output: e.g. "5 passed, 1 failed in 0.45s"
            metrics = self._parse_pytest_summary(raw_out)
            is_truncated = len(raw_out) > max_chars

            return {
                "success": (res.returncode == 0),
                "returncode": res.returncode,
                "metrics": metrics,
                "output": raw_out[:max_chars],
                "error": raw_out[:max_chars] if res.returncode != 0 else None,
                "is_truncated": is_truncated,
                "total_chars": len(raw_out),
                "elapsed_seconds": round(elapsed, 3),
            }
        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - t0
            out = e.output or ""
            return {
                "success": False,
                "returncode": 124,
                "metrics": {"passed": 0, "failed": 0, "errors": 1},
                "output": out[:max_chars],
                "error": f"Pytest execution timed out after {timeout_sec}s.",
                "timed_out": True,
                "elapsed_seconds": round(elapsed, 3),
            }
        except Exception as e:
            elapsed = time.time() - t0
            return {
                "success": False,
                "returncode": -1,
                "metrics": {"passed": 0, "failed": 0, "errors": 1},
                "output": "",
                "error": str(e),
                "elapsed_seconds": round(elapsed, 3),
            }

    @staticmethod
    def _parse_pytest_summary(output: str) -> dict[str, Any]:
        """Extracts passed, failed, error, and skipped counts from pytest output."""
        summary = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "total": 0}
        # Matches pytest summary line: "==== 2 passed, 1 failed, 1 error in 0.23s ===="
        for line in reversed(output.splitlines()):
            if "passed" in line or "failed" in line or "error" in line:
                m_pass = re.search(r"(\d+)\s+passed", line)
                if m_pass:
                    summary["passed"] = int(m_pass.group(1))

                m_fail = re.search(r"(\d+)\s+failed", line)
                if m_fail:
                    summary["failed"] = int(m_fail.group(1))

                m_err = re.search(r"(\d+)\s+error", line)
                if m_err:
                    summary["errors"] = int(m_err.group(1))

                m_skip = re.search(r"(\d+)\s+skipped", line)
                if m_skip:
                    summary["skipped"] = int(m_skip.group(1))

                summary["total"] = (
                    summary["passed"] + summary["failed"] + summary["errors"] + summary["skipped"]
                )
                if summary["total"] > 0:
                    break
        return summary

    # -------------------------------------------------------------------------
    # String Formatting Dispatcher for System 1 Prompts
    # -------------------------------------------------------------------------

    def execute(self, tool_name: str, params: dict[str, Any]) -> str:
        """Deterministic dispatcher for named tool calls, returning clean Markdown."""
        name = tool_name.lower().strip()

        if name in ("python_syntax_check", "syntax_check"):
            code = params.get("code", "")
            if not code and "filename" in params:
                read_res = self.read_file(params["filename"], max_chars=None)
                code = read_res.get("content", "")
            if not code:
                return "Error: No code provided for syntax checking."

            res = self.syntax_check(code)
            if res["valid"]:
                return "### Python Syntax Check: PASSED\nNo syntax or AST parsing errors detected."
            return (
                f"### Python Syntax Check: FAILED\n"
                f"- **Error:** {res.get('error')}\n"
                f"- **Location:** Line {res.get('line')}, Col {res.get('offset')}\n"
                f"- **Code snippet:** `{res.get('text')}`\n"
            )

        elif name in ("python_eval", "eval_code", "python_run"):
            code = params.get("code", "")
            filename = params.get("filename", "_snippet.py")
            timeout = params.get("timeout")
            if not code and filename and filename != "_snippet.py":
                file_res = self.read_file(filename, max_chars=None)
                if file_res.get("success"):
                    code = file_res.get("content", "")
            if not code:
                return "Error: Code parameter cannot be empty."

            res = self.eval_code(code, filename=filename, timeout=timeout)
            status_str = "PASSED (Code 0)" if res.get("success") else f"FAILED (Code {res.get('returncode')})"
            trunc_str = f" *(Truncated at {len(res['output'])} of {res['total_chars']} chars)*" if res.get("is_truncated") else ""
            err_str = f"\n**Error:** {res['error']}" if res.get("error") else ""

            return (
                f"### Python Execution: {status_str}\n"
                f"*Duration: {res.get('elapsed_seconds', 0.0)}s*{err_str}\n\n"
                f"```text\n{res.get('output', '').strip()}\n```\n{trunc_str}".strip()
            )

        elif name in ("pytest_run", "run_pytest"):
            target = params.get("target", ".")
            timeout = params.get("timeout")
            res = self.run_pytest(target=target, timeout=timeout)
            metrics = res.get("metrics", {})

            status_str = "PASSED" if res.get("success") else "FAILED"
            summary_badge = (
                f"**Passed:** {metrics.get('passed', 0)} | "
                f"**Failed:** {metrics.get('failed', 0)} | "
                f"**Errors:** {metrics.get('errors', 0)}"
            )

            return (
                f"### Pytest Suite: {status_str}\n"
                f"{summary_badge} *(Duration: {res.get('elapsed_seconds', 0.0)}s)*\n\n"
                f"```text\n{res.get('output', '').strip()}\n```"
            )

        elif name in ("sandbox_write_file", "write_file"):
            fn = params.get("filename")
            content = params.get("content", "")
            if not fn:
                return "Error: 'filename' parameter is required."
            res = self.write_file(fn, content)
            if res.get("success"):
                return f"Successfully wrote {res['size_bytes']} bytes to sandbox file `{fn}`."
            return f"Error writing file `{fn}`: {res.get('error')}"

        elif name in ("sandbox_read_file", "read_file"):
            fn = params.get("filename")
            if not fn:
                return "Error: 'filename' parameter is required."
            res = self.read_file(
                fn,
                start_line=params.get("start_line"),
                end_line=params.get("end_line"),
                max_chars=params.get("max_chars", 3500),
            )
            if "error" in res:
                return f"Error reading file `{fn}`: {res['error']}"
            preview_meta = res.get("preview", "")
            header = f"### Sandbox File: `{fn}` ({preview_meta})\n\n" if preview_meta else f"### Sandbox File: `{fn}`\n\n"
            return f"{header}```python\n{res['content']}\n```"

        elif name in ("sandbox_list_files", "list_files"):
            files = self.list_files()
            if not files:
                return "Sandbox is currently empty."
            lines = [f"# Sandbox Files ({len(files)} items)"]
            for f in files:
                lines.append(f"- `{f['filename']}` ({f['size_bytes']} bytes)")
            return "\n".join(lines)

        else:
            return f"Error: Unknown Sandbox tool '{tool_name}'."
