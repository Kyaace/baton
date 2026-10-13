"""Unit tests for Baton Subsystem A: Sandboxed Python REPL & Verification Engine."""

import shutil
import tempfile
from pathlib import Path

import pytest

from baton.tools.sandbox import SandboxTool, windows_safe_run


@pytest.fixture
def temp_sandbox_dir():
    temp_dir = Path(tempfile.mkdtemp(prefix="baton_test_sandbox_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sandbox(temp_sandbox_dir):
    return SandboxTool(sandbox_dir=temp_sandbox_dir, default_timeout=5)


def test_syntax_check_valid_and_invalid(sandbox):
    # Valid syntax
    valid_code = "def add(a: int, b: int) -> int:\n    return a + b\n\nresult = add(2, 3)"
    res = sandbox.syntax_check(valid_code)
    assert res["valid"] is True
    assert "Syntax valid" in res["message"]

    # Invalid syntax
    invalid_code = "def broken(\n    return 42"
    res_inv = sandbox.syntax_check(invalid_code)
    assert res_inv["valid"] is False
    assert res_inv["line"] is not None
    assert "SyntaxError" in res_inv["message"]


def test_file_staging_and_traversal_protection(sandbox):
    # Safe write & read
    write_res = sandbox.write_file("test_module.py", "CONSTANT = 1337\n")
    assert write_res["success"] is True

    read_res = sandbox.read_file("test_module.py")
    assert read_res["content"] == "CONSTANT = 1337\n"
    assert read_res["is_truncated"] is False

    # List files
    file_list = sandbox.list_files()
    assert len(file_list) == 1
    assert file_list[0]["filename"] == "test_module.py"

    # Traversal attempt outside sandbox
    with pytest.raises(PermissionError):
        sandbox._resolve_safe_path("../../escaped.py")

    write_bad = sandbox.write_file("../../escaped.py", "malicious = True")
    assert write_bad["success"] is False
    assert "Access denied" in write_bad["error"]

    read_bad = sandbox.read_file("../../../etc/passwd")
    assert "error" in read_bad
    assert "Access denied" in read_bad["error"]


def test_eval_code_clean_execution(sandbox):
    code = "import math\nprint(f'PI={round(math.pi, 4)}')\n"
    res = sandbox.eval_code(code)
    assert res["success"] is True
    assert res["returncode"] == 0
    assert "PI=3.1416" in res["output"]
    assert res["elapsed_seconds"] >= 0.0


def test_eval_code_syntax_error_fast_fails(sandbox):
    bad_code = "for i in range(10\n    print(i)"
    res = sandbox.eval_code(bad_code)
    assert res["success"] is False
    assert res["returncode"] == 1
    assert "SyntaxError" in res["stderr"]


def test_eval_code_timeout_and_process_termination(sandbox):
    # Code that attempts to run indefinitely
    infinite_loop = "import time\nwhile True:\n    time.sleep(0.1)\n"
    res = sandbox.eval_code(infinite_loop, timeout=2)
    assert res["success"] is False
    assert res["returncode"] == 124
    assert res.get("timed_out") is True
    assert "timed out after 2s" in res["error"]


def test_run_pytest_in_sandbox(sandbox):
    # 1. Stage passing test
    test_code = """
def test_addition():
    assert 1 + 1 == 2

def test_subtraction():
    assert 5 - 3 == 2
"""
    sandbox.write_file("test_math.py", test_code)

    res = sandbox.run_pytest(target="test_math.py", timeout=15)
    assert res["success"] is True
    assert res["returncode"] == 0
    assert res["metrics"]["passed"] == 2
    assert res["metrics"]["failed"] == 0

    # 2. Stage failing test
    fail_code = """
def test_failing():
    assert 2 * 2 == 5
"""
    sandbox.write_file("test_fail.py", fail_code)

    res_fail = sandbox.run_pytest(target="test_fail.py", timeout=15)
    assert res_fail["success"] is False
    assert res_fail["returncode"] == 1
    assert res_fail["metrics"]["failed"] == 1


def test_execute_dispatcher_markdown_output(sandbox):
    # Syntax check dispatcher
    out_valid = sandbox.execute("python_syntax_check", {"code": "x = 42"})
    assert "Python Syntax Check: PASSED" in out_valid

    out_invalid = sandbox.execute("python_syntax_check", {"code": "def foo("})
    assert "Python Syntax Check: FAILED" in out_invalid

    # Eval dispatcher
    out_eval = sandbox.execute("python_eval", {"code": "print('DISPATCHER_OUTPUT')"})
    assert "Python Execution: PASSED" in out_eval
    assert "DISPATCHER_OUTPUT" in out_eval

    # Write & read dispatcher
    out_write = sandbox.execute("sandbox_write_file", {"filename": "data.txt", "content": "hello world"})
    assert "Successfully wrote" in out_write

    out_read = sandbox.execute("sandbox_read_file", {"filename": "data.txt"})
    assert "hello world" in out_read


def test_system1_execute_task_fast_verify_with_sandbox(tmp_path: Path):
    """Verify that System1Runner automatically stages upstream code and runs tests on fast_verify tasks."""
    from unittest.mock import MagicMock, patch
    from baton.models import Task, TaskType, Tier
    from baton.system1 import System1Runner

    sandbox = SandboxTool(sandbox_dir=tmp_path / "sandbox")
    dossiers_dir = tmp_path / "dossiers"
    runner = System1Runner(artifacts_dir=dossiers_dir, sandbox_tool=sandbox)

    # 1. Create simulated upstream synthesized python artifact
    synth_file = tmp_path / "calc_solution.py"
    synth_file.write_text("def add(a, b):\n    return a + b\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8")

    upstream = [
        {
            "task_id": "synth_1",
            "title": "Synthesize Calculator",
            "artifact_path": str(synth_file),
            "result_payload": "Synthesized calculator module.",
        }
    ]

    verify_task = Task(
        task_id="verify_calc",
        job_id="job_1",
        title="Verify Calculator Syntax and Unit Tests",
        tier=Tier.SYSTEM_1,
        task_type=TaskType.FAST_VERIFY,
        prompt_template="Validate the synthesized calculator code.",
    )

    mock_gen = MagicMock(return_value={
        "response": "VERIFIED: All tests passed and syntax valid.",
        "tokens_generated": 45,
        "elapsed_seconds": 0.5,
    })
    runner.generate = mock_gen

    summary, art_path, tokens, elapsed = runner.execute_task(verify_task, upstream_dossiers=upstream)

    assert "VERIFIED" in summary
    assert Path(art_path).exists()
    content = Path(art_path).read_text(encoding="utf-8")
    assert "# Verification Dossier:" in content
    assert "Baton System 1 (Verifier)" in content

    # Verify that the generated prompt presented verification data and verifier persona
    call_args = mock_gen.call_args
    prompt_sent = call_args[0][0]
    sys_prompt_sent = call_args[1]["system_prompt"]
    assert "Verification & Test Data" in prompt_sent
    assert "Syntax Verification" in prompt_sent
    assert "Test Suite Execution" in prompt_sent
    assert "Baton System 1: an autonomous, rigorous Code Verifier" in sys_prompt_sent


def test_system1_execute_task_named_sandbox_tool(tmp_path: Path):
    """Verify System1Runner tool dispatch for explicit sandbox tool names."""
    from unittest.mock import MagicMock
    from baton.models import Task, TaskType, Tier
    from baton.system1 import System1Runner

    sandbox = SandboxTool(sandbox_dir=tmp_path / "sandbox")
    runner = System1Runner(artifacts_dir=tmp_path / "dossiers", sandbox_tool=sandbox)

    task = Task(
        task_id="eval_task",
        job_id="job_1",
        title="Run Math Check",
        tier=Tier.SYSTEM_1,
        task_type=TaskType.FAST_RECON,
        tool_name="python_eval",
        tool_params={"code": "import math\nprint(f'SQRT={math.isqrt(144)}')"},
    )

    runner.generate = MagicMock(return_value={
        "response": "Calculated SQRT=12 successfully.",
        "tokens_generated": 20,
        "elapsed_seconds": 0.2,
    })

    summary, art_path, tokens, elapsed = runner.execute_task(task)
    assert "Calculated" in summary
    assert Path(art_path).exists()
    assert "# Verification Dossier:" in Path(art_path).read_text(encoding="utf-8")

