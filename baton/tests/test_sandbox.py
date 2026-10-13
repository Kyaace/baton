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


def test_system1_local_repair_syntax_error(tmp_path: Path):
    """Verify that System 1 automatically fixes syntax errors during fast_verify and updates the artifact."""
    from unittest.mock import MagicMock
    from baton.models import Task, TaskType, Tier
    from baton.system1 import System1Runner

    sandbox = SandboxTool(sandbox_dir=tmp_path / "sandbox")
    runner = System1Runner(artifacts_dir=tmp_path / "dossiers", sandbox_tool=sandbox, max_repair_retries=2)

    broken_code = "def compute_velocity(distance, time_val\n    return distance / time_val\n\ndef test_velocity():\n    assert compute_velocity(100, 20) == 5\n"
    synth_file = tmp_path / "velocity.py"
    synth_file.write_text(broken_code, encoding="utf-8")

    upstream = [{"task_id": "synth_1", "title": "Synth Velocity", "artifact_path": str(synth_file), "result_payload": "Synth done"}]
    verify_task = Task(
        task_id="verify_vel",
        job_id="job_1",
        title="Verify Velocity",
        tier=Tier.SYSTEM_1,
        task_type=TaskType.FAST_VERIFY,
    )

    fixed_code = "```python\ndef compute_velocity(distance, time_val):\n    return distance / time_val\n\ndef test_velocity():\n    assert compute_velocity(100, 20) == 5\n```"
    # First call: patch generation; Second call: verifier dossier compilation
    runner.generate = MagicMock(side_effect=[
        {"response": fixed_code, "tokens_generated": 40, "elapsed_seconds": 0.3},
        {"response": "VERIFIED: Repaired syntax error successfully.", "tokens_generated": 30, "elapsed_seconds": 0.2},
    ])

    summary, art_path, tokens, elapsed = runner.execute_task(verify_task, upstream_dossiers=upstream)

    # Verify that the original artifact on disk was repaired
    repaired_disk_content = synth_file.read_text(encoding="utf-8")
    assert "def compute_velocity(distance, time_val):" in repaired_disk_content

    # Verify report was generated
    assert Path(art_path).exists()
    dossier = Path(art_path).read_text(encoding="utf-8")
    assert "VERIFIED" in dossier


def test_system1_local_repair_failing_assertion(tmp_path: Path):
    """Verify that System 1 automatically repairs failing pytest assertions."""
    from unittest.mock import MagicMock
    from baton.models import Task, TaskType, Tier
    from baton.system1 import System1Runner

    sandbox = SandboxTool(sandbox_dir=tmp_path / "sandbox")
    runner = System1Runner(artifacts_dir=tmp_path / "dossiers", sandbox_tool=sandbox, max_repair_retries=2)

    buggy_code = "def multiply(a, b):\n    return a + b\n\ndef test_multiply():\n    assert multiply(3, 4) == 12\n"
    synth_file = tmp_path / "calc.py"
    synth_file.write_text(buggy_code, encoding="utf-8")

    upstream = [{"task_id": "synth_1", "title": "Synth Calc", "artifact_path": str(synth_file), "result_payload": "Synth done"}]
    verify_task = Task(
        task_id="verify_calc",
        job_id="job_1",
        title="Verify Calc",
        tier=Tier.SYSTEM_1,
        task_type=TaskType.FAST_VERIFY,
    )

    fixed_code = "```python\ndef multiply(a, b):\n    return a * b\n\ndef test_multiply():\n    assert multiply(3, 4) == 12\n```"
    runner.generate = MagicMock(side_effect=[
        {"response": fixed_code, "tokens_generated": 35, "elapsed_seconds": 0.3},
        {"response": "VERIFIED: Repaired arithmetic bug in multiplication.", "tokens_generated": 25, "elapsed_seconds": 0.2},
    ])

    summary, art_path, tokens, elapsed = runner.execute_task(verify_task, upstream_dossiers=upstream)

    # Check that disk content has the fixed multiplier
    assert "return a * b" in synth_file.read_text(encoding="utf-8")


def test_system1_verification_exhausted_raises_error(tmp_path: Path):
    """Verify that when local repair retries fail to resolve errors, VerificationFailedError is raised."""
    from unittest.mock import MagicMock
    from baton.models import Task, TaskType, Tier, VerificationFailedError
    from baton.system1 import System1Runner

    sandbox = SandboxTool(sandbox_dir=tmp_path / "sandbox")
    runner = System1Runner(artifacts_dir=tmp_path / "dossiers", sandbox_tool=sandbox, max_repair_retries=2)

    broken_code = "def impossible():\n    assert 1 == 2\n\ndef test_imp():\n    impossible()\n"
    synth_file = tmp_path / "impossible.py"
    synth_file.write_text(broken_code, encoding="utf-8")

    upstream = [{"task_id": "synth_1", "title": "Synth Impossible", "artifact_path": str(synth_file), "result_payload": "Synth done"}]
    verify_task = Task(
        task_id="verify_imp",
        job_id="job_1",
        title="Verify Impossible",
        tier=Tier.SYSTEM_1,
        task_type=TaskType.FAST_VERIFY,
    )

    # Emits answers that still fail
    still_broken = "```python\ndef impossible():\n    assert False\n\ndef test_imp():\n    impossible()\n```"
    runner.generate = MagicMock(return_value={"response": still_broken, "tokens_generated": 20, "elapsed_seconds": 0.2})

    with pytest.raises(VerificationFailedError) as exc_info:
        runner.execute_task(verify_task, upstream_dossiers=upstream)

    assert exc_info.value.artifact_path is not None
    assert "impossible.py" in exc_info.value.target_code or "impossible" in exc_info.value.target_code


def test_briefing_dossier_compact_rendering():
    from baton.models import BriefingDossier

    dossier = BriefingDossier(
        summary="Keplerian orbit propagation requires solving Kepler's equation iteratively.",
        core_facts=["Gravitational parameter of Earth is 398600.4418 km^3/s^2", "Mean motion n = sqrt(mu / a^3)"],
        equations_or_apis=["M = E - e * sin(E)", "def solve_kepler(M: float, e: float) -> float"],
        open_questions=["Near-parabolic eccentricity stability"],
        source_citations=["Fundamentals of Astrodynamics (Bate, Mueller, White)"],
    )

    md = dossier.to_compact_markdown()
    assert "## Summary: Keplerian orbit propagation" in md
    assert "### Core Facts" in md
    assert "- Gravitational parameter" in md
    assert "### Equations & APIs" in md
    assert "`M = E - e * sin(E)`" in md
    assert "`def solve_kepler(M: float, e: float) -> float`" in md
    assert "### Open Questions" in md
    assert "### Citations" in md


def test_strip_conversational_fluff():
    from baton.system1 import strip_conversational_fluff

    raw_with_fluff = (
        "Sure, here is the briefing dossier you requested!\n\n"
        "## Summary: Kepler's equation is transcendental.\n"
        "- Fact 1\n\n"
        "I hope this helps! Let me know if you need any other calculations."
    )
    cleaned = strip_conversational_fluff(raw_with_fluff)
    assert not cleaned.startswith("Sure")
    assert not cleaned.endswith("calculations.")
    assert cleaned.startswith("## Summary: Kepler's equation is transcendental.")


def test_parse_structured_dossier_json_and_fallback():
    from baton.system1 import parse_structured_dossier
    from baton.models import BriefingDossier, VerificationDossier

    # 1. Valid JSON in markdown code fence
    json_block = """```json
    {
        "summary": "Orbital resonance verified.",
        "core_facts": ["Period ratio is 2:1"],
        "equations_or_apis": ["T1 / T2 = 2.0"],
        "open_questions": [],
        "source_citations": ["JPL Ephemeris"]
    }
    ```"""
    md, model = parse_structured_dossier(json_block, is_verify=False, is_sdr=False)
    assert isinstance(model, BriefingDossier)
    assert "## Summary: Orbital resonance verified." in md
    assert "`T1 / T2 = 2.0`" in md

    # 2. Valid Verification JSON
    verify_json = """{
        "verdict": "VERIFIED",
        "syntax_valid": true,
        "tests_passed": 5,
        "tests_failed": 0,
        "test_errors": 0,
        "key_findings": ["All unit assertions hold."],
        "remediation_points": []
    }"""
    md_v, model_v = parse_structured_dossier(verify_json, is_verify=True, is_sdr=False)
    assert isinstance(model_v, VerificationDossier)
    assert "## Verdict: VERIFIED" in md_v
    assert "Tests: 5 passed" in md_v

    # 3. Fallback on non-JSON raw markdown
    raw_md = "## Freeform Dossier\n- Finding A\n- Finding B"
    md_f, model_f = parse_structured_dossier(raw_md)
    assert model_f is None
    assert "## Freeform Dossier" in md_f


def test_system1_executes_task_with_structured_json(tmp_path: Path):
    import json
    from unittest.mock import patch
    from baton.models import Task, TaskType, Tier
    from baton.system1 import System1Runner

    runner = System1Runner(artifacts_dir=tmp_path, structured_output=True)
    task = Task(
        task_id="recon_structured",
        job_id="job_struct",
        title="Extract Ephemeris Formulas",
        tier=Tier.SYSTEM_1,
        task_type=TaskType.FAST_RECON,
        prompt_template="Formulas for solar coordinates",
    )

    simulated_json = json.dumps({
        "summary": "Solar elevation is calculated using solar declination and hour angle.",
        "core_facts": ["Solar constant = 1361 W/m^2", "Obliquity of ecliptic = 23.44 deg"],
        "equations_or_apis": ["sin(alt) = sin(lat)*sin(dec) + cos(lat)*cos(dec)*cos(ha)"],
        "open_questions": [],
        "source_citations": ["NOAA Solar Position Calculator"],
    })

    with patch.object(runner, "generate") as mock_gen:
        mock_gen.return_value = {
            "response": simulated_json,
            "tokens_generated": 65,
            "elapsed_seconds": 1.1,
        }

        summary, art_path, tokens, elapsed = runner.execute_task(task)

        # Check format was requested as "json"
        assert mock_gen.call_args[1]["format"] == "json"
        assert "## Summary: Solar elevation" in summary
        content = Path(art_path).read_text(encoding="utf-8")
        assert "### Equations & APIs" in content
        assert "sin(alt)" in content


