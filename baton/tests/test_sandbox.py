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
    # Valid syntax with substance
    valid_code = "def add(a: int, b: int) -> int:\n    return a + b\n\nresult = add(2, 3)"
    res = sandbox.syntax_check(valid_code)
    assert res["valid"] is True
    assert res["has_substance"] is True
    assert "Syntax valid" in res["message"]

    # Valid syntax but header/imports only (lacks substance)
    header_only = '"""Module docstring"""\nimport math\nfrom typing import List\n# -------------------\n'
    res_header = sandbox.syntax_check(header_only)
    assert res_header["valid"] is True
    assert res_header["has_substance"] is False
    assert "only imports, comments, or docstrings" in res_header["message"]

    # Invalid syntax
    invalid_code = "def broken(\n    return 42"
    res_inv = sandbox.syntax_check(invalid_code)
    assert res_inv["valid"] is False
    assert res_inv["has_substance"] is False
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
    assert "SyntaxError" in res["error"]


def test_eval_code_runtime_error_populates_error_traceback(sandbox):
    runtime_err_code = "import nonexistent_module_xyz\n"
    res = sandbox.eval_code(runtime_err_code)
    assert res["success"] is False
    assert res["returncode"] != 0
    assert res.get("error") is not None
    assert "ModuleNotFoundError" in res["error"]
    assert "nonexistent_module_xyz" in res["error"]


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


def test_fast_verify_rejects_header_only_skeleton(tmp_path: Path):
    """Verify that fast_verify rejects files that only have imports and comments without substantive code."""
    from unittest.mock import MagicMock
    from baton.models import Task, TaskType, Tier, VerificationFailedError
    from baton.system1 import System1Runner

    sandbox = SandboxTool(sandbox_dir=tmp_path / "sandbox")
    runner = System1Runner(artifacts_dir=tmp_path / "dossiers", sandbox_tool=sandbox, max_repair_retries=1)

    # 21-line truncated pong skeleton like the real-world bug
    skeleton = (
        '"""Pong game implemented with Pygame."""\n'
        'from __future__ import annotations\n'
        'import math\n'
        'import random\n'
        '# ----------------------------------------------------------------\n'
    )
    synth_file = tmp_path / "pong_skeleton.py"
    synth_file.write_text(skeleton, encoding="utf-8")

    upstream = [{"task_id": "synth_1", "title": "Synth Pong", "artifact_path": str(synth_file), "result_payload": "Synth done"}]
    verify_task = Task(
        task_id="verify_pong",
        job_id="job_1",
        title="Verify Pong",
        tier=Tier.SYSTEM_1,
        task_type=TaskType.FAST_VERIFY,
    )

    # Local repair fails to provide substantive code
    runner.generate = MagicMock(return_value={"response": "```python\n# Still no code\nimport math\n```", "tokens_generated": 10, "elapsed_seconds": 0.1})

    with pytest.raises(VerificationFailedError) as exc_info:
        runner.execute_task(verify_task, upstream_dossiers=upstream)

    assert "Substance Verification" in str(exc_info.value)
    assert "pong_skeleton.py" in str(exc_info.value)


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


def test_python_eval_reads_staged_file_when_code_empty(tmp_path: Path):
    """Verify that python_eval reads code from workspace file when code argument is omitted."""
    from baton.tools.sandbox import SandboxTool
    sandbox = SandboxTool(sandbox_dir=tmp_path / "sandbox_ws")
    sandbox.write_file("game.py", "print('PONG_INITIALIZED')\n")

    res = sandbox.execute("python_eval", {"filename": "game.py"})
    assert "PASSED (Code 0)" in res
    assert "PONG_INITIALIZED" in res


def test_system1_recon_autonomous_recovery_on_typo_or_miss(tmp_path: Path):
    """Verify that System 1 autonomously recovers from a recon tool miss (e.g. misspelled article)."""
    import json
    from unittest.mock import MagicMock, patch
    from baton.models import Task, TaskType, Tier
    from baton.system1 import System1Runner

    # 1. Mock zim_tool execution: first returns miss, second (recovered) returns article
    def mock_zim_exec(tool_name, params):
        title = params.get("title") or params.get("page_title")
        if title == "Fall Equanox":
            return "Section '' in article 'Fall Equanox' was empty or not found in [wikipedia]."
        elif title == "Autumnal equinox":
            return "# Autumnal equinox (Lead Section)\nThe autumnal equinox is an astronomical event..."
        return "Not found"

    mock_zim = MagicMock()
    mock_zim.execute.side_effect = mock_zim_exec
    mock_zim.reader.suggest_titles.return_value = ["Autumnal equinox", "Equinox"]

    runner = System1Runner(artifacts_dir=tmp_path, zim_tool=mock_zim, structured_output=True)
    task = Task(
        task_id="recon_fall_equanox",
        job_id="job_equinox",
        title="Fall Equanox Wikipedia Recon",
        tier=Tier.SYSTEM_1,
        task_type=TaskType.FAST_RECON,
        tool_name="wiki_read_section",
        tool_params={"page_title": "Fall Equanox"},
        prompt_template="Extract detailed information about Fall Equanox from Wikipedia.",
    )

    # 2. Mock generate calls:
    # 1st call: recovery prompt -> emits JSON repair action
    # 2nd call: dossier distillation prompt -> emits structured briefing dossier
    recovery_json = json.dumps({
        "diagnosis": "Typo in 'Equanox'. Closest canonical Wikipedia article is 'Autumnal equinox'.",
        "tool": "wiki_read_lead",
        "params": {"title": "Autumnal equinox"},
    })
    dossier_json = json.dumps({
        "summary": "Autumnal equinox occurs in September in the Northern Hemisphere.",
        "core_facts": ["Occurs around September 22-23", "Sun crosses the celestial equator"],
        "equations_or_apis": ["delta = 23.45 * sin(360 * (284 + N)/365)"],
        "open_questions": [],
        "source_citations": ["Wikipedia: Autumnal equinox"],
    })

    with patch.object(runner, "generate", side_effect=[
        {"response": recovery_json, "tokens_generated": 40, "elapsed_seconds": 0.5},
        {"response": dossier_json, "tokens_generated": 100, "elapsed_seconds": 1.0},
    ]) as mock_gen:
        summary, art_path, tokens, elapsed = runner.execute_task(task)

        assert mock_gen.call_count == 2
        assert "## Summary: Autumnal equinox" in summary
        assert art_path is not None
        content = Path(art_path).read_text(encoding="utf-8")
        assert "- Occurs around September 22-23" in content
        assert "delta = 23.45 * sin" in content


def test_slice_text_by_lines_utilities():
    from baton.tools.sandbox import slice_text_by_lines

    # 1. Empty text
    empty_res = slice_text_by_lines("")
    assert empty_res["content"] == ""
    assert empty_res["is_truncated"] is False
    assert empty_res["total_lines"] == 0

    # 2. Multi-line slice
    text = "line1\nline2\nline3\nline4\nline5\n"
    res = slice_text_by_lines(text, start_line=2, end_line=4)
    assert res["content"] == "line2\nline3\nline4\n"
    assert res["start_line"] == 2
    assert res["end_line"] == 4
    assert res["total_lines"] == 5
    assert res["is_truncated"] is True
    assert res["preview"] == f"Lines 2-4 of 5 ({len(res['content'])} of {len(text)} chars)"

    # 3. Single line result
    single_res = slice_text_by_lines(text, start_line=3, end_line=3)
    assert single_res["content"] == "line3\n"
    assert single_res["start_line"] == 3
    assert single_res["end_line"] == 3
    assert single_res["preview"] == f"Line 3 of 5 ({len(single_res['content'])} of {len(text)} chars)"

    # 4. Out of bounds start_line
    oob_res = slice_text_by_lines(text, start_line=10, end_line=20)
    assert oob_res["content"] == ""
    assert oob_res["is_truncated"] is True

    # 5. Char budget capping
    capped_res = slice_text_by_lines(text, max_chars=12)
    assert capped_res["is_truncated"] is True
    assert len(capped_res["content"]) <= 12


def test_sandbox_read_file_line_range(sandbox):
    content = "\n".join([f"def func_{i}(): return {i}" for i in range(1, 21)])
    sandbox.write_file("funcs.py", content)

    # Read specific line range
    res = sandbox.read_file("funcs.py", start_line=5, end_line=8)
    assert res["success"] is True
    assert res["start_line"] == 5
    assert res["end_line"] == 8
    assert res["total_lines"] == 20
    assert "def func_5(): return 5" in res["content"]
    assert "def func_8(): return 8" in res["content"]
    assert "def func_9" not in res["content"]
    assert res["preview"] == f"Lines 5-8 of 20 ({len(res['content'])} of {len(content)} chars)"

    # Execute dispatcher
    md = sandbox.execute("sandbox_read_file", {"filename": "funcs.py", "start_line": 5, "end_line": 8})
    assert "### Sandbox File: `funcs.py` (Lines 5-8 of 20" in md
    assert "def func_5" in md





