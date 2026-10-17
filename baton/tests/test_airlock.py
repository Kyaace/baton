"""Unit tests for System 1 / System 2 Inbound/Outbound Airlocks and the Fast-Patch Escalation Ladder."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from baton.db import Database
from baton.models import (
    BriefingDossier,
    DAGPlan,
    Task,
    TaskPlan,
    TaskStatus,
    TaskType,
    Tier,
    VerificationFailedError,
)
from baton.orchestrator import Orchestrator
from baton.system1 import (
    System1Runner,
    generate_diagnostic_dossier,
    repair_dag_plan_json,
    validate_delimiters,
    validate_embedded_code_syntax,
)
from baton.system2 import System2Runner
from baton.tools.sandbox import SandboxTool


# =============================================================================
# 1. Inbound Pre-Flight Airlock Tests
# =============================================================================

def test_validate_delimiters_balanced():
    clean_text = (
        "## Summary\n"
        "Here is a formula: f(x) = [a + b] * {c - d}.\n"
        "```python\n"
        "def compute():\n"
        "    return (1 + 2)\n"
        "```\n"
    )
    is_valid, reason = validate_delimiters(clean_text)
    assert is_valid is True
    assert reason == ""


def test_validate_delimiters_unclosed_fence():
    broken_fence = (
        "## Summary\n"
        "Here is broken code:\n"
        "```python\n"
        "def broken():\n"
        "    pass\n"
    )
    is_valid, reason = validate_delimiters(broken_fence)
    assert is_valid is False
    assert "Unclosed code fence" in reason


def test_validate_delimiters_mismatched_brackets():
    mismatched = "Equations: value = (10 + [20) - 5]"
    is_valid, reason = validate_delimiters(mismatched)
    assert is_valid is False
    assert "Mismatched delimiter" in reason


def test_validate_embedded_code_syntax_valid():
    text = (
        "### Equations\n"
        "```python\n"
        "def calculate_julian_date(y: int, m: int, d: int) -> float:\n"
        "    return float(y * 365 + m * 30 + d)\n"
        "```\n"
    )
    is_valid, errors = validate_embedded_code_syntax(text)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_embedded_code_syntax_invalid():
    broken_code = (
        "### Equations\n"
        "```python\n"
        "def broken_fn(x, y\n"
        "    return x +\n"
        "```\n"
    )
    is_valid, errors = validate_embedded_code_syntax(broken_code)
    assert is_valid is False
    assert len(errors) > 0
    assert "syntax error" in errors[0]


def test_briefing_dossier_compact_truncate():
    # Construct a dossier with large lists
    dossier = BriefingDossier(
        summary="Planetary ephemeris calculation constants and algorithms.",
        core_facts=[f"Fact item number {i} with vital planetary parameters and constants." for i in range(25)],
        equations_or_apis=[f"def orbit_eqn_{i}(t): return {i} * t" for i in range(15)],
        open_questions=[f"Uncertainty parameter {i} regarding perturbations" for i in range(10)],
        source_citations=[f"Astronomical Almanac Section {i}, Page {i*10}" for i in range(15)],
    )

    raw_words = dossier.word_count
    assert raw_words > 300, f"Expected >300 words, got {raw_words}"

    truncated = dossier.compact_truncate(max_words=150)
    truncated_words = len(truncated.split())
    assert truncated_words <= 160  # Account for truncation suffix
    assert "Planetary ephemeris" in truncated
    assert "Fact item number 0" in truncated


# =============================================================================
# 2. Outbound Post-Flight DAG Airlock Tests
# =============================================================================

def test_dag_plan_validate_linear():
    plan = DAGPlan(
        project="Valid Linear Pipeline",
        goal="Test linear acyclic graph",
        tasks=[
            TaskPlan(id="t1", title="Scout Ephemeris", tier=Tier.SYSTEM_1, depends_on=[]),
            TaskPlan(id="t2", title="Synthesize Code", tier=Tier.SYSTEM_2, depends_on=["t1"]),
            TaskPlan(id="t3", title="Verify Module", tier=Tier.SYSTEM_1, depends_on=["t2"]),
        ],
    )
    # Should execute without raising
    plan.validate_dag()


def test_dag_plan_validate_cycle_detection():
    cyclic_plan = DAGPlan(
        project="Cyclic Graph Failure",
        goal="Detect circular dependency loop",
        tasks=[
            TaskPlan(id="node_a", title="Node A", tier=Tier.SYSTEM_1, depends_on=["node_b"]),
            TaskPlan(id="node_b", title="Node B", tier=Tier.SYSTEM_2, depends_on=["node_a"]),
        ],
    )
    with pytest.raises(ValueError, match="Circular dependency detected"):
        cyclic_plan.validate_dag()


def test_dag_plan_validate_dangling_dependency():
    dangling_plan = DAGPlan(
        project="Dangling Dependency",
        goal="Detect missing parent task ID",
        tasks=[
            TaskPlan(id="node_1", title="Task 1", tier=Tier.SYSTEM_1, depends_on=["non_existent_node"]),
        ],
    )
    with pytest.raises(ValueError, match="dangling dependencies"):
        dangling_plan.validate_dag()


def test_dag_plan_validate_duplicate_ids():
    dup_plan = DAGPlan(
        project="Duplicate IDs",
        goal="Detect collision on task ID",
        tasks=[
            TaskPlan(id="task_x", title="First Instance", tier=Tier.SYSTEM_1),
            TaskPlan(id="task_x", title="Second Instance", tier=Tier.SYSTEM_2),
        ],
    )
    with pytest.raises(ValueError, match="Duplicate task ID"):
        dup_plan.validate_dag()


def test_repair_dag_plan_json_one_shot(tmp_path: Path):
    runner = System1Runner(artifacts_dir=tmp_path / "dossiers")

    # Mock runner.generate returning valid repaired JSON
    repaired_json = json.dumps({
        "project": "Repaired Ephemeris",
        "goal": "Calculate lunar coordinates",
        "tasks": [
            {
                "id": "t1_recon",
                "title": "Recon ZIM",
                "type": "fast_recon",
                "tier": "system_1",
                "depends_on": [],
                "priority": 10,
            },
            {
                "id": "t2_synth",
                "title": "Synthesize Ephemeris",
                "type": "deep_synthesis",
                "tier": "system_2",
                "depends_on": ["t1_recon"],
                "priority": 1,
            },
        ],
    })
    runner.generate = MagicMock(return_value={"response": repaired_json})

    raw_malformed = "{'project': 'Repaired Ephemeris', tasks: [missing quotes and brackets..."
    plan, msg = repair_dag_plan_json(runner, raw_malformed, "JSONDecodeError: Expecting property name enclosed in double quotes")

    assert plan is not None
    assert plan.project == "Repaired Ephemeris"
    assert len(plan.tasks) == 2
    assert "successfully repaired" in msg


# =============================================================================
# 3. Fast-Patch Escalation Ladder Tests
# =============================================================================

def test_generate_diagnostic_dossier_under_150_words():
    raw_pytest_trace = (
        "============================= FAILURES =============================\n"
        "_____________________ test_orbital_velocity ________________________\n"
        "def test_orbital_velocity():\n"
        ">       assert compute_velocity(100.0) == 7.8\n"
        "E       AssertionError: assert 8.2 == 7.8\n"
        "E         +8.2\n"
        "E         -7.8\n"
        "tests/test_orbit.py:24: AssertionError\n"
        "======================= 1 failed in 0.12s =========================\n"
    )

    diag = generate_diagnostic_dossier(raw_pytest_trace, max_words=150)
    assert "Root Cause Analysis" in diag
    assert "AssertionError" in diag
    assert len(diag.split()) <= 150


def test_system1_runner_max_repair_retries_default_3(tmp_path: Path):
    runner = System1Runner(artifacts_dir=tmp_path / "dossiers")
    assert runner.max_repair_retries == 3


def test_system1_inbound_airlock_density_trimming(tmp_path: Path):
    runner = System1Runner(artifacts_dir=tmp_path / "dossiers")
    task = Task(
        task_id="recon_density",
        job_id="job_density",
        title="Extract Solar Constants",
        tier=Tier.SYSTEM_1,
        task_type=TaskType.FAST_RECON,
    )

    # Return a model that exceeds 300 words
    oversized_response = json.dumps({
        "summary": "Detailed astronomical constants for solar system barycenter and planetary orbits.",
        "core_facts": [f"Astronomical constant entry {i}: value = {i * 1.618033} km/s." for i in range(30)],
        "equations_or_apis": [f"def get_param_{i}(): return {i}" for i in range(15)],
        "open_questions": ["Is perturbation accurate?"],
        "source_citations": ["JPL Ephemeris DE440"],
    })

    # When re-prompted for compression, return a tight dossier
    compressed_response = json.dumps({
        "summary": "Core astronomical constants for solar barycenter.",
        "core_facts": ["Speed of light c = 299792 km/s", "Gravitational constant G = 6.674e-11"],
        "equations_or_apis": ["def get_barycenter(t): return 0.0"],
        "open_questions": [],
        "source_citations": ["JPL Ephemeris"],
    })

    call_count = 0
    def mock_gen(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"response": oversized_response, "tokens_generated": 350}
        return {"response": compressed_response, "tokens_generated": 100}

    runner.generate = MagicMock(side_effect=mock_gen)

    summary, art_path, tokens, elapsed = runner.execute_task(task)
    assert Path(art_path).exists()
    content = Path(art_path).read_text(encoding="utf-8")
    assert len(content.split()) <= 350
    assert "Core astronomical constants" in content
    # Verify that the re-compression loop was triggered
    assert call_count >= 2


def test_orchestrator_tier2_escalation_enqueues_system2(tmp_path: Path):
    db = Database(tmp_path / "test_esc.db")
    job = db.create_job("Project Ephemeris", "Calculate orbital positions")
    orch = Orchestrator(db=db, progress_dir=tmp_path / "prog")

    v_task = orch.add_manual_task(job.job_id, "Verify Solar Model", Tier.SYSTEM_1)
    err = VerificationFailedError(
        message="Root Cause Analysis:\nAssertionError: assert 8.2 == 7.8 in compute_velocity",
        failure_trace="AssertionError: assert 8.2 == 7.8\n> assert compute_velocity(100.0) == 7.8",
        target_code="def compute_velocity(r): return 8.2",
    )

    repair_task = orch._enqueue_system2_repair(v_task, err)
    assert repair_task is not None
    assert repair_task.tier == Tier.SYSTEM_2
    assert repair_task.task_type == TaskType.DEEP_SYNTHESIS
    assert "Fast-Patch Ladder Tier 2 Escalation" in repair_task.prompt_template
    assert "AssertionError: assert 8.2 == 7.8" in repair_task.prompt_template


def test_generate_diagnostic_dossier_prioritizes_primary_failure():
    """Verify that generate_diagnostic_dossier prioritizes the original failure over downstream retry errors."""
    primary_err = "AttributeError: 'Paddle' object has no attribute 'top'\n  File 'breakout.py', line 210, in update"
    retry_trace = (
        "### Syntax Verification (`breakout.py`):\nPASSED\n\n"
        "### Execution (`breakout.py`): FAILED\nAttributeError: 'Paddle' object has no attribute 'top'\n\n"
        "### Local Patch (Attempt 1/3): FAILED (Syntax)\nSyntaxError: unterminated string literal on line 435\n\n"
        "### Local Patch (Attempt 2/3): FAILED (Syntax)\nSyntaxError: invalid syntax on line 440"
    )

    diag = generate_diagnostic_dossier(retry_trace, primary_failure=primary_err, max_words=150)
    assert "Root Cause Analysis" in diag
    assert "AttributeError: 'Paddle' object has no attribute 'top'" in diag


def test_system1_local_repair_skips_oversized_files(tmp_path: Path):
    """Verify that files over 12,000 characters skip System 1 single-pass rewrite to prevent HTTP 400 crashes."""
    sandbox = SandboxTool(sandbox_dir=tmp_path / "sandbox")
    runner = System1Runner(artifacts_dir=tmp_path / "dossiers", sandbox_tool=sandbox)

    # Create 15,000 character python file with deliberate runtime failure
    large_code = "# Large file\n" + ("x = 1\n" * 2000) + "raise ValueError('Initial deliberate bug')\n"
    art_file = tmp_path / "large_game.py"
    art_file.write_text(large_code, encoding="utf-8")

    verify_task = Task(
        task_id="verify_large",
        job_id="job_test",
        title="Verify Large Game",
        tier=Tier.SYSTEM_1,
        task_type=TaskType.FAST_VERIFY,
    )
    upstream = [{"artifact_path": str(art_file), "title": "Large Game"}]

    with patch.object(runner, "generate") as mock_gen:
        mock_gen.return_value = {
            "response": '{"verdict": "FAILED", "key_findings": ["ValueError: Initial deliberate bug"], "remediation_points": ["Fix bug"]}',
            "tokens_generated": 50,
        }
        with pytest.raises(VerificationFailedError) as exc_info:
            runner.execute_task(verify_task, upstream_dossiers=upstream)

        # Ensure local repair prompt was NOT generated (only dossier distillation was called)
        assert mock_gen.call_count == 1
        err = exc_info.value
        assert "Initial deliberate bug" in err.message or "Initial deliberate bug" in err.failure_trace
        assert err.target_code == large_code


def test_system1_local_repair_handles_http_400_cleanly(tmp_path: Path):
    """Verify that HTTP 400 or network exceptions in local repair do not crash the runner and preserve primary failure."""
    import requests
    sandbox = SandboxTool(sandbox_dir=tmp_path / "sandbox")
    runner = System1Runner(artifacts_dir=tmp_path / "dossiers", sandbox_tool=sandbox)

    failing_code = "raise RuntimeError('Original primary failure')\n"
    art_file = tmp_path / "small_test.py"
    art_file.write_text(failing_code, encoding="utf-8")

    verify_task = Task(
        task_id="verify_small",
        job_id="job_test",
        title="Verify Small",
        tier=Tier.SYSTEM_1,
        task_type=TaskType.FAST_VERIFY,
    )
    upstream = [{"artifact_path": str(art_file), "title": "Small"}]

    call_count = 0
    def mock_gen_with_400(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call is local repair generation: simulate HTTP 400 Bad Request
            resp = requests.Response()
            resp.status_code = 400
            raise requests.exceptions.HTTPError("400 Client Error: Bad Request", response=resp)
        # Second call is dossier distillation
        return {
            "response": '{"verdict": "FAILED", "key_findings": ["RuntimeError: Original primary failure"], "remediation_points": ["Fix"]}',
            "tokens_generated": 50,
        }

    with patch.object(runner, "generate", side_effect=mock_gen_with_400):
        with pytest.raises(VerificationFailedError) as exc_info:
            runner.execute_task(verify_task, upstream_dossiers=upstream)

        err = exc_info.value
        assert "Original primary failure" in err.message or "Original primary failure" in err.failure_trace
        assert err.target_code == failing_code

