"""Tests for Baton Orchestrator deterministic execution loop."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from baton.db import Database
from baton.models import DAGPlan, TaskPlan, Tier, TaskType, TaskStatus, JobStatus
from baton.orchestrator import Orchestrator


class MockSystem1:
    def execute_task(self, task, upstream):
        art_path = f"artifacts/{task.task_id}.md"
        return f"Mock Dossier for {task.title}", art_path, 150, 1.2


class MockSystem2:
    def synthesize(self, task, upstream):
        art_path = f"artifacts/{task.task_id}.py"
        return f"Mock Code for {task.title}", art_path, 2400, 45.0

    def plan_project(self, project, goal):
        return DAGPlan(
            project=project,
            goal=goal,
            tasks=[
                TaskPlan(id="recon_a", title="Recon A", tier=Tier.SYSTEM_1, type=TaskType.FAST_RECON),
                TaskPlan(id="synth_b", title="Synth B", tier=Tier.SYSTEM_2, type=TaskType.DEEP_SYNTHESIS, depends_on=["recon_a"]),
            ]
        )


class MockEngineManager:
    def __init__(self):
        self.active_tier = None

    def switch_to_tier(self, tier):
        self.active_tier = tier
        return True

    def get_status(self):
        return {"active_tier": self.active_tier}


@pytest.fixture
def mock_orchestrator(tmp_path: Path):
    db = Database(tmp_path / "mock_baton.db")
    engine_mgr = MockEngineManager()
    sys1 = MockSystem1()
    sys2 = MockSystem2()
    return Orchestrator(db=db, engine_manager=engine_mgr, system1=sys1, system2=sys2, auto_verify=False)


def test_orchestrator_end_to_end_relay(mock_orchestrator: Orchestrator):
    # Submit a project
    job = mock_orchestrator.submit_goal("Solar Model", "Compute solar elevation", plan_with_system2=True)
    assert job.status == JobStatus.PENDING

    tasks = mock_orchestrator.db.get_job_tasks(job.job_id)
    assert len(tasks) == 2

    # Step 1: Should run recon_a on System 1
    t1 = mock_orchestrator.step()
    assert t1 is not None
    assert t1.task_id == "recon_a"
    assert t1.status == TaskStatus.COMPLETED
    assert mock_orchestrator.engine_mgr.active_tier == Tier.SYSTEM_1

    # Step 2: Should automatically switch to System 2 and run synth_b
    t2 = mock_orchestrator.step()
    assert t2 is not None
    assert t2.task_id == "synth_b"
    assert t2.status == TaskStatus.COMPLETED
    assert mock_orchestrator.engine_mgr.active_tier == Tier.SYSTEM_2

    # Step 3: Queue should be empty now
    t3 = mock_orchestrator.step()
    assert t3 is None

    # Job should be completed
    updated_job = mock_orchestrator.db.get_job(job.job_id)
    assert updated_job.status == JobStatus.COMPLETED

    # Verify artifacts recorded
    artifacts = mock_orchestrator.db.get_artifacts(job_id=job.job_id)
    assert len(artifacts) == 2
    types = {a.artifact_type.value for a in artifacts}
    assert "dossier" in types
    assert "code" in types


def test_orchestrator_recon_synth_verify_pipeline(mock_orchestrator: Orchestrator):
    """Verify 3-stage relay (System 1 Recon -> System 2 Synth -> System 1 Verifier)."""
    job = mock_orchestrator.db.create_job("Full Pipeline", "Recon, Synth, and Verify")

    mock_orchestrator.db.create_task(
        title="Recon Task",
        tier=Tier.SYSTEM_1,
        task_type=TaskType.FAST_RECON,
        task_id="recon_1",
        job_id=job.job_id,
        priority=10,
    )
    mock_orchestrator.db.create_task(
        title="Synth Task",
        tier=Tier.SYSTEM_2,
        task_type=TaskType.DEEP_SYNTHESIS,
        task_id="synth_2",
        job_id=job.job_id,
        depends_on=["recon_1"],
        priority=5,
    )
    mock_orchestrator.db.create_task(
        title="Verify Task",
        tier=Tier.SYSTEM_1,
        task_type=TaskType.FAST_VERIFY,
        task_id="verify_3",
        job_id=job.job_id,
        depends_on=["synth_2"],
        priority=1,
    )

    # Step 1: Recon (System 1)
    t1 = mock_orchestrator.step()
    assert t1 is not None
    assert t1.task_id == "recon_1"
    assert t1.status == TaskStatus.COMPLETED

    # Step 2: Synth (System 2)
    t2 = mock_orchestrator.step()
    assert t2 is not None
    assert t2.task_id == "synth_2"
    assert t2.status == TaskStatus.COMPLETED

    # Step 3: Verify (System 1)
    t3 = mock_orchestrator.step()
    assert t3 is not None
    assert t3.task_id == "verify_3"
    assert t3.status == TaskStatus.COMPLETED

    # Verify that verify_3 recorded ArtifactType.REPORT
    artifacts = mock_orchestrator.db.get_artifacts(job_id=job.job_id)
    assert len(artifacts) == 3
    verify_art = [a for a in artifacts if a.task_id == "verify_3"][0]
    assert verify_art.artifact_type.value == "report"
    assert verify_art.title == "Verification: Verify Task"


def test_orchestrator_auto_verify_injection(tmp_path: Path):
    """Verify that when a System 2 synthesis task completes, a downstream FAST_VERIFY task is automatically injected."""
    db = Database(tmp_path / "auto_verify_baton.db")
    engine_mgr = MockEngineManager()
    sys1 = MockSystem1()
    sys2 = MockSystem2()
    orch = Orchestrator(db=db, engine_manager=engine_mgr, system1=sys1, system2=sys2, auto_verify=True)

    job = db.create_job("Auto Verify Job", "Test automatic verification injection")
    db.create_task(
        title="Synth Code",
        tier=Tier.SYSTEM_2,
        task_type=TaskType.DEEP_SYNTHESIS,
        task_id="synth_main",
        job_id=job.job_id,
        priority=10,
    )
    db.create_task(
        title="Consume Code",
        tier=Tier.SYSTEM_1,
        task_type=TaskType.FAST_RECON,
        task_id="downstream_step",
        job_id=job.job_id,
        depends_on=["synth_main"],
        priority=5,
    )

    # Step 1: Run synth_main (System 2)
    t1 = orch.step()
    assert t1 is not None
    assert t1.task_id == "synth_main"
    assert t1.status == TaskStatus.COMPLETED

    # Check that verify_synth_main was automatically injected and downstream_step was rewired
    verify_task = db.get_task("verify_synth_main")
    assert verify_task is not None
    assert verify_task.task_type == TaskType.FAST_VERIFY
    assert verify_task.status == TaskStatus.READY

    downstream = db.get_task("downstream_step")
    assert downstream.status == TaskStatus.BLOCKED
    assert "verify_synth_main" in downstream.depends_on
    assert "synth_main" not in downstream.depends_on

    # Step 2: Run the auto-injected verify task (System 1)
    t2 = orch.step()
    assert t2 is not None
    assert t2.task_id == "verify_synth_main"
    assert t2.status == TaskStatus.COMPLETED

    # Now downstream_step should be unblocked and READY
    downstream_after = db.get_task("downstream_step")
    assert downstream_after.status == TaskStatus.READY

    # Step 3: Run downstream task
    t3 = orch.step()
    assert t3 is not None
    assert t3.task_id == "downstream_step"
    assert t3.status == TaskStatus.COMPLETED


def test_orchestrator_auto_repair_escalation(tmp_path: Path):
    """Verify that when verification fails, Orchestrator enqueues a System 2 repair task and keeps dependents blocked."""
    from baton.models import VerificationFailedError

    db = Database(tmp_path / "repair_baton.db")
    engine_mgr = MockEngineManager()
    sys1 = MockSystem1()
    sys2 = MockSystem2()
    orch = Orchestrator(db=db, engine_manager=engine_mgr, system1=sys1, system2=sys2, auto_verify=False, auto_repair=True)

    job = db.create_job("Repair Escalation Job", "Test escalation to System 2 repair")
    db.create_task(
        title="Verify Broken Module",
        tier=Tier.SYSTEM_1,
        task_type=TaskType.FAST_VERIFY,
        task_id="verify_target",
        job_id=job.job_id,
        priority=10,
    )
    db.create_task(
        title="Deploy Code",
        tier=Tier.SYSTEM_1,
        task_type=TaskType.FAST_RECON,
        task_id="deploy_step",
        job_id=job.job_id,
        depends_on=["verify_target"],
        priority=5,
    )

    # Mock sys1 to simulate verification failure after retries
    sys1.execute_task = MagicMock(side_effect=VerificationFailedError(
        message="Verification failed: 2 tests broken",
        failure_trace="AssertionError: 1 != 2",
        target_code="def broken(): pass",
    ))

    # Step 1: Step the queue on verify_target -> should fail and trigger repair
    t1 = orch.step()
    assert t1 is not None
    assert t1.task_id == "verify_target"
    assert t1.status == TaskStatus.FAILED

    # Check that a System 2 repair task was automatically created and is READY
    repair_task = db.get_task("repair_verify_target")
    assert repair_task is not None
    assert repair_task.tier == Tier.SYSTEM_2
    assert repair_task.task_type == TaskType.DEEP_SYNTHESIS
    assert repair_task.status == TaskStatus.READY
    assert "AssertionError: 1 != 2" in repair_task.prompt_template

    # Check that deploy_step is still BLOCKED and rewired to wait on the repair task
    deploy = db.get_task("deploy_step")
    assert deploy.status == TaskStatus.BLOCKED
    assert "repair_verify_target" in deploy.depends_on

    # Step 2: Run the repair task on System 2
    t2 = orch.step()
    assert t2 is not None
    assert t2.task_id == "repair_verify_target"
    assert t2.status == TaskStatus.COMPLETED

    # Now deploy_step is unblocked and READY
    deploy_after = db.get_task("deploy_step")
    assert deploy_after.status == TaskStatus.READY


def test_tier_sticky_scheduling_drains_warm_system1(tmp_path: Path):
    """Verify that when System 1 is warm, all runnable System 1 tasks drain before switching to System 2,
    even if the System 2 task has a higher scalar priority."""
    db = Database(tmp_path / "sticky_sys1.db")
    engine_mgr = MockEngineManager()
    engine_mgr.active_tier = Tier.SYSTEM_1  # System 1 is warm
    orch = Orchestrator(db=db, engine_manager=engine_mgr, system1=MockSystem1(), system2=MockSystem2(), auto_verify=False, tier_sticky=True)

    job = db.create_job("Sticky Sys1 Job", "Test batching on System 1")
    # Low priority Sys1 tasks
    db.create_task(title="Sys1 Recon A", tier=Tier.SYSTEM_1, task_id="sys1_a", job_id=job.job_id, priority=3)
    db.create_task(title="Sys1 Recon B", tier=Tier.SYSTEM_1, task_id="sys1_b", job_id=job.job_id, priority=4)
    # High priority Sys2 task
    db.create_task(title="Sys2 Synth X", tier=Tier.SYSTEM_2, task_id="sys2_x", job_id=job.job_id, priority=10)

    # Step 1: Should run sys1_b (prio 4) because System 1 is warm
    t1 = orch.step()
    assert t1 is not None
    assert t1.task_id == "sys1_b"
    assert engine_mgr.active_tier == Tier.SYSTEM_1

    # Step 2: Should run sys1_a (prio 3) because System 1 is still warm
    t2 = orch.step()
    assert t2 is not None
    assert t2.task_id == "sys1_a"
    assert engine_mgr.active_tier == Tier.SYSTEM_1

    # Step 3: All Sys1 tasks drained -> now switch to System 2 for sys2_x
    t3 = orch.step()
    assert t3 is not None
    assert t3.task_id == "sys2_x"
    assert engine_mgr.active_tier == Tier.SYSTEM_2


def test_tier_sticky_scheduling_drains_warm_system2(tmp_path: Path):
    """Verify that when System 2 (Colibrì) is warm, all runnable System 2 tasks drain before switching to System 1,
    preventing premature unloading of 284B MoE weights."""
    db = Database(tmp_path / "sticky_sys2.db")
    engine_mgr = MockEngineManager()
    engine_mgr.active_tier = Tier.SYSTEM_2  # Colibrì is warm
    orch = Orchestrator(db=db, engine_manager=engine_mgr, system1=MockSystem1(), system2=MockSystem2(), auto_verify=False, tier_sticky=True)

    job = db.create_job("Sticky Sys2 Job", "Test batching on System 2")
    db.create_task(title="Sys2 Synth A", tier=Tier.SYSTEM_2, task_id="sys2_a", job_id=job.job_id, priority=3)
    db.create_task(title="Sys2 Synth B", tier=Tier.SYSTEM_2, task_id="sys2_b", job_id=job.job_id, priority=4)
    db.create_task(title="Sys1 Urgent Recon", tier=Tier.SYSTEM_1, task_id="sys1_urgent", job_id=job.job_id, priority=10)

    # Step 1: Drains sys2_b on warm System 2
    t1 = orch.step()
    assert t1 is not None
    assert t1.task_id == "sys2_b"
    assert engine_mgr.active_tier == Tier.SYSTEM_2

    # Step 2: Drains sys2_a on warm System 2
    t2 = orch.step()
    assert t2 is not None
    assert t2.task_id == "sys2_a"
    assert engine_mgr.active_tier == Tier.SYSTEM_2

    # Step 3: Now switches to System 1 for sys1_urgent
    t3 = orch.step()
    assert t3 is not None
    assert t3.task_id == "sys1_urgent"
    assert engine_mgr.active_tier == Tier.SYSTEM_1


def test_tier_sticky_cold_start_prioritizes_system1(tmp_path: Path):
    """Verify that on cold start (active_tier is None), System 1 tasks run first to unblock downstream nodes."""
    db = Database(tmp_path / "cold_start.db")
    engine_mgr = MockEngineManager()  # active_tier is None
    orch = Orchestrator(db=db, engine_manager=engine_mgr, system1=MockSystem1(), system2=MockSystem2(), auto_verify=False, tier_sticky=True)

    job = db.create_job("Cold Start Job", "Test cold start priority")
    db.create_task(title="Sys2 Task High", tier=Tier.SYSTEM_2, task_id="sys2_high", job_id=job.job_id, priority=10)
    db.create_task(title="Sys1 Recon Low", tier=Tier.SYSTEM_1, task_id="sys1_low", job_id=job.job_id, priority=2)

    # Cold start should pick Sys1
    t1 = orch.step()
    assert t1 is not None
    assert t1.task_id == "sys1_low"
    assert engine_mgr.active_tier == Tier.SYSTEM_1


def test_tier_sticky_starvation_threshold(tmp_path: Path):
    """Verify that when a warm tier hits the starvation threshold, it yields to the waiting other tier."""
    db = Database(tmp_path / "starvation.db")
    engine_mgr = MockEngineManager()
    engine_mgr.active_tier = Tier.SYSTEM_1
    # Starvation threshold of 2 tasks
    orch = Orchestrator(db=db, engine_manager=engine_mgr, system1=MockSystem1(), system2=MockSystem2(), auto_verify=False, tier_sticky=True, starvation_threshold=2)

    job = db.create_job("Starvation Job", "Test starvation threshold")
    db.create_task(title="Sys1 Task 1", tier=Tier.SYSTEM_1, task_id="sys1_1", job_id=job.job_id, priority=5)
    db.create_task(title="Sys1 Task 2", tier=Tier.SYSTEM_1, task_id="sys1_2", job_id=job.job_id, priority=5)
    db.create_task(title="Sys1 Task 3", tier=Tier.SYSTEM_1, task_id="sys1_3", job_id=job.job_id, priority=5)
    db.create_task(title="Sys2 Waiting", tier=Tier.SYSTEM_2, task_id="sys2_wait", job_id=job.job_id, priority=10)

    # Step 1: Sys1 runs (count = 1)
    t1 = orch.step()
    assert t1 is not None
    assert t1.task_id == "sys1_1"

    # Step 2: Sys1 runs (count = 2)
    t2 = orch.step()
    assert t2 is not None
    assert t2.task_id == "sys1_2"

    # Step 3: Count has reached 2 (threshold), so it yields to Sys2
    t3 = orch.step()
    assert t3 is not None
    assert t3.task_id == "sys2_wait"
    assert engine_mgr.active_tier == Tier.SYSTEM_2


def test_tier_sticky_disabled_uses_scalar_priority(tmp_path: Path):
    """Verify that if tier_sticky=False, execution strictly obeys scalar priority."""
    db = Database(tmp_path / "no_sticky.db")
    engine_mgr = MockEngineManager()
    engine_mgr.active_tier = Tier.SYSTEM_1
    orch = Orchestrator(db=db, engine_manager=engine_mgr, system1=MockSystem1(), system2=MockSystem2(), auto_verify=False, tier_sticky=False)

    job = db.create_job("Scalar Job", "Strict scalar priority")
    db.create_task(title="Sys1 Low", tier=Tier.SYSTEM_1, task_id="sys1_low", job_id=job.job_id, priority=1)
    db.create_task(title="Sys2 High", tier=Tier.SYSTEM_2, task_id="sys2_high", job_id=job.job_id, priority=10)

    t1 = orch.step()
    assert t1 is not None
    assert t1.task_id == "sys2_high"
    assert engine_mgr.active_tier == Tier.SYSTEM_2


def test_orchestrator_in_progress_concurrency_lock(tmp_path: Path):
    """Verify that Orchestrator.step() refuses to start when any task is currently IN_PROGRESS."""
    db = Database(tmp_path / "concurrency_lock.db")
    engine_mgr = MockEngineManager()
    orch = Orchestrator(db=db, engine_manager=engine_mgr, system1=MockSystem1(), system2=MockSystem2(), auto_verify=False)

    job = db.create_job("Lock Job", "Test cross-runner concurrency lock")
    t1 = db.create_task(title="Running Task", tier=Tier.SYSTEM_2, task_id="task_running", job_id=job.job_id)
    t2 = db.create_task(title="Ready Task", tier=Tier.SYSTEM_2, task_id="task_ready", job_id=job.job_id)

    # Claim task 1 so it becomes IN_PROGRESS
    claimed = db.claim_specific_task(t1.task_id)
    assert claimed is not None
    assert claimed.status == TaskStatus.IN_PROGRESS
    assert len(db.get_in_progress_tasks()) == 1

    # Attempting to step should return None (locked out)
    assert orch.step() is None
    assert orch.step(task_id=t2.task_id) is None

    # Complete task 1
    db.mark_task_completed(t1.task_id, result_payload="done")
    assert len(db.get_in_progress_tasks()) == 0

    # Now step succeeds and claims task 2
    stepped = orch.step()
    assert stepped is not None
    assert stepped.task_id == t2.task_id
    assert stepped.status == TaskStatus.COMPLETED



