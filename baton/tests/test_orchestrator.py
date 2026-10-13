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
    return Orchestrator(db=db, engine_manager=engine_mgr, system1=sys1, system2=sys2)


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

