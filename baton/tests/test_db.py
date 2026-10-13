"""Unit tests for Baton database, DAG validation, and priority queue."""

import pytest
from pathlib import Path
from baton.models import Tier, TaskType, TaskStatus, JobStatus, TaskPlan, DAGPlan, ArtifactType
from baton.db import Database


@pytest.fixture
def temp_db(tmp_path: Path):
    db_file = tmp_path / "test_baton.db"
    return Database(db_file)


def test_create_and_get_job(temp_db: Database):
    job = temp_db.create_job(title="Solar Ephemeris", goal="Compute solar elevation")
    assert job.title == "Solar Ephemeris"
    assert job.status == JobStatus.PENDING

    retrieved = temp_db.get_job(job.job_id)
    assert retrieved is not None
    assert retrieved.job_id == job.job_id
    assert retrieved.goal == "Compute solar elevation"


def test_dag_cycle_detection(temp_db: Database):
    # Construct a cyclic plan A -> B -> A
    cycle_plan = [
        TaskPlan(id="task_a", title="Task A", depends_on=["task_b"]),
        TaskPlan(id="task_b", title="Task B", depends_on=["task_a"]),
    ]
    with pytest.raises(ValueError, match="Cycle detected"):
        temp_db.validate_dag(cycle_plan)


def test_ingest_dag_and_dependency_cascading(temp_db: Database):
    job = temp_db.create_job(title="Orbital Tracker", goal="Build orbital tracker")

    plan = DAGPlan(
        project="Orbital Tracker",
        goal="Build orbital tracker",
        tasks=[
            TaskPlan(
                id="recon_1",
                title="Recon orbital mechanics",
                tier=Tier.SYSTEM_1,
                type=TaskType.FAST_RECON,
                depends_on=[],
                priority=10,
            ),
            TaskPlan(
                id="recon_2",
                title="Recon constants",
                tier=Tier.SYSTEM_1,
                type=TaskType.FAST_RECON,
                depends_on=[],
                priority=5,
            ),
            TaskPlan(
                id="synth_1",
                title="Synthesize Python module",
                tier=Tier.SYSTEM_2,
                type=TaskType.DEEP_SYNTHESIS,
                depends_on=["recon_1", "recon_2"],
                priority=1,
            ),
            TaskPlan(
                id="verify_1",
                title="Verify unit tests",
                tier=Tier.SYSTEM_1,
                type=TaskType.FAST_VERIFY,
                depends_on=["synth_1"],
                priority=1,
            ),
        ],
    )

    tasks = temp_db.ingest_dag_plan(job.job_id, plan)
    assert len(tasks) == 4

    # Check runnable tasks: recon_1 and recon_2 should be READY, synth_1 and verify_1 should be BLOCKED
    runnable = temp_db.get_runnable_tasks(job_id=job.job_id)
    assert len(runnable) == 2
    # Ordered by priority DESC: recon_1 (10) before recon_2 (5)
    assert runnable[0].task_id == "recon_1"
    assert runnable[1].task_id == "recon_2"

    synth_task = temp_db.get_task("synth_1")
    assert synth_task is not None
    assert synth_task.status == TaskStatus.BLOCKED

    # Claim and complete recon_1
    claimed = temp_db.claim_next_task(tier=Tier.SYSTEM_1, job_id=job.job_id)
    assert claimed is not None
    assert claimed.task_id == "recon_1"
    assert claimed.status == TaskStatus.IN_PROGRESS

    temp_db.mark_task_completed("recon_1", result_payload="Orbital formula found")
    
    # synth_1 should still be BLOCKED because recon_2 is not done
    synth_task = temp_db.get_task("synth_1")
    assert synth_task.status == TaskStatus.BLOCKED

    # Complete recon_2
    claimed_2 = temp_db.claim_next_task(tier=Tier.SYSTEM_1, job_id=job.job_id)
    assert claimed_2.task_id == "recon_2"
    temp_db.mark_task_completed("recon_2", result_payload="Constants extracted")

    # Now synth_1 should have automatically cascaded to READY!
    synth_task = temp_db.get_task("synth_1")
    assert synth_task.status == TaskStatus.READY

    # Verify upstream dossiers retrieval
    dossiers = temp_db.get_upstream_dossiers("synth_1")
    assert len(dossiers) == 2
    assert any(d["result_payload"] == "Orbital formula found" for d in dossiers)
    assert any(d["result_payload"] == "Constants extracted" for d in dossiers)

    # Claim and complete synth_1
    claimed_s = temp_db.claim_next_task(tier=Tier.SYSTEM_2, job_id=job.job_id)
    assert claimed_s.task_id == "synth_1"
    temp_db.mark_task_completed("synth_1", result_payload="class OrbitalTracker: pass")

    # Now verify_1 should be READY
    verify_task = temp_db.get_task("verify_1")
    assert verify_task.status == TaskStatus.READY

    # Complete verify_1 and check job completion
    temp_db.mark_task_completed("verify_1", result_payload="All tests passed")
    updated_job = temp_db.get_job(job.job_id)
    assert updated_job.status == JobStatus.COMPLETED


def test_artifacts_tracking(temp_db: Database):
    job = temp_db.create_job(title="Art Job", goal="Test artifacts")
    plan = DAGPlan(
        project="Art",
        goal="Art goal",
        tasks=[TaskPlan(id="art_task", title="Make art", tier=Tier.SYSTEM_1, type=TaskType.FAST_RECON)]
    )
    temp_db.ingest_dag_plan(job.job_id, plan)

    art = temp_db.record_artifact(
        task_id="art_task",
        job_id=job.job_id,
        artifact_type=ArtifactType.DOSSIER,
        title="Solar Dossier",
        file_path="artifacts/solar_dossier.md",
        content_preview="Summary of solar formulas..."
    )
    assert art.artifact_id.startswith("art_")

    artifacts = temp_db.get_artifacts(job_id=job.job_id)
    assert len(artifacts) == 1
    assert artifacts[0].title == "Solar Dossier"


def test_delete_task_cascades_readiness(temp_db: Database):
    job = temp_db.create_job(title="Delete Job", goal="Test task deletion")
    plan = DAGPlan(
        project="Test",
        goal="Test deletion",
        tasks=[
            TaskPlan(id="task_a", title="Task A", tier=Tier.SYSTEM_1, type=TaskType.FAST_RECON),
            TaskPlan(id="task_b", title="Task B", tier=Tier.SYSTEM_2, type=TaskType.DEEP_SYNTHESIS, depends_on=["task_a"]),
        ],
    )
    temp_db.ingest_dag_plan(job.job_id, plan)

    # task_b should be BLOCKED on task_a
    tb = temp_db.get_task("task_b")
    assert tb.status == TaskStatus.BLOCKED

    # Delete task_a
    deleted = temp_db.delete_task("task_a")
    assert deleted is True

    # task_a should be gone
    assert temp_db.get_task("task_a") is None

    # task_b should now be unblocked and READY
    tb_after = temp_db.get_task("task_b")
    assert tb_after.status == TaskStatus.READY


def test_database_wal_mode_and_pragmas(temp_db: Database):
    """Verify that SQLite initializes with WAL mode, foreign keys, and busy timeout."""
    with temp_db.get_connection() as conn:
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert journal_mode.lower() == "wal"

        foreign_keys = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        assert foreign_keys == 1

        busy_timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
        assert busy_timeout == 5000

        synchronous = conn.execute("PRAGMA synchronous;").fetchone()[0]
        # 1 corresponds to NORMAL
        assert synchronous == 1


def test_database_concurrent_reader_and_writer(tmp_path: Path):
    """Verify that a reader can read concurrently while a writer holds an active write transaction."""
    import threading
    import time

    db_path = tmp_path / "test_concurrent.db"
    db = Database(db_path)

    writer_started = threading.Event()
    reader_done = threading.Event()
    read_results = []
    errors = []

    def writer_worker():
        try:
            conn = db.get_connection()
            with conn:
                conn.execute(
                    "INSERT INTO jobs (job_id, title, goal, status) VALUES (?, ?, ?, ?)",
                    ("job_write", "Concurrent Job", "Testing WAL", "pending"),
                )
                writer_started.set()
                # Hold the transaction open while reader queries
                time.sleep(0.3)
            conn.close()
        except Exception as e:
            errors.append(e)

    def reader_worker():
        try:
            writer_started.wait(timeout=2.0)
            # Reader opens connection while writer's transaction is still active
            conn = db.get_connection()
            rows = conn.execute("SELECT * FROM jobs").fetchall()
            read_results.append(len(rows))
            conn.close()
        except Exception as e:
            errors.append(e)
        finally:
            reader_done.set()

    t_writer = threading.Thread(target=writer_worker)
    t_reader = threading.Thread(target=reader_worker)

    t_writer.start()
    t_reader.start()

    t_writer.join(timeout=3.0)
    t_reader.join(timeout=3.0)

    assert not errors, f"Errors occurred during concurrent access: {errors}"
    assert reader_done.is_set()
    # In WAL mode, snapshot isolation allows reader to read without getting 'database is locked'
    assert len(read_results) == 1
