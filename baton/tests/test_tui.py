"""Unit tests for Baton Textual TUI."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from textual.widgets import DataTable, Markdown, Static

from baton.db import Database
from baton.models import TaskProgress, TaskStatus, Tier
from baton.orchestrator import Orchestrator
from baton.tui_textual import BatonApp


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_tui_mount_and_render_in_progress(tmp_path: Path):
    db = Database(tmp_path / "test_tui.db")
    job = db.create_job("Project Ephemeris", "Calculate coordinates")
    
    class MockEngMgr:
        active_tier = Tier.SYSTEM_2
        def get_status(self):
            return {
                "active_tier": "system_2",
                "ollama": {"online": False},
                "colibri": {"online": True},
            }

    orch = Orchestrator(db=db, engine_manager=MockEngMgr(), progress_dir=tmp_path / "prog")
    
    # Add a task and set to in_progress
    t = orch.add_manual_task(job.job_id, "Derive Ephemeris", Tier.SYSTEM_2)
    db.claim_specific_task(t.task_id)

    # Set up progress
    prog = TaskProgress(
        task_id=t.task_id,
        tier=Tier.SYSTEM_2,
        started_at=100.0,
        tokens_generated=42,
        tokens_per_second=0.75,
        reasoning_text="Integrating planetary perturbations...",
        content_text="def compute_earth_pos(jd):",
    )
    orch.active_progress[t.task_id] = prog
    prog.save_to_file(tmp_path / "prog" / f"{t.task_id}.json")

    app = BatonApp(orchestrator=orch)
    async with app.run_test() as pilot:
        # Check engine banner has running indicator
        banner = app.query_one("#engine-bar", Static)
        assert "RUNNING" in str(banner.render())

        # Check queue table
        table = app.query_one("#queue-table", DataTable)
        assert table.row_count == 1

        # Check inspector markdown displays live reasoning and draft code
        md = app.query_one("#inspector-content", Markdown)
        md_text = md.source or ""
        assert "Integrating planetary perturbations" in md_text
        assert "compute_earth_pos" in md_text
        assert "IN PROGRESS" in md_text


def test_edit_task_modal_initialization(tmp_path: Path):
    from baton.tui_textual import EditTaskModal

    db = Database(tmp_path / "test_modal.db")
    job = db.create_job("Modal Job", "Test modal init")
    orch = Orchestrator(db=db)
    t = orch.add_manual_task(job.job_id, "Test Edit Task", Tier.SYSTEM_1, prompt="Sample prompt")

    modal = EditTaskModal(t, db)
    assert modal.target_task.task_id == t.task_id
    assert modal.target_task.title == "Test Edit Task"


@pytest.mark.anyio
async def test_tui_delete_task_action(tmp_path: Path):
    db = Database(tmp_path / "test_del_tui.db")
    job = db.create_job("Delete Job", "Delete task")
    orch = Orchestrator(db=db, progress_dir=tmp_path / "prog")
    t = orch.add_manual_task(job.job_id, "Task To Delete", Tier.SYSTEM_1)

    app = BatonApp(orchestrator=orch)
    async with app.run_test() as pilot:
        table = app.query_one("#queue-table", DataTable)
        assert table.row_count == 1
        assert app.selected_task_id == t.task_id

        # Trigger delete action (hotkey 'd')
        app.action_delete_selected()
        await pilot.pause()

        # Task should be gone from table and DB
        assert table.row_count == 0
        assert db.get_task(t.task_id) is None


@pytest.mark.anyio
async def test_tui_stop_task_action(tmp_path: Path):
    db = Database(tmp_path / "test_stop_tui.db")
    job = db.create_job("Stop Job", "Stop task action")
    orch = Orchestrator(db=db, progress_dir=tmp_path / "prog")
    t = orch.add_manual_task(job.job_id, "Task To Stop", Tier.SYSTEM_2)
    db.claim_specific_task(t.task_id)

    app = BatonApp(orchestrator=orch)
    async with app.run_test() as pilot:
        assert app.is_running_task is False
        app.is_running_task = True  # emulate background runner active

        # Press 's' to trigger stop action
        await pilot.press("s")
        await pilot.pause()

        assert app.is_running_task is False
        stopped_task = db.get_task(t.task_id)
        assert stopped_task.status == TaskStatus.FAILED
        assert "Stopped by user" in stopped_task.error_message


@pytest.mark.anyio
async def test_tui_verify_task_action(tmp_path: Path):
    db = Database(tmp_path / "test_verify_tui.db")
    job = db.create_job("Verify Job", "Test verify keybinding")
    orch = Orchestrator(db=db, progress_dir=tmp_path / "prog")
    t = orch.add_manual_task(job.job_id, "Synthesize Planetary Model", Tier.SYSTEM_2)

    app = BatonApp(orchestrator=orch)
    async with app.run_test() as pilot:
        table = app.query_one("#queue-table", DataTable)
        assert table.row_count == 1
        assert app.selected_task_id == t.task_id

        # Press 'v' to queue verification task
        await pilot.press("v")
        await pilot.pause()

        # Should now have 2 tasks in table
        assert table.row_count == 2
        all_tasks = db.get_all_tasks()
        verify_tasks = [tsk for tsk in all_tasks if tsk.task_id != t.task_id]
        assert len(verify_tasks) == 1
        verify_task = verify_tasks[0]
        assert verify_task.tier == Tier.SYSTEM_1
        assert verify_task.task_type.value == "fast_verify"
        assert t.task_id in verify_task.depends_on
        assert "Verify: Synthesize Planetary Model" in verify_task.title



