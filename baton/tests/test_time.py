"""Unit tests for Time & Alarm bundle, parsing, and scheduling mechanics."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from baton.chat import BatonChatSession, TIME_TOOLS_SCHEMA, TOOL_BUNDLES
from baton.db import Database
from baton.models import Task, TaskStatus, Tier
from baton.orchestrator import Orchestrator
from baton.tools.time_tool import TimeTool, parse_time_spec


@pytest.fixture
def temp_db(tmp_path: Path) -> Database:
    db_file = tmp_path / "test_time_baton.db"
    return Database(db_path=db_file)


@pytest.fixture
def orchestrator(temp_db: Database, tmp_path: Path) -> Orchestrator:
    return Orchestrator(db=temp_db, progress_dir=tmp_path / "progress")


@pytest.fixture
def time_tool(orchestrator: Orchestrator) -> TimeTool:
    return TimeTool(db=orchestrator.db, orchestrator=orchestrator)


# =============================================================================
# 1. parse_time_spec tests
# =============================================================================

def test_parse_time_spec_relative_shorthand():
    before = datetime.now()
    t_30s = parse_time_spec("30s")
    assert (t_30s - before).total_seconds() >= 29

    t_15m = parse_time_spec("15m")
    assert 890 <= (t_15m - before).total_seconds() <= 910

    t_2h = parse_time_spec("2h")
    assert 7190 <= (t_2h - before).total_seconds() <= 7210

    t_1d = parse_time_spec("1d")
    assert 86390 <= (t_1d - before).total_seconds() <= 86410


def test_parse_time_spec_natural_language():
    before = datetime.now()
    t_10m = parse_time_spec("in 10 minutes")
    assert 590 <= (t_10m - before).total_seconds() <= 610

    t_1h = parse_time_spec("in 1 hour")
    assert 3590 <= (t_1h - before).total_seconds() <= 3610

    t_plus = parse_time_spec("+45s")
    assert 44 <= (t_plus - before).total_seconds() <= 46


def test_parse_time_spec_iso8601():
    parsed = parse_time_spec("2030-05-20T15:30:00")
    assert parsed.year == 2030
    assert parsed.month == 5
    assert parsed.day == 20
    assert parsed.hour == 15
    assert parsed.minute == 30


def test_parse_time_spec_time_of_day():
    now = datetime.now()
    # Test setting a time 2 hours ahead
    future_time = now + timedelta(hours=2)
    spec = future_time.strftime("%H:%M")
    parsed = parse_time_spec(spec)
    assert parsed.hour == future_time.hour
    assert parsed.minute == future_time.minute
    assert parsed.day == future_time.day

    # Test setting a time 2 hours in the past -> should rollover to tomorrow
    past_time = now - timedelta(hours=2)
    spec_past = past_time.strftime("%H:%M")
    parsed_past = parse_time_spec(spec_past)
    assert parsed_past.hour == past_time.hour
    assert parsed_past.minute == past_time.minute
    assert parsed_past.date() == (now + timedelta(days=1)).date()


def test_parse_time_spec_invalid():
    with pytest.raises(ValueError, match="Could not parse time specification"):
        parse_time_spec("not a valid time spec at all")


# =============================================================================
# 2. TimeTool Methods & Formatting
# =============================================================================

def test_time_tool_get_current_time(time_tool: TimeTool):
    info = time_tool.get_current_time()
    assert "utc_iso" in info
    assert "iso_utc" in info
    assert "local_iso" in info
    assert "iso_local" in info
    assert "timezone" in info
    assert "epoch_seconds" in info
    assert "formatted" in info
    assert info["epoch_seconds"] > 1700000000


def test_time_tool_format_markdown(time_tool: TimeTool):
    md = time_tool.format_time_markdown()
    assert "Local Time:" in md
    assert "UTC Time:" in md
    assert "ISO-8601 Local:" in md


# =============================================================================
# 3. Database Scheduling Integration
# =============================================================================

def test_db_scheduled_tasks_filtering(temp_db: Database):
    # Task 1: No schedule (immediately runnable)
    t1 = temp_db.create_task(title="Immediate Task 1", tier=Tier.SYSTEM_1)

    # Task 2: Scheduled in the past (immediately runnable)
    past = datetime.now() - timedelta(minutes=5)
    t2 = temp_db.create_task(title="Past Task 2", tier=Tier.SYSTEM_1, scheduled_at=past)

    # Task 3: Scheduled in future (should NOT be runnable)
    future = datetime.now() + timedelta(hours=1)
    t3 = temp_db.create_task(title="Future Task 3", tier=Tier.SYSTEM_1, scheduled_at=future)

    runnable = temp_db.get_runnable_tasks()
    runnable_ids = [t.task_id for t in runnable]
    assert t1.task_id in runnable_ids
    assert t2.task_id in runnable_ids
    assert t3.task_id not in runnable_ids

    # Check get_scheduled_tasks
    scheduled = temp_db.get_scheduled_tasks()
    sched_ids = [t.task_id for t in scheduled]
    assert t3.task_id in sched_ids
    assert t1.task_id not in sched_ids

    # Check get_next_scheduled_task
    next_t = temp_db.get_next_scheduled_task()
    assert next_t is not None
    assert next_t.task_id == t3.task_id

    # Test update_task clear_schedule
    temp_db.update_task(t3.task_id, clear_schedule=True)
    t3_refreshed = temp_db.get_task(t3.task_id)
    assert t3_refreshed is not None
    assert t3_refreshed.scheduled_at is None
    assert t3.task_id in [t.task_id for t in temp_db.get_runnable_tasks()]


# =============================================================================
# 4. TimeTool Operations & Execution
# =============================================================================

def test_time_tool_schedule_and_cancel(time_tool: TimeTool, temp_db: Database):
    task = temp_db.create_task(title="Alarm Target", tier=Tier.SYSTEM_1)

    msg = time_tool.schedule_task_alarm(task.task_id, "30m")
    assert "Alarm Scheduled for Task" in msg
    assert task.task_id in msg

    # Verify task in DB has scheduled_at
    updated = temp_db.get_task(task.task_id)
    assert updated is not None
    assert updated.scheduled_at is not None

    # List alarms
    alarms_md = time_tool.list_alarms()
    assert "Active Task Alarms" in alarms_md
    assert task.task_id in alarms_md

    # Cancel alarm
    cancel_md = time_tool.cancel_alarm(task.task_id)
    assert "Alarm cancelled" in cancel_md

    refreshed = temp_db.get_task(task.task_id)
    assert refreshed is not None
    assert refreshed.scheduled_at is None


def test_time_tool_create_scheduled_task(time_tool: TimeTool, temp_db: Database):
    msg = time_tool.create_scheduled_task(
        title="Check Satellites",
        when="1h",
        tier="system_1",
        prompt="Scan overhead passes",
    )
    assert "New Scheduled Task Created" in msg
    assert "Check Satellites" in msg

    # Verify it is in list_alarms
    alarms = time_tool.list_alarms()
    assert "Check Satellites" in alarms


def test_time_tool_dispatcher(time_tool: TimeTool, temp_db: Database):
    res_time = time_tool.execute("get_current_time", {})
    assert "Current System Time" in res_time

    t = temp_db.create_task(title="Dispatch target", tier=Tier.SYSTEM_1)
    res_sched = time_tool.execute("schedule_task_alarm", {"task_id": t.task_id, "when": "15m"})
    assert "Alarm Scheduled" in res_sched

    res_list = time_tool.execute("list_alarms", {})
    assert "Dispatch target" in res_list

    res_cancel = time_tool.execute("cancel_alarm", {"task_id": t.task_id})
    assert "Alarm cancelled" in res_cancel


# =============================================================================
# 5. Chat Session Tool Shelf & Slash Commands
# =============================================================================

def test_chat_session_time_bundle_integration(temp_db: Database, tmp_path: Path):
    orch = Orchestrator(db=temp_db, progress_dir=tmp_path / "progress")
    session = BatonChatSession(
        db=temp_db,
        orchestrator=orch,
        model_name="qwen3:14b",
    )

    # Time bundle is on the shelf initially
    assert "time" in TOOL_BUNDLES
    assert "time" not in session.active_bundles

    # Activate time bundle via shelf tool call
    res_act = session.execute_tool("activate_tool_bundle", {"bundle_name": "time"})
    assert res_act["success"] is True
    assert "time" in session.active_bundles

    # Check active tools include time tools
    active_tools = session.get_active_tools_schema()
    tool_names = {t["function"]["name"] for t in active_tools}
    assert "get_current_time" in tool_names
    assert "schedule_task_alarm" in tool_names
    assert "create_scheduled_task" in tool_names
    assert "list_alarms" in tool_names
    assert "cancel_alarm" in tool_names

    # Test executing tool via session dispatcher
    res_raw = session.execute_tool("get_current_time", {})
    assert "result" in res_raw
    assert "Current System Time" in res_raw["result"]

    # Test create_scheduled_task through session dispatcher
    res_create = session.execute_tool(
        "create_scheduled_task",
        {"title": "Chat Created Alarm", "when": "45m"},
    )
    assert "result" in res_create
    assert "New Scheduled Task Created" in res_create["result"]

    # Test slash commands
    assert session.handle_slash_command("/time") is True
    assert session.handle_slash_command("/alarms") is True


def test_orchestrator_wait_for_scheduled(orchestrator: Orchestrator, temp_db: Database):
    # Create a task scheduled 0.2 seconds in the future
    soon = datetime.now() + timedelta(seconds=0.2)
    orchestrator.create_task(
        title="Immediate Scheduled Task",
        tier=Tier.SYSTEM_1,
        scheduled_at=soon,
    )

    # Mock the system1 runner so it doesn't need Ollama
    orchestrator.system1.run_recon = MagicMock(return_value="Task executed successfully")

    # Run with wait_for_scheduled=True and a small max_wait
    completed = orchestrator.run_all(max_steps=5, wait_for_scheduled=True, max_wait_seconds=2.0)
    assert len(completed) == 1
    assert completed[0].title == "Immediate Scheduled Task"
