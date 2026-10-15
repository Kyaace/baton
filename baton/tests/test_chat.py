"""Unit tests for Baton System 1 Copilot Chat (baton/chat.py)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from baton.chat import (
    BATON_QUEUE_TOOLS_SCHEMA,
    BATON_TOOLS_SCHEMA,
    CORE_TOOLS_SCHEMA,
    NEWSBOAT_TOOLS_SCHEMA,
    RSS_TOOLS_SCHEMA,
    SANDBOX_TOOLS_SCHEMA,
    SDR_TOOLS_SCHEMA,
    SHELF_TOOLS_SCHEMA,
    TOOL_BUNDLES,
    WIKIPEDIA_TOOLS_SCHEMA,
    BatonChatSession,
)
from baton.db import Database
from baton.models import Task, TaskStatus, Tier
from baton.orchestrator import Orchestrator


@pytest.fixture
def temp_db(tmp_path: Path) -> Database:
    db_file = tmp_path / "test_chat_baton.db"
    return Database(db_path=db_file)


@pytest.fixture
def chat_session(temp_db: Database, tmp_path: Path) -> BatonChatSession:
    orch = Orchestrator(db=temp_db, progress_dir=tmp_path / "progress")
    session = BatonChatSession(
        db=temp_db,
        orchestrator=orch,
        model_name="qwen3:14b",
        ollama_url="http://127.0.0.1:11434",
    )
    return session


def test_tool_schemas_validity():
    """Verify that all function definitions match standard JSON schema specs."""
    assert len(BATON_TOOLS_SCHEMA) >= 20
    assert len(SHELF_TOOLS_SCHEMA) == 4
    assert len(BATON_QUEUE_TOOLS_SCHEMA) == 10
    assert len(SANDBOX_TOOLS_SCHEMA) == 6
    assert len(SDR_TOOLS_SCHEMA) == 6

    names = set()
    for item in BATON_TOOLS_SCHEMA:
        assert item["type"] == "function"
        fn = item["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"
        assert "properties" in fn["parameters"]
        names.add(fn["name"])

    required_tools = {
        "list_tasks",
        "get_task",
        "add_task",
        "delete_task",
        "retry_failed_tasks",
        "list_artifacts",
        "read_artifact",
        "queue_status",
        "step_queue",
        "start_orchestration",
        "list_tool_bundles",
        "activate_tool_bundle",
        "deactivate_tool_bundle",
        "tool_help",
        "wiki_search",
        "wiki_read_section",
        "wiki_read_lead",
        "wiki_infobox",
        "rss_list_feeds",
        "rss_latest",
        "rss_search",
        "rss_read_article",
        "python_eval",
        "python_syntax_check",
        "pytest_run",
        "sandbox_write_file",
        "sandbox_read_file",
        "sandbox_list_files",
        "sdr_device_info",
        "sdr_spectrum_scan",
        "sdr_record_rf",
        "sdr_decode_telemetry",
        "sdr_adsb_tracking",
        "sdr_decode_packet",
        "get_current_time",
        "schedule_task_alarm",
        "create_scheduled_task",
        "list_alarms",
        "cancel_alarm",
    }
    assert required_tools.issubset(names)


def test_tool_shelf_dynamic_activation_and_deactivation(chat_session: BatonChatSession):
    """Verify that only active tool bundles are compiled into prompt schema, saving context."""
    # 1. Initial state: Baton Chat defaults to having 'core' and 'baton' bundles active
    assert chat_session.active_bundles == {"core", "baton"}
    initial_schema = chat_session.get_active_tools_schema()
    initial_names = {t["function"]["name"] for t in initial_schema}
    assert "list_tasks" in initial_names
    assert "add_task" in initial_names
    assert "activate_tool_bundle" in initial_names
    assert "rss_search" not in initial_names
    assert "wiki_search" not in initial_names
    assert "python_eval" not in initial_names
    assert "get_current_time" not in initial_names
    assert len(initial_schema) == len(SHELF_TOOLS_SCHEMA) + len(BATON_QUEUE_TOOLS_SCHEMA)

    # 2. Inspect shelf tool
    shelf_res = chat_session.execute_tool("list_tool_bundles", {})
    assert len(shelf_res["shelf"]) == 9  # core, baton, wikipedia, newsboat, sandbox, sdr, time, docs, vault
    assert "core" in shelf_res["active_bundles"]
    assert "baton" in shelf_res["active_bundles"]

    # 3. Deactivate 'baton' bundle to simulate switching to pure shelf/research mode
    deact_baton = chat_session.execute_tool("deactivate_tool_bundle", {"bundle_name": "baton"})
    assert deact_baton["success"] is True
    assert "baton" not in chat_session.active_bundles
    assert chat_session.active_bundles == {"core"}

    contracted_schema = chat_session.get_active_tools_schema()
    assert len(contracted_schema) == len(SHELF_TOOLS_SCHEMA)
    contracted_names = {t["function"]["name"] for t in contracted_schema}
    assert "add_task" not in contracted_names
    assert "delete_task" not in contracted_names

    # 4. Activate Sandbox bundle
    act_sandbox = chat_session.execute_tool("activate_tool_bundle", {"bundle_name": "sandbox"})
    assert act_sandbox["success"] is True
    assert "sandbox" in chat_session.active_bundles

    expanded_schema = chat_session.get_active_tools_schema()
    expanded_names = {t["function"]["name"] for t in expanded_schema}
    assert "python_eval" in expanded_names
    assert "python_syntax_check" in expanded_names
    assert "pytest_run" in expanded_names
    assert len(expanded_schema) == len(SHELF_TOOLS_SCHEMA) + len(SANDBOX_TOOLS_SCHEMA)

    # 5. Deactivate Sandbox bundle
    deact_sandbox = chat_session.execute_tool("deactivate_tool_bundle", {"bundle_name": "sandbox"})
    assert deact_sandbox["success"] is True
    assert "sandbox" not in chat_session.active_bundles

    # 6. Activate Newsboat bundle (resolves to 'rss')
    act_res = chat_session.execute_tool("activate_tool_bundle", {"bundle_name": "newsboat"})
    assert act_res["success"] is True
    assert act_res["activated_bundle"] == "rss"
    assert "rss" in chat_session.active_bundles

    expanded_schema = chat_session.get_active_tools_schema()
    expanded_names = {t["function"]["name"] for t in expanded_schema}
    assert "rss_search" in expanded_names
    assert "rss_latest" in expanded_names
    assert len(expanded_schema) == len(SHELF_TOOLS_SCHEMA) + len(RSS_TOOLS_SCHEMA)

    # 7. Deactivate RSS bundle
    deact_res = chat_session.execute_tool("deactivate_tool_bundle", {"bundle_name": "rss"})
    assert deact_res["success"] is True
    assert "rss" not in chat_session.active_bundles

    # 8. Activate SDR bundle
    act_sdr = chat_session.execute_tool("activate_tool_bundle", {"bundle_name": "sdr"})
    assert act_sdr["success"] is True
    assert "sdr" in chat_session.active_bundles

    sdr_schema = chat_session.get_active_tools_schema()
    sdr_names = {t["function"]["name"] for t in sdr_schema}
    assert "sdr_spectrum_scan" in sdr_names
    assert "sdr_device_info" in sdr_names
    assert len(sdr_schema) == len(SHELF_TOOLS_SCHEMA) + len(SDR_TOOLS_SCHEMA)

    # 9. Deactivate SDR bundle
    deact_sdr = chat_session.execute_tool("deactivate_tool_bundle", {"bundle_name": "sdr"})
    assert deact_sdr["success"] is True
    assert "sdr" not in chat_session.active_bundles

    assert len(chat_session.get_active_tools_schema()) == len(SHELF_TOOLS_SCHEMA)

    # 10. Reactivate 'baton' bundle
    act_baton = chat_session.execute_tool("activate_tool_bundle", {"bundle_name": "baton"})
    assert act_baton["success"] is True
    assert "baton" in chat_session.active_bundles
    re_names = {t["function"]["name"] for t in chat_session.get_active_tools_schema()}
    assert "add_task" in re_names


def test_research_agent_session_isolation(temp_db: Database, tmp_path: Path):
    """Verify that a research agent initialized without 'baton' bundle cannot mutate the queue."""
    orch = Orchestrator(db=temp_db, progress_dir=tmp_path / "progress")
    research_session = BatonChatSession(
        db=temp_db,
        orchestrator=orch,
        initial_bundles={"core", "library"},
    )
    assert research_session.active_bundles == {"core", "library"}
    schema = research_session.get_active_tools_schema()
    names = {t["function"]["name"] for t in schema}

    # Research tools present
    assert "wiki_search" in names
    assert "wiki_read_section" in names
    assert "activate_tool_bundle" in names

    # Queue mutation tools absent
    assert "add_task" not in names
    assert "delete_task" not in names
    assert "retry_failed_tasks" not in names
    assert "queue_status" not in names
    assert "step_queue" not in names
    assert "list_tasks" not in names

    # Backwards compatibility: initial_bundles with 'wikipedia' resolves to 'library'
    legacy_session = BatonChatSession(
        db=temp_db,
        orchestrator=orch,
        initial_bundles={"core", "wikipedia"},
    )
    assert legacy_session.active_bundles == {"core", "library"}


def test_execute_tool_add_list_and_get_task(chat_session: BatonChatSession):
    """Verify add_task, list_tasks, and get_task tool dispatches."""
    # 1. Add Task
    add_res = chat_session.execute_tool(
        "add_task",
        {
            "title": "Compute Lunar Semi-Major Axis",
            "tier": "system_2",
            "prompt": "Derive orbital eccentricity and semi-major axis.",
            "priority": 8,
            "task_id": "test_lunar_axis",
        },
    )
    assert add_res["success"] is True
    assert add_res["task_id"] == "test_lunar_axis"
    assert add_res["tier"] == "system_2"
    assert add_res["status"] == "ready"
    assert add_res["priority"] == 8

    # 2. List Tasks
    list_res = chat_session.execute_tool("list_tasks", {})
    assert list_res["count"] == 1
    assert list_res["tasks"][0]["task_id"] == "test_lunar_axis"

    # 3. Get Task Details
    get_res = chat_session.execute_tool("get_task", {"task_id": "test_lunar_axis"})
    assert get_res["task_id"] == "test_lunar_axis"
    assert "Derive orbital eccentricity" in get_res["prompt"]
    assert get_res["priority"] == 8


def test_execute_tool_delete_and_status(chat_session: BatonChatSession):
    """Verify delete_task and queue_status tool dispatches."""
    chat_session.execute_tool(
        "add_task",
        {
            "title": "Temporary Recon Task",
            "tier": "system_1",
            "prompt": "Quick fact extraction.",
            "task_id": "temp_recon_task",
        },
    )

    status_res = chat_session.execute_tool("queue_status", {})
    assert status_res["total_tasks"] == 1
    assert status_res["status_counts"].get("ready") == 1

    del_res = chat_session.execute_tool("delete_task", {"task_id": "temp_recon_task"})
    assert del_res["success"] is True

    status_res_after = chat_session.execute_tool("queue_status", {})
    assert status_res_after["total_tasks"] == 0


def test_execute_tool_newsboat_dispatch(chat_session: BatonChatSession):
    """Verify dispatching Newsboat tools through execute_tool."""
    mock_nb = MagicMock()
    mock_nb.list_feeds.return_value = [{"title": "Ars Technica", "total_articles": 5}]
    mock_nb.get_latest_articles.return_value = [{"id": 1, "title": "Test AI Article"}]
    mock_nb.search_articles.return_value = [{"id": 1, "title": "Test AI Article", "snippet": "AI news..."}]
    mock_nb.read_article.return_value = {"id": 1, "title": "Test AI Article", "content": "Full text body"}

    chat_session._newsboat_tool = mock_nb

    feeds_res = chat_session.execute_tool("rss_list_feeds", {})
    assert "feeds" in feeds_res
    assert feeds_res["feeds"][0]["title"] == "Ars Technica"

    latest_res = chat_session.execute_tool("rss_latest", {"limit": 5, "since": "24h"})
    assert "articles" in latest_res
    assert latest_res["articles"][0]["title"] == "Test AI Article"
    mock_nb.get_latest_articles.assert_called_with(feed_query=None, unread_only=False, limit=5, since="24h")

    search_res = chat_session.execute_tool("rss_search", {"query": "AI", "since": "48"})
    assert "results" in search_res
    assert search_res["results"][0]["snippet"] == "AI news..."
    mock_nb.search_articles.assert_called_with("AI", feed_query=None, limit=5, since="48")

    read_res = chat_session.execute_tool("rss_read_article", {"id": 1})
    assert read_res["content"] == "Full text body"

    # Verify actionable error message when newsboat_tool is not available
    with patch("baton.chat.NewsboatTool", side_effect=FileNotFoundError("cache.db not found")):
        chat_session._newsboat_tool = None
        err_res = chat_session.execute_tool("rss_search", {"query": "test"})
        assert "cache.db" in err_res["error"]
        assert "reload_newsboat.bat" in err_res["error"]


def test_execute_tool_sandbox_dispatch(chat_session: BatonChatSession, tmp_path: Path):
    """Verify tool execution dispatch for Sandbox tools."""
    mock_sandbox = MagicMock()
    mock_sandbox.syntax_check.return_value = {"valid": True, "message": "Syntax valid."}
    mock_sandbox.eval_code.return_value = {
        "success": True,
        "returncode": 0,
        "output": "Result: 42\n",
        "elapsed_seconds": 0.05,
    }
    mock_sandbox.run_pytest.return_value = {
        "success": True,
        "metrics": {"passed": 3, "failed": 0},
    }
    mock_sandbox.write_file.return_value = {"success": True, "size_bytes": 100}
    mock_sandbox.read_file.return_value = {"content": "code content"}
    mock_sandbox.list_files.return_value = [{"filename": "test.py", "size_bytes": 100}]

    chat_session._sandbox_tool = mock_sandbox

    # 1. Syntax check
    syn_res = chat_session.execute_tool("python_syntax_check", {"code": "x = 42"})
    assert syn_res["valid"] is True
    mock_sandbox.syntax_check.assert_called_once_with("x = 42")

    # 2. Eval code
    eval_res = chat_session.execute_tool("python_eval", {"code": "print(42)"})
    assert eval_res["success"] is True
    mock_sandbox.eval_code.assert_called_once_with(
        code="print(42)", filename="_snippet.py", timeout=None
    )

    # 3. Pytest run
    pytest_res = chat_session.execute_tool("pytest_run", {"target": "tests/"})
    assert pytest_res["success"] is True
    mock_sandbox.run_pytest.assert_called_once_with(target="tests/", timeout=None)

    # 4. Write file
    w_res = chat_session.execute_tool("sandbox_write_file", {"filename": "app.py", "content": "print('ok')"})
    assert w_res["success"] is True
    mock_sandbox.write_file.assert_called_once_with("app.py", "print('ok')")

    # 5. Read file
    r_res = chat_session.execute_tool("sandbox_read_file", {"filename": "app.py"})
    assert r_res["content"] == "code content"

    # 6. List files
    l_res = chat_session.execute_tool("sandbox_list_files", {})
    assert len(l_res["files"]) == 1


def test_execute_tool_sdr_dispatch(chat_session: BatonChatSession):
    """Verify tool execution dispatch for SDR tools."""
    mock_sdr = MagicMock()
    mock_sdr.get_device_info.return_value = {"status": "connected", "tuner": "R828D"}
    mock_sdr.spectrum_scan.return_value = {"status": "success", "band_mhz": "144-148", "peak_signals": []}
    mock_sdr.record_rf.return_value = {"status": "success", "file_path": "test.wav"}
    mock_sdr.decode_telemetry.return_value = {"status": "success", "packet_count": 2}
    mock_sdr.adsb_tracking.return_value = {"status": "success", "total_aircraft_detected": 1}
    mock_sdr.decode_digital_packet.return_value = {"status": "success", "message_count": 1}

    chat_session._sdr_tool = mock_sdr

    info_res = chat_session.execute_tool("sdr_device_info", {})
    assert info_res["status"] == "connected"

    scan_res = chat_session.execute_tool("sdr_spectrum_scan", {"start_freq_mhz": 144.0, "stop_freq_mhz": 148.0})
    assert scan_res["status"] == "success"

    rec_res = chat_session.execute_tool("sdr_record_rf", {"frequency_mhz": 144.39})
    assert rec_res["status"] == "success"

    telem_res = chat_session.execute_tool("sdr_decode_telemetry", {"frequency_mhz": 433.92})
    assert telem_res["packet_count"] == 2

    adsb_res = chat_session.execute_tool("sdr_adsb_tracking", {"duration_seconds": 5})
    assert adsb_res["total_aircraft_detected"] == 1

    pkt_res = chat_session.execute_tool("sdr_decode_packet", {"frequency_mhz": 144.390, "protocol": "aprs"})
    assert pkt_res["message_count"] == 1


def test_slash_command_handlers(chat_session: BatonChatSession):
    """Verify built-in direct slash commands without LLM overhead."""
    mock_sdr = MagicMock()
    mock_sdr.get_device_info.return_value = {"status": "connected", "tuner": "R828D"}
    mock_sdr.execute.return_value = "### SDR Test Result"
    chat_session._sdr_tool = mock_sdr

    assert chat_session.handle_slash_command("/help") is True
    assert chat_session.handle_slash_command("/tasks") is True
    assert chat_session.handle_slash_command("/status") is True
    assert chat_session.handle_slash_command("/artifacts") is True
    assert chat_session.handle_slash_command("/clear") is True
    assert chat_session.handle_slash_command("/shelf") is True
    assert chat_session.handle_slash_command("/sdr") is True
    assert chat_session.handle_slash_command("/adsb") is True
    assert chat_session.handle_slash_command("/telemetry") is True

    # Test bundle toggle (newsboat resolves to rss)
    assert chat_session.handle_slash_command("/bundle newsboat") is True
    assert "rss" in chat_session.active_bundles

    assert chat_session.handle_slash_command("/bundle newsboat") is True
    assert "rss" not in chat_session.active_bundles

    assert chat_session.handle_slash_command("/bundle all") is True
    assert chat_session.active_bundles == set(TOOL_BUNDLES.keys())

    assert chat_session.handle_slash_command("/bundle core") is True
    assert chat_session.active_bundles == {"core"}

    assert chat_session.model_name == "qwen3:14b"
    assert chat_session.handle_slash_command("/model qwen2.5-coder:14b") is True
    assert chat_session.model_name == "qwen2.5-coder:14b"


def test_send_message_with_mocked_ollama_tool_call(chat_session: BatonChatSession):
    """Verify multi-turn tool calling loop using native Ollama tool call schema."""
    # First response: Ollama returns a tool call to add_task
    tool_call_payload = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {
                        "name": "add_task",
                        "arguments": {
                            "title": "Calculate Solar Declination",
                            "tier": "system_1",
                            "prompt": "Extract solar declination formula from Spencer (1971).",
                            "priority": 7,
                        },
                    },
                }
            ],
        }
    }

    # Second response: Ollama sees the tool result and provides final confirmation
    final_reply_payload = {
        "message": {
            "role": "assistant",
            "content": "I have created the task 'Calculate Solar Declination' and queued it as System 1.",
            "tool_calls": [],
        }
    }

    mock_res_1 = MagicMock()
    mock_res_1.json.return_value = tool_call_payload
    mock_res_1.raise_for_status.return_value = None

    mock_res_2 = MagicMock()
    mock_res_2.json.return_value = final_reply_payload
    mock_res_2.raise_for_status.return_value = None

    recorded_calls = []

    def mock_on_tool_call(name: str, args: dict):
        recorded_calls.append((name, args))

    with patch("requests.post", side_effect=[mock_res_1, mock_res_2]) as mock_post:
        response = chat_session.send_message(
            "Please create a task to extract solar declination formula.",
            on_tool_call=mock_on_tool_call,
        )

        assert mock_post.call_count == 2
        assert len(recorded_calls) == 1
        assert recorded_calls[0][0] == "add_task"
        assert recorded_calls[0][1]["title"] == "Calculate Solar Declination"

        # Check final reply
        assert "I have created the task" in response

        # Verify task actually exists in SQLite database
        tasks = chat_session.db.get_all_tasks()
        assert len(tasks) == 1
        assert tasks[0].title == "Calculate Solar Declination"
        assert tasks[0].tier == Tier.SYSTEM_1
        assert tasks[0].priority == 7


def test_chat_session_logging(temp_db: Database, tmp_path: Path):
    """Verify that chat logs record user input, slash commands, tool calls, and responses."""
    logs_dir = tmp_path / "logs"
    session = BatonChatSession(
        db=temp_db,
        model_name="qwen3:14b",
        logs_dir=logs_dir,
    )

    assert session.session_log_path.exists()
    assert session.latest_log_path.exists()

    # 1. Test slash command logging
    session.handle_slash_command("/bundle sdr")
    session_text = session.session_log_path.read_text(encoding="utf-8")
    assert "[SLASH_COMMAND]" in session_text
    assert "/bundle sdr" in session_text

    # 2. Test message and mock response logging
    mock_reply = {
        "message": {
            "role": "assistant",
            "content": "Hello! I am ready to assist with task planning.",
            "tool_calls": [],
        },
        "total_duration": 500000000,
        "eval_count": 12,
        "prompt_eval_count": 45,
    }
    mock_res = MagicMock()
    mock_res.json.return_value = mock_reply
    mock_res.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_res):
        session.send_message("What can you do?")

    session_text_after = session.session_log_path.read_text(encoding="utf-8")
    latest_text = session.latest_log_path.read_text(encoding="utf-8")

    for text in (session_text_after, latest_text):
        assert "[USER]" in text
        assert "What can you do?" in text
        assert "[AGENT_RESPONSE]" in text
        assert "Hello! I am ready to assist" in text
        assert "[TELEMETRY]" in text
        assert "eval_tokens" in text or "eval_count" in text


def test_library_bundle_activation_and_dynamic_mounted_corpora(chat_session: BatonChatSession):
    """Verify activating 'library' (or legacy 'wikipedia') returns mounted corpora and dynamic metadata."""
    # 1. Activate via legacy name 'wikipedia'
    res = chat_session.execute_tool("activate_tool_bundle", {"bundle_name": "wikipedia"})
    assert res["success"] is True
    assert res["activated_bundle"] == "library"
    assert "library" in chat_session.active_bundles
    assert "mounted_corpora" in res
    assert isinstance(res["mounted_corpora"], list)
    assert "default_corpus" in res

    # 2. Deactivate via 'library'
    deact = chat_session.execute_tool("deactivate_tool_bundle", {"bundle_name": "library"})
    assert deact["success"] is True
    assert "library" not in chat_session.active_bundles

    # 3. Activate via 'library'
    res2 = chat_session.execute_tool("activate_tool_bundle", {"bundle_name": "library"})
    assert res2["success"] is True
    assert res2["activated_bundle"] == "library"
    assert "library" in chat_session.active_bundles
    assert "mounted_corpora" in res2


def test_tool_help_curated_and_fallback(chat_session: BatonChatSession):
    """Verify tool_help returns curated cheatsheets when available, and falls back to schema/docstring."""
    # 1. Curated cheatsheet for add_task
    help_add = chat_session.execute_tool("tool_help", {"tool_name": "add_task"})
    assert help_add["source"] == "curated_cheatsheet"
    assert "canonical_dag_example" in help_add
    assert "parameters" in help_add
    assert "tier" in help_add["parameters"]

    # 2. Curated cheatsheet for start_orchestration
    help_orch = chat_session.execute_tool("tool_help", {"tool_name": "start_orchestration"})
    assert help_orch["source"] == "curated_cheatsheet"
    assert "max_steps" in help_orch["parameters"]

    # 3. Curated cheatsheet for rss_search and rss_read_article
    help_rss = chat_session.execute_tool("tool_help", {"tool_name": "rss_search"})
    assert help_rss["source"] == "curated_cheatsheet"
    assert "since" in help_rss["parameters"]
    assert "query" in help_rss["parameters"]

    help_read = chat_session.execute_tool("tool_help", {"tool_name": "rss_read_article"})
    assert help_read["source"] == "curated_cheatsheet"
    assert "id" in help_read["parameters"]

    # 4. Schema fallback for python_eval (which has no curated entry)
    help_eval = chat_session.execute_tool("tool_help", {"tool_name": "python_eval"})
    assert help_eval["source"] == "schema_fallback"
    assert "parameters" in help_eval
    assert "code" in help_eval["parameters"]

    # 5. Unknown tool error
    help_err = chat_session.execute_tool("tool_help", {"tool_name": "completely_fictional_tool"})
    assert "error" in help_err
    assert "active_tools" in help_err


def test_add_task_tier_normalization_and_parameter_aliasing(chat_session: BatonChatSession):
    """Verify add_task normalizes tier variants and aliases description/dependencies/tool."""
    # 1. System1 variant with 'description' alias instead of prompt
    t1_res = chat_session.execute_tool(
        "add_task",
        {
            "title": "Orbital Recon Alpha",
            "tier": "system1",  # no underscore
            "description": "Gather semi-major axis formulas.",
            "dependencies": [],  # dependencies alias
        },
    )
    assert t1_res["success"] is True
    assert t1_res["tier"] == "system_1"
    assert t1_res["status"] == "ready"

    # Verify prompt in database was populated from 'description'
    t1_db = chat_session.db.get_task(t1_res["task_id"])
    assert t1_db is not None
    assert t1_db.prompt_template == "Gather semi-major axis formulas."

    # 2. System2 variant with 'System2' capitalization and 'instructions'
    t2_res = chat_session.execute_tool(
        "add_task",
        {
            "title": "Orbital Synthesis Beta",
            "tier": "System2",  # capitalized
            "instructions": "Synthesize Python Kepler propagator.",
            "depends_on": [t1_res["task_id"]],
        },
    )
    assert t2_res["success"] is True
    assert t2_res["tier"] == "system_2"
    assert t2_res["status"] == "blocked"

    t2_db = chat_session.db.get_task(t2_res["task_id"])
    assert t2_db is not None
    assert t2_db.prompt_template == "Synthesize Python Kepler propagator."


def test_fuzzy_dependency_resolution_by_title(chat_session: BatonChatSession):
    """Verify that specifying task titles in dependencies automatically resolves to upstream task_ids."""
    # Step 1: Create upstream task with title
    recon_res = chat_session.execute_tool(
        "add_task",
        {
            "title": "Boids Simulation Research",
            "tier": "system_1",
            "prompt": "Extract boid flocking rules.",
        },
    )
    assert recon_res["success"] is True
    recon_id = recon_res["task_id"]

    # Step 2: Create downstream task passing title string in dependencies
    synth_res = chat_session.execute_tool(
        "add_task",
        {
            "title": "Boids Simulation Script Development",
            "tier": "system_2",
            "prompt": "Write simulation code.",
            "dependencies": ["Boids Simulation Research"],  # title instead of task_id
        },
    )
    assert synth_res["success"] is True
    assert synth_res["depends_on"] == [recon_id]
    assert "resolved_dependencies" in synth_res
    assert synth_res["resolved_dependencies"][0]["from_title"] == "Boids Simulation Research"
    assert synth_res["resolved_dependencies"][0]["to_id"] == recon_id

    # Downstream task should be blocked because upstream recon is ready, not completed
    assert synth_res["status"] == "blocked"


def test_activate_and_deactivate_bundle_parameter_aliasing(chat_session: BatonChatSession):
    """Verify activate/deactivate accept 'tool_bundle' and 'bundle' aliases."""
    # Activate with tool_bundle
    act_res = chat_session.execute_tool("activate_tool_bundle", {"tool_bundle": "sandbox"})
    assert act_res["success"] is True
    assert "sandbox" in chat_session.active_bundles

    # Deactivate with bundle
    deact_res = chat_session.execute_tool("deactivate_tool_bundle", {"bundle": "sandbox"})
    assert deact_res["success"] is True
    assert "sandbox" not in chat_session.active_bundles


def test_start_orchestration_tool_and_slash_commands(chat_session: BatonChatSession):
    """Verify start_orchestration tool and slash command runner."""
    # 1. Empty queue returns cleanly
    empty_res = chat_session.execute_tool("start_orchestration", {"max_steps": 2})
    assert empty_res["success"] is True
    assert empty_res["steps_executed"] == 0

    # 2. Add a ready task and execute via start_orchestration
    t_res = chat_session.execute_tool(
        "add_task",
        {
            "title": "Auto Orchestrate Target",
            "tier": "system_1",
            "prompt": "Simple quick task.",
        },
    )
    assert t_res["status"] == "ready"

    # Mock system1 execution so it doesn't need live Ollama
    chat_session.orchestrator.system1.execute_task = MagicMock(
        return_value=("Dossier summary text", "dossier.md", 50, 0.5)
    )

    orch_res = chat_session.execute_tool("start_orchestration", {"max_steps": 3})
    assert orch_res["success"] is True
    assert orch_res["steps_executed"] == 1
    assert orch_res["executed_tasks"][0]["task_id"] == t_res["task_id"]
    assert orch_res["executed_tasks"][0]["status"] == "completed"

    # 3. Test slash commands /start and /help <tool>
    assert chat_session.handle_slash_command("/start 1") is True
    assert chat_session.handle_slash_command("/help add_task") is True
    assert chat_session.handle_slash_command("/toolhelp start_orchestration") is True


def test_start_orchestration_and_step_queue_concurrency_lock(chat_session: BatonChatSession):
    """Verify that start_orchestration and step_queue yield cleanly when another runner is in progress."""
    t1 = chat_session.db.create_task(title="Running Elsewhere", tier=Tier.SYSTEM_1, task_id="in_prog_t1")
    t2 = chat_session.db.create_task(title="Queued Task", tier=Tier.SYSTEM_1, task_id="queued_t2")

    # Claim task 1 to simulate TUI / another process running it
    claimed = chat_session.db.claim_specific_task(t1.task_id)
    assert claimed is not None
    assert claimed.status == TaskStatus.IN_PROGRESS

    # step_queue should report busy
    sq_res = chat_session.execute_tool("step_queue", {})
    assert sq_res["success"] is False
    assert "actively running in another process" in sq_res["message"]

    # start_orchestration should pause cleanly
    so_res = chat_session.execute_tool("start_orchestration", {"max_steps": 3})
    assert so_res["success"] is False
    assert "actively running in another process" in so_res["message"]

    # Clear in_progress task
    chat_session.db.mark_task_completed(t1.task_id, result_payload="Done")

    # Now step_queue succeeds
    chat_session.orchestrator.system1.execute_task = MagicMock(
        return_value=("Dossier payload", "artifact.md", 100, 1.0)
    )
    sq_res2 = chat_session.execute_tool("step_queue", {})
    assert sq_res2["success"] is True
    assert sq_res2["task_id"] == t2.task_id




