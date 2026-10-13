"""Unit tests for Baton streaming TaskProgress and live metrics."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from baton.models import TaskProgress, Tier, format_duration


def test_format_duration():
    assert format_duration(12.5) == "12.5s"
    assert format_duration(95.0) == "1m 35s"
    assert format_duration(3665.0) == "1h 01m 05s"


def test_task_progress_persistence(tmp_path: Path):
    prog_file = tmp_path / "test_task.json"
    p = TaskProgress(
        task_id="task_123",
        tier=Tier.SYSTEM_2,
        started_at=time.time() - 30.0,
        tokens_generated=25,
        tokens_per_second=0.83,
        reasoning_text="Evaluating ephemeris parameters...",
        content_text="def solve_kepler():",
    )
    p.save_to_file(prog_file)
    assert prog_file.exists()

    loaded = TaskProgress.load_from_file(prog_file)
    assert loaded is not None
    assert loaded.task_id == "task_123"
    assert loaded.tier == Tier.SYSTEM_2
    assert loaded.tokens_generated == 25
    assert "Evaluating" in loaded.reasoning_text
    assert "solve_kepler" in loaded.content_text
    assert loaded.elapsed_seconds >= 30.0


def test_system2_streaming_sse(tmp_path: Path):
    from baton.system2 import System2Runner

    runner = System2Runner(artifacts_dir=tmp_path)
    runner._discovered_model = "deepseek-v4-colibri"

    sse_lines = [
        b'data: {"choices":[{"delta":{"reasoning_content":"Thinking about "}}]}\n\n',
        b'data: {"choices":[{"delta":{"reasoning_content":"orbits..."}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"print(\\"hello\\")"}}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    mock_resp = MagicMock()
    mock_resp.iter_lines.return_value = sse_lines
    mock_resp.raise_for_status.return_value = None

    updates = []
    progress = TaskProgress(task_id="test_stream", tier=Tier.SYSTEM_2, started_at=time.time())

    with patch("requests.post", return_value=mock_resp):
        res = runner.chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            progress=progress,
            progress_callback=lambda p: updates.append((p.reasoning_text, p.content_text)),
        )

    assert res["content"] == 'print("hello")'
    assert res["reasoning_content"] == "Thinking about orbits..."
    assert len(updates) >= 2
    assert progress.is_completed is True


def test_orchestrator_progress_tracking(tmp_path: Path):
    from baton.db import Database
    from baton.models import JobStatus, TaskType
    from baton.orchestrator import Orchestrator

    class MockSys1:
        def execute_task(self, task, upstream, progress_callback=None):
            if progress_callback:
                p = TaskProgress(task_id=task.task_id, tier=Tier.SYSTEM_1, started_at=time.time(), content_text="Drafting...")
                progress_callback(p)
            return "Mock summary", str(tmp_path / "mock.md"), 100, 1.0

    class MockSys2:
        pass

    class MockEngMgr:
        active_tier = Tier.SYSTEM_1
        def switch_to_tier(self, t):
            self.active_tier = t
            return True

    db = Database(tmp_path / "test.db")
    orch = Orchestrator(db=db, engine_manager=MockEngMgr(), system1=MockSys1(), system2=MockSys2(), progress_dir=tmp_path / "prog")
    job = db.create_job("Test", "Goal")
    t = orch.add_manual_task(job.job_id, "Test S1", Tier.SYSTEM_1)

    t_res = orch.step()
    assert t_res is not None

    prog = orch.get_active_progress(t.task_id)
    assert prog is not None
    assert prog.is_completed is True
    assert prog.tokens_generated == 100


def test_check_degenerate_repetition():
    from baton.system2 import check_degenerate_repetition

    # Valid normal text should not trigger
    assert check_degenerate_repetition("import numpy as np\ndef foo(x):\n    return x * 2") is False
    assert check_degenerate_repetition("0.0") is False

    # 60 repeated zeros (the exact failure mode observed)
    infinite_zeros = "MEAN_LONGITUDE_RATE = 0." + ("0" * 65)
    assert check_degenerate_repetition(infinite_zeros) is True

    # Repeating period-2 pattern (e.g. 0.0.0.0...)
    repeating_period2 = "RATE = " + ("0." * 35)
    assert check_degenerate_repetition(repeating_period2) is True


def test_system2_repetition_breaks_stream(tmp_path: Path):
    from baton.system2 import System2Runner

    runner = System2Runner(artifacts_dir=tmp_path)
    runner._discovered_model = "deepseek-v4-colibri"

    # Stream valid text followed by degenerate zeros
    sse_lines = [
        b'data: {"choices":[{"delta":{"content":"import numpy as np\\n"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"MEAN_LONGITUDE = 0."}}]}\n\n',
    ]
    # Add 70 individual '0' chunks
    for _ in range(70):
        sse_lines.append(b'data: {"choices":[{"delta":{"content":"0"}}]}\n\n')
    sse_lines.append(b'data: [DONE]\n\n')

    mock_resp = MagicMock()
    mock_resp.iter_lines.return_value = sse_lines
    mock_resp.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_resp):
        res = runner.chat_completion(
            messages=[{"role": "user", "content": "test"}],
            stream=True,
        )

    # Should break cleanly before finishing all chunks and return what was generated
    assert "import numpy as np" in res["content"]
    assert "MEAN_LONGITUDE" in res["content"]


def test_system2_cancellation_active(tmp_path: Path):
    import pytest
    from baton.models import TaskCancelledError
    from baton.system2 import System2Runner

    runner = System2Runner(artifacts_dir=tmp_path)
    runner._discovered_model = "deepseek-v4-colibri"

    mock_resp = MagicMock()
    # Emulate cancel_active called mid-stream
    def mock_iter_lines(decode_unicode=True):
        yield b'data: {"choices":[{"delta":{"content":"starting code..."}}]}\n\n'
        runner.cancel_active()
        yield b'data: {"choices":[{"delta":{"content":"more code"}}]}\n\n'

    mock_resp.iter_lines = mock_iter_lines
    mock_resp.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(TaskCancelledError):
            runner.chat_completion(
                messages=[{"role": "user", "content": "test"}],
                stream=True,
            )


def test_orchestrator_stop_current_task(tmp_path: Path):
    from baton.db import Database
    from baton.models import TaskStatus
    from baton.orchestrator import Orchestrator

    db = Database(tmp_path / "test_stop.db")
    job = db.create_job("Stop Job", "Test stopping")
    orch = Orchestrator(db=db, progress_dir=tmp_path / "prog")
    t = orch.add_manual_task(job.job_id, "Active Task", Tier.SYSTEM_2)
    db.claim_specific_task(t.task_id)

    # Task is IN_PROGRESS
    assert db.get_task(t.task_id).status == TaskStatus.IN_PROGRESS

    stopped_id = orch.stop_current_task()
    assert stopped_id == t.task_id

    # Task should now be FAILED with "Stopped by user"
    updated_t = db.get_task(t.task_id)
    assert updated_t.status == TaskStatus.FAILED
    assert "Stopped by user" in updated_t.error_message


def test_system2_finish_reason_and_truncation(tmp_path: Path):
    from baton.system2 import System2Runner

    runner = System2Runner(artifacts_dir=tmp_path)
    runner._discovered_model = "deepseek-v4-colibri"

    # Case 1: Stream ends on length limit
    sse_lines_length = [
        b'data: {"choices":[{"delta":{"content":"partially generated code..."},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n',
        b'data: [DONE]\n\n',
    ]
    mock_resp_length = MagicMock()
    mock_resp_length.iter_lines.return_value = sse_lines_length
    mock_resp_length.raise_for_status.return_value = None

    prog = TaskProgress(task_id="t_len", tier=Tier.SYSTEM_2, started_at=time.time())
    with patch("requests.post", return_value=mock_resp_length):
        res = runner.chat_completion([{"role": "user", "content": "gen"}], progress=prog)

    assert res["finish_reason"] == "length"
    assert res["is_truncated"] is True
    assert prog.finish_reason == "length"
    assert prog.is_truncated is True

    # Case 2: Stream ends cleanly on stop
    sse_lines_stop = [
        b'data: {"choices":[{"delta":{"content":"return True\\n"},"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    ]
    mock_resp_stop = MagicMock()
    mock_resp_stop.iter_lines.return_value = sse_lines_stop
    mock_resp_stop.raise_for_status.return_value = None

    prog_stop = TaskProgress(task_id="t_stop", tier=Tier.SYSTEM_2, started_at=time.time())
    with patch("requests.post", return_value=mock_resp_stop):
        res_stop = runner.chat_completion([{"role": "user", "content": "gen"}], progress=prog_stop)

    assert res_stop["finish_reason"] == "stop"
    assert res_stop["is_truncated"] is False
    assert prog_stop.is_truncated is False


def test_system2_chat_completion_max_tokens_payload(tmp_path: Path):
    from baton.system2 import System2Runner

    runner = System2Runner(artifacts_dir=tmp_path)
    runner._discovered_model = "deepseek-v4-colibri"

    mock_resp = MagicMock()
    mock_resp.iter_lines.return_value = [
        b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    ]
    mock_resp.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_resp) as mock_post:
        runner.chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8192,
            stream=True,
        )
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["max_tokens"] == 8192


def test_system2_synthesize_seamless_continuation(tmp_path: Path):
    from baton.models import Task, TaskType
    from baton.system2 import System2Runner

    runner = System2Runner(artifacts_dir=tmp_path)
    runner._discovered_model = "deepseek-v4-colibri"

    task = Task(
        task_id="test_continuation",
        job_id="job_1",
        title="Continuation Test",
        tier=Tier.SYSTEM_2,
        task_type=TaskType.DEEP_SYNTHESIS,
        prompt_template="Write full script",
    )

    # First turn gets cut off mid-line with finish_reason: "length"
    sse_turn1 = [
        b'data: {"choices":[{"delta":{"content":"def solve():\\n    x = 100 + "},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n',
        b'data: [DONE]\n\n',
    ]
    mock_resp1 = MagicMock()
    mock_resp1.iter_lines.return_value = sse_turn1
    mock_resp1.raise_for_status.return_value = None

    # Second turn continues from where it left off and finishes cleanly with "stop"
    sse_turn2 = [
        b'data: {"choices":[{"delta":{"content":"200\\n    return x\\n"},"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    ]
    mock_resp2 = MagicMock()
    mock_resp2.iter_lines.return_value = sse_turn2
    mock_resp2.raise_for_status.return_value = None

    posted_messages = []

    def mock_post(*args, **kwargs):
        posted_messages.append(kwargs["json"]["messages"])
        if len(posted_messages) == 1:
            return mock_resp1
        return mock_resp2

    with patch("requests.post", side_effect=mock_post):
        summary, art_path, tokens, elapsed = runner.synthesize(task, auto_continue=True, max_continuations=2)

    # Verify both turns occurred
    assert len(posted_messages) == 2
    # Verify second turn passed trailing assistant message with first turn output
    second_msgs = posted_messages[1]
    assert second_msgs[-1]["role"] == "assistant"
    assert second_msgs[-1]["content"] == "def solve():\n    x = 100 + "

    # Verify output stitched together seamlessly
    content = Path(art_path).read_text(encoding="utf-8")
    assert content == "def solve():\n    x = 100 + 200\n    return x"
    assert "across 2 turns" in summary


def test_system2_synthesize_max_continuations_budget_clamp(tmp_path: Path):
    from baton.models import Task, TaskType
    from baton.system2 import System2Runner

    runner = System2Runner(artifacts_dir=tmp_path)
    runner._discovered_model = "deepseek-v4-colibri"

    task = Task(
        task_id="test_clamp",
        job_id="job_1",
        title="Clamp Test",
        tier=Tier.SYSTEM_2,
        task_type=TaskType.DEEP_SYNTHESIS,
        prompt_template="Generate infinite terms",
    )

    # Stream always returns length limited
    def make_length_resp():
        r = MagicMock()
        r.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"chunk "},"finish_reason":"length"}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        r.raise_for_status.return_value = None
        return r

    with patch("requests.post", side_effect=lambda *args, **kwargs: make_length_resp()):
        summary, art_path, tokens, elapsed = runner.synthesize(task, auto_continue=True, max_continuations=2)

    assert "[TRUNCATED: hit context budget ceiling after 2 continuation(s)]" in summary
    assert Path(art_path).exists()



