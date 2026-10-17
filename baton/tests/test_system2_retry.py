"""Unit tests for System 2 Colibrì 429 exponential backoff and retry logic."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import requests

from baton.models import TaskCancelledError
from baton.system2 import System2Runner


def test_post_with_retry_succeeds_first_attempt():
    runner = System2Runner()
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = runner._post_with_retry(
            endpoint="http://127.0.0.1:8000/v1/chat/completions",
            payload={"messages": []},
            headers={},
            stream=False,
            timeout=10.0,
        )
        assert res == mock_resp
        assert mock_post.call_count == 1


def test_post_with_retry_backs_off_on_429_then_succeeds():
    runner = System2Runner()

    mock_resp_429 = MagicMock(spec=requests.Response)
    mock_resp_429.status_code = 429

    mock_resp_200 = MagicMock(spec=requests.Response)
    mock_resp_200.status_code = 200

    with patch("requests.post", side_effect=[mock_resp_429, mock_resp_200]) as mock_post:
        res = runner._post_with_retry(
            endpoint="http://127.0.0.1:8000/v1/chat/completions",
            payload={"messages": []},
            headers={},
            stream=False,
            timeout=10.0,
            max_retries=2,
            backoff_delays=(0.01, 0.02),
        )
        assert res == mock_resp_200
        assert mock_post.call_count == 2


def test_post_with_retry_exhausts_retries_and_raises():
    runner = System2Runner()

    mock_resp_429 = MagicMock(spec=requests.Response)
    mock_resp_429.status_code = 429
    mock_resp_429.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "429 Client Error: Too Many Requests"
    )

    with patch("requests.post", return_value=mock_resp_429) as mock_post:
        with pytest.raises(requests.exceptions.HTTPError) as exc_info:
            runner._post_with_retry(
                endpoint="http://127.0.0.1:8000/v1/chat/completions",
                payload={"messages": []},
                headers={},
                stream=False,
                timeout=10.0,
                max_retries=2,
                backoff_delays=(0.01, 0.02),
            )
        assert "429 Client Error" in str(exc_info.value)
        # Initial attempt + 2 retries = 3 total attempts
        assert mock_post.call_count == 3


def test_post_with_retry_cancelled_raises_task_cancelled():
    runner = System2Runner()
    runner._cancel_requested = True

    with pytest.raises(TaskCancelledError):
        runner._post_with_retry(
            endpoint="http://127.0.0.1:8000/v1/chat/completions",
            payload={"messages": []},
            headers={},
            stream=False,
            timeout=10.0,
        )


def test_system2_salvages_code_from_reasoning_monologue(tmp_path):
    from pathlib import Path
    from baton.models import Task, TaskType, Tier

    runner = System2Runner(artifacts_dir=tmp_path)
    task = Task(
        task_id="synth_salvage",
        job_id="job_salvage",
        title="Synthesize Salvage Function",
        tier=Tier.SYSTEM_2,
        task_type=TaskType.DEEP_SYNTHESIS,
        prompt_template="Create a Python function",
    )

    # Mock chat_completion returning empty content, but code inside reasoning
    reasoning_code = (
        "Let me think about this problem.\n"
        "Here is the function:\n"
        "```python\n"
        "def test_salvaged():\n"
        "    return 42\n"
        "```\n"
    )
    mock_res = {
        "content": "",
        "reasoning_content": reasoning_code,
        "tokens_generated": 500,
        "elapsed_seconds": 1.0,
        "finish_reason": "stop",
    }
    with patch.object(runner, "chat_completion", return_value=mock_res):
        summary, art_path, tokens, elapsed = runner.synthesize(task)
        assert art_path is not None
        assert art_path.endswith(".py")
        content = Path(art_path).read_text(encoding="utf-8")
        assert "def test_salvaged():" in content
        assert "return 42" in content


def test_system2_repair_code_replaces_lines_and_succeeds(tmp_path: Path):
    """Verify that System2Runner.repair_code reads code, executes replace_lines, and exits on Code 0."""
    from baton.models import Task, TaskType, Tier
    from baton.tools.sandbox import SandboxTool

    sandbox = SandboxTool(sandbox_dir=tmp_path / "sandbox")
    runner = System2Runner(artifacts_dir=tmp_path / "artifacts")

    # Create broken file in sandbox
    broken_code = "def compute():\n    return 1 / 0\n\nif __name__ == '__main__':\n    print(compute())\n"
    sandbox.write_file("calc.py", broken_code)

    task = Task(
        task_id="repair_verify_calc",
        job_id="job_1",
        title="Repair: Compute Calc",
        tier=Tier.SYSTEM_2,
        task_type=TaskType.DEEP_SYNTHESIS,
        prompt_template="ZeroDivisionError on line 2",
        tool_params={"target_artifact": "calc.py", "is_repair": True},
    )

    # Turn 1: model calls sandbox_read_file
    turn1_res = {
        "content": "Let me inspect the file around line 2.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "sandbox_read_file",
                    "arguments": {"filename": "calc.py", "start_line": 1, "end_line": 5},
                },
            }
        ],
        "tokens_generated": 50,
        "elapsed_seconds": 0.2,
    }

    # Turn 2: model calls sandbox_replace_lines with fix
    turn2_res = {
        "content": "Replacing line 2 with working logic.",
        "tool_calls": [
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "sandbox_replace_lines",
                    "arguments": {
                        "filename": "calc.py",
                        "start_line": 2,
                        "end_line": 2,
                        "replacement_content": "    return 100",
                    },
                },
            }
        ],
        "tokens_generated": 60,
        "elapsed_seconds": 0.2,
    }

    with patch.object(runner, "chat_completion", side_effect=[turn1_res, turn2_res]) as mock_chat:
        summary, art_path, tokens, elapsed, success = runner.repair_code(task, sandbox_tool=sandbox)

        assert success is True
        assert mock_chat.call_count == 2
        assert "PASSED (Code 0)" in summary

        # Verify code on disk was patched and runs
        read_res = sandbox.read_file("calc.py", max_chars=None)
        assert "return 100" in read_res["content"]
        assert "return 1 / 0" not in read_res["content"]


def test_system2_repair_code_exhausts_edit_budget(tmp_path: Path):
    """Verify that System2Runner.repair_code respects the 3-edit budget and exits on failure."""
    from baton.models import Task, TaskType, Tier
    from baton.tools.sandbox import SandboxTool

    sandbox = SandboxTool(sandbox_dir=tmp_path / "sandbox")
    runner = System2Runner(artifacts_dir=tmp_path / "artifacts")

    broken_code = "def compute():\n    return 1 / 0\n\nif __name__ == '__main__':\n    print(compute())\n"
    sandbox.write_file("calc_fail.py", broken_code)

    task = Task(
        task_id="repair_verify_calc_fail",
        job_id="job_1",
        title="Repair: Compute Calc Fail",
        tier=Tier.SYSTEM_2,
        task_type=TaskType.DEEP_SYNTHESIS,
        prompt_template="ZeroDivisionError",
        tool_params={"target_artifact": "calc_fail.py", "is_repair": True},
    )

    # 3 broken edits that still crash
    broken_turn = {
        "content": "Trying another broken edit",
        "tool_calls": [
            {
                "id": "call_fail",
                "type": "function",
                "function": {
                    "name": "sandbox_replace_lines",
                    "arguments": {
                        "filename": "calc_fail.py",
                        "start_line": 2,
                        "end_line": 2,
                        "replacement_content": "    raise ValueError('still broken')",
                    },
                },
            }
        ],
        "tokens_generated": 30,
        "elapsed_seconds": 0.1,
    }

    with patch.object(runner, "chat_completion", return_value=broken_turn) as mock_chat:
        summary, art_path, tokens, elapsed, success = runner.repair_code(
            task, sandbox_tool=sandbox, max_edits=3
        )
        assert success is False
        assert mock_chat.call_count == 3
        assert "exhausted 3 edit attempts" in summary

