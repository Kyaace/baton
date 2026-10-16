"""Unit tests for System 2 Colibrì 429 exponential backoff and retry logic."""

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

