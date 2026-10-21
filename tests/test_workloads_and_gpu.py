import subprocess
from unittest.mock import patch
from benchmark.workloads import (
    SINGLE_REQUEST_PROMPT,
    AGENT_SYSTEM_PROMPT,
    get_concurrent_prompts,
    get_agent_turn_script,
)
from benchmark.gpu_monitor import query_gpu_memory_mb


def test_single_and_concurrent_prompts():
    assert SINGLE_REQUEST_PROMPT == "Explain concurrent users in two sentences."
    prompts_32 = get_concurrent_prompts(32)
    assert len(prompts_32) == 32
    assert len(set(prompts_32)) == 32  # All 32 prompts must be distinct


def test_agent_turn_script_has_five_turns_and_heavy_prefix():
    # System prompt + tool definitions should be substantial (~600 tokens / >1800 chars)
    assert len(AGENT_SYSTEM_PROMPT) >= 1800
    turns = get_agent_turn_script(session_id=1)
    assert len(turns) == 5
    for idx, turn in enumerate(turns):
        assert "user_message" in turn
        assert isinstance(turn["user_message"], str)
        if idx < 4:
            assert "tool_result" in turn


def test_gpu_monitor_parses_nvidia_smi_and_handles_missing_binary():
    with patch("subprocess.check_output", return_value=b"3456\n"):
        assert query_gpu_memory_mb(0) == 3456.0

    with patch("subprocess.check_output", side_effect=FileNotFoundError("no nvidia-smi")):
        assert query_gpu_memory_mb(0) == 0.0

    with patch(
        "subprocess.check_output",
        side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
    ):
        assert query_gpu_memory_mb(0) == 0.0

    with patch("subprocess.check_output", return_value=b"N/A\n"):
        assert query_gpu_memory_mb(0) == 0.0
