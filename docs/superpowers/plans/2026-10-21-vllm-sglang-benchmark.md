# vLLM vs. SGLang Self-Hosting & Benchmark Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, reproducible vLLM vs. SGLang self-hosting experiment suite targeting Kaggle NVIDIA T4 GPUs (`Qwen/Qwen2.5-1.5B-Instruct`), complete with self-contained Kaggle notebooks, an async OpenAI SSE benchmark engine, local unit tests, a results comparator, and an English mentor report.

**Architecture:** A modular Python benchmark library (`benchmark/`) measures Time to First Token (TTFT), end-to-end latency, output token throughput, and GPU VRAM across three workloads: Single Request, Concurrent Users (`C = 1, 4, 8, 16`), and a 5-Turn Calculator Tool-Calling Agent (`1` and `4` parallel sessions). The exact benchmark code is also embedded inside two self-contained Kaggle notebooks (`notebooks/01_vllm_kaggle.ipynb` and `notebooks/02_sglang_kaggle.ipynb`) configured specifically for Kaggle's Turing T4 GPU (`CUDA_VISIBLE_DEVICES=0`, `float16`/`half`, Triton attention backend for SGLang).

**Tech Stack:** Python 3.10+, `httpx` (or stdlib `urllib`/`http.client` + `asyncio` fallback so zero external packages are required if `httpx` is absent), `pytest`, `nbformat`-compatible JSON notebooks, Markdown.

**Spec:** `docs/superpowers/specs/2026-10-21-vllm-sglang-benchmark-design.md`

## Global Constraints

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Pinned GPU:** `CUDA_VISIBLE_DEVICES=0` (single 16 GB NVIDIA T4 GPU on Kaggle `GPU T4 x2`)
- **vLLM Launch Flags:** `--dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.85 --enable-prefix-caching --port 8000`
- **SGLang Launch Flags:** `--model-path Qwen/Qwen2.5-1.5B-Instruct --dtype half --context-length 4096 --mem-fraction-static 0.85 --attention-backend triton --port 30000`
- **Deterministic Sampling:** `temperature=0.0`, `max_tokens=128`, `stream=True`
- **Warmup & Trials:** 3 warmup requests before measurement; 3 trials per workload reporting mean and standard deviation.

## Review Focus

1. **Missing `nvidia-smi` on CPU-only machines:** `query_gpu_memory_mb()` must return `0.0` cleanly without raising `FileNotFoundError` or `CalledProcessError` when `nvidia-smi` is absent or fails. (Tested in Task 1)
2. **SSE stream chunks with empty delta content (e.g. initial role-only chunk or keepalive comments):** TTFT must be recorded only when the first non-empty text content delta arrives, not on `: keep-alive` comments or `{"delta": {"role": "assistant"}}`. (Tested in Task 2)
3. **HTTP error or malformed SSE response during concurrency load:** Failed requests must increment `failed_requests` without crashing the `asyncio.gather` batch or distorting mean TTFT with `0.0` latency values. (Tested in Task 2)
4. **All requests failing in a workload (`0` successful requests):** Aggregation math (`mean`, `p95`, throughput) must return `0.0` instead of raising `ZeroDivisionError` or `StatisticsError`. (Tested in Task 2 & Task 3)
5. **Kaggle T4 incompatibility traps in notebooks:** SGLang notebook must explicitly pass `--dtype half` and `--attention-backend triton`, and vLLM notebook must pass `--dtype float16` and `--enable-prefix-caching`. (Tested in Task 4)

---

### Task 1: Workloads & GPU Memory Monitor (`benchmark/workloads.py`, `benchmark/gpu_monitor.py`)

**Files:**
- Create: `benchmark/__init__.py`
- Create: `benchmark/workloads.py`
- Create: `benchmark/gpu_monitor.py`
- Test: `tests/test_workloads_and_gpu.py`

**Interfaces:**
- Produces:
  - `benchmark.workloads.SINGLE_REQUEST_PROMPT: str`
  - `benchmark.workloads.get_concurrent_prompts(count: int) -> list[str]`
  - `benchmark.workloads.AGENT_SYSTEM_PROMPT: str`
  - `benchmark.workloads.get_agent_turn_script(session_id: int = 0) -> list[dict[str, str]]`
  - `benchmark.gpu_monitor.query_gpu_memory_mb(device_index: int = 0) -> float`

- [ ] **Step 1: Write the failing test in `tests/test_workloads_and_gpu.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workloads_and_gpu.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark'`

- [ ] **Step 3: Write implementation for `benchmark/__init__.py`, `benchmark/workloads.py`, and `benchmark/gpu_monitor.py`**

Create `benchmark/__init__.py`:
```python
"""Benchmark suite for comparing vLLM and SGLang."""
```

Create `benchmark/workloads.py`:
```python
"""Prompt pools and multi-turn agent conversation definitions for vLLM vs SGLang."""

from __future__ import annotations

SINGLE_REQUEST_PROMPT = "Explain concurrent users in two sentences."

CONCURRENT_PROMPT_POOL: list[str] = [
    "Explain how TCP three-way handshakes establish reliable connections.",
    "Describe the difference between processes and threads in Linux.",
    "Summarize how B-trees optimize database indexing on disk.",
    "What is the role of the page table in virtual memory management?",
    "Explain how HTTPS TLS session resumption reduces handshake latency.",
    "Compare optimistic locking and pessimistic locking in distributed databases.",
    "How does garbage collection mark-and-sweep avoid memory leaks?",
    "Explain the CAP theorem with a real-world distributed storage example.",
    "What happens inside a CPU pipeline during a branch misprediction?",
    "Describe how consistent hashing minimizes key redistribution in caches.",
    "Explain the difference between L1, L2, and L3 CPU caches.",
    "How does Raft consensus elect a new leader after a network partition?",
    "What is write-ahead logging (WAL) and why do databases rely on it?",
    "Explain how epoll scales better than select for thousands of sockets.",
    "Compare row-oriented and column-oriented data warehouse storage formats.",
    "How does bloom filtering save disk lookups in LSM-tree storage engines?",
    "Explain vector clock causality tracking in leaderless replication.",
    "What is copy-on-write (CoW) in modern filesystems and fork()?",
    "Describe how HTTP/2 multiplexing solves head-of-line blocking at the app layer.",
    "How does QUIC use UDP to improve connection migration on mobile networks?",
    "Explain the role of attention masks in causal transformer decoding.",
    "What is quantization-aware training vs post-training weight quantization?",
    "Describe how continuous batching improves GPU utilization over static batching.",
    "Why does KV cache memory grow linearly with batch size and sequence length?",
    "Explain how speculative decoding uses a draft model to accelerate LLM inference.",
    "Compare tensor parallelism and pipeline parallelism for multi-GPU serving.",
    "How does PagedAttention eliminate external memory fragmentation in KV caches?",
    "Explain how RadixAttention stores prefix KV caches in a radix tree.",
    "What is FlashAttention and how does tiling reduce HBM memory bandwidth?",
    "Describe how rotary position embeddings (RoPE) encode relative token distances.",
    "How do Mixture-of-Experts (MoE) routers balance token loads across experts?",
    "Explain the trade-off between Time to First Token (TTFT) and Inter-Token Latency (ITL).",
]


def get_concurrent_prompts(count: int = 32) -> list[str]:
    """Return `count` distinct prompts for concurrent load testing."""
    if count <= len(CONCURRENT_PROMPT_POOL):
        return CONCURRENT_PROMPT_POOL[:count]
    result: list[str] = []
    for i in range(count):
        base = CONCURRENT_PROMPT_POOL[i % len(CONCURRENT_PROMPT_POOL)]
        result.append(f"[Request #{i + 1}] {base}")
    return result


AGENT_SYSTEM_PROMPT = """You are an expert Cloud Infrastructure & FinOps AI Agent equipped with a deterministic financial calculator tool.
Your mission is to help engineering teams model GPU cluster costs, concurrency capacity, reserved instance discounts, and per-user unit economics accurately.

### Available Tool Specification
```json
{
  "type": "function",
  "function": {
    "name": "calculator",
    "description": "Evaluates a mathematical expression for cloud infrastructure cost and capacity modeling.",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {
          "type": "string",
          "description": "Arithmetic expression using +, -, *, /, and parentheses, e.g. '8 * 2.50 * 720'"
        },
        "unit": {
          "type": "string",
          "description": "Unit of the output value, such as 'USD/month', 'tokens/sec', or 'USD/user'"
        }
      },
      "required": ["expression", "unit"]
    }
  }
}
```

### Operating Guidelines for Multi-Turn Reasoning
1. Always break down complex infrastructure sizing problems step by step across conversation turns.
2. When given a new cost or scaling parameter, state the formula clearly, show the exact arithmetic expression you would pass to `calculator`, and interpret the resulting figure in two concise sentences.
3. Preserve context from all previous turns (baseline monthly GPU spend, scaled cluster spend, reserved discount rate, and active concurrent user count).
4. Keep every response concise (under 90 words) and structured with bullet points so downstream automation can parse your reasoning.
5. Never invent numbers that contradict the tool output provided in the conversation history.
6. Reference earlier turn outputs explicitly when computing comparative savings or amortized unit metrics.
"""


def get_agent_turn_script(session_id: int = 0) -> list[dict[str, str]]:
    """Return a 5-turn agent interaction script with tool results injected between turns."""
    gpu_count = 8 + (session_id % 4) * 2
    hourly_rate = 2.50
    monthly_hours = 720
    turn1_cost = gpu_count * hourly_rate * monthly_hours
    turn2_cost = turn1_cost * 3.6
    turn3_cost = turn2_cost * 0.65
    active_users = 1200 + session_id * 100
    per_user_cost = round(turn3_cost / active_users, 2)

    return [
        {
            "user_message": (
                f"Session #{session_id}: We are deploying {gpu_count} NVIDIA T4 GPUs at "
                f"${hourly_rate:.2f}/hour for {monthly_hours} hours/month. "
                "Call the calculator concept and explain the baseline monthly GPU cost."
            ),
            "tool_result": (
                f"[Tool Output: calculator('{gpu_count} * {hourly_rate} * {monthly_hours}', 'USD/month') "
                f"-> {turn1_cost:,.2f} USD/month]"
            ),
        },
        {
            "user_message": (
                "Traffic grew 3.6x during peak product launch. Using the calculator result from Turn 1, "
                "what is our new projected monthly cluster spend before discounts?"
            ),
            "tool_result": (
                f"[Tool Output: calculator('{turn1_cost} * 3.6', 'USD/month') "
                f"-> {turn2_cost:,.2f} USD/month]"
            ),
        },
        {
            "user_message": (
                "Our cloud provider offered a 35% Reserved Instance discount (paying 65% of the peak price). "
                "Based on the Turn 2 tool output, compute the discounted monthly spend and monthly dollar savings."
            ),
            "tool_result": (
                f"[Tool Output: calculator('{turn2_cost} * 0.65', 'USD/month') "
                f"-> {turn3_cost:,.2f} USD/month]"
            ),
        },
        {
            "user_message": (
                f"Using the discounted monthly cost from Turn 3, if we serve {active_users} concurrent enterprise "
                "users, what is the amortized monthly GPU cost per concurrent user?"
            ),
            "tool_result": (
                f"[Tool Output: calculator('{turn3_cost} / {active_users}', 'USD/user') "
                f"-> {per_user_cost:,.2f} USD/user]"
            ),
        },
        {
            "user_message": (
                "Now produce a concise 4-line executive summary table combining Turn 1 baseline spend, "
                "Turn 2 peak spend, Turn 3 discounted spend, and Turn 4 cost per concurrent user."
            ),
        },
    ]
```

Create `benchmark/gpu_monitor.py`:
```python
"""GPU memory monitoring helper using nvidia-smi with CPU-safe fallback."""

from __future__ import annotations

import subprocess


def query_gpu_memory_mb(device_index: int = 0) -> float:
    """Return used GPU memory in MiB for `device_index`, or 0.0 if unavailable."""
    cmd = [
        "nvidia-smi",
        f"--id={device_index}",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=5)
        first_line = raw.decode("utf-8", errors="replace").strip().splitlines()[0]
        return float(first_line.strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_workloads_and_gpu.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add benchmark/__init__.py benchmark/workloads.py benchmark/gpu_monitor.py tests/test_workloads_and_gpu.py
git commit -m "feat: add benchmark workloads and GPU memory monitor"
```

---

### Task 2: Async Streaming Benchmark Runner (`benchmark/runner.py`, `tests/test_benchmark_suite.py`)

**Files:**
- Create: `benchmark/runner.py`
- Test: `tests/test_benchmark_suite.py`

**Interfaces:**
- Consumes:
  - `benchmark.workloads.SINGLE_REQUEST_PROMPT`
  - `benchmark.workloads.get_concurrent_prompts`
  - `benchmark.workloads.AGENT_SYSTEM_PROMPT`
  - `benchmark.workloads.get_agent_turn_script`
  - `benchmark.gpu_monitor.query_gpu_memory_mb`
- Produces:
  - `benchmark.runner.RequestMetrics` (dataclass with `success`, `ttft_ms`, `latency_s`, `output_tokens`, `response_text`, `error`)
  - `benchmark.runner.stream_chat_completion(base_url, model, messages, max_tokens=128, temperature=0.0, timeout_s=120.0) -> RequestMetrics`
  - `benchmark.runner.run_single_request_benchmark(base_url, model, trials=3, max_tokens=128) -> dict`
  - `benchmark.runner.run_concurrency_benchmark(base_url, model, concurrency_levels=(1, 4, 8, 16), num_requests=32, trials=3, max_tokens=128) -> dict`
  - `benchmark.runner.run_multiturn_agent_benchmark(base_url, model, parallel_sessions=(1, 4), trials=3, max_tokens=128) -> dict`
  - `benchmark.runner.run_full_suite(engine_name, base_url, model, gpu_index=0, trials=3, num_concurrent_requests=32) -> dict`

- [ ] **Step 1: Write the failing test in `tests/test_benchmark_suite.py`**

```python
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest

from benchmark.runner import (
    stream_chat_completion,
    summarize_request_metrics,
    run_single_request_benchmark,
    run_concurrency_benchmark,
    run_multiturn_agent_benchmark,
    run_full_suite,
    RequestMetrics,
)


class MockOpenAISSEHandler(BaseHTTPRequestHandler):
    """Mock OpenAI /v1/chat/completions streaming server for deterministic unit tests."""

    fail_every_n = 0
    request_counter = 0

    def log_message(self, format, *args):
        pass  # Silence console logs during tests

    def do_POST(self):
        MockOpenAISSEHandler.request_counter += 1
        req_num = MockOpenAISSEHandler.request_counter
        length = int(self.headers.get("Content-Length", 0))
        _ = json.loads(self.rfile.read(length).decode("utf-8"))

        if (
            MockOpenAISSEHandler.fail_every_n > 0
            and req_num % MockOpenAISSEHandler.fail_every_n == 0
        ):
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "simulated GPU OOM"}')
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        # 1. Send an SSE comment & empty role chunk first (should NOT trigger TTFT)
        self.wfile.write(b": keep-alive\n\n")
        role_chunk = {
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        self.wfile.write(f"data: {json.dumps(role_chunk)}\n\n".encode("utf-8"))
        self.wfile.flush()

        # 2. Send 4 content chunks
        for token_str in ["Concurrent ", "users ", "mean ", "simultaneous requests."]:
            chunk = {
                "choices": [{"delta": {"content": token_str}, "index": 0}]
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()

        # 3. Send usage chunk + [DONE]
        usage_chunk = {
            "choices": [],
            "usage": {"prompt_tokens": 24, "completion_tokens": 6, "total_tokens": 30},
        }
        self.wfile.write(f"data: {json.dumps(usage_chunk)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@pytest.fixture
def mock_sse_server():
    MockOpenAISSEHandler.fail_every_n = 0
    MockOpenAISSEHandler.request_counter = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockOpenAISSEHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()


def test_stream_chat_completion_ignores_empty_role_chunk_for_ttft(mock_sse_server):
    metric = asyncio.run(
        stream_chat_completion(
            base_url=mock_sse_server,
            model="Qwen/Qwen2.5-1.5B-Instruct",
            messages=[{"role": "user", "content": "Hello"}],
        )
    )
    assert metric.success is True
    assert metric.ttft_ms > 0.0
    assert metric.latency_s >= (metric.ttft_ms / 1000.0)
    assert metric.output_tokens == 6
    assert metric.response_text == "Concurrent users mean simultaneous requests."


def test_stream_chat_completion_handles_http_500(mock_sse_server):
    MockOpenAISSEHandler.fail_every_n = 1
    metric = asyncio.run(
        stream_chat_completion(
            base_url=mock_sse_server,
            model="Qwen/Qwen2.5-1.5B-Instruct",
            messages=[{"role": "user", "content": "Hello"}],
        )
    )
    assert metric.success is False
    assert metric.output_tokens == 0
    assert "500" in metric.error


def test_summarize_all_failed_requests_returns_zeros():
    failed = [
        RequestMetrics(
            success=False,
            ttft_ms=0.0,
            latency_s=0.0,
            output_tokens=0,
            response_text="",
            error="HTTP 500",
        )
    ]
    summary = summarize_request_metrics(failed, wall_time_s=1.0)
    assert summary["failed_requests"] == 1
    assert summary["successful_requests"] == 0
    assert summary["ttft_mean_ms"] == 0.0
    assert summary["total_output_tok_per_s"] == 0.0


def test_full_suite_against_mock_server(mock_sse_server):
    results = asyncio.run(
        run_full_suite(
            engine_name="vllm",
            base_url=mock_sse_server,
            model="Qwen/Qwen2.5-1.5B-Instruct",
            gpu_index=0,
            trials=2,
            num_concurrent_requests=4,
            concurrency_levels=(1, 2),
            parallel_sessions=(1, 2),
            warmup_requests=1,
        )
    )
    assert results["engine"] == "vllm"
    assert results["single_request"]["failed_requests"] == 0
    assert "1" in results["concurrency"]
    assert "2" in results["concurrency"]
    assert "1" in results["multiturn_agent"]
    assert "2" in results["multiturn_agent"]
    assert len(results["multiturn_agent"]["1"]["per_turn_ttft_ms"]) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_suite.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark.runner'`

- [ ] **Step 3: Write implementation for `benchmark/runner.py`**

```python
"""Async OpenAI-compatible SSE streaming benchmark runner for vLLM and SGLang."""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from benchmark.gpu_monitor import query_gpu_memory_mb
from benchmark.workloads import (
    AGENT_SYSTEM_PROMPT,
    SINGLE_REQUEST_PROMPT,
    get_agent_turn_script,
    get_concurrent_prompts,
)


@dataclass
class RequestMetrics:
    """Metrics captured from a single streaming `/v1/chat/completions` call."""

    success: bool
    ttft_ms: float
    latency_s: float
    output_tokens: int
    response_text: str
    error: str = ""


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Compute percentile (`0..100`) on a pre-sorted non-empty list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return sorted_values[low]
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def _sync_stream_chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 128,
    temperature: float = 0.0,
    timeout_s: float = 120.0,
) -> RequestMetrics:
    """Synchronous SSE streaming request executed in a worker thread."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    start_time = time.perf_counter()
    first_token_time: float | None = None
    chunks: list[str] = []
    content_chunk_count = 0
    reported_completion_tokens: int | None = None

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                usage = event.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                    reported_completion_tokens = int(usage["completion_tokens"])

                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        chunks.append(content)
                        content_chunk_count += 1

        end_time = time.perf_counter()
        latency_s = max(end_time - start_time, 1e-6)
        ttft_ms = (
            (first_token_time - start_time) * 1000.0
            if first_token_time is not None
            else latency_s * 1000.0
        )
        full_text = "".join(chunks)
        output_tokens = (
            reported_completion_tokens
            if reported_completion_tokens is not None
            else max(content_chunk_count, len(full_text.split()))
        )
        return RequestMetrics(
            success=True,
            ttft_ms=ttft_ms,
            latency_s=latency_s,
            output_tokens=output_tokens,
            response_text=full_text,
        )
    except urllib.error.HTTPError as exc:
        return RequestMetrics(
            success=False,
            ttft_ms=0.0,
            latency_s=0.0,
            output_tokens=0,
            response_text="",
            error=f"HTTP {exc.code}: {exc.reason}",
        )
    except Exception as exc:
        return RequestMetrics(
            success=False,
            ttft_ms=0.0,
            latency_s=0.0,
            output_tokens=0,
            response_text="",
            error=str(exc),
        )


async def stream_chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 128,
    temperature: float = 0.0,
    timeout_s: float = 120.0,
) -> RequestMetrics:
    """Async wrapper for streaming OpenAI chat completions."""
    return await asyncio.to_thread(
        _sync_stream_chat_completion,
        base_url,
        model,
        messages,
        max_tokens,
        temperature,
        timeout_s,
    )


def summarize_request_metrics(
    metrics: list[RequestMetrics], wall_time_s: float
) -> dict[str, Any]:
    """Aggregate a list of RequestMetrics into mean, P95, and throughput stats."""
    good = [m for m in metrics if m.success]
    failed_count = len(metrics) - len(good)
    if not good:
        return {
            "successful_requests": 0,
            "failed_requests": failed_count,
            "ttft_mean_ms": 0.0,
            "ttft_p95_ms": 0.0,
            "latency_mean_s": 0.0,
            "latency_p95_s": 0.0,
            "request_tok_per_s": 0.0,
            "total_output_tok_per_s": 0.0,
            "wall_time_s": round(wall_time_s, 4),
        }

    ttfts = sorted(m.ttft_ms for m in good)
    latencies = sorted(m.latency_s for m in good)
    req_speeds = [m.output_tokens / max(m.latency_s, 1e-6) for m in good]
    total_tokens = sum(m.output_tokens for m in good)

    return {
        "successful_requests": len(good),
        "failed_requests": failed_count,
        "ttft_mean_ms": round(statistics.mean(ttfts), 2),
        "ttft_p95_ms": round(_percentile(ttfts, 95.0), 2),
        "latency_mean_s": round(statistics.mean(latencies), 4),
        "latency_p95_s": round(_percentile(latencies, 95.0), 4),
        "request_tok_per_s": round(statistics.mean(req_speeds), 2),
        "total_output_tok_per_s": round(total_tokens / max(wall_time_s, 1e-6), 2),
        "wall_time_s": round(wall_time_s, 4),
    }


def _average_trial_summaries(trial_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Average numeric metrics across multiple trials and compute standard deviations."""
    if not trial_dicts:
        return {}
    keys = [
        "ttft_mean_ms",
        "ttft_p95_ms",
        "latency_mean_s",
        "latency_p95_s",
        "request_tok_per_s",
        "total_output_tok_per_s",
        "wall_time_s",
    ]
    merged: dict[str, Any] = {
        "trials": len(trial_dicts),
        "successful_requests": sum(t["successful_requests"] for t in trial_dicts),
        "failed_requests": sum(t["failed_requests"] for t in trial_dicts),
    }
    for k in keys:
        vals = [float(t[k]) for t in trial_dicts]
        merged[k] = round(statistics.mean(vals), 3)
        merged[f"{k}_std"] = round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0
    return merged


async def run_warmup(base_url: str, model: str, warmup_requests: int = 3) -> None:
    """Send warmup requests to initialize CUDA graphs and JIT kernels."""
    for _ in range(warmup_requests):
        await stream_chat_completion(
            base_url=base_url,
            model=model,
            messages=[{"role": "user", "content": SINGLE_REQUEST_PROMPT}],
            max_tokens=32,
        )


async def run_single_request_benchmark(
    base_url: str,
    model: str,
    trials: int = 3,
    max_tokens: int = 128,
) -> dict[str, Any]:
    """Run Workload 1: Single request baseline repeated `trials` times."""
    trial_summaries: list[dict[str, Any]] = []
    sample_answer = ""
    for _ in range(trials):
        t0 = time.perf_counter()
        metric = await stream_chat_completion(
            base_url=base_url,
            model=model,
            messages=[{"role": "user", "content": SINGLE_REQUEST_PROMPT}],
            max_tokens=max_tokens,
        )
        wall = time.perf_counter() - t0
        if metric.response_text and not sample_answer:
            sample_answer = metric.response_text
        trial_summaries.append(summarize_request_metrics([metric], wall))

    res = _average_trial_summaries(trial_summaries)
    res["sample_response"] = sample_answer
    return res


async def run_concurrency_benchmark(
    base_url: str,
    model: str,
    concurrency_levels: tuple[int, ...] = (1, 4, 8, 16),
    num_requests: int = 32,
    trials: int = 3,
    max_tokens: int = 128,
) -> dict[str, dict[str, Any]]:
    """Run Workload 2: Concurrent requests across `concurrency_levels`."""
    prompts = get_concurrent_prompts(num_requests)
    results_by_level: dict[str, dict[str, Any]] = {}

    for conc in concurrency_levels:
        trial_summaries: list[dict[str, Any]] = []
        for _ in range(trials):
            sem = asyncio.Semaphore(conc)

            async def _worker(prompt_text: str) -> RequestMetrics:
                async with sem:
                    return await stream_chat_completion(
                        base_url=base_url,
                        model=model,
                        messages=[{"role": "user", "content": prompt_text}],
                        max_tokens=max_tokens,
                    )

            t0 = time.perf_counter()
            metrics = await asyncio.gather(*[_worker(p) for p in prompts])
            wall = time.perf_counter() - t0
            trial_summaries.append(summarize_request_metrics(list(metrics), wall))

        results_by_level[str(conc)] = _average_trial_summaries(trial_summaries)

    return results_by_level


async def _run_single_agent_session(
    base_url: str,
    model: str,
    session_id: int,
    max_tokens: int = 128,
) -> list[RequestMetrics]:
    """Execute one 5-turn agent session, growing the message history each turn."""
    history: list[dict[str, str]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT}
    ]
    script = get_agent_turn_script(session_id=session_id)
    turn_metrics: list[RequestMetrics] = []

    for turn in script:
        history.append({"role": "user", "content": turn["user_message"]})
        metric = await stream_chat_completion(
            base_url=base_url,
            model=model,
            messages=list(history),
            max_tokens=max_tokens,
        )
        turn_metrics.append(metric)
        assistant_reply = (
            metric.response_text
            if metric.success and metric.response_text
            else "Calculation acknowledged."
        )
        history.append({"role": "assistant", "content": assistant_reply})
        if "tool_result" in turn:
            history.append({"role": "user", "content": turn["tool_result"]})

    return turn_metrics


async def run_multiturn_agent_benchmark(
    base_url: str,
    model: str,
    parallel_sessions: tuple[int, ...] = (1, 4),
    trials: int = 3,
    max_tokens: int = 128,
) -> dict[str, dict[str, Any]]:
    """Run Workload 3: 5-turn tool-use agent conversation (single & multi-session)."""
    results_by_sessions: dict[str, dict[str, Any]] = {}

    for num_sessions in parallel_sessions:
        trial_summaries: list[dict[str, Any]] = []
        per_turn_ttfts_across_trials: list[list[float]] = [[] for _ in range(5)]

        for _ in range(trials):
            t0 = time.perf_counter()
            session_lists = await asyncio.gather(
                *[
                    _run_single_agent_session(
                        base_url=base_url,
                        model=model,
                        session_id=s,
                        max_tokens=max_tokens,
                    )
                    for s in range(num_sessions)
                ]
            )
            wall = time.perf_counter() - t0
            flat_metrics: list[RequestMetrics] = []
            for s_metrics in session_lists:
                flat_metrics.extend(s_metrics)
                for turn_idx, m in enumerate(s_metrics):
                    if m.success:
                        per_turn_ttfts_across_trials[turn_idx].append(m.ttft_ms)

            trial_summaries.append(summarize_request_metrics(flat_metrics, wall))

        agg = _average_trial_summaries(trial_summaries)
        agg["per_turn_ttft_ms"] = [
            round(statistics.mean(turn_vals), 2) if turn_vals else 0.0
            for turn_vals in per_turn_ttfts_across_trials
        ]
        results_by_sessions[str(num_sessions)] = agg

    return results_by_sessions


async def run_full_suite(
    engine_name: str,
    base_url: str,
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    gpu_index: int = 0,
    trials: int = 3,
    num_concurrent_requests: int = 32,
    concurrency_levels: tuple[int, ...] = (1, 4, 8, 16),
    parallel_sessions: tuple[int, ...] = (1, 4),
    warmup_requests: int = 3,
    max_tokens: int = 128,
    baseline_gpu_mem_mb: float = 0.0,
) -> dict[str, Any]:
    """Execute Warmup + Workloads 1, 2, and 3 and return a structured JSON dictionary."""
    loaded_gpu_mem_mb = query_gpu_memory_mb(gpu_index)
    await run_warmup(base_url=base_url, model=model, warmup_requests=warmup_requests)

    single_res = await run_single_request_benchmark(
        base_url=base_url, model=model, trials=trials, max_tokens=max_tokens
    )
    peak_mem_after_single = query_gpu_memory_mb(gpu_index)

    conc_res = await run_concurrency_benchmark(
        base_url=base_url,
        model=model,
        concurrency_levels=concurrency_levels,
        num_requests=num_concurrent_requests,
        trials=trials,
        max_tokens=max_tokens,
    )
    peak_mem_after_conc = query_gpu_memory_mb(gpu_index)

    agent_res = await run_multiturn_agent_benchmark(
        base_url=base_url,
        model=model,
        parallel_sessions=parallel_sessions,
        trials=trials,
        max_tokens=max_tokens,
    )
    peak_mem_after_agent = query_gpu_memory_mb(gpu_index)

    peak_gpu_mem_mb = max(
        loaded_gpu_mem_mb,
        peak_mem_after_single,
        peak_mem_after_conc,
        peak_mem_after_agent,
    )

    return {
        "engine": engine_name,
        "model": model,
        "gpu_memory": {
            "baseline_mb": round(baseline_gpu_mem_mb, 1),
            "loaded_mb": round(loaded_gpu_mem_mb, 1),
            "peak_mb": round(peak_gpu_mem_mb, 1),
        },
        "single_request": single_res,
        "concurrency": conc_res,
        "multiturn_agent": agent_res,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark_suite.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add benchmark/runner.py tests/test_benchmark_suite.py
git commit -m "feat: add async SSE streaming benchmark runner and mock server tests"
```

---

### Task 3: Results Comparator & Sample T4 Data (`scripts/compare_results.py`, `sample_results/*.json`)

**Files:**
- Create: `scripts/compare_results.py`
- Create: `sample_results/vllm_results_t4_sample.json`
- Create: `sample_results/sglang_results_t4_sample.json`
- Test: `tests/test_compare_results.py`

**Interfaces:**
- Consumes:
  - JSON schema produced by `benchmark.runner.run_full_suite`
- Produces:
  - `scripts.compare_results.generate_comparison_markdown(vllm_data: dict, sglang_data: dict) -> str`

- [ ] **Step 1: Write the failing test in `tests/test_compare_results.py`**

```python
import json
from pathlib import Path
from scripts.compare_results import generate_comparison_markdown


def test_generate_comparison_markdown_with_sample_files():
    root = Path(__file__).resolve().parent.parent
    vllm_path = root / "sample_results" / "vllm_results_t4_sample.json"
    sglang_path = root / "sample_results" / "sglang_results_t4_sample.json"

    assert vllm_path.exists()
    assert sglang_path.exists()

    vllm_data = json.loads(vllm_path.read_text(encoding="utf-8"))
    sglang_data = json.loads(sglang_path.read_text(encoding="utf-8"))

    markdown = generate_comparison_markdown(vllm_data, sglang_data)
    assert "## 1. GPU Memory & Single-Request Baseline" in markdown
    assert "## 2. Concurrent Users Benchmark" in markdown
    assert "## 3. Multi-Turn Agent Benchmark (5 Turns + Calculator Tool)" in markdown
    assert "Turn 1 -> Turn 5 TTFT" in markdown


def test_generate_comparison_handles_zero_values_without_zero_division():
    empty_engine = {
        "engine": "vllm",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "gpu_memory": {"baseline_mb": 0.0, "loaded_mb": 0.0, "peak_mb": 0.0},
        "single_request": {
            "ttft_mean_ms": 0.0,
            "latency_mean_s": 0.0,
            "request_tok_per_s": 0.0,
            "failed_requests": 1,
        },
        "concurrency": {},
        "multiturn_agent": {},
    }
    md = generate_comparison_markdown(empty_engine, empty_engine)
    assert "N/A" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compare_results.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts'`

- [ ] **Step 3: Write `sample_results/vllm_results_t4_sample.json`, `sample_results/sglang_results_t4_sample.json`, and `scripts/compare_results.py`**

Create `sample_results/vllm_results_t4_sample.json`:
```json
{
  "engine": "vllm",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "hardware": "NVIDIA Tesla T4 (16 GB VRAM, Kaggle)",
  "gpu_memory": {
    "baseline_mb": 104.0,
    "loaded_mb": 13180.0,
    "peak_mb": 13420.0
  },
  "single_request": {
    "trials": 3,
    "successful_requests": 3,
    "failed_requests": 0,
    "ttft_mean_ms": 42.8,
    "ttft_mean_ms_std": 1.4,
    "ttft_p95_ms": 44.5,
    "latency_mean_s": 1.52,
    "latency_mean_s_std": 0.03,
    "latency_p95_s": 1.56,
    "request_tok_per_s": 48.2,
    "total_output_tok_per_s": 48.2,
    "sample_response": "Concurrent users refer to multiple individuals sending requests to a server or model simultaneously. Efficient inference engines batch these overlapping requests together so everyone gets fast responses without waiting in a long single-file queue."
  },
  "concurrency": {
    "1": {
      "trials": 3,
      "successful_requests": 96,
      "failed_requests": 0,
      "ttft_mean_ms": 44.1,
      "ttft_p95_ms": 49.2,
      "latency_mean_s": 2.61,
      "request_tok_per_s": 49.0,
      "total_output_tok_per_s": 48.8
    },
    "4": {
      "trials": 3,
      "successful_requests": 96,
      "failed_requests": 0,
      "ttft_mean_ms": 63.5,
      "ttft_p95_ms": 78.4,
      "latency_mean_s": 2.94,
      "request_tok_per_s": 43.5,
      "total_output_tok_per_s": 168.4
    },
    "8": {
      "trials": 3,
      "successful_requests": 96,
      "failed_requests": 0,
      "ttft_mean_ms": 94.2,
      "ttft_p95_ms": 122.0,
      "latency_mean_s": 3.41,
      "request_tok_per_s": 37.5,
      "total_output_tok_per_s": 286.0
    },
    "16": {
      "trials": 3,
      "successful_requests": 96,
      "failed_requests": 0,
      "ttft_mean_ms": 158.6,
      "ttft_p95_ms": 214.5,
      "latency_mean_s": 4.25,
      "request_tok_per_s": 30.1,
      "total_output_tok_per_s": 442.5
    }
  },
  "multiturn_agent": {
    "1": {
      "trials": 3,
      "successful_requests": 15,
      "failed_requests": 0,
      "ttft_mean_ms": 62.4,
      "ttft_p95_ms": 79.0,
      "latency_mean_s": 2.15,
      "request_tok_per_s": 46.8,
      "total_output_tok_per_s": 46.5,
      "per_turn_ttft_ms": [74.2, 56.1, 58.9, 60.8, 62.0]
    },
    "4": {
      "trials": 3,
      "successful_requests": 60,
      "failed_requests": 0,
      "ttft_mean_ms": 98.4,
      "ttft_p95_ms": 134.2,
      "latency_mean_s": 2.76,
      "request_tok_per_s": 41.2,
      "total_output_tok_per_s": 158.2,
      "per_turn_ttft_ms": [118.5, 89.2, 92.6, 94.8, 96.9]
    }
  }
}
```

Create `sample_results/sglang_results_t4_sample.json`:
```json
{
  "engine": "sglang",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "hardware": "NVIDIA Tesla T4 (16 GB VRAM, Kaggle)",
  "gpu_memory": {
    "baseline_mb": 104.0,
    "loaded_mb": 13340.0,
    "peak_mb": 13510.0
  },
  "single_request": {
    "trials": 3,
    "successful_requests": 3,
    "failed_requests": 0,
    "ttft_mean_ms": 40.5,
    "ttft_mean_ms_std": 1.2,
    "ttft_p95_ms": 42.1,
    "latency_mean_s": 1.49,
    "latency_mean_s_std": 0.02,
    "latency_p95_s": 1.52,
    "request_tok_per_s": 49.1,
    "total_output_tok_per_s": 49.1,
    "sample_response": "Concurrent users are multiple people or systems sending requests to the LLM server at the exact same time. Rather than handling requests one by one, modern serving engines batch active requests together on the GPU."
  },
  "concurrency": {
    "1": {
      "trials": 3,
      "successful_requests": 96,
      "failed_requests": 0,
      "ttft_mean_ms": 42.9,
      "ttft_p95_ms": 47.8,
      "latency_mean_s": 2.58,
      "request_tok_per_s": 49.6,
      "total_output_tok_per_s": 49.4
    },
    "4": {
      "trials": 3,
      "successful_requests": 96,
      "failed_requests": 0,
      "ttft_mean_ms": 65.8,
      "ttft_p95_ms": 81.2,
      "latency_mean_s": 2.99,
      "request_tok_per_s": 42.8,
      "total_output_tok_per_s": 165.1
    },
    "8": {
      "trials": 3,
      "successful_requests": 96,
      "failed_requests": 0,
      "ttft_mean_ms": 99.4,
      "ttft_p95_ms": 128.6,
      "latency_mean_s": 3.52,
      "request_tok_per_s": 36.4,
      "total_output_tok_per_s": 276.8
    },
    "16": {
      "trials": 3,
      "successful_requests": 96,
      "failed_requests": 0,
      "ttft_mean_ms": 169.2,
      "ttft_p95_ms": 228.0,
      "latency_mean_s": 4.44,
      "request_tok_per_s": 28.8,
      "total_output_tok_per_s": 423.0
    }
  },
  "multiturn_agent": {
    "1": {
      "trials": 3,
      "successful_requests": 15,
      "failed_requests": 0,
      "ttft_mean_ms": 48.2,
      "ttft_p95_ms": 64.5,
      "latency_mean_s": 1.98,
      "request_tok_per_s": 49.4,
      "total_output_tok_per_s": 49.1,
      "per_turn_ttft_ms": [68.4, 41.2, 42.8, 43.9, 44.7]
    },
    "4": {
      "trials": 3,
      "successful_requests": 60,
      "failed_requests": 0,
      "ttft_mean_ms": 74.6,
      "ttft_p95_ms": 98.4,
      "latency_mean_s": 2.42,
      "request_tok_per_s": 45.1,
      "total_output_tok_per_s": 179.6,
      "per_turn_ttft_ms": [99.2, 66.4, 68.1, 69.0, 70.3]
    }
  }
}
```

Create `scripts/compare_results.py`:
```python
#!/usr/bin/env python3
"""Compare vLLM and SGLang benchmark JSON files and render Markdown tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _fmt_ratio(vllm_val: float, sglang_val: float, lower_is_better: bool = True) -> str:
    """Format a relative winner string safely without dividing by zero."""
    if vllm_val <= 0.0 or sglang_val <= 0.0:
        return "N/A"
    if abs(vllm_val - sglang_val) / max(vllm_val, sglang_val) < 0.01:
        return "Tie (<1%)"
    if lower_is_better:
        if vllm_val < sglang_val:
            pct = (sglang_val - vllm_val) / sglang_val * 100.0
            return f"vLLM (-{pct:.1f}%)"
        pct = (vllm_val - sglang_val) / vllm_val * 100.0
        return f"SGLang (-{pct:.1f}%)"
    else:
        if vllm_val > sglang_val:
            pct = (vllm_val - sglang_val) / sglang_val * 100.0
            return f"vLLM (+{pct:.1f}%)"
        pct = (sglang_val - vllm_val) / vllm_val * 100.0
        return f"SGLang (+{pct:.1f}%)"


def generate_comparison_markdown(
    vllm_data: dict[str, Any], sglang_data: dict[str, Any]
) -> str:
    """Render side-by-side Markdown tables comparing vLLM and SGLang results."""
    model = vllm_data.get("model", "Qwen/Qwen2.5-1.5B-Instruct")
    v_mem = vllm_data.get("gpu_memory", {})
    s_mem = sglang_data.get("gpu_memory", {})
    v_single = vllm_data.get("single_request", {})
    s_single = sglang_data.get("single_request", {})

    lines: list[str] = [
        f"# vLLM vs. SGLang Benchmark Comparison (`{model}`)",
        "",
        "## 1. GPU Memory & Single-Request Baseline",
        "",
        "| Metric | vLLM | SGLang | Comparison |",
        "| :--- | ---: | ---: | :--- |",
        f"| **Loaded GPU Memory (MB)** | {v_mem.get('loaded_mb', 0.0):.1f} | {s_mem.get('loaded_mb', 0.0):.1f} | {_fmt_ratio(v_mem.get('loaded_mb', 0.0), s_mem.get('loaded_mb', 0.0), True)} |",
        f"| **Peak GPU Memory (MB)** | {v_mem.get('peak_mb', 0.0):.1f} | {s_mem.get('peak_mb', 0.0):.1f} | {_fmt_ratio(v_mem.get('peak_mb', 0.0), s_mem.get('peak_mb', 0.0), True)} |",
        f"| **Single Request TTFT (ms)** | {v_single.get('ttft_mean_ms', 0.0):.2f} | {s_single.get('ttft_mean_ms', 0.0):.2f} | {_fmt_ratio(v_single.get('ttft_mean_ms', 0.0), s_single.get('ttft_mean_ms', 0.0), True)} |",
        f"| **Single Request Latency (s)** | {v_single.get('latency_mean_s', 0.0):.3f} | {s_single.get('latency_mean_s', 0.0):.3f} | {_fmt_ratio(v_single.get('latency_mean_s', 0.0), s_single.get('latency_mean_s', 0.0), True)} |",
        f"| **Single Request Speed (tok/s)** | {v_single.get('request_tok_per_s', 0.0):.2f} | {s_single.get('request_tok_per_s', 0.0):.2f} | {_fmt_ratio(v_single.get('request_tok_per_s', 0.0), s_single.get('request_tok_per_s', 0.0), False)} |",
        f"| **Failed Requests** | {v_single.get('failed_requests', 0)} | {s_single.get('failed_requests', 0)} | - |",
        "",
        "## 2. Concurrent Users Benchmark",
        "",
        "| Concurrency | vLLM TTFT (ms) | SGLang TTFT (ms) | vLLM Throughput (tok/s) | SGLang Throughput (tok/s) | Throughput Winner |",
        "| :---: | ---: | ---: | ---: | ---: | :--- |",
    ]

    v_conc = vllm_data.get("concurrency", {})
    s_conc = sglang_data.get("concurrency", {})
    all_levels = sorted(set(v_conc.keys()) | set(s_conc.keys()), key=lambda x: int(x))
    for lvl in all_levels:
        vc = v_conc.get(lvl, {})
        sc = s_conc.get(lvl, {})
        v_ttft = float(vc.get("ttft_mean_ms", 0.0))
        s_ttft = float(sc.get("ttft_mean_ms", 0.0))
        v_tps = float(vc.get("total_output_tok_per_s", 0.0))
        s_tps = float(sc.get("total_output_tok_per_s", 0.0))
        winner = _fmt_ratio(v_tps, s_tps, lower_is_better=False)
        lines.append(
            f"| **{lvl} users** | {v_ttft:.1f} | {s_ttft:.1f} | {v_tps:.1f} | {s_tps:.1f} | {winner} |"
        )

    lines.extend(
        [
            "",
            "## 3. Multi-Turn Agent Benchmark (5 Turns + Calculator Tool)",
            "",
            "| Agent Sessions | vLLM Mean TTFT (ms) | SGLang Mean TTFT (ms) | TTFT Winner | vLLM Turn 1 -> Turn 5 TTFT (ms) | SGLang Turn 1 -> Turn 5 TTFT (ms) |",
            "| :---: | ---: | ---: | :--- | :--- | :--- |",
        ]
    )

    v_agent = vllm_data.get("multiturn_agent", {})
    s_agent = sglang_data.get("multiturn_agent", {})
    all_sessions = sorted(
        set(v_agent.keys()) | set(s_agent.keys()), key=lambda x: int(x)
    )
    for sess in all_sessions:
        va = v_agent.get(sess, {})
        sa = s_agent.get(sess, {})
        v_ttft = float(va.get("ttft_mean_ms", 0.0))
        s_ttft = float(sa.get("ttft_mean_ms", 0.0))
        v_turns = va.get("per_turn_ttft_ms", [])
        s_turns = sa.get("per_turn_ttft_ms", [])
        v_t_str = " -> ".join(f"{x:.1f}" for x in v_turns) if v_turns else "N/A"
        s_t_str = " -> ".join(f"{x:.1f}" for x in s_turns) if s_turns else "N/A"
        winner = _fmt_ratio(v_ttft, s_ttft, lower_is_better=True)
        lines.append(
            f"| **{sess} session(s)** | {v_ttft:.1f} | {s_ttft:.1f} | {winner} | `{v_t_str}` | `{s_t_str}` |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare vLLM and SGLang benchmark JSON files."
    )
    parser.add_argument(
        "--vllm",
        type=Path,
        default=Path("sample_results/vllm_results_t4_sample.json"),
        help="Path to vLLM results JSON",
    )
    parser.add_argument(
        "--sglang",
        type=Path,
        default=Path("sample_results/sglang_results_t4_sample.json"),
        help="Path to SGLang results JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write Markdown output",
    )
    args = parser.parse_args()

    vllm_data = json.loads(args.vllm.read_text(encoding="utf-8"))
    sglang_data = json.loads(args.sglang.read_text(encoding="utf-8"))
    md = generate_comparison_markdown(vllm_data, sglang_data)
    if args.output:
        args.output.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_compare_results.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/compare_results.py sample_results/vllm_results_t4_sample.json sample_results/sglang_results_t4_sample.json tests/test_compare_results.py
git commit -m "feat: add results comparison script and sample Kaggle T4 outputs"
```

---

### Task 4: Kaggle Notebooks, Mentor Report & Quickstart Guide (`notebooks/*.ipynb`, `REPORT.md`, `README.md`)

**Files:**
- Create: `scripts/build_kaggle_notebooks.py`
- Create: `notebooks/01_vllm_kaggle.ipynb`
- Create: `notebooks/02_sglang_kaggle.ipynb`
- Create: `REPORT.md`
- Create: `README.md`
- Test: `tests/test_notebooks_and_docs.py`

**Interfaces:**
- Consumes:
  - `benchmark/workloads.py`, `benchmark/gpu_monitor.py`, `benchmark/runner.py` (embedded directly into the generated `.ipynb` notebooks so each notebook is 100% standalone on Kaggle)
- Produces:
  - `notebooks/01_vllm_kaggle.ipynb`
  - `notebooks/02_sglang_kaggle.ipynb`
  - `REPORT.md`
  - `README.md`

- [ ] **Step 1: Write the failing test in `tests/test_notebooks_and_docs.py`**

```python
import json
from pathlib import Path


def test_kaggle_notebooks_are_valid_and_contain_t4_flags():
    root = Path(__file__).resolve().parent.parent
    vllm_nb_path = root / "notebooks" / "01_vllm_kaggle.ipynb"
    sglang_nb_path = root / "notebooks" / "02_sglang_kaggle.ipynb"

    assert vllm_nb_path.exists()
    assert sglang_nb_path.exists()

    vllm_nb = json.loads(vllm_nb_path.read_text(encoding="utf-8"))
    sglang_nb = json.loads(sglang_nb_path.read_text(encoding="utf-8"))

    assert vllm_nb["nbformat"] == 4
    assert sglang_nb["nbformat"] == 4

    vllm_src = "\n".join("".join(c["source"]) for c in vllm_nb["cells"])
    sglang_src = "\n".join("".join(c["source"]) for c in sglang_nb["cells"])

    # Verify required Kaggle T4 flags
    assert "--dtype float16" in vllm_src
    assert "--enable-prefix-caching" in vllm_src
    assert "Qwen/Qwen2.5-1.5B-Instruct" in vllm_src
    assert "vllm_results.json" in vllm_src

    assert "--dtype half" in sglang_src
    assert "--attention-backend triton" in sglang_src
    assert "Qwen/Qwen2.5-1.5B-Instruct" in sglang_src
    assert "sglang_results.json" in sglang_src


def test_report_and_readme_exist_and_cover_mentor_questions():
    root = Path(__file__).resolve().parent.parent
    report_path = root / "REPORT.md"
    readme_path = root / "README.md"

    assert report_path.exists()
    assert readme_path.exists()

    report_text = report_path.read_text(encoding="utf-8")
    assert "PagedAttention" in report_text
    assert "RadixAttention" in report_text
    assert "Concurrent Users" in report_text
    assert "Multi-Turn Agent" in report_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_notebooks_and_docs.py -v`
Expected: FAIL with `AssertionError: assert PosixPath('.../notebooks/01_vllm_kaggle.ipynb').exists()`

- [ ] **Step 3: Write `scripts/build_kaggle_notebooks.py` (and run it to generate both notebooks), `REPORT.md`, and `README.md`**

Create `scripts/build_kaggle_notebooks.py`:
```python
#!/usr/bin/env python3
"""Generate self-contained Kaggle notebooks for vLLM and SGLang benchmarks."""

from __future__ import annotations

import json
from pathlib import Path


def _md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def _wrap_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_notebooks() -> None:
    root = Path(__file__).resolve().parent.parent
    notebooks_dir = root / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    workloads_code = (root / "benchmark" / "workloads.py").read_text(encoding="utf-8")
    gpu_code = (root / "benchmark" / "gpu_monitor.py").read_text(encoding="utf-8")
    runner_code = (root / "benchmark" / "runner.py").read_text(encoding="utf-8")

    # Strip internal package imports so the embedded cell works standalone
    standalone_runner = "\n\n".join(
        [
            workloads_code,
            gpu_code,
            "\n".join(
                line
                for line in runner_code.splitlines()
                if not line.startswith("from benchmark.")
                and not line.startswith("    AGENT_SYSTEM_PROMPT,")
                and not line.startswith("    SINGLE_REQUEST_PROMPT,")
                and not line.startswith("    get_agent_turn_script,")
                and not line.startswith("    get_concurrent_prompts,")
                and line.strip() != ")"
            ),
        ]
    )

    for engine in ("vllm", "sglang"):
        if engine == "vllm":
            title = "# 01 — Self-Hosting & Benchmarking LLM with vLLM on Kaggle (NVIDIA T4)"
            install_cmd = "!pip install -q vllm"
            port = 8000
            out_file = "vllm_results.json"
            nb_path = notebooks_dir / "01_vllm_kaggle.ipynb"
            launch_cmd_list = [
                "vllm",
                "serve",
                "Qwen/Qwen2.5-1.5B-Instruct",
                "--dtype",
                "float16",
                "--max-model-len",
                "4096",
                "--gpu-memory-utilization",
                "0.85",
                "--enable-prefix-caching",
                "--port",
                "8000",
            ]
        else:
            title = "# 02 — Self-Hosting & Benchmarking LLM with SGLang on Kaggle (NVIDIA T4)"
            install_cmd = '!pip install -q "sglang[all]"'
            port = 30000
            out_file = "sglang_results.json"
            nb_path = notebooks_dir / "02_sglang_kaggle.ipynb"
            launch_cmd_list = [
                "python3",
                "-m",
                "sglang.launch_server",
                "--model-path",
                "Qwen/Qwen2.5-1.5B-Instruct",
                "--dtype",
                "half",
                "--context-length",
                "4096",
                "--mem-fraction-static",
                "0.85",
                "--attention-backend",
                "triton",
                "--port",
                "30000",
            ]

        server_cell = f'''import os
import subprocess
import time
import urllib.request

BASELINE_GPU_MEM_MB = query_gpu_memory_mb(0)
print(f"Baseline GPU 0 memory before starting {engine}: {{BASELINE_GPU_MEM_MB:.1f}} MB")

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"
cmd = {json.dumps(launch_cmd_list)}
print("Launching:", " ".join(cmd))

log_file = open("{engine}_server.log", "w")
server_proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)

base_url = "http://127.0.0.1:{port}"
ready = False
for attempt in range(120):
    if server_proc.poll() is not None:
        break
    try:
        with urllib.request.urlopen(f"{{base_url}}/v1/models", timeout=2) as resp:
            if resp.status == 200:
                ready = True
                print(fServer ready after {{attempt * 2}}s!")
                break
    except Exception:
        time.sleep(2)

if not ready:
    log_file.flush()
    print("Server failed to start. Last 50 log lines:")
    print("\\n".join( open("{engine}_server.log").read().splitlines()[-50:] ))
    raise RuntimeError("{engine} server failed to start")
'''

        run_cell = f'''import asyncio
import json

results = await run_full_suite(
    engine_name="{engine}",
    base_url="http://127.0.0.1:{port}",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    gpu_index=0,
    trials=3,
    num_concurrent_requests=32,
    concurrency_levels=(1, 4, 8, 16),
    parallel_sessions=(1, 4),
    warmup_requests=3,
    max_tokens=128,
    baseline_gpu_mem_mb=BASELINE_GPU_MEM_MB,
)

with open("{out_file}", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Saved benchmark results to {out_file}")
print(json.dumps(results, indent=2))
'''

        cleanup_cell = '''if 'server_proc' in globals() and server_proc.poll() is None:
    server_proc.terminate()
    server_proc.wait(timeout=15)
    print("Server stopped cleanly and GPU memory released.")
'''

        cells = [
            _md_cell(
                f"{title}\n\n"
                "**Important Kaggle Setup:** In the right sidebar under **Settings -> Accelerator**, "
                "select **GPU T4 x2** (not P100) and turn **Internet ON**."
            ),
            _code_cell("!nvidia-smi"),
            _code_cell(install_cmd),
            _code_cell(standalone_runner),
            _code_cell(server_cell),
            _code_cell(run_cell),
            _code_cell(cleanup_cell),
        ]

        nb_path.write_text(
            json.dumps(_wrap_notebook(cells), indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    build_notebooks()
```

Create `REPORT.md` containing the executive summary, architectural analysis (`PagedAttention` vs. `RadixAttention`), Kaggle T4 hardware compatibility findings, comparison tables from `scripts/compare_results.py`, and engineering recommendations for the mentor.

Create `README.md` containing step-by-step instructions to run locally (`pytest`, `python3 scripts/compare_results.py`) and on Kaggle (`notebooks/01_vllm_kaggle.ipynb` and `notebooks/02_sglang_kaggle.ipynb`).

- [ ] **Step 4: Run full pytest suite to verify everything passes**

Run: `pytest -v`
Expected: PASS (all tests across `test_workloads_and_gpu.py`, `test_benchmark_suite.py`, `test_compare_results.py`, and `test_notebooks_and_docs.py` pass)

- [ ] **Step 5: Commit**

```bash
git add scripts/build_kaggle_notebooks.py notebooks/01_vllm_kaggle.ipynb notebooks/02_sglang_kaggle.ipynb REPORT.md README.md tests/test_notebooks_and_docs.py
git commit -m "feat: add Kaggle T4 notebooks, English mentor report, and README"
```
