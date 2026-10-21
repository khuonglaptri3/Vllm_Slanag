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
