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
        if not chunks or first_token_time is None:
            return RequestMetrics(
                success=False,
                ttft_ms=0.0,
                latency_s=0.0,
                output_tokens=0,
                response_text="",
                error="Empty response stream (no content chunks received)",
            )
        latency_s = max(end_time - start_time, 1e-6)
        ttft_ms = (first_token_time - start_time) * 1000.0
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
