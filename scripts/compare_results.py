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
