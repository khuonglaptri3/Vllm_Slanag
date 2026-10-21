#!/usr/bin/env python3
"""Generate two self-contained Kaggle notebooks (one for vLLM, one for SGLang)."""

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


SINGLE_ENGINE_FORMATTER_CODE = '''
def format_single_engine_markdown(data: dict) -> str:
    """Format a single engine's benchmark results into ready-to-copy Markdown tables."""
    engine = data.get("engine", "unknown").upper()
    model = data.get("model", "Qwen/Qwen2.5-1.5B-Instruct")
    mem = data.get("gpu_memory", {})
    single = data.get("single_request", {})
    conc = data.get("concurrency", {})
    agent = data.get("multiturn_agent", {})

    lines = [
        f"# {engine} Benchmark Results (`{model}`)",
        "",
        "## 1. GPU Memory & Single-Request Baseline",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| **Baseline GPU Memory (MB)** | {mem.get('baseline_mb', 0.0):.1f} |",
        f"| **Loaded GPU Memory (MB)** | {mem.get('loaded_mb', 0.0):.1f} |",
        f"| **Peak GPU Memory (MB)** | {mem.get('peak_mb', 0.0):.1f} |",
        f"| **Single Request TTFT Mean (ms)** | {single.get('ttft_mean_ms', 0.0):.2f} ± {single.get('ttft_mean_ms_std', 0.0):.2f} |",
        f"| **Single Request TTFT P95 (ms)** | {single.get('ttft_p95_ms', 0.0):.2f} |",
        f"| **Single Request Total Latency (s)** | {single.get('latency_mean_s', 0.0):.3f} ± {single.get('latency_mean_s_std', 0.0):.3f} |",
        f"| **Single Request Output Speed (tok/s)** | {single.get('request_tok_per_s', 0.0):.2f} |",
        f"| **Failed Requests** | {single.get('failed_requests', 0)} |",
        "",
        f"**Sample Response:** _{single.get('sample_response', '')}_",
        "",
        "## 2. Concurrent Users Benchmark (32 Prompts per Batch)",
        "",
        "| Concurrency | Mean TTFT (ms) | P95 TTFT (ms) | Mean Latency (s) | Per-Req Speed (tok/s) | Total Server Throughput (tok/s) | Failed |",
        "| :---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]

    for lvl in sorted(conc.keys(), key=lambda x: int(x)):
        c = conc[lvl]
        lines.append(
            f"| **{lvl} users** | {c.get('ttft_mean_ms', 0.0):.1f} | {c.get('ttft_p95_ms', 0.0):.1f} | "
            f"{c.get('latency_mean_s', 0.0):.3f} | {c.get('request_tok_per_s', 0.0):.1f} | "
            f"**{c.get('total_output_tok_per_s', 0.0):.1f}** | {c.get('failed_requests', 0)} |"
        )

    lines.extend([
        "",
        "## 3. Multi-Turn Agent Benchmark (5 Turns + Calculator Tool)",
        "",
        "| Parallel Agent Sessions | Mean TTFT (ms) | P95 TTFT (ms) | Mean Latency (s) | Total Throughput (tok/s) | Turn 1 -> Turn 5 TTFT (ms) | Failed |",
        "| :---: | ---: | ---: | ---: | ---: | :--- | :---: |",
    ])

    for sess in sorted(agent.keys(), key=lambda x: int(x)):
        a = agent[sess]
        turns = a.get("per_turn_ttft_ms", [])
        t_str = " -> ".join(f"{x:.1f}" for x in turns) if turns else "N/A"
        lines.append(
            f"| **{sess} session(s)** | {a.get('ttft_mean_ms', 0.0):.1f} | {a.get('ttft_p95_ms', 0.0):.1f} | "
            f"{a.get('latency_mean_s', 0.0):.3f} | **{a.get('total_output_tok_per_s', 0.0):.1f}** | "
            f"`{t_str}` | {a.get('failed_requests', 0)} |"
        )

    return "\\n".join(lines) + "\\n"
'''.strip()


def build_notebooks() -> None:
    root = Path(__file__).resolve().parent.parent
    notebooks_dir = root / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    workloads_code = (root / "benchmark" / "workloads.py").read_text(encoding="utf-8")
    gpu_code = (root / "benchmark" / "gpu_monitor.py").read_text(encoding="utf-8")
    runner_code = (root / "benchmark" / "runner.py").read_text(encoding="utf-8")

    # Strip internal package imports and duplicate __future__ imports so the embedded cell works standalone
    gpu_lines = [
        line
        for line in gpu_code.splitlines()
        if not line.startswith("from __future__ import")
    ]
    runner_lines: list[str] = []
    skip_import_block = False
    for line in runner_code.splitlines():
        if line.startswith("from __future__ import"):
            continue
        if line.startswith("from benchmark.gpu_monitor"):
            continue
        if line.startswith("from benchmark.workloads import ("):
            skip_import_block = True
            continue
        if skip_import_block:
            if line.strip() == ")":
                skip_import_block = False
            continue
        runner_lines.append(line)

    standalone_runner = "\n\n".join(
        [
            workloads_code,
            "\n".join(gpu_lines),
            "\n".join(runner_lines),
            SINGLE_ENGINE_FORMATTER_CODE,
        ]
    )

    for engine in ("vllm", "sglang"):
        if engine == "vllm":
            title = "# 01 — All-in-One vLLM Self-Hosting & Benchmark on Kaggle (NVIDIA T4)"
            install_cmd = "!pip install -q vllm && pip uninstall -y -q torchaudio"
            port = 8000
            out_file = "vllm_results.json"
            summary_file = "vllm_summary.md"
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
            title = "# 02 — All-in-One SGLang Self-Hosting & Benchmark on Kaggle (NVIDIA T4)"
            install_cmd = '!pip install -q "sglang[all]" && pip uninstall -y -q torchaudio'
            port = 30000
            out_file = "sglang_results.json"
            summary_file = "sglang_summary.md"
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

        server_cell = f'''# Launch command: {" ".join(launch_cmd_list)}
import os
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
                print(f"Server ready after {{attempt * 2}}s!")
                break
    except Exception:
        time.sleep(2)

if not ready:
    log_file.flush()
    print("Server failed to start. Last 50 log lines:")
    print("\\n".join(open("{engine}_server.log").read().splitlines()[-50:]))
    raise RuntimeError("{engine} server failed to start")
'''

        run_cell = f'''import json
from IPython.display import Markdown, display

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

summary_md = format_single_engine_markdown(results)
with open("{summary_file}", "w", encoding="utf-8") as f:
    f.write(summary_md)

print("Saved benchmark results to {out_file} and {summary_file}\\n")
print(summary_md)
display(Markdown(summary_md))
'''

        cleanup_cell = '''if 'server_proc' in globals() and server_proc.poll() is None:
    server_proc.terminate()
    server_proc.wait(timeout=15)
    print("Server stopped cleanly and GPU memory released.")
if 'log_file' in globals() and not log_file.closed:
    log_file.close()
'''

        cells = [
            _md_cell(
                f"{title}\n\n"
                "This notebook is **100% self-contained** — you only need to upload this `.ipynb` file to Kaggle, "
                "click **Run All**, and take the formatted results table at the bottom.\n\n"
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
