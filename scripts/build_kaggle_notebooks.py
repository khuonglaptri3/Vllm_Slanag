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
