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
