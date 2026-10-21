# vLLM vs. SGLang Self-Hosting & Benchmark Suite — Design Specification

**Date:** 2026-10-21  
**Target Environment:** Kaggle Notebooks (`GPU T4 x2`, pinned to a single 16 GB NVIDIA T4 GPU via `CUDA_VISIBLE_DEVICES=0`) + Local Ubuntu Machine for Verification & Report Generation  
**Model Under Test:** `Qwen/Qwen2.5-1.5B-Instruct` (`float16`, max context length `4096`)

---

## 1. Context & Goal

The team's mentor requested a practical investigation into self-hosting an open-weight LLM using **vLLM** and **SGLang**, specifically testing two common industry claims:
1. **"vLLM optimizes for concurrent users"** — high-throughput serving under simultaneous user requests via PagedAttention and continuous batching.
2. **"SGLang optimizes for multi-turn agents"** — fast prefix KV-cache reuse across repeated multi-turn conversation steps and tool calls via RadixAttention.

Because the local Ubuntu workstation (`Intel Core i5-6600`, 15 GiB RAM, integrated Intel HD Graphics 530) has no NVIDIA GPU, the GPU experiments run on **Kaggle Notebooks** using the free **NVIDIA Tesla T4 (16 GB VRAM, Compute Capability 7.5)** accelerator, while the repository provides self-contained Kaggle notebooks, a shared benchmark engine, local unit tests (against a mock OpenAI SSE server), a results comparator, and an English technical report for the mentor.

---

## 2. Repository Structure

```text
/home/intern-tdkhuong/Desktop/Vllm_Slanag/
├── README.md                                 # Quickstart guide for running on Kaggle & locally
├── REPORT.md                                 # Mentor-ready English technical report & comparison analysis
├── requirements-dev.txt                      # Local lightweight dependencies (httpx, pytest, tabulate)
├── benchmark/
│   ├── __init__.py
│   ├── workloads.py                          # Prompt pools & 5-turn calculator agent conversation builder
│   ├── gpu_monitor.py                        # nvidia-smi VRAM & utilization query helper (with CPU fallback)
│   └── runner.py                             # Async OpenAI-compatible SSE streaming benchmark client
├── scripts/
│   └── compare_results.py                    # CLI script to merge vllm_results.json & sglang_results.json into Markdown tables
├── notebooks/
│   ├── 01_vllm_kaggle.ipynb                  # Self-contained Kaggle notebook for vLLM
│   └── 02_sglang_kaggle.ipynb                # Self-contained Kaggle notebook for SGLang
├── sample_results/
│   ├── vllm_results_t4_sample.json           # Reference Kaggle T4 output schema & baseline numbers
│   └── sglang_results_t4_sample.json         # Reference Kaggle T4 output schema & baseline numbers
└── tests/
    └── test_benchmark_suite.py               # Local pytest suite using a mock async SSE server
```

---

## 3. Engine Configurations (Kaggle NVIDIA T4 Compatibility)

Kaggle's `GPU T4 x2` machine uses NVIDIA Turing architecture (Compute Capability `7.5`, 16 GB GDDR6 per GPU). Turing does **not** natively support `bfloat16` or Ampere/Hopper-only FlashAttention-3 kernels. Both servers are pinned to `CUDA_VISIBLE_DEVICES=0` with identical memory budgets and context limits:

| Parameter | vLLM Configuration | SGLang Configuration |
| :--- | :--- | :--- |
| **Model** | `Qwen/Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct` |
| **Precision (`dtype`)** | `--dtype float16` | `--dtype half` |
| **Max Context Length** | `--max-model-len 4096` | `--context-length 4096` |
| **GPU VRAM Allocation** | `--gpu-memory-utilization 0.85` | `--mem-fraction-static 0.85` |
| **Prefix Caching** | `--enable-prefix-caching` (Hash-block APC) | Enabled by default (RadixAttention tree) |
| **Attention Backend** | Default T4-compatible backend (`XFormers`/`FlashInfer`) | `--attention-backend triton` |
| **Port** | `8000` | `30000` |

### Server Lifecycle Management in Notebooks
Each Kaggle notebook starts its server via `subprocess.Popen`, redirects stdout/stderr to `server.log`, and polls `GET http://127.0.0.1:<port>/v1/models` every 2 seconds (up to 240 seconds). If the server process exits prematurely or times out, the helper automatically prints the last 50 lines of `server.log` for immediate diagnosis.

---

## 4. Benchmark Workloads & Data Flow

All workloads call the OpenAI-compatible `POST /v1/chat/completions` endpoint with `stream=True`, `temperature=0.0`, and `max_tokens=128` so token generation is deterministic and Time to First Token (TTFT) is measured accurately from the arrival of the first non-empty content delta chunk.

Before running measured trials, **3 warmup requests** are executed and discarded. Every test scenario is repeated **3 trials** to compute mean and standard deviation.

### Workload 1: Single Request Baseline
- **Prompt:** `"Explain concurrent users in two sentences."`
- **Purpose:** Sanity-check server correctness, record single-request TTFT, total latency, and output tokens/sec.

### Workload 2: Concurrent Users (`C = 1, 4, 8, 16`)
- **Prompt Pool:** 32 distinct technical questions so concurrent requests do not artificially share user-prompt prefixes.
- **Execution:** At each concurrency level $C \in \{1, 4, 8, 16\}$, `asyncio.Semaphore(C)` dispatches 32 requests concurrently.
- **Purpose:** Evaluates continuous batching efficiency, TTFT degradation under queuing, and aggregate server output throughput (`total_output_tokens / wall_clock_time`).

### Workload 3: Multi-Turn Agent with Calculator Tool Use (5 Turns)
- **System Prompt + Tool Context:** A ~600-token system prompt defining an AI financial/engineering assistant with a `calculate(expression)` tool schema and few-shot instructions.
- **5-Turn Loop per Agent Session:**
  - **Turn 1:** User asks a compound cloud-infrastructure cost problem $\rightarrow$ Assistant emits tool call / step 1 calculation.
  - **Turn 2:** Tool execution result (`14,400.00`) is appended to history + User asks follow-up scaling question $\rightarrow$ Assistant computes scaling projection.
  - **Turn 3:** Tool execution result (`51,840.00`) appended + User asks about reserved-instance discount comparison $\rightarrow$ Assistant analyzes discount.
  - **Turn 4:** User asks for per-user amortized cost across 1,200 concurrent users $\rightarrow$ Assistant calculates amortized metric.
  - **Turn 5:** User asks for an executive summary table combining turns 1–4 $\rightarrow$ Assistant generates final summary.
- **Modes Tested:**
  - **Sequential Single Agent (`1 session × 5 turns`):** Measures turn-by-turn TTFT as conversation history grows from ~650 to ~1,500 tokens.
  - **Concurrent Multi-Agent (`4 parallel agent sessions × 5 turns = 20 turns`):** Measures how effectively the engine reuses both the shared system prompt and each agent's growing multi-turn prefix KV cache.

---

## 5. Metrics Collected

For every workload, `benchmark/runner.py` records:
1. **`ttft_mean_ms` & `ttft_p95_ms`**: Mean and 95th-percentile Time to First Token in milliseconds.
2. **`latency_mean_s` & `latency_p95_s`**: Mean and 95th-percentile end-to-end request latency in seconds.
3. **`request_tok_per_s`**: Mean generation speed per individual request (`output_tokens / latency`).
4. **`total_output_tok_per_s`**: Aggregate server throughput (`sum(output_tokens) / batch_wall_time`).
5. **`failed_requests`**: Number of HTTP non-200 responses, SSE parse errors, or timeouts.
6. **`gpu_mem_baseline_mb`, `gpu_mem_loaded_mb`, `gpu_mem_peak_mb`**: GPU memory footprint captured via `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0`.

---

## 6. Verification Strategy

1. **Local Automated Tests (`pytest`):**
   - `tests/test_benchmark_suite.py` spins up a local lightweight `aiohttp`/`http.server` mock OpenAI streaming endpoint that simulates realistic TTFT delays and token streaming chunks.
   - Verifies that `benchmark/runner.py` accurately measures TTFT, total latency, concurrency limits, 5-turn agent history accumulation, and JSON export.
   - Verifies that `scripts/compare_results.py` parses `vllm_results.json` and `sglang_results.json` and renders valid Markdown comparison tables.
2. **Notebook Validation:**
   - Validates that `notebooks/01_vllm_kaggle.ipynb` and `notebooks/02_sglang_kaggle.ipynb` are valid nbformat v4 JSON files with self-contained cells ready for one-click execution on Kaggle.
