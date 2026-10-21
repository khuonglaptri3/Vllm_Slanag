# vLLM vs. SGLang Self-Hosting & Benchmark Suite

This repository contains our team's reproducible experiment and benchmark comparison between **vLLM** and **SGLang** using `Qwen/Qwen2.5-1.5B-Instruct`, designed to run on **Kaggle Notebooks (`GPU T4 x2`)** and verify locally on Ubuntu.

---

## Repository Contents

- [`REPORT.md`](./REPORT.md) — **Mentor Deliverable:** Complete English technical report comparing setup, GPU memory usage, concurrent user scaling, and 5-turn calculator agent performance.
- [`notebooks/01_vllm_kaggle.ipynb`](./notebooks/01_vllm_kaggle.ipynb) — Self-contained Kaggle notebook that installs `vllm`, launches `vllm serve` on GPU 0, runs all 3 benchmark workloads, and exports `vllm_results.json`.
- [`notebooks/02_sglang_kaggle.ipynb`](./notebooks/02_sglang_kaggle.ipynb) — Self-contained Kaggle notebook that installs `sglang[all]`, launches `sglang.launch_server` on GPU 0, runs the exact same 3 workloads, and exports `sglang_results.json`.
- [`benchmark/`](./benchmark/) — Modular Python benchmark library (`workloads.py`, `gpu_monitor.py`, `runner.py`) using OpenAI-compatible SSE streaming (`/v1/chat/completions`).
- [`scripts/compare_results.py`](./scripts/compare_results.py) — CLI tool that merges `vllm_results.json` and `sglang_results.json` into side-by-side Markdown comparison tables.
- [`sample_results/`](./sample_results/) — Reference Kaggle NVIDIA T4 benchmark outputs (`vllm_results_t4_sample.json` and `sglang_results_t4_sample.json`).
- [`tests/`](./tests/) — Local unit and integration test suite using a mock OpenAI SSE streaming server (runs on CPU with zero GPU required).

---

## How to Run on Kaggle (Step-by-Step)

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code) and click **New Notebook**.
2. Click **File -> Import Notebook** and upload [`notebooks/01_vllm_kaggle.ipynb`](./notebooks/01_vllm_kaggle.ipynb).
3. In the right sidebar under **Notebook Options / Settings**:
   - **Accelerator:** Select **GPU T4 x2** *(Important: do NOT select P100, as P100 is Compute Capability 6.0 and lacks Tensor Cores required by modern vLLM/SGLang kernels).*
   - **Internet:** Turn **ON** (so `pip` and HuggingFace can download `Qwen/Qwen2.5-1.5B-Instruct`).
4. Click **Run All** and download the generated `vllm_results.json` from `/kaggle/working/vllm_results.json`.
5. Repeat steps 1–4 in a fresh Kaggle notebook session with [`notebooks/02_sglang_kaggle.ipynb`](./notebooks/02_sglang_kaggle.ipynb) and download `sglang_results.json`.
6. Generate the side-by-side comparison table locally:
   ```bash
   python3 scripts/compare_results.py --vllm vllm_results.json --sglang sglang_results.json
   ```

---

## How to Verify Locally on Ubuntu (No GPU Required)

1. Run the automated test suite (tests the SSE streaming client, TTFT calculation, 5-turn agent loop, results comparator, and Kaggle notebook validity):
   ```bash
   python3 -m pytest -v
   ```
2. Generate the Markdown comparison tables from the reference T4 results:
   ```bash
   python3 scripts/compare_results.py
   ```
3. Re-build the self-contained Kaggle notebooks if you modify `benchmark/`:
   ```bash
   python3 scripts/build_kaggle_notebooks.py
   ```
