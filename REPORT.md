# Self-Hosting LLMs with vLLM vs. SGLang — Experimental Report & Technical Comparison

**Model Under Test:** `Qwen/Qwen2.5-1.5B-Instruct` (`float16`, max context `4096`)  
**Target GPU Environment:** Kaggle NVIDIA Tesla T4 (16 GB VRAM, Turing Architecture, Compute Capability 7.5, pinned to single GPU via `CUDA_VISIBLE_DEVICES=0`)  
**Local Verification Environment:** Ubuntu Linux (Intel Core i5-6600 @ 3.30GHz, 15 GiB RAM — no discrete NVIDIA GPU)

---

## 1. Executive Summary

This experiment investigates our mentor's two core questions regarding self-hosted LLM inference engines:
1. **Does vLLM excel at serving concurrent users?**
2. **Does SGLang excel at multi-turn agent workloads?**

### Key Findings from Live Kaggle T4 Runs (`results/sllm/` & `results/slang/`)
- **Both engines are production-grade and expose drop-in OpenAI-compatible `/v1/chat/completions` servers**, completing all 465 requests per engine with **0 failed requests**.
- **vLLM (`v0.32.0`) outperformed SGLang across all measured latency, throughput, startup, and memory metrics on Tesla T4:**
  - **Startup & Memory:** vLLM initialized **2.67x faster** (`63.2s` vs. `168.5s`) and consumed **~1.09 GB less peak GPU VRAM** (`12,385 MB` vs. `13,475 MB`), while SGLang reserved a **+10.5% larger KV cache pool** (`345,689` vs. `312,784` tokens).
  - **Single-Request Speed:** vLLM achieved a **44.2% lower Mean TTFT** (`30.38 ms` vs. `54.47 ms`) and **39.6% faster per-request token generation** (`67.82 tok/s` vs. `48.60 tok/s`).
  - **Concurrent Users (`1, 4, 8, 16` simultaneous users with distinct prompts):** vLLM scaled total server throughput by **7.31x** (from **`67.0 tok/s`** to **`489.7 tok/s`**) while keeping Mean TTFT between **`26.3 ms` and `53.8 ms`**. SGLang scaled by **10.51x** from its lower single-stream baseline (`42.3 tok/s` $\rightarrow$ `444.3 tok/s`), finishing **10.2%–58.5% behind vLLM** in total throughput.
  - **Multi-Turn Agents (5-turn calculator tool-use conversation with growing history):** Both engines demonstrated immediate Turn 1 $\rightarrow$ Turn 2 TTFT drops from prefix caching (`52.4 ms -> 42.6 ms` in vLLM `-18.7%`; `132.5 ms -> 126.2 ms` in SGLang `-4.8%`, with SGLang's `UnifiedRadixCache` matching `564` to `1,211` cached tokens per turn). However, on Turing T4 (`CC 7.5`), vLLM's `torch.compile` (`inductor`) + piecewise/full CUDA graphs delivered **2.75x–3.00x faster Mean TTFT** (`53.0 ms` vs. `145.6 ms` at 1 session; `71.4 ms` vs. `214.3 ms` at 4 sessions) and **10.6%–33.3% higher multi-turn throughput**.

---

## 2. Architectural Comparison: Why Do They Behave Differently?

| Feature | vLLM (`PagedAttention` + Block APC) | SGLang (`RadixAttention` + Cache-Aware Scheduler) |
| :--- | :--- | :--- |
| **Core KV-Cache Innovation** | **PagedAttention:** Partitions KV cache into fixed-size virtual memory blocks (e.g., 16 tokens/block) to eliminate external fragmentation and maximize batch size. | **RadixAttention:** Organizes KV cache pages in a **Radix Tree (`RustUnifiedTreeCore`)** with LRU eviction (`page_size=1`), retaining KV state across completed requests automatically. |
| **Multi-Turn Prefix Reuse** | **Hash-Block Automatic Prefix Caching (`--enable-prefix-caching`):** Hashes full token blocks; partial block tails or dynamic mid-turn boundaries require full-block alignment. | **Token-Granular Radix Tree Matching:** Matches arbitrary shared prefixes across system prompts, few-shot examples, branching agent trees, and multi-turn chat histories. |
| **Scheduling Strategy** | First-Come-First-Served (FCFS) continuous batching optimized for maximum token throughput across independent users. | **Cache-Aware Scheduling:** Groups requests sharing radix tree prefixes to maximize GPU L2/HBM cache hits. |
| **Compilation & CUDA Graphs (T4 Run)** | `VLLM_COMPILE` (`inductor` backend, `19.75s` AOT compile) + `FULL_AND_PIECEWISE` CUDA graphs captured up to batch size `512` in `15s`. | Default `enable_torch_compile=False`; `breakable` prefill CUDA graphs (`132.2s` capture) and `full` decode CUDA graphs for `bs=[1, 2, 4, 8]` (`3.2s` capture). |
| **Structured Output / Tool Calls** | Uses `outlines` / `xgrammar` FSM decoding. | Uses compressed finite-state machines (**Jump-Forward Decoding** / `xgrammar`) that emit multi-token JSON boilerplate in a single step. |

---

## 3. Hardware & Setup Findings (Local Ubuntu vs. Kaggle T4)

### 3.1 Why Kaggle Was Required
Running `nvidia-smi` on our local Ubuntu machine showed no NVIDIA GPU (only integrated `Intel HD Graphics 530` with an `Intel i5-6600` 4-core CPU and 15 GiB RAM). Because both PagedAttention and RadixAttention are GPU HBM memory-management architectures, we targeted **Kaggle Notebooks (`GPU T4 x2`)** as recommended (`"Không có server thì thử lên Kaggle"`).

### 3.2 Critical Kaggle T4 Compatibility Rules
When deploying vLLM or SGLang on Kaggle, five hardware and container constraints must be respected:
1. **Select `GPU T4 x2`, NOT `GPU P100`:** Kaggle's P100 uses Pascal architecture (Compute Capability `6.0`), which lacks Tensor Cores and is unsupported by modern vLLM/SGLang attention kernels. Tesla T4 is Turing (Compute Capability `7.5`).
2. **Use `float16` (`half`), NOT `bfloat16`:** Turing GPUs (`CC 7.5`) do not have native hardware `bfloat16` support (introduced in Ampere `CC 8.0`). Passing `--dtype float16` (vLLM) and `--dtype half` (SGLang) is mandatory.
3. **Use `--attention-backend triton` and `--sampling-backend pytorch` on T4:** Neither FlashAttention-2 nor FlashInfer top-k/top-p sampling supports Compute Capability `7.5`. vLLM `v0.32.0` automatically falls back to `TRITON_ATTN` and PyTorch sampling, while SGLang requires `--attention-backend triton --sampling-backend pytorch` and `SGLANG_FORCE_FUSED_OP_BACKEND=native`.
4. **Remove Pre-installed `torchaudio` on Kaggle Python 3.13 (`pip uninstall -y torchaudio`):** Kaggle's base image ships with `torchaudio` built for CUDA 12.8, whereas `pip install vllm` / `sglang[all]` upgrades PyTorch to CUDA 13.0. Uninstalling the unused `torchaudio` package prevents `transformers` from raising a fatal CUDA version mismatch error on startup.
5. **Expose `nvidia/cuda_nvrtc/lib` in `LD_LIBRARY_PATH` for SGLang:** SGLang's dependency chain pulls `torch-c-dlpack-ext` and `flashinfer`, which dynamically link against `libnvrtc.so.13` bundled inside `/usr/local/lib/python3.13/dist-packages/nvidia/cuda_nvrtc/lib`.

---

## 4. Benchmark Results (`Qwen/Qwen2.5-1.5B-Instruct` on Single NVIDIA T4 16GB)

All tests ran with `temperature=0.0`, `max_tokens=128`, `stream=True`, 3 warmup requests, and 3 measured trials per workload (465 total requests per engine, **0 failed requests**). Real Kaggle T4 benchmark artifacts and server logs are archived in [`results/sllm/`](results/sllm/) and [`results/slang/`](results/slang/).

### 4.1 GPU Memory, Startup & Single-Request Baseline

| Metric | vLLM ([`results/sllm/`](results/sllm/vllm_results.json)) | SGLang ([`results/slang/`](results/slang/sglang_results.json)) | Comparison |
| :--- | ---: | ---: | :--- |
| **Baseline GPU Memory (MB)** | 0.0 | 3.0 | - |
| **Loaded GPU Memory (MB)** | **12383.0** | 13409.0 | **vLLM (-7.7% / -1,026 MB)** |
| **Peak GPU Memory (MB)** | **12385.0** | 13475.0 | **vLLM (-8.1% / -1,090 MB)** |
| **Allocated KV Cache Capacity** | 312,784 tokens (8.35 GiB) | **345,689 tokens (9.24 GiB)** | **SGLang (+10.5% capacity)** |
| **Engine Startup Time (s)** | **63.18 s** | 168.51 s | **vLLM (2.67x faster)** |
| **Single Request TTFT Mean (ms)** | **30.38 ± 4.88** | 54.47 ± 7.06 | **vLLM (-44.2%)** |
| **Single Request TTFT P95 (ms)** | **30.38** | 54.47 | **vLLM (-44.2%)** |
| **Single Request Latency (s)** | 0.826 ± 0.005 | **0.679 ± 0.010** | SGLang (-17.8%, shorter reply) |
| **Single Request Speed (tok/s)** | **67.82 ± 0.40** | 48.60 ± 0.71 | **vLLM (+39.6%)** |
| **Failed Requests** | 0 | 0 | - |

> **Note on Single-Request Latency vs. Token Speed:** SGLang's wall-clock latency (`0.679s` vs. `0.826s`) was slightly lower only because its 2-sentence answer was shorter (~33 tokens vs. ~56 tokens in vLLM). Normalized per generated token, **vLLM streamed tokens 39.6% faster (`67.82 tok/s` vs. `48.60 tok/s`)** and delivered the first token **44.2% sooner (`30.38 ms` vs. `54.47 ms`)**.

---

### 4.2 Scenario A — Concurrent Users Benchmark (32 Distinct Prompts per Batch × 3 Trials)

| Concurrency | vLLM Mean TTFT (ms) | SGLang Mean TTFT (ms) | vLLM P95 TTFT (ms) | SGLang P95 TTFT (ms) | vLLM Per-Req (tok/s) | SGLang Per-Req (tok/s) | vLLM Throughput (tok/s) | SGLang Throughput (tok/s) | Throughput Winner |
| :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **1 users** | **26.3** | 44.6 | **27.7** | 57.3 | **67.0** | 42.3 | **67.0** | 42.3 | **vLLM (+58.5%)** |
| **4 users** | **45.7** | 340.0 | **53.6** | 2128.8 | **64.5** | 56.7 | **257.8** | 213.2 | **vLLM (+20.9%)** |
| **8 users** | **52.0** | 136.0 | **59.7** | 152.1 | **61.3** | 55.3 | **489.5** | 442.2 | **vLLM (+10.7%)** |
| **16 users** | **53.8** | 139.1 | **62.0** | 154.8 | **61.3** | 55.6 | **489.7** | 444.3 | **vLLM (+10.2%)** |

**Analysis (Concurrent Users):**
- **Throughput Scaling:** vLLM scales server throughput from **`67.0 tok/s` to `489.7 tok/s`** while keeping Mean TTFT tightly bounded between **`26.3 ms` and `53.8 ms`** (`P95 <= 62.0 ms`). SGLang scales from **`42.3 tok/s` to `444.3 tok/s`** with steady-state Mean TTFT of **`136–139 ms`** at higher concurrency.
- **Root Cause of SGLang's Concurrency = 4 TTFT Spike (`340.0 ms` Mean / `2,128.8 ms` P95):** Inspecting [`results/slang/sglang_server.log` (line 562)](results/slang/sglang_server.log) reveals that when the first `concurrency=4` batch arrived at `09:06:30`, SGLang triggered a serving-time Triton JIT kernel compilation:
  > `Triton kernel '_fwd_kernel' took 6.08 s to compile after serving started. Serving-time compilation can stall the engine; pre-compile it during engine init.`
  Because the 3 warmup requests ran sequentially (`concurrency=1`), the multi-sequence Triton prefill kernel was compiled on the fly during the first `concurrency=4` trial (`ttft_p95_ms_std: 3516.3 ms`). Once cached, subsequent batches at `concurrency=8` and `16` dropped back to `136.0 ms` and `139.1 ms`.
- **Why `8 users` and `16 users` Plateau at Identical Throughput in Both Engines:** In `benchmark/runner.py`, `stream_chat_completion()` wraps blocking `urllib.request.urlopen` calls in `asyncio.to_thread()`, which uses Python's default `ThreadPoolExecutor` (`max_workers = min(32, os.cpu_count() + 4)`). On Kaggle's 4-vCPU container (`4 + 4 = 8` threads), the client thread pool dispatches at most **8 concurrent HTTP streams** even when the asyncio semaphore is `16` (confirmed in `sglang_server.log` lines 951–1099 where `#running-req` caps at `8`).

---

### 4.3 Scenario B — Multi-Turn Agent Benchmark (5 Turns + Calculator Tool Use)

Each agent session starts with a ~440-token FinOps System Prompt + JSON `calculator` tool schema and executes 5 sequential turns where tool outputs and assistant responses accumulate in the conversation history (~450 $\rightarrow$ ~1,300 input tokens).

| Agent Sessions | vLLM Mean TTFT (ms) | SGLang Mean TTFT (ms) | vLLM Throughput (tok/s) | SGLang Throughput (tok/s) | TTFT Winner | vLLM Turn 1 -> Turn 5 TTFT (ms) | SGLang Turn 1 -> Turn 5 TTFT (ms) |
| :---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| **1 session(s)** | **53.0** | 145.6 | **61.8** | 46.4 | **vLLM (-63.6%)** | `52.4 -> 42.6 -> 55.9 -> 51.8 -> 62.4` | `132.5 -> 126.2 -> 133.8 -> 155.6 -> 180.0` |
| **4 session(s)** | **71.4** | 214.3 | **196.0** | 177.2 | **vLLM (-66.7%)** | `64.2 -> 66.5 -> 67.3 -> 80.8 -> 78.3` | `180.4 -> 165.4 -> 205.9 -> 241.2 -> 278.7` |

**Analysis (Multi-Turn Agents & Prefix Caching):**
- **Prefix Caching Verified in Both Server Logs:**
  - **vLLM (`--enable-prefix-caching`):** Turn 2 TTFT in single-session mode drops by **18.7%** (`52.4 ms` $\rightarrow$ `42.6 ms`), and `results/sllm/vllm_server.log` records a **90.4% prefix cache hit rate** across the multi-turn agent workload (`196.0 tok/s` aggregate throughput at 4 sessions).
  - **SGLang (`UnifiedRadixCache`):** `results/slang/sglang_server.log` (lines 1100–1121) confirms exact token-level Radix Tree hits across all 5 turns of a session:
    - Turn 1: `#new-token: 442, #cached-token: 5` (`132.5 ms`)
    - Turn 2: `#new-token: 82, #cached-token: 564` (`126.2 ms`, **-4.8% TTFT drop**)
    - Turn 3: `#new-token: 96, #cached-token: 773` (`133.8 ms`)
    - Turn 4: `#new-token: 88, #cached-token: 996` (`155.6 ms`)
    - Turn 5: `#new-token: 88, #cached-token: 1211` (`180.0 ms`)
    Furthermore, when Trials 2 and 3 replayed the deterministic conversation script (`temperature=0.0`), SGLang's Radix Tree matched almost the entire sequence (`#new-token: 1, #cached-token: 1083`).
- **Why vLLM Still Achieved Lower Multi-Turn TTFT on Tesla T4:**
  - While SGLang's RadixCache avoided recomputing cached prefix tokens, SGLang ran without `torch.compile` (`enable_torch_compile=False`) and fell back to `cuda graph: False` whenever a prefill had `#new-token: 1` (smaller than its minimum captured prefill CUDA graph size of `4` tokens).
  - By contrast, vLLM's V1 engine compiled the model backbone with `torch.compile` (`inductor`) and used `FULL_AND_PIECEWISE` CUDA graphs starting at `1` token, giving vLLM a **2.75x–3.00x TTFT advantage** and **10.6%–33.3% higher throughput** on Turing T4 hardware.

---

## 5. Practical Conclusion & Recommendation for Our Team

1. **On NVIDIA Tesla T4 (Turing `CC 7.5`), vLLM `v0.32.0` is the clear winner out-of-the-box:**
   - vLLM works without manual `LD_LIBRARY_PATH` or sampling-backend workarounds, starts **2.67x faster**, uses **~1.09 GB less GPU memory**, and delivers higher throughput and lower TTFT across **both** concurrent user batches and multi-turn agent sessions thanks to its default `torch.compile` + piecewise CUDA graph pipeline and block-level Automatic Prefix Caching (`--enable-prefix-caching`).
2. **When SGLang's `RadixAttention` shines (and how to tune it):**
   - SGLang's `UnifiedRadixCache` (`page_size=1`) excels at token-exact prefix sharing across branching agent workflows without block-boundary constraints. However, to match vLLM's kernel execution speed, SGLang should be deployed on **Ampere/Hopper GPUs (`CC >= 8.0`)** where `FlashInfer` / `FlashAttention-3` are supported, with `--enable-torch-compile` enabled and multi-batch warmup performed at startup to avoid serving-time Triton JIT stalls.
3. **Both engines expose the same OpenAI API:**
   - Because our application code talks to `/v1/chat/completions`, we can decouple our agent code from the serving engine and switch between vLLM (`port 8000`) and SGLang (`port 30000`) via a single `BASE_URL` environment variable.
