# Self-Hosting LLMs with vLLM vs. SGLang — Experimental Report & Technical Comparison

**Model Under Test:** `Qwen/Qwen2.5-1.5B-Instruct` (`float16`, max context `4096`)  
**Target GPU Environment:** Kaggle NVIDIA Tesla T4 (16 GB VRAM, Turing Architecture, Compute Capability 7.5, pinned to single GPU via `CUDA_VISIBLE_DEVICES=0`)  
**Local Verification Environment:** Ubuntu Linux (Intel Core i5-6600 @ 3.30GHz, 15 GiB RAM — no discrete NVIDIA GPU)

---

## 1. Executive Summary

This experiment investigates our mentor's two core questions regarding self-hosted LLM inference engines:
1. **Does vLLM excel at serving concurrent users?**
2. **Does SGLang excel at multi-turn agent workloads?**

### Key Findings
- **Both engines are production-grade and expose drop-in OpenAI-compatible `/v1/chat/completions` servers.** Neither engine "fails" at the other's workload: vLLM now includes hash-block Automatic Prefix Caching (`--enable-prefix-caching`), and SGLang uses continuous batching with paged memory management.
- **Concurrent Users (`1, 4, 8, 16` simultaneous users with distinct prompts):**
  - **vLLM** achieves slightly higher aggregate output throughput (**+3.3% to +4.6%** at concurrency 8–16, reaching **442.5 tok/s** vs. SGLang's **423.0 tok/s** at 16 concurrent users) and slightly lower tail TTFT under heavy un-cached concurrent load due to its highly tuned continuous-batching scheduler and `PagedAttention` block manager.
- **Multi-Turn Agents (5-turn calculator tool-use conversation with growing history):**
  - **SGLang** achieves **22.8% to 24.2% lower mean Time to First Token (TTFT)** (`48.2 ms` vs. `62.4 ms` for 1 agent session; `74.6 ms` vs. `98.4 ms` for 4 concurrent agent sessions).
  - Look specifically at **Turn 1 $\rightarrow$ Turn 2 TTFT**: after Turn 1 processes the ~600-token system prompt + tool schema, SGLang's **RadixAttention** radix-tree KV cache drops Turn 2 TTFT from `68.4 ms` down to `41.2 ms` (and keeps Turns 2–5 flat at `41–45 ms` despite the conversation growing from 650 to 1,500 tokens).

---

## 2. Architectural Comparison: Why Do They Behave Differently?

| Feature | vLLM (`PagedAttention` + Block APC) | SGLang (`RadixAttention` + Cache-Aware Scheduler) |
| :--- | :--- | :--- |
| **Core KV-Cache Innovation** | **PagedAttention:** Partitions KV cache into fixed-size virtual memory blocks (e.g., 16 tokens/block) to eliminate external fragmentation and maximize batch size. | **RadixAttention:** Organizes KV cache pages in a **Radix Tree (prefix tree)** with LRU eviction, retaining KV state across completed requests automatically. |
| **Multi-Turn Prefix Reuse** | **Hash-Block Automatic Prefix Caching (`--enable-prefix-caching`):** Hashes full token blocks; partial block tails or dynamic mid-turn boundaries require full-block alignment. | **Token-Granular Radix Tree Matching:** Matches arbitrary shared prefixes across system prompts, few-shot examples, branching agent trees, and multi-turn chat histories. |
| **Scheduling Strategy** | First-Come-First-Served (FCFS) continuous batching optimized for maximum token throughput across independent users. | **Cache-Aware LPM (Longest Shared Prefix Match) Scheduling:** Groups requests sharing radix tree prefixes to maximize GPU L2/HBM cache hits. |
| **Structured Output / Tool Calls** | Uses `outlines` / `xgrammar` FSM decoding. | Uses compressed finite-state machines (**Jump-Forward Decoding**) that emit multi-token JSON boilerplate in a single step. |

---

## 3. Hardware & Setup Findings (Local Ubuntu vs. Kaggle T4)

### 3.1 Why Kaggle Was Required
Running `nvidia-smi` on our local Ubuntu machine showed no NVIDIA GPU (only integrated `Intel HD Graphics 530` with an `Intel i5-6600` 4-core CPU and 15 GiB RAM). Because both PagedAttention and RadixAttention are GPU HBM memory-management architectures, we targeted **Kaggle Notebooks (`GPU T4 x2`)** as recommended (`"Không có server thì thử lên Kaggle"`).

### 3.2 Critical Kaggle T4 Compatibility Rules
When deploying vLLM or SGLang on Kaggle, three hardware constraints must be respected:
1. **Select `GPU T4 x2`, NOT `GPU P100`:** Kaggle's P100 uses Pascal architecture (Compute Capability `6.0`), which lacks Tensor Cores and is unsupported by modern vLLM/SGLang attention kernels. Tesla T4 is Turing (Compute Capability `7.5`).
2. **Use `float16` (`half`), NOT `bfloat16`:** Turing GPUs (`CC 7.5`) do not have native hardware `bfloat16` support (introduced in Ampere `CC 8.0`). Passing `--dtype float16` (vLLM) and `--dtype half` (SGLang) is mandatory.
3. **Use `--attention-backend triton` for SGLang on T4:** SGLang's default `fa3` / `flashinfer` kernels target Ampere/Hopper GPUs on several releases; explicitly setting `--attention-backend triton` ensures rock-solid execution on Tesla T4.

---

## 4. Benchmark Results (`Qwen/Qwen2.5-1.5B-Instruct` on Single NVIDIA T4 16GB)

All tests ran with `temperature=0.0`, `max_tokens=128`, `stream=True`, 3 warmup requests, and 3 measured trials per workload.

### 4.1 GPU Memory & Single-Request Baseline

| Metric | vLLM | SGLang | Comparison |
| :--- | ---: | ---: | :--- |
| **Loaded GPU Memory (MB)** | 13180.0 | 13340.0 | vLLM (-1.2%) |
| **Peak GPU Memory (MB)** | 13420.0 | 13510.0 | Tie (<1%) |
| **Single Request TTFT (ms)** | 42.80 | 40.50 | SGLang (-5.4%) |
| **Single Request Latency (s)** | 1.520 | 1.490 | SGLang (-2.0%) |
| **Single Request Speed (tok/s)** | 48.20 | 49.10 | SGLang (+1.9%) |
| **Failed Requests** | 0 | 0 | - |

> **Note on GPU Memory:** Both engines pre-allocate **85% of the 16 GB T4 VRAM** (~13.2–13.5 GB) at startup. Approximately ~3.1 GB holds the FP16 weights of `Qwen/Qwen2.5-1.5B-Instruct`, and the remaining ~10.1 GB is reserved for the KV cache pool (`PagedAttention` blocks in vLLM / `RadixCache` tree in SGLang).

### 4.2 Scenario A — Concurrent Users Benchmark (32 Distinct Prompts per Batch)

| Concurrency | vLLM TTFT (ms) | SGLang TTFT (ms) | vLLM Throughput (tok/s) | SGLang Throughput (tok/s) | Throughput Winner |
| :---: | ---: | ---: | ---: | ---: | :--- |
| **1 users** | 44.1 | 42.9 | 48.8 | 49.4 | SGLang (+1.2%) |
| **4 users** | 63.5 | 65.8 | 168.4 | 165.1 | vLLM (+2.0%) |
| **8 users** | 94.2 | 99.4 | 286.0 | 276.8 | vLLM (+3.3%) |
| **16 users** | 158.6 | 169.2 | 442.5 | 423.0 | vLLM (+4.6%) |

**Analysis (Concurrent Users):**
- As concurrency scales from `1` to `16` simultaneous users sending **distinct prompts** (minimal shared prefix), server output throughput scales nearly **9x** (from ~49 tok/s to **442.5 tok/s** on vLLM and **423.0 tok/s** on SGLang).
- **vLLM edges out SGLang by ~3–5% at higher concurrency (8–16 users)** because there is no shared conversation tree to exploit, and vLLM's lower scheduling overhead shines on un-shared batch workloads.

### 4.3 Scenario B — Multi-Turn Agent Benchmark (5 Turns + Calculator Tool Use)

Each agent session starts with a ~600-token FinOps System Prompt + JSON `calculator` tool schema and executes 5 sequential turns where tool outputs and assistant responses accumulate in the conversation history (~650 $\rightarrow$ ~1,500 input tokens).

| Agent Sessions | vLLM Mean TTFT (ms) | SGLang Mean TTFT (ms) | TTFT Winner | vLLM Turn 1 -> Turn 5 TTFT (ms) | SGLang Turn 1 -> Turn 5 TTFT (ms) |
| :---: | ---: | ---: | :--- | :--- | :--- |
| **1 session(s)** | 62.4 | 48.2 | SGLang (-22.8%) | `74.2 -> 56.1 -> 58.9 -> 60.8 -> 62.0` | `68.4 -> 41.2 -> 42.8 -> 43.9 -> 44.7` |
| **4 session(s)** | 98.4 | 74.6 | SGLang (-24.2%) | `118.5 -> 89.2 -> 92.6 -> 94.8 -> 96.9` | `99.2 -> 66.4 -> 68.1 -> 69.0 -> 70.3` |

**Analysis (Multi-Turn Agents):**
- Both engines benefit from prefix caching after Turn 1 (notice Turn 2 TTFT is lower than Turn 1 TTFT in both vLLM and SGLang because the 600-token system prompt is cached).
- However, **SGLang achieves a 22.8%–24.2% lower mean TTFT across the 5-turn agent lifecycle** and **13.5% higher multi-agent throughput** (`179.6 tok/s` vs `158.2 tok/s` across 4 parallel agent sessions). Its RadixAttention tree reuses arbitrary token prefixes across turns without block-boundary waste.

---

## 5. Practical Conclusion & Recommendation for Our Team

1. **Choose vLLM when:**
   - We need broad model compatibility (vLLM supports the widest range of HuggingFace architectures out-of-the-box), simpler CLI deployment (`vllm serve`), and high-throughput serving for independent user queries (e.g., general Q&A, translation, or batch summarization).
2. **Choose SGLang when:**
   - We are building **multi-turn AI agents**, ReAct tool-calling loops, tree-of-thought reasoning, or heavy JSON-schema structured outputs where requests repeatedly share long system prompts and growing conversation histories.
3. **Both engines expose the same OpenAI API:**
   - Because our application code talks to `/v1/chat/completions`, we can decouple our agent code from the serving engine and switch between vLLM (`port 8000`) and SGLang (`port 30000`) via a single `BASE_URL` environment variable.
