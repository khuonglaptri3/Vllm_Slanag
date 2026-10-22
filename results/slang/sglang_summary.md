# SGLANG Benchmark Results (`Qwen/Qwen2.5-1.5B-Instruct`)

## 1. GPU Memory & Single-Request Baseline

| Metric | Value |
| :--- | ---: |
| **Baseline GPU Memory (MB)** | 3.0 |
| **Loaded GPU Memory (MB)** | 13409.0 |
| **Peak GPU Memory (MB)** | 13475.0 |
| **Single Request TTFT Mean (ms)** | 54.47 ± 7.06 |
| **Single Request TTFT P95 (ms)** | 54.47 |
| **Single Request Total Latency (s)** | 0.679 ± 0.010 |
| **Single Request Output Speed (tok/s)** | 48.60 |
| **Failed Requests** | 0 |

**Sample Response:** _Concurrent users refer to the number of active users accessing a system or application simultaneously, which is an important metric for measuring the performance and scalability of a service._

## 2. Concurrent Users Benchmark (32 Prompts per Batch)

| Concurrency | Mean TTFT (ms) | P95 TTFT (ms) | Mean Latency (s) | Per-Req Speed (tok/s) | Total Server Throughput (tok/s) | Failed |
| :---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **1 users** | 44.6 | 57.3 | 3.029 | 42.3 | **42.3** | 0 |
| **4 users** | 340.0 | 2128.8 | 2.440 | 56.7 | **213.2** | 0 |
| **8 users** | 136.0 | 152.1 | 2.314 | 55.3 | **442.2** | 0 |
| **16 users** | 139.1 | 154.8 | 2.304 | 55.6 | **444.3** | 0 |

## 3. Multi-Turn Agent Benchmark (5 Turns + Calculator Tool)

| Parallel Agent Sessions | Mean TTFT (ms) | P95 TTFT (ms) | Mean Latency (s) | Total Throughput (tok/s) | Turn 1 -> Turn 5 TTFT (ms) | Failed |
| :---: | ---: | ---: | ---: | ---: | :--- | :---: |
| **1 session(s)** | 145.6 | 179.0 | 2.713 | **46.4** | `132.5 -> 126.2 -> 133.8 -> 155.6 -> 180.0` | 0 |
| **4 session(s)** | 214.3 | 355.5 | 2.819 | **177.2** | `180.4 -> 165.4 -> 205.9 -> 241.2 -> 278.7` | 0 |
