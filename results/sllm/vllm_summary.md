# VLLM Benchmark Results (`Qwen/Qwen2.5-1.5B-Instruct`)

## 1. GPU Memory & Single-Request Baseline

| Metric | Value |
| :--- | ---: |
| **Baseline GPU Memory (MB)** | 0.0 |
| **Loaded GPU Memory (MB)** | 12383.0 |
| **Peak GPU Memory (MB)** | 12385.0 |
| **Single Request TTFT Mean (ms)** | 30.38 ± 4.88 |
| **Single Request TTFT P95 (ms)** | 30.38 |
| **Single Request Total Latency (s)** | 0.826 ± 0.005 |
| **Single Request Output Speed (tok/s)** | 67.82 |
| **Failed Requests** | 0 |

**Sample Response:** _Concurrent users refer to the number of active users accessing a system or application simultaneously at any given time. This metric is crucial for understanding how well an application can handle multiple user interactions and is often used as a key performance indicator (KPI) in software development and operations._

## 2. Concurrent Users Benchmark (32 Prompts per Batch)

| Concurrency | Mean TTFT (ms) | P95 TTFT (ms) | Mean Latency (s) | Per-Req Speed (tok/s) | Total Server Throughput (tok/s) | Failed |
| :---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **1 users** | 26.3 | 27.7 | 1.911 | 67.0 | **67.0** | 0 |
| **4 users** | 45.7 | 53.6 | 1.985 | 64.5 | **257.8** | 0 |
| **8 users** | 52.0 | 59.7 | 2.089 | 61.3 | **489.5** | 0 |
| **16 users** | 53.8 | 62.0 | 2.090 | 61.3 | **489.7** | 0 |

## 3. Multi-Turn Agent Benchmark (5 Turns + Calculator Tool)

| Parallel Agent Sessions | Mean TTFT (ms) | P95 TTFT (ms) | Mean Latency (s) | Total Throughput (tok/s) | Turn 1 -> Turn 5 TTFT (ms) | Failed |
| :---: | ---: | ---: | ---: | ---: | :--- | :---: |
| **1 session(s)** | 53.0 | 63.3 | 1.863 | **61.8** | `52.4 -> 42.6 -> 55.9 -> 51.8 -> 62.4` | 0 |
| **4 session(s)** | 71.4 | 101.1 | 2.112 | **196.0** | `64.2 -> 66.5 -> 67.3 -> 80.8 -> 78.3` | 0 |
