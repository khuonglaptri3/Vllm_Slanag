"""Prompt pools and multi-turn agent conversation definitions for vLLM vs SGLang."""

from __future__ import annotations

SINGLE_REQUEST_PROMPT = "Explain concurrent users in two sentences."

CONCURRENT_PROMPT_POOL: list[str] = [
    "Explain how TCP three-way handshakes establish reliable connections.",
    "Describe the difference between processes and threads in Linux.",
    "Summarize how B-trees optimize database indexing on disk.",
    "What is the role of the page table in virtual memory management?",
    "Explain how HTTPS TLS session resumption reduces handshake latency.",
    "Compare optimistic locking and pessimistic locking in distributed databases.",
    "How does garbage collection mark-and-sweep avoid memory leaks?",
    "Explain the CAP theorem with a real-world distributed storage example.",
    "What happens inside a CPU pipeline during a branch misprediction?",
    "Describe how consistent hashing minimizes key redistribution in caches.",
    "Explain the difference between L1, L2, and L3 CPU caches.",
    "How does Raft consensus elect a new leader after a network partition?",
    "What is write-ahead logging (WAL) and why do databases rely on it?",
    "Explain how epoll scales better than select for thousands of sockets.",
    "Compare row-oriented and column-oriented data warehouse storage formats.",
    "How does bloom filtering save disk lookups in LSM-tree storage engines?",
    "Explain vector clock causality tracking in leaderless replication.",
    "What is copy-on-write (CoW) in modern filesystems and fork()?",
    "Describe how HTTP/2 multiplexing solves head-of-line blocking at the app layer.",
    "How does QUIC use UDP to improve connection migration on mobile networks?",
    "Explain the role of attention masks in causal transformer decoding.",
    "What is quantization-aware training vs post-training weight quantization?",
    "Describe how continuous batching improves GPU utilization over static batching.",
    "Why does KV cache memory grow linearly with batch size and sequence length?",
    "Explain how speculative decoding uses a draft model to accelerate LLM inference.",
    "Compare tensor parallelism and pipeline parallelism for multi-GPU serving.",
    "How does PagedAttention eliminate external memory fragmentation in KV caches?",
    "Explain how RadixAttention stores prefix KV caches in a radix tree.",
    "What is FlashAttention and how does tiling reduce HBM memory bandwidth?",
    "Describe how rotary position embeddings (RoPE) encode relative token distances.",
    "How do Mixture-of-Experts (MoE) routers balance token loads across experts?",
    "Explain the trade-off between Time to First Token (TTFT) and Inter-Token Latency (ITL).",
]


def get_concurrent_prompts(count: int = 32) -> list[str]:
    """Return `count` distinct prompts for concurrent load testing."""
    if count <= len(CONCURRENT_PROMPT_POOL):
        return CONCURRENT_PROMPT_POOL[:count]
    result: list[str] = []
    for i in range(count):
        base = CONCURRENT_PROMPT_POOL[i % len(CONCURRENT_PROMPT_POOL)]
        result.append(f"[Request #{i + 1}] {base}")
    return result


AGENT_SYSTEM_PROMPT = """You are an expert Cloud Infrastructure & FinOps AI Agent equipped with a deterministic financial calculator tool.
Your mission is to help engineering teams model GPU cluster costs, concurrency capacity, reserved instance discounts, and per-user unit economics accurately.

### Available Tool Specification
```json
{
  "type": "function",
  "function": {
    "name": "calculator",
    "description": "Evaluates a mathematical expression for cloud infrastructure cost and capacity modeling.",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {
          "type": "string",
          "description": "Arithmetic expression using +, -, *, /, and parentheses, e.g. '8 * 2.50 * 720'"
        },
        "unit": {
          "type": "string",
          "description": "Unit of the output value, such as 'USD/month', 'tokens/sec', or 'USD/user'"
        }
      },
      "required": ["expression", "unit"]
    }
  }
}
```

### Operating Guidelines for Multi-Turn Reasoning
1. Always break down complex infrastructure sizing problems step by step across conversation turns.
2. When given a new cost or scaling parameter, state the formula clearly, show the exact arithmetic expression you would pass to `calculator`, and interpret the resulting figure in two concise sentences.
3. Preserve context from all previous turns (baseline monthly GPU spend, scaled cluster spend, reserved discount rate, and active concurrent user count).
4. Keep every response concise (under 90 words) and structured with bullet points so downstream automation can parse your reasoning.
5. Never invent numbers that contradict the tool output provided in the conversation history.
6. Reference earlier turn outputs explicitly when computing comparative savings or amortized unit metrics.
7. Format all currency values in USD with two decimal places and thousand separators so financial reports remain consistent across all five turns.
"""


def get_agent_turn_script(session_id: int = 0) -> list[dict[str, str]]:
    """Return a 5-turn agent interaction script with tool results injected between turns."""
    gpu_count = 8 + (session_id % 4) * 2
    hourly_rate = 2.50
    monthly_hours = 720
    turn1_cost = gpu_count * hourly_rate * monthly_hours
    turn2_cost = turn1_cost * 3.6
    turn3_cost = turn2_cost * 0.65
    active_users = 1200 + session_id * 100
    per_user_cost = round(turn3_cost / active_users, 2)

    return [
        {
            "user_message": (
                f"Session #{session_id}: We are deploying {gpu_count} NVIDIA T4 GPUs at "
                f"${hourly_rate:.2f}/hour for {monthly_hours} hours/month. "
                "Call the calculator concept and explain the baseline monthly GPU cost."
            ),
            "tool_result": (
                f"[Tool Output: calculator('{gpu_count} * {hourly_rate} * {monthly_hours}', 'USD/month') "
                f"-> {turn1_cost:,.2f} USD/month]"
            ),
        },
        {
            "user_message": (
                "Traffic grew 3.6x during peak product launch. Using the calculator result from Turn 1, "
                "what is our new projected monthly cluster spend before discounts?"
            ),
            "tool_result": (
                f"[Tool Output: calculator('{turn1_cost} * 3.6', 'USD/month') "
                f"-> {turn2_cost:,.2f} USD/month]"
            ),
        },
        {
            "user_message": (
                "Our cloud provider offered a 35% Reserved Instance discount (paying 65% of the peak price). "
                "Based on the Turn 2 tool output, compute the discounted monthly spend and monthly dollar savings."
            ),
            "tool_result": (
                f"[Tool Output: calculator('{turn2_cost} * 0.65', 'USD/month') "
                f"-> {turn3_cost:,.2f} USD/month]"
            ),
        },
        {
            "user_message": (
                f"Using the discounted monthly cost from Turn 3, if we serve {active_users} concurrent enterprise "
                "users, what is the amortized monthly GPU cost per concurrent user?"
            ),
            "tool_result": (
                f"[Tool Output: calculator('{turn3_cost} / {active_users}', 'USD/user') "
                f"-> {per_user_cost:,.2f} USD/user]"
            ),
        },
        {
            "user_message": (
                "Now produce a concise 4-line executive summary table combining Turn 1 baseline spend, "
                "Turn 2 peak spend, Turn 3 discounted spend, and Turn 4 cost per concurrent user."
            ),
        },
    ]
