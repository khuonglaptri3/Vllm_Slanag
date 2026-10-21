"""GPU memory monitoring helper using nvidia-smi with CPU-safe fallback."""

from __future__ import annotations

import subprocess


def query_gpu_memory_mb(device_index: int = 0) -> float:
    """Return used GPU memory in MiB for `device_index`, or 0.0 if unavailable."""
    cmd = [
        "nvidia-smi",
        f"--id={device_index}",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=5)
        first_line = raw.decode("utf-8", errors="replace").strip().splitlines()[0]
        return float(first_line.strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return 0.0
