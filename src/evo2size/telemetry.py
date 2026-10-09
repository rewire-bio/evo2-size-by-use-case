"""Timers and a background nvidia-smi sampler."""
from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager
from pathlib import Path


class GpuSampler:
    """Write nvidia-smi readings every 500 ms to a CSV until stopped."""

    FIELDS = "timestamp,memory.used,memory.total,utilization.gpu,power.draw"

    def __init__(self, path: Path):
        self.path = path
        self.process: subprocess.Popen | None = None

    def __enter__(self):
        self.handle = self.path.open("w")
        self.process = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,nounits", "-lms", "500"],
            stdout=self.handle, stderr=subprocess.DEVNULL,
        )
        return self

    def __exit__(self, *exc):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.handle.close()


class Timings(dict):
    @contextmanager
    def measure(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self[name] = round(time.perf_counter() - start, 3)
