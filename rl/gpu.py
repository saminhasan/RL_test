"""NVIDIA GPU readings through nvidia-smi (no extra packages), and a guard that
slows training down while the GPU runs hotter than a set temperature.

Used by the trainer (rl/dqn.py) and the dashboard (rl/dashboard.py); no torch here.
"""
from __future__ import annotations

import shutil
import subprocess
import time

FIELDS = {
    "name":        "name",
    "temp":        "temperature.gpu",
    "util":        "utilization.gpu",
    "mem_used":    "memory.used",
    "mem_total":   "memory.total",
    "clock":       "clocks.gr",
    "clock_max":   "clocks.max.gr",
}


def query() -> dict | None:
    """The first GPU's readings, numbers where nvidia-smi gives numbers; None without nvidia-smi."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=" + ",".join(FIELDS.values()), "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    values = [v.strip() for v in out.splitlines()[0].split(",")] if out.strip() else []
    if len(values) != len(FIELDS):
        return None
    reading = {}
    for key, v in zip(FIELDS, values):
        try:
            reading[key] = float(v)
        except ValueError:
            reading[key] = v if key == "name" else None
    return reading


class ThermalGuard:
    """Call once per training step. While the GPU is hotter than `max_temp` it
    sleeps a little after each step, more the longer it stays hot, and eases
    off again once it has cooled 2 degrees below. max_temp 0 turns it off."""

    def __init__(self, max_temp: float, every: float = 15.0, step: float = 0.1, most: float = 2.0) -> None:
        self.max_temp, self.every, self.step, self.most = max_temp, every, step, most
        self.pause = 0.0                 # seconds slept after each training step
        self.temp: float | None = None   # the last reading
        self._next = 0.0

    def __call__(self) -> None:
        if self.max_temp <= 0:
            return
        now = time.monotonic()
        if now >= self._next:
            self._next = now + self.every
            reading = query()
            self.temp = reading["temp"] if reading else None
            if self.temp is not None:
                if self.temp > self.max_temp:
                    self.pause = min(self.most, self.pause + self.step)
                elif self.temp < self.max_temp - 2:
                    self.pause = max(0.0, self.pause - self.step / 2)
        if self.pause:
            time.sleep(self.pause)
