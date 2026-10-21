"""What a training run is doing, written down for the dashboard (rl/dashboard.py).

    runs/<run>/status.json     the latest state, rewritten every few minutes
    runs/<run>/metrics.jsonl   one JSON object per line, appended as things happen:
      {"kind": "train", "step", "t", "win", "revealed", "return", "clicks", "td", "q", "sps", "lr", ...}
                       rolling means over the last 1,000 training games (with exploration)
      {"kind": "eval",  "step", "t", "level", "games", "win", "win_margin", "efficiency", "revealed", "lr"}
                       greedy play on the fixed evaluation boards
      {"kind": "event", "step", "t", "text", "tag"}
                       tag "session" where a (resumed) run starts, "lr" / "reward" / "level"
                       where training changed, "stop" where it stopped

A run resumed from a checkpoint replays the steps after that checkpoint; read()
keeps only the newer copy of those steps. Writing never stops training: an
error is reported once and the journal carries on. No torch here.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

TB_TAGS = {
    "charts/win_rate": "win", "charts/revealed_fraction": "revealed", "charts/episodic_return": "return",
    "charts/episodic_length": "clicks", "losses/td_loss": "td", "losses/q_values": "q", "charts/SPS": "sps",
    "charts/epsilon": "epsilon", "charts/level": "level",
}


def _clean(value):
    """JSON-safe: numpy numbers to plain ones, NaN and infinity to None."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def replace_with_retry(tmp: Path, path: Path, tries: int = 20) -> None:
    """os.replace, retried: on Windows it fails while another process (the
    dashboard, a virus scan, a copy) has the target open for a moment."""
    for i in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.25)


class Journal:
    def __init__(self, run_dir: Path) -> None:
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metrics, self.status_file = self.dir / "metrics.jsonl", self.dir / "status.json"
        self._warned = False

    @property
    def new(self) -> bool:
        return not self.metrics.exists()

    def _guard(self, fn) -> None:
        try:
            fn()
        except OSError as e:
            if not self._warned:
                print(f"journal: could not write ({e}); training carries on", flush=True)
                self._warned = True

    def log(self, kind: str, step: int, **fields) -> None:
        row = _clean({"kind": kind, "step": int(step), "t": round(time.time(), 1), **fields})

        def write():
            with self.metrics.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        self._guard(write)

    def event(self, step: int, text: str, tag: str = "note") -> None:
        self.log("event", step, text=text, tag=tag)

    def status(self, **fields) -> None:
        def write():
            tmp = self.status_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(_clean({"updated": time.time(), **fields}), indent=1), encoding="utf-8")
            replace_with_retry(tmp, self.status_file)
        self._guard(write)

    def backfill(self, eval_history: list[dict], events_dir: Path | None = None) -> None:
        """Start a new journal from what a run already has: the checkpoint's
        evaluations and, if tensorboard can read them, the logged training curves."""
        rows = [] if events_dir is None else _tensorboard_rows(Path(events_dir))
        rows += [{"kind": "eval", **e} for e in eval_history]
        rows.sort(key=lambda r: (r["step"], r["kind"] != "train"))

        def write():
            with self.metrics.open("a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(_clean(r)) + "\n")
        self._guard(write)


def _tensorboard_rows(events_dir: Path) -> list[dict]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return []
    files = sorted(events_dir.glob("events.out.tfevents.*"), key=lambda p: int(p.name.split(".")[3]))
    rows: dict[int, dict] = {}
    for path in files:
        acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
        acc.Reload()
        found: dict[int, dict] = {}
        for tag, key in TB_TAGS.items():
            if tag in acc.Tags().get("scalars", []):
                for ev in acc.Scalars(tag):
                    found.setdefault(ev.step, {"kind": "train", "step": ev.step, "t": round(ev.wall_time, 1)})[key] = ev.value
        if found:
            first = min(found)
            rows = {s: r for s, r in rows.items() if s < first}   # a resumed session replaces what came after
            rows.update(found)
    return [r for r in rows.values() if "win" in r]


def read(run_dir: Path) -> dict:
    """{"train": [...], "eval": [...], "events": [...]}, each in step order,
    with the steps a resumed session replayed kept only once (the newer copy)."""
    out = {"train": [], "eval": [], "events": []}
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue                     # a line cut short by a crash
            kind = row.get("kind")
            if kind == "event" and row.get("tag") == "session":
                s = row["step"]
                out["train"] = [r for r in out["train"] if r["step"] <= s]
                out["eval"] = [r for r in out["eval"] if r["step"] <= s]
                out["events"] = [r for r in out["events"] if r["step"] <= s or r.get("tag") in ("session", "stop")]
            if kind in ("train", "eval"):
                out[kind].append(row)
            elif kind == "event":
                out["events"].append(row)
    for rows in out.values():
        rows.sort(key=lambda r: r["step"])
    return out


def read_status(run_dir: Path) -> dict | None:
    for _ in range(5):
        try:
            return json.loads((Path(run_dir) / "status.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (PermissionError, json.JSONDecodeError):
            time.sleep(0.1)                  # being replaced right now
    return None
