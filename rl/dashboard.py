"""A dashboard for a long training run, on the local network.

    python -m rl.dashboard                                   # http://<this pc>:8765
    python -m rl.dashboard --start-task "RL training" --log "..\\train.log"

It shows what the run writes to runs/<run>/ (rl/journal.py): win rate on the
evaluation boards next to bayes, the training curves, the GPU and the log.
The newest run is shown unless --run names one. Three buttons, which need the
token printed at start (open the page once as http://<pc>:8765/?token=<token>,
the browser remembers it):

    Pause      the run saves and stops (models/<run>.stop), and models/PAUSED
               keeps the training task from starting it again at logon
    Resume     removes models/PAUSED and starts the Windows task --start-task
    Benchmark  plays a copy of the latest checkpoint against bayes on
               --benchmark-games boards, with the loss breakdown, while
               training carries on; results are kept in runs/<run>/benchmarks.jsonl

No torch here, so the dashboard keeps running whatever training does.
"""
from __future__ import annotations

import argparse
import hmac
import json
import math
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from rl import gpu, journal

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR, MODELS_DIR = ROOT / "runs", ROOT / "models"
PAGE = Path(__file__).resolve().parent / "dashboard.html"
PAUSED = MODELS_DIR / "PAUSED"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
RANGES = {"7d": 7 * 86400, "24h": 86400}


# -- the run on disk ------------------------------------------------------------

def newest_run() -> str | None:
    found = sorted(RUNS_DIR.glob("*/status.json"), key=lambda p: p.stat().st_mtime)
    return found[-1].parent.name if found else None


def process_alive(pid) -> bool:
    """True if `pid` is a running Python process."""
    if not pid:
        return False
    if os.name != "nt":
        try:
            os.kill(int(pid), 0)
            return True
        except OSError:
            return False
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.windll.kernel32
    k32.OpenProcess.restype = wintypes.HANDLE
    handle = k32.OpenProcess(0x1000, False, int(pid))            # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        if not k32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:   # STILL_ACTIVE
            return False
        name = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        k32.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(size))
        return "python" in Path(name.value).name.lower()            # not some later process with the same id
    finally:
        k32.CloseHandle(handle)


class History:
    """journal.read(), re-read only when metrics.jsonl changed."""

    def __init__(self) -> None:
        self._key, self._rows, self._lock = None, None, threading.Lock()

    def get(self, run: str) -> dict:
        path = RUNS_DIR / run / "metrics.jsonl"
        try:
            st = path.stat()
            key = (run, st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            key = (run, None, None)
        with self._lock:
            if key != self._key:
                self._rows, self._key = journal.read(RUNS_DIR / run), key
            return self._rows


def thin(rows: list[dict], most: int) -> list[dict]:
    """At most `most` rows: means over runs of neighbours, at the last step of each."""
    if len(rows) <= most:
        return rows
    k = math.ceil(len(rows) / most)
    out = []
    for i in range(0, len(rows), k):
        chunk = rows[i:i + k]
        row = dict(chunk[-1])
        for key, v in row.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and key not in ("step", "t", "level"):
                vals = [r[key] for r in chunk if isinstance(r.get(key), (int, float))]
                row[key] = sum(vals) / len(vals) if vals else None
        out.append(row)
    return out


def tail(path: Path | None, lines: int = 80) -> list[str]:
    if not path or not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 48_000))
        text = f.read().decode("utf-8", errors="replace").replace("﻿", "")
    return text.splitlines()[-lines:]


# -- benchmark --------------------------------------------------------------------

class Benchmark:
    """One benchmark at a time, in its own process (rl.evaluate --json)."""

    def __init__(self, games: int) -> None:
        self.games, self.proc, self.started, self.run, self.error = games, None, None, None, None
        self._lock = threading.Lock()

    def start(self, run: str, status: dict) -> str:
        with self._lock:
            if self.proc and self.proc.poll() is None:
                return "a benchmark is already running"
            latest = Path((status.get("checkpoints") or {}).get("latest", MODELS_DIR / f"{run}-latest.pt"))
            if not latest.exists():
                return "no checkpoint saved yet"
            copy = MODELS_DIR / f"{run}-benchmark.pt"
            shutil.copyfile(latest, copy)          # a fixed copy: training keeps saving over the latest
            out = RUNS_DIR / run / "benchmark.json"
            out.unlink(missing_ok=True)
            cmd = [sys.executable, "-m", "rl.evaluate", "--model", str(copy), "--level", str(status.get("target_level", 4)),
                   "--games", str(self.games), "--breakdown", "--json", str(out)]
            if status.get("bf16") is False:
                cmd.append("--no-bf16")
            log = (RUNS_DIR / run / "benchmark.log").open("w", encoding="utf-8")
            self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=NO_WINDOW,
                                         env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"})
            log.close()
            self.started, self.run, self.error = time.time(), run, None
            return ""

    def state(self) -> dict:
        with self._lock:
            if self.proc is None:
                return {"running": False, "error": self.error}
            if self.proc.poll() is None:
                return {"running": True, "started": self.started, "games": self.games}
            out = RUNS_DIR / self.run / "benchmark.json"
            if out.exists():
                result = json.loads(out.read_text())
                with (RUNS_DIR / self.run / "benchmarks.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({**result, "seconds": round(time.time() - self.started)}) + "\n")
                out.unlink()
            else:
                log = tail(RUNS_DIR / self.run / "benchmark.log", 6)
                self.error = "the benchmark failed: " + (log[-1] if log else f"exit code {self.proc.returncode}")
            self.proc = None
            return {"running": False, "error": self.error}


def benchmarks(run: str) -> list[dict]:
    path = RUNS_DIR / run / "benchmarks.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# -- the server ---------------------------------------------------------------------

class Dashboard:
    def __init__(self, args) -> None:
        self.args, self.token = args, load_token(Path(args.token_file))
        self.history, self.bench = History(), Benchmark(args.benchmark_games)
        self.resume_asked = 0.0
        self._gpu, self._gpu_at = None, 0.0

    def run_name(self) -> str | None:
        return self.args.run or newest_run()

    def gpu(self) -> dict | None:
        if time.time() - self._gpu_at > 5:
            self._gpu, self._gpu_at = gpu.query(), time.time()
        return self._gpu

    def phase(self, run: str | None, status: dict | None) -> str:
        status = status or {}
        if process_alive(status.get("pid")):
            if (MODELS_DIR / f"{run}.stop").exists() or status.get("state") == "saving":
                return "stopping"
            if status.get("state") in ("evaluating", "starting"):
                return status["state"]
            return "stalled" if time.time() - status.get("updated", 0) > 30 * 60 else "training"
        if time.time() - self.resume_asked < 180:
            return "starting"
        if PAUSED.exists():
            return "paused"
        return "stopped" if status.get("state") == "stopped" else "down"

    def state(self) -> dict:
        run = self.run_name()
        status = journal.read_status(RUNS_DIR / run) if run else None
        return {"run": run, "phase": self.phase(run, status), "status": status, "gpu": self.gpu(),
                "paused": PAUSED.exists(), "benchmark": self.bench.state(), "host": socket.gethostname(),
                "can_start": bool(self.args.start_task), "now": time.time()}

    def history_for(self, rng: str) -> dict:
        run = self.run_name()
        if not run:
            return {"train": [], "eval": [], "events": [], "benchmarks": []}
        rows = self.history.get(run)
        since = 0
        if rng in RANGES:
            cut = time.time() - RANGES[rng]
            since = next((r["step"] for r in rows["train"] if r.get("t", 0) >= cut), math.inf)
        elif rng == "reward":
            since = max((e["step"] for e in rows["events"] if e.get("tag") == "reward"), default=0)
        pick = lambda key: [r for r in rows[key] if r["step"] >= since]   # noqa: E731
        return {"since": since if since != math.inf else None, "train": thin(pick("train"), 900),
                "eval": pick("eval"), "events": pick("events"), "benchmarks": benchmarks(run)}

    # -- controls

    def pause(self) -> tuple[bool, str]:
        PAUSED.write_text(time.strftime("paused from the dashboard %Y-%m-%d %H:%M:%S\n"))
        run = self.run_name()
        status = journal.read_status(RUNS_DIR / run) if run else None
        if run and process_alive((status or {}).get("pid")):
            (MODELS_DIR / f"{run}.stop").touch()
            return True, "Saving and stopping. It finishes the current step (or evaluation) first."
        return True, "Paused. Training stays off until you resume."

    def resume(self) -> tuple[bool, str]:
        run = self.run_name()
        status = journal.read_status(RUNS_DIR / run) if run else None
        if process_alive((status or {}).get("pid")):
            return False, "Training is already running."
        if not self.args.start_task:
            return False, "No training task configured (--start-task)."
        PAUSED.unlink(missing_ok=True)
        if run:
            (MODELS_DIR / f"{run}.stop").unlink(missing_ok=True)
        # the task's window stays open after training ends, so end it before running it again
        subprocess.run(["schtasks", "/end", "/tn", self.args.start_task], capture_output=True, creationflags=NO_WINDOW)
        res = subprocess.run(["schtasks", "/run", "/tn", self.args.start_task], capture_output=True, text=True,
                             creationflags=NO_WINDOW)
        if res.returncode != 0:
            return False, "Could not start the task: " + (res.stderr or res.stdout).strip()
        self.resume_asked = time.time()
        return True, "Starting. Loading the checkpoint takes a minute or two."

    def benchmark(self) -> tuple[bool, str]:
        run = self.run_name()
        status = journal.read_status(RUNS_DIR / run) if run else None
        if not run or not status:
            return False, "No run to benchmark."
        err = self.bench.start(run, status)
        return (False, err) if err else (True, f"Benchmark started on {self.bench.games:,} boards.")


def load_token(path: Path) -> str:
    if path.exists() and path.read_text().strip():
        return path.read_text().strip()
    token = secrets.token_urlsafe(18)
    path.write_text(token + "\n")
    return token


def make_handler(app: Dashboard):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MinesweeperDashboard/1"

        def log_message(self, *a) -> None:
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _authorized(self, query: dict) -> bool:
            given = self.headers.get("X-Token") or (query.get("token") or [""])[0]
            return bool(given) and hmac.compare_digest(given.encode(), app.token.encode())

        def do_GET(self) -> None:
            url = urlparse(self.path)
            q = parse_qs(url.query)
            try:
                if url.path == "/":
                    csp = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                           "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'")
                    self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8", {"Content-Security-Policy": csp})
                elif url.path == "/api/state":
                    self._json({**app.state(), "authorized": self._authorized(q)})
                elif url.path == "/api/history":
                    self._json(app.history_for((q.get("range") or ["all"])[0]))
                elif url.path == "/api/log":
                    self._json({"lines": tail(Path(app.args.log) if app.args.log else None)})
                elif url.path in ("/api/download/latest", "/api/download/best"):
                    if not self._authorized(q):
                        return self._json({"ok": False, "message": "token needed"}, 403)
                    run = app.run_name()
                    path = MODELS_DIR / f"{run}-{url.path.rsplit('/', 1)[1]}.pt"
                    if not run or not path.exists():
                        return self._json({"ok": False, "message": "no such checkpoint"}, 404)
                    self._send(200, path.read_bytes(), "application/octet-stream",
                               {"Content-Disposition": f'attachment; filename="{path.name}"'})
                else:
                    self._json({"ok": False, "message": "not found"}, 404)
            except (ConnectionError, TimeoutError):
                pass

        def do_POST(self) -> None:
            url = urlparse(self.path)
            actions = {"/api/pause": app.pause, "/api/resume": app.resume, "/api/benchmark": app.benchmark}
            if url.path not in actions:
                return self._json({"ok": False, "message": "not found"}, 404)
            if not self._authorized({}):
                return self._json({"ok": False, "message": "This needs the dashboard token."}, 403)
            try:
                ok, message = actions[url.path]()
            except OSError as e:
                ok, message = False, str(e)
            self._json({"ok": ok, "message": message}, 200 if ok else 409)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = every network this PC is on")
    ap.add_argument("--run", default="", help="the run to show (default: the newest)")
    ap.add_argument("--log", default="", help="a training log file to show")
    ap.add_argument("--start-task", default="", help="Windows scheduled task that starts training (for Resume)")
    ap.add_argument("--benchmark-games", type=int, default=2000)
    ap.add_argument("--token-file", default=str(ROOT / "dashboard-token.txt"))
    args = ap.parse_args()

    app = Dashboard(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    server.daemon_threads = True
    print(f"dashboard on http://{socket.gethostname()}:{args.port}/?token={app.token}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
