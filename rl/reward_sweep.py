"""Persistent, sequential reward experiments on reserved validation boards.

    python -m rl.reward_sweep --phase inference
    python -m rl.reward_sweep --phase screen
    python -m rl.reward_sweep --phase screen --only risk_only --clicks 1024 --validation-games 20

The screen warm-starts every candidate from the same frozen checkpoint. Each
variant records its source/config/checkpoint hashes, exact child commands, logs,
and paired validation results. Completed variants are skipped on restart.
Interrupted training resumes its own latest checkpoint; the trainer restarts
replay and RNG streams on resume, so this is recorded as a distinct session,
not advertised as bitwise equivalent to uninterrupted training.

Create experiments/reward-study-20261024/STOP to finish the current evaluation
or cleanly stop the current training child and launch no further jobs. Remove
STOP before resuming. The original user run is never selected or stopped.
No test boards are opened here. Confirmation across fresh training seeds is a
separate phase, selected after inspecting the screening results.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from rl.experiment_eval import evaluate, sha256, source_hashes, write_json

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR, MODELS_DIR = ROOT / "runs", ROOT / "models"
STUDY = ROOT / "experiments" / "reward-study-20261024"
INIT = MODELS_DIR / "reward-study-20261024-baseline-best.pt"
VALIDATION_SEED0 = 1_108_081_562_368
STUDY_NAME = "reward-study-20261024"


def screen_candidates() -> dict[str, dict]:
    return {
        "old_precision_control": {"fp32_head": False, "reward_reveal": 0.01},
        "precise_ratio100": {"reward_reveal": 0.01},
        "risk_only": {"reward_reveal": 0.0},
        "ratio10": {"reward_reveal": 0.1},
        "ratio1000": {"reward_reveal": 0.001},
        "reveal_progress": {"reward_reveal": 0.01, "reward_reveal_scale": 1.0,
                            "reward_reveal_scale_start": 0.0},
        "late2": {"reward_late": 0.01, "late_power": 2.0},
        "late10": {"reward_late": 0.01, "late_power": 10.0},
        "burst2": {"reward_burst": 0.01, "burst_rate": 2.0},
        "burst5": {"reward_burst": 0.01, "burst_rate": 5.0},
        "future_win1": {"reward_win": 1.0, "gamma": 1.0, "n_step": 1},
        "future_win3": {"reward_win": 1.0, "gamma": 1.0, "n_step": 3},
        "future_win_discount": {"reward_win": 1.0, "gamma": 0.999, "n_step": 3},
    }


def inference_variants() -> dict[str, dict]:
    return {
        "bf16": {},
        "fp32_head": {"fp32_head": True},
        "fp32": {"no_bf16": True},
        "symmetry_bf16": {"symmetries": True},
        "symmetry_fp32": {"symmetries": True, "no_bf16": True},
    }


def selected_names(available: dict, only: list[str] | None) -> list[str]:
    if only is None:
        return list(available)
    unknown = set(only) - set(available)
    if unknown:
        raise ValueError("unknown variant(s): " + ", ".join(sorted(unknown)))
    if len(set(only)) != len(only):
        raise ValueError("--only contains duplicate variants")
    return only


def training_config(name: str, *, clicks: int = 500_000, init: Path = INIT,
                    validation_games: int = 2000) -> dict:
    candidates = screen_candidates()
    if name not in candidates:
        raise ValueError(f"unknown candidate: {name}")
    if clicks < 1:
        raise ValueError("clicks must be positive")
    return {
        "exp_name": f"{STUDY_NAME}-{name}-s101-n{clicks}",
        "seed": 101, "init": str(init.resolve()), "curriculum": "2",
        "total_timesteps": clicks, "num_envs": 64, "channels": 64, "layers": 16,
        "n_step": 1, "per_alpha": 0.0, "buffer_size": 1_000_000,
        "loss": "bce", "reward_mine": 1.0, "reward_win": 0.0,
        "reward_progress": 0.0, "reward_reveal": 0.0,
        "gamma": 0.0, "wall": 0.0, "augment": True, "ema": 0.999,
        "deterministic": True, "fp32_head": True, "learning_rate": 3e-5,
        "lr_mode": "exp", "lr_floor": 1e-5, "lr_patience": 0,
        "start_e": 0.02, "end_e": 0.005, "exploration_fraction": 0.1,
        "learning_starts": 10_000, "eval_every": min(100_000, clicks),
        "eval_games": min(500, validation_games),
        "eval_breakdown": True, "checkpoint_every": min(100_000, clicks),
        "eval_seed0": VALIDATION_SEED0,
        **candidates[name],
    }


def command_line(config: dict, python: str = sys.executable) -> list[str]:
    """Argument lists avoid shell quoting and preserve explicit boolean settings."""
    command = [python, "-m", "rl.dqn"]
    for key, value in config.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            command.append(flag if value else "--no-" + key.replace("_", "-"))
        else:
            command.extend((flag, str(value)))
    return command


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def campaign_exp_name(config: dict, study_dir: Path, sources: dict, init_hash: str) -> str:
    """Different campaigns/sources cannot accidentally adopt each other's runs."""
    identity = {"study_dir": str(study_dir.resolve()), "source_sha256": sources,
                "init_sha256": init_hash, "training_config": config}
    return config["exp_name"] + "-p" + fingerprint(identity)[:12]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict | None:
    for attempt in range(10):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (PermissionError, json.JSONDecodeError):
            if attempt == 9:
                raise
            time.sleep(0.1)
    return None


def open_manifest(path: Path, protocol: dict) -> dict:
    existing = read_json(path)
    expected = fingerprint(protocol)
    if existing:
        if existing["protocol_sha256"] != expected:
            raise ValueError(f"protocol changed for {path}; use a new study directory rather than mixing results")
        return existing
    entry = {"protocol": protocol, "protocol_sha256": expected,
             "state": "pending", "created_utc": utc_now(), "sessions": []}
    write_json(path, entry)
    return entry


def discover_run(exp_name: str, runs_dir: Path = RUNS_DIR) -> str | None:
    pattern = re.compile(re.escape(exp_name) + r"-c64x16-\d{8}-\d{6}")
    names = [p.name for p in runs_dir.glob(exp_name + "-c64x16-*")
             if p.is_dir() and pattern.fullmatch(p.name)]
    if len(names) > 1:
        raise ValueError(f"multiple candidate runs for {exp_name}; refusing ambiguous resume")
    return names[0] if names else None


def process_identity(pid: int | None):
    """Creation identity prevents an unrelated process reusing a stale PID."""
    if not pid:
        return None
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.GetProcessTimes.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
                                          ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
                                          ctypes.POINTER(wintypes.FILETIME))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return None
        try:
            exit_code = wintypes.DWORD()
            created, exited = wintypes.FILETIME(), wintypes.FILETIME()
            kernel_time, user_time = wintypes.FILETIME(), wintypes.FILETIME()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)) or exit_code.value != 259:
                return None
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                          ctypes.byref(kernel_time), ctypes.byref(user_time)):
                return None
            return [int(pid), created.dwHighDateTime, created.dwLowDateTime]
        finally:
            kernel.CloseHandle(handle)
    try:
        # Linux process start ticks, after the possibly space-containing name.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else [int(pid), fields[19]]
    except (OSError, IndexError):
        return None


def request_stop(exp_name: str) -> str | None:
    run = discover_run(exp_name)
    if run:
        (MODELS_DIR / f"{run}.stop").touch()
    return run


class CampaignLock:
    """Atomic first creation plus a held OS lock makes stale recovery race safe.

    The released file is retained: unlinking a Windows file after closing its
    handle can race with the next owner acquiring it. An owner creation identity
    distinguishes crashed runners from a later process with a reused PID.
    """

    def __init__(self, path: Path, phase: str):
        self.path, self.phase, self.fd = path, phase, None
        self.owner = None

    @staticmethod
    def lock_file(fd: int, release: bool = False) -> None:
        if os.name == "nt":
            import msvcrt
            # Windows byte locks are mandatory even for read-only observers.
            # Reserve a byte beyond the short JSON payload so PID/status reads
            # remain possible while the campaign is locked.
            os.lseek(fd, 4096, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK if release else msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN if release else fcntl.LOCK_EX | fcntl.LOCK_NB)

    def read_owner(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def write_owner(self) -> None:
        raw = (json.dumps(self.owner, sort_keys=True) + "\n").encode()
        os.lseek(self.fd, 0, os.SEEK_SET)
        os.ftruncate(self.fd, 0)
        os.write(self.fd, raw)
        os.fsync(self.fd)

    def occupied(self, owner: dict) -> RuntimeError:
        return RuntimeError(
            f"campaign runner already active (PID {owner.get('pid', 'initializing')}, "
            f"phase {owner.get('phase', 'unknown')}) at {self.path.parent}; "
            "inspect status.json or create STOP to stop it, then rerun after it exits")

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError:
            fd = os.open(self.path, os.O_RDWR)
        try:
            try:
                self.lock_file(fd)
            except OSError:
                raise self.occupied(self.read_owner()) from None
            owner = self.read_owner()
            identity = owner.get("process_identity")
            if not owner.get("released") and identity and process_identity(owner.get("pid")) == identity:
                self.lock_file(fd, release=True)
                raise self.occupied(owner)
            self.fd = fd
            identity = process_identity(os.getpid())
            if identity is None:
                raise RuntimeError("could not identify this runner process; refusing an unowned campaign lock")
            self.owner = {"pid": os.getpid(), "process_identity": identity,
                          "phase": self.phase, "created_utc": utc_now(), "released": False}
            self.write_owner()
            return self
        except BaseException:
            os.close(fd)
            self.fd = None
            raise

    def __exit__(self, *exception):
        if self.fd is not None:
            try:
                self.owner.update(released=True, released_utc=utc_now())
                self.write_owner()
            finally:
                self.lock_file(self.fd, release=True)
                os.close(self.fd)
                self.fd = None


class StudyRunner:
    def __init__(self, args):
        self.args = args
        self.root = args.study_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.stop = self.root / "STOP"
        self.sources = {**source_hashes(), "rl/reward_sweep.py": sha256(Path(__file__))}
        self.init_hash = sha256(args.init)
        self.current_exp = None

    def update(self, path: Path, entry: dict, state: str, **fields) -> None:
        entry.update(state=state, updated_utc=utc_now(), **fields)
        write_json(path, entry)
        write_json(self.root / "status.json", {
            "phase": self.args.phase, "variant": entry["protocol"]["name"],
            "state": state, "updated_utc": entry["updated_utc"],
            "manifest": str(path), "protocol_sha256": entry["protocol_sha256"],
            "run": entry.get("run"), "progress": entry.get("progress"),
            "child_pid": entry.get("child_pid"), "result": entry.get("result"),
            "stop_requested": self.stop.exists(),
        })

    def protocol(self, name: str, config: dict) -> dict:
        return {"schema": 1, "phase": self.args.phase, "name": name,
                "source_sha256": self.sources,
                "init": str(self.args.init.resolve()), "init_sha256": self.init_hash,
                "config": config, "split": "validation", "level": 2,
                "validation_games": self.args.validation_games,
                "python": str(Path(self.args.python).resolve())}

    def check_sources(self) -> None:
        current = {**source_hashes(), "rl/reward_sweep.py": sha256(Path(__file__))}
        if current != self.sources or sha256(self.args.init) != self.init_hash:
            raise RuntimeError("source or frozen initialization changed during the study; stopped before the next job")

    def evaluate(self, models: list[Path], output: Path, **settings) -> dict:
        self.check_sources()
        args = SimpleNamespace(
            model=[str(p) for p in models], split="validation", level=2,
            games=self.args.validation_games, replicate=0, output=output,
            cache_dir=self.root / "bayes-cache", device=self.args.device,
            n_envs=256, jobs=self.args.jobs, no_breakdown=False,
            no_bf16=False, symmetries=False, fp32_head=False,
            bootstrap_samples=20_000, statistics_seed=20261024,
        )
        for key, value in settings.items():
            setattr(args, key, value)
        return evaluate(args)

    def inference(self, name: str, settings: dict) -> None:
        directory = self.root / "inference" / f"{name}-n{self.args.validation_games}"
        path = directory / "manifest.json"
        entry = open_manifest(path, self.protocol(name, settings))
        if entry["state"] == "complete" and Path(entry["result"]).exists():
            print(f"skip completed inference {name}", flush=True)
            return
        self.update(path, entry, "evaluating")
        output = directory / "result.json"
        self.evaluate([self.args.init], output, **settings)
        self.update(path, entry, "complete", result=str(output), result_sha256=sha256(output))

    def monitor(self, path: Path, entry: dict, config: dict, process=None) -> None:
        identity = entry.get("child_identity")
        while process.poll() is None if process is not None else (
                identity is not None and process_identity(entry.get("child_pid")) == identity):
            run = discover_run(config["exp_name"])
            status = read_json(RUNS_DIR / run / "status.json") if run else None
            if self.stop.exists():
                request_stop(config["exp_name"])
            self.update(path, entry, "stopping" if self.stop.exists() else "training",
                        run=run, progress=status)
            time.sleep(self.args.poll_seconds)
        if process is not None and process.returncode != 0:
            self.update(path, entry, "failed", returncode=process.returncode)
            raise RuntimeError(f"training exited {process.returncode}; see {path.parent / 'train.log'}")

    def screen(self, name: str, changes: dict) -> None:
        config = training_config(name, clicks=self.args.clicks, init=self.args.init,
                                 validation_games=self.args.validation_games)
        config["exp_name"] = campaign_exp_name(config, self.root, self.sources, self.init_hash)
        directory = self.root / "screen" / f"{name}-s101-n{self.args.clicks}-v{self.args.validation_games}"
        path = directory / "manifest.json"
        entry = open_manifest(path, self.protocol(name, config))
        if entry["state"] == "complete" and Path(entry["result"]).exists():
            print(f"skip completed candidate {name}", flush=True)
            return
        self.current_exp = config["exp_name"]
        run = discover_run(config["exp_name"])
        status = read_json(RUNS_DIR / run / "status.json") if run else None
        if entry.get("child_identity") and process_identity(entry.get("child_pid")) == entry["child_identity"]:
            print(f"monitor existing child {entry['child_pid']} for {name}", flush=True)
            self.monitor(path, entry, config)
            run = discover_run(config["exp_name"])
            status = read_json(RUNS_DIR / run / "status.json") if run else None
        if not status or status.get("step", 0) < self.args.clicks:
            if self.stop.exists():
                self.update(path, entry, "paused", run=run, progress=status)
                return
            self.check_sources()
            actual = dict(config)
            latest = MODELS_DIR / f"{run}-latest.pt" if run else None
            if latest and latest.exists():
                actual.pop("init")
                actual["resume"] = str(latest)
            elif run:
                raise RuntimeError(f"{run} exists without a resumable checkpoint; refusing a duplicate run")
            command = command_line(actual, self.args.python)
            entry["sessions"].append({"started_utc": utc_now(), "command": command,
                                      "resumed": "resume" in actual})
            self.update(path, entry, "starting", run=run)
            with (directory / "train.log").open("a", encoding="utf-8") as log:
                log.write("\n" + json.dumps(entry["sessions"][-1]) + "\n")
                log.flush()
                env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT),
                       "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.update(path, entry, "training", child_pid=process.pid,
                            child_identity=process_identity(process.pid))
                self.monitor(path, entry, config, process)
            run = discover_run(config["exp_name"])
            status = read_json(RUNS_DIR / run / "status.json") if run else None
        if not status or status.get("step", 0) < self.args.clicks:
            self.update(path, entry, "paused", run=run, progress=status)
            if not self.stop.exists():
                raise RuntimeError(f"{name} stopped before its budget; rerun the same command to resume")
            return
        if self.stop.exists():
            self.update(path, entry, "trained", run=run, progress=status)
            return
        self.update(path, entry, "evaluating", run=run, progress=status)
        snapshots = []
        for label in ("best", "latest"):
            source = MODELS_DIR / f"{run}-{label}.pt"
            if not source.exists():
                raise RuntimeError(f"missing {source}")
            destination = directory / "checkpoints" / f"{label}.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp = destination.with_suffix(".tmp")
            shutil.copyfile(source, temp)
            os.replace(temp, destination)
            snapshots.append(destination)
        output = directory / "result.json"
        self.evaluate(snapshots, output)
        self.update(path, entry, "complete", result=str(output), result_sha256=sha256(output),
                    checkpoint_sha256={p.name: sha256(p) for p in snapshots})
        self.current_exp = None

    def _run_locked(self) -> None:
        variants = inference_variants() if self.args.phase == "inference" else screen_candidates()
        names = selected_names(variants, self.args.only)
        try:
            for name in names:
                if self.stop.exists():
                    print(f"STOP exists at {self.stop}; no further jobs started", flush=True)
                    break
                print(f"{self.args.phase}: {name}", flush=True)
                if self.args.phase == "inference":
                    self.inference(name, variants[name])
                else:
                    self.screen(name, variants[name])
        except KeyboardInterrupt:
            self.stop.touch()
            if self.current_exp:
                request_stop(self.current_exp)
            raise

    def run(self) -> None:
        with CampaignLock(self.root / "runner.lock", self.args.phase):
            self._run_locked()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("inference", "screen"), required=True)
    parser.add_argument("--only", nargs="+", help="run only these named variants in the selected phase")
    parser.add_argument("--study-dir", type=Path, default=STUDY)
    parser.add_argument("--init", type=Path, default=INIT)
    parser.add_argument("--clicks", type=int, default=500_000)
    parser.add_argument("--validation-games", type=int, default=2000)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="auto", help="evaluation device; training uses its default CUDA device")
    parser.add_argument("--jobs", type=int, default=8, help="Bayes evaluation worker processes")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    variants = inference_variants() if args.phase == "inference" else screen_candidates()
    try:
        selected_names(variants, args.only)
    except ValueError as error:
        parser.error(str(error))
    if args.clicks < 1 or not 1 <= args.validation_games <= 1_000_000 or args.jobs < 1 or args.poll_seconds <= 0:
        parser.error("clicks, games, jobs and poll-seconds must be positive (games at most one million)")
    if not args.init.is_file():
        parser.error(f"frozen initialization does not exist: {args.init}")
    StudyRunner(args).run()


if __name__ == "__main__":
    main()
