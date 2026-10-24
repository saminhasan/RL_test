"""Queue six safety/opening experiments after the reward screen releases its lock.

    python -m rl.auxiliary_sweep --wait-for-screen
    python -m rl.auxiliary_sweep --only safety_control --clicks 1024 --validation-games 20 --device cpu --study-dir experiments/aux-smoke

The main campaign lock is held throughout training and validation. While waiting
only auxiliary-status.json is updated, preserving the running screen's status.
Global study/STOP is bridged into the current child's model/STOP; removing STOP
and rerunning resumes that child's latest model, with the trainer's empty-replay
resume semantics. Only validation boards are used, including policy-grid tuning.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rl.experiment_eval import sha256, source_hashes, write_json
from rl.reward_sweep import (CampaignLock, ROOT, STUDY, command_line, open_manifest,
                             process_identity, read_json, selected_names, utc_now)

DEFAULT_INIT = STUDY / "screen" / "risk_only-s101-n500000-v2000" / "checkpoints" / "latest.pt"
EXPECTED_INIT_SHA = "8ec3c4a583fe1508f212273824174d908831289b7416f8d5bbbfbc76c542d4ad"


def candidates():
    return {
        "safety_control": {"aux": "none", "opening_lambda": 0.0},
        "zero_representation": {"aux": "zero", "opening_lambda": 0.0},
        "zero_policy": {"aux": "zero", "opening_lambda": 0.01},
        "public_rules": {"aux": "none", "opening_lambda": 0.0, "clue_weight": 1.0, "budget_weight": 0.1},
        "zero_public": {"aux": "zero", "opening_lambda": 0.01, "clue_weight": 1.0, "budget_weight": 0.1},
        "zero_public_strong": {"aux": "zero", "opening_lambda": 0.01, "clue_weight": 10.0, "budget_weight": 1.0},
    }


def policy_grid(config):
    return [0, 0.001, 0.003, 0.01, 0.03, 0.1] if config["aux"] == "zero" else [0]


def training_config(args, name, output):
    return {"output_dir": str(output), "init": str(args.init.resolve()), "seed": args.seed,
            "total_timesteps": args.clicks, "num_envs": 64,
            "eval_every": min(100_000, args.clicks), "eval_games": min(500, args.validation_games),
            "checkpoint_every": min(100_000, args.clicks), "jobs": args.jobs,
            "device": args.device, **candidates()[name]}


def child_command(config, command="train"):
    # Reuse the boolean-safe argument-list builder, without invoking a shell.
    return [sys.executable, "-m", "rl.safety_opening", command, *command_line(config)[3:]]


def complete_result(path, grid, games, *, checkpoint_hash=None, sources=None):
    """The evaluator writes after each lambda; a partial artifact is resumable."""
    result = read_json(path)
    return bool(result and result.get("protocol", {}).get("games") == games
                and result["protocol"].get("split") == "validation"
                and [policy["opening_lambda"] for policy in result.get("policies", [])] == grid
                and (checkpoint_hash is None or result.get("checkpoint", {}).get("sha256") == checkpoint_hash)
                and (sources is None or result["protocol"].get("source_sha256") == sources))


class AuxiliaryRunner:
    def __init__(self, args):
        self.args, self.root = args, args.study_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.stop, self.owns_lock = self.root / "STOP", False
        self.current_output = None
        self.sources = {**source_hashes(), **{name: sha256(ROOT / name) for name in
            ("rl/reward_sweep.py", "rl/auxiliary_sweep.py", "rl/safety_opening.py", "rl/public_constraints.py")}}
        self.init_hash = sha256(args.init)
        if args.init.resolve() == DEFAULT_INIT.resolve() and self.init_hash != EXPECTED_INIT_SHA:
            raise ValueError("frozen pure-risk initialization hash changed")

    def status(self, state, **fields):
        row = {"phase": "auxiliary", "state": state, "updated_utc": utc_now(),
               "pid": os.getpid(), "stop_requested": self.stop.exists(), **fields}
        write_json(self.root / "auxiliary-status.json", row)
        if self.owns_lock:
            write_json(self.root / "status.json", row)

    def sources_unchanged(self):
        if any(sha256(ROOT / name) != digest for name, digest in self.sources.items()) or sha256(self.args.init) != self.init_hash:
            raise RuntimeError("source or frozen initialization changed; start a separate campaign")

    def run_child(self, command, manifest, entry, output, *, training):
        self.sources_unchanged()
        identity = entry.get("child_identity")
        attached = identity is not None and process_identity(entry.get("child_pid")) == identity
        process = None
        if not attached:
            log = (manifest.parent / ("train.log" if training else "evaluate.log")).open("a", encoding="utf-8")
            log.write("\n" + json.dumps(command) + "\n")
            log.flush()
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
            log.close()
            identity = process_identity(process.pid)
            entry.update(child_pid=process.pid, child_identity=identity, command=command)
            entry.setdefault("commands", []).append({"utc": utc_now(), "argv": command})
            write_json(manifest, entry)
        while process.poll() is None if process is not None else (
                identity is not None and process_identity(entry.get("child_pid")) == identity):
            if self.stop.exists() and training:
                (output / "STOP").touch()
            self.status("stopping" if self.stop.exists() else "training" if training else "evaluating",
                        variant=entry["protocol"]["name"], child_pid=entry.get("child_pid"),
                        progress=read_json(output / "status.json"), manifest=str(manifest))
            time.sleep(2)
        entry.update(child_pid=None, child_identity=None)
        write_json(manifest, entry)
        if process is not None and process.returncode:
            raise RuntimeError(f"child exited {process.returncode}; inspect {manifest.parent}")

    def case(self, name):
        directory = self.root / "auxiliary" / f"{name}-s{self.args.seed}-n{self.args.clicks}-v{self.args.validation_games}"
        output, manifest = directory / "model", directory / "manifest.json"
        self.current_output = output
        config = training_config(self.args, name, output)
        protocol = {"schema": 1, "name": name, "config": config, "source_sha256": self.sources,
                    "init_sha256": self.init_hash, "split": "validation", "games": self.args.validation_games,
                    "grid": policy_grid(config)}
        entry = open_manifest(manifest, protocol)
        if entry["state"] == "complete" and all(Path(path).exists() for path in entry["results"]):
            # Recheck the current source models below before skipping children:
            # a manually resumed model may have changed after completion.
            print(f"recheck completed auxiliary {name}", flush=True)
        print(f"auxiliary: {name}", flush=True)
        output.mkdir(parents=True, exist_ok=True)
        # If a prior runner died, finish monitoring its exact recorded child.
        identity = entry.get("child_identity")
        if identity is not None and process_identity(entry.get("child_pid")) == identity:
            self.run_child(entry["command"], manifest, entry, output, training=entry.get("state") == "training")
        status = read_json(output / "status.json") or {}
        if status.get("step", 0) < self.args.clicks:
            if self.stop.exists():
                return
            (output / "STOP").unlink(missing_ok=True)
            actual = dict(config)
            if (output / "latest.pt").exists():
                actual.pop("init")
                actual["resume"] = str(output / "latest.pt")
            entry["state"] = "training"
            write_json(manifest, entry)
            self.run_child(child_command(actual), manifest, entry, output, training=True)
            status = read_json(output / "status.json") or {}
        if self.stop.exists():
            entry["state"] = "paused"
            write_json(manifest, entry)
            return
        if status.get("step", 0) < self.args.clicks:
            raise RuntimeError(f"{name} stopped below its click budget; rerun to resume")
        results = {}
        eval_sources = {name: digest for name, digest in self.sources.items()
                        if name not in ("rl/reward_sweep.py", "rl/auxiliary_sweep.py")}
        for label in ("best", "latest"):
            artifact = directory / f"{label}-validation.json"
            source, checkpoint = output / f"{label}.pt", directory / "checkpoints" / f"{label}.pt"
            model_hash = sha256(source)
            if not checkpoint.exists() or sha256(checkpoint) != model_hash:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint.with_suffix(".tmp")
                shutil.copyfile(source, temporary)
                os.replace(temporary, checkpoint)
                if sha256(checkpoint) != model_hash or sha256(source) != model_hash:
                    raise RuntimeError(f"checkpoint changed while freezing it: {source}")
            complete = lambda: complete_result(artifact, policy_grid(config), self.args.validation_games,
                                                checkpoint_hash=model_hash, sources=eval_sources)
            if not complete():
                command = child_command({"model": str(checkpoint), "output": str(artifact), "games": self.args.validation_games,
                    "split": "validation", "cache_dir": str(self.root / "bayes-cache"), "device": self.args.device,
                    "jobs": self.args.jobs}, "evaluate") + ["--grid", *map(str, policy_grid(config))]
                entry["state"] = "evaluating"
                write_json(manifest, entry)
                self.run_child(command, manifest, entry, output, training=False)
            if not complete():
                raise RuntimeError(f"incomplete validation result: {artifact}")
            results[str(artifact)] = sha256(artifact)
            if self.stop.exists():
                return
        entry.update(state="complete", results=results, completed_utc=utc_now(),
                     checkpoint_sha256={path.name: sha256(path) for path in (directory / "checkpoints").glob("*.pt")})
        write_json(manifest, entry)
        self.status("complete", variant=name, results=results)

    def run(self):
        lock = CampaignLock(self.root / "runner.lock", "auxiliary")
        while not self.stop.exists():
            try:
                lock.__enter__()
                break
            except RuntimeError as error:
                if not self.args.wait_for_screen:
                    raise
                print(f"waiting for screen: {error}", flush=True)
                self.status("waiting_for_screen", owner=read_json(self.root / "runner.lock"))
                for _ in range(45):
                    if self.stop.exists():
                        self.status("stopped_while_waiting")
                        return
                    time.sleep(1)
        else:
            self.status("stopped")
            return
        self.owns_lock = True
        try:
            for name in selected_names(candidates(), self.args.only):
                if self.stop.exists():
                    break
                self.case(name)
        except KeyboardInterrupt:
            self.stop.touch()
            if self.current_output is not None:
                (self.current_output / "STOP").touch()
            raise
        finally:
            lock.__exit__(None, None, None)
            self.owns_lock = False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study-dir", type=Path, default=STUDY)
    parser.add_argument("--init", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--clicks", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--validation-games", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--wait-for-screen", action="store_true")
    args = parser.parse_args()
    try:
        selected_names(candidates(), args.only)
    except ValueError as error:
        parser.error(str(error))
    if args.clicks < 1 or not 1 <= args.validation_games <= 1_000_000 or args.jobs < 1:
        parser.error("clicks/games/jobs must be positive; games at most one million")
    AuxiliaryRunner(args).run()


if __name__ == "__main__":
    main()
