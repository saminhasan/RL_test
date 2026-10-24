"""Freeze one validation-selected recipe, repeat training, then open test boards.

    python -m rl.study_io confirm --wait-for-sweeps

Selection uses only the 13 main and six auxiliary campaign manifests. Three
fresh-seed repeats share the selected recipe's frozen pretrained initialization;
this tests warm-start repeatability, not training from scratch. All repeats are
trained and checkpoint hashes sealed before any test evaluation starts. The
selected policy, checkpoint rule and inference settings never change on test.
Global study/STOP stops this stage; its own status is confirmation-status.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from rl.auxiliary_sweep import candidates as auxiliary_candidates
from rl.experiment_eval import board_seeds, sha256, source_hashes, write_json
from rl.reward_sweep import (CampaignLock, MODELS_DIR, ROOT, RUNS_DIR, STUDY, command_line,
    discover_run, fingerprint, open_manifest, process_identity, read_json, screen_candidates, utc_now)


def choose_winner(options):
    if not options:
        raise ValueError("empty validation selection pool")
    if any(option["split"] != "validation" for option in options):
        raise ValueError("selection may use validation only")
    ordered = sorted(options, key=lambda option: option["identity"])
    return max(ordered, key=lambda option: (option["summary"]["win"], option["summary"].get("efficiency") or -1))


def write_selection_plan(path, payload):
    existing = read_json(path)
    digest = fingerprint(payload)
    if existing:
        if existing["plan_sha256"] != fingerprint({k: v for k, v in existing.items() if k != "plan_sha256"}):
            raise ValueError("selection plan seal is invalid")
        if existing["plan_sha256"] != digest:
            raise ValueError("selection plan is already frozen; do not retune after selection")
        return existing
    result = {**payload, "plan_sha256": digest}
    write_json(path, result)
    return result


def success_criterion(rows, seeds):
    """Every one of at least three predeclared repeats must beat Bayes."""
    indexed = {row["seed"]: row for row in rows}
    return (len(seeds) >= 3 and len(indexed) == len(rows) == len(seeds) and set(indexed) == set(seeds)
            and all(row["paired_vs_bayes"]["win_difference"] > 0 and
                    row["paired_vs_bayes"]["win_difference_ci95"][0] > 0 for row in rows))


def seal_training(path, plan, frozen):
    if len(frozen) != len(plan["seeds"]) or {row["seed"] for row in frozen} != set(plan["seeds"]):
        raise ValueError("all predeclared repeats must finish before the test split opens")
    if any(sha256(Path(row["path"])) != row["sha256"] for row in frozen):
        raise ValueError("a trained checkpoint changed before the training seal")
    return write_selection_plan(path, {"selection_plan_sha256": plan["plan_sha256"], "checkpoints": frozen})


def valid_test_result(result, frozen, plan):
    if not result:
        return False
    q, expected = plan["winner"]["kind"] == "q", board_seeds("test", 2, plan["test_games"])
    protocol = result.get("protocol", {})
    if any(protocol.get(key) != value for key, value in {"split": "test", "level": 2,
           "games": plan["test_games"], "replicate": 0, "requested_seed_start": expected.start,
           "requested_seed_stop": expected.stop}.items()):
        return False
    if not protocol.get("source_sha256") or any(plan["source_sha256"].get(name) != digest
            for name, digest in protocol["source_sha256"].items()):
        return False
    policies = result.get("models" if q else "policies", [])
    if len(policies) != 1 or len(policies[0].get("records", [])) != plan["test_games"]:
        return False
    policy, flags = policies[0], plan["winner"]["inference"]
    digest = policy.get("sha256") if q else result.get("checkpoint", {}).get("sha256")
    if digest != frozen["sha256"]:
        return False
    if flags.get("no_bf16") and result.get("runtime", {}).get("bf16"):
        return False
    return (result["runtime"].get("symmetries") == flags.get("symmetries", False) and
            result["runtime"].get("fp32_head_override") == flags.get("fp32_head", False)
            if q else policy.get("opening_lambda") == flags["opening_lambda"])


def source_inventory():
    return {**source_hashes(), **{name: sha256(ROOT / name) for name in
        ("rl/reward_sweep.py", "rl/auxiliary_sweep.py", "rl/safety_opening.py",
         "rl/public_constraints.py", "rl/confirm_study.py", "rl/study_io.py")}}


def selection_pool(root):
    options, inputs, completed = [], {}, 0
    for family, names in (("screen", screen_candidates()), ("auxiliary", auxiliary_candidates())):
        for name in names:
            paths = list((root / family).glob(f"{name}-s*-n500000-v2000/manifest.json"))
            if len(paths) > 1:
                raise ValueError(f"ambiguous screening manifests for {family}/{name}")
            if not paths or (entry := read_json(paths[0]))["state"] != "complete":
                continue
            completed += 1
            manifest = paths[0]
            protocol = entry["protocol"]
            if fingerprint(protocol) != entry["protocol_sha256"]:
                raise ValueError(f"screening manifest seal mismatch: {manifest}")
            inputs[str(manifest)] = sha256(manifest)
            artifacts = [Path(entry["result"])] if family == "screen" else [Path(path) for path in entry["results"]]
            for artifact in artifacts:
                result = read_json(artifact)
                expected_hash = entry["result_sha256"] if family == "screen" else entry["results"][str(artifact)]
                if sha256(artifact) != expected_hash:
                    raise ValueError(f"screening result hash mismatch: {artifact}")
                if result["protocol"]["split"] != "validation" or result["protocol"]["games"] != 2000:
                    raise ValueError(f"selection artifact is not the fixed validation protocol: {artifact}")
                if any(protocol["source_sha256"].get(name) != digest or sha256(ROOT / name) != digest
                       for name, digest in result["protocol"]["source_sha256"].items()):
                    raise ValueError(f"validation source provenance mismatch: {artifact}")
                inputs[str(artifact)] = sha256(artifact)
                policies = result["models"] if family == "screen" else result["policies"]
                for policy in policies:
                    label = Path(policy["path"] if family == "screen" else result["checkpoint"]["path"]).stem
                    if label not in ("best", "latest"):
                        raise ValueError("only best/latest checkpoints belong in confirmation selection")
                    model_path = Path(policy["path"] if family == "screen" else result["checkpoint"]["path"])
                    model_hash = policy["sha256"] if family == "screen" else result["checkpoint"]["sha256"]
                    if sha256(model_path) != model_hash:
                        raise ValueError(f"validation checkpoint changed: {model_path}")
                    runtime = result["runtime"]
                    inference = ({"no_bf16": not runtime["bf16"], "symmetries": runtime["symmetries"],
                                  "fp32_head": runtime["fp32_head_override"]} if family == "screen" else
                                 {"opening_lambda": policy["opening_lambda"], "no_bf16": not runtime["bf16"]})
                    options.append({"identity": f"{family}/{name}/{label}/{inference}", "name": name,
                        "kind": "q" if family == "screen" else "aux", "checkpoint_label": label,
                        "split": "validation", "summary": policy["summary"], "inference": inference,
                        "config": protocol["config"], "init_sha256": protocol["init_sha256"],
                        "manifest": str(manifest), "artifact": str(artifact)})
    return (options if completed == 19 else None), inputs, completed


class Confirmation:
    def __init__(self, args):
        self.args, self.root = args, args.study_dir.resolve()
        self.directory = self.root / "confirmation"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stop, self.locked = self.root / "STOP", False
        self.current, self.plan = None, None

    def status(self, state, **fields):
        row = {"phase": "confirmation", "state": state, "pid": os.getpid(), "updated_utc": utc_now(), **fields}
        write_json(self.root / "confirmation-status.json", row)
        if self.locked:
            write_json(self.root / "status.json", row)

    def check(self):
        if source_inventory() != self.plan["source_sha256"]:
            raise RuntimeError("confirmation sources changed after the plan was frozen")
        winner = self.plan["winner"]
        if sha256(Path(winner["config"]["init"])) != winner["init_sha256"]:
            raise RuntimeError("shared initialization changed after selection")
        sealed = read_json(self.directory / "selection-plan.json")
        write_selection_plan(self.directory / "selection-plan.json", {k: v for k, v in self.plan.items() if k != "plan_sha256"})
        if sealed != self.plan:
            raise RuntimeError("selection plan changed")

    def child(self, command, manifest, entry, train_output=None, q_exp=None):
        self.check()
        identity, process = entry.get("child_identity"), None
        if identity is None or process_identity(entry.get("child_pid")) != identity:
            with (manifest.parent / "process.log").open("a", encoding="utf-8") as log:
                log.write("\n" + json.dumps(command) + "\n")
                log.flush()
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), env={**os.environ,
                    "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
            identity = process_identity(process.pid)
            entry.update(child_pid=process.pid, child_identity=identity, command=command)
            entry.setdefault("commands", []).append(command)
            write_json(manifest, entry)
        while process.poll() is None if process is not None else (
                identity is not None and process_identity(entry.get("child_pid")) == identity):
            if self.stop.exists():
                if train_output:
                    (train_output / "STOP").touch()
                if q_exp and (run := discover_run(q_exp)):
                    (MODELS_DIR / f"{run}.stop").touch()
            self.status(entry["state"], seed=entry["protocol"]["seed"], child_pid=entry.get("child_pid"))
            time.sleep(3)
        entry.update(child_pid=None, child_identity=None)
        write_json(manifest, entry)
        if process is not None and process.returncode:
            raise RuntimeError(f"confirmation child exited {process.returncode}; see {manifest.parent}")

    def training(self, seed):
        winner, budget = self.plan["winner"], self.plan["training_clicks"]
        directory = self.directory / f"seed-{seed}"
        config = {**winner["config"], "seed": seed, "total_timesteps": budget}
        config["eval_every"] = min(config.get("eval_every", 100000), budget)
        config["checkpoint_every"] = min(config.get("checkpoint_every", 100000), budget)
        q = winner["kind"] == "q"
        exp = f"reward-confirm-{self.plan['plan_sha256'][:12]}-s{seed}"
        if q:
            config.update(exp_name=exp, cuda=self.args.device != "cpu")
        else:
            config.update(output_dir=str(directory / "model"), device=self.args.device)
        manifest = directory / "manifest.json"
        entry = open_manifest(manifest, {"plan_sha256": self.plan["plan_sha256"], "seed": seed, "config": config})
        self.current = (directory / "model", exp if q else None)
        if entry.get("child_identity") and process_identity(entry.get("child_pid")) == entry["child_identity"]:
            self.child(entry["command"], manifest, entry, None if q else directory / "model", exp if q else None)
        run = discover_run(exp) if q else None
        output = RUNS_DIR / run if run else directory / "model"
        progress = read_json(output / "status.json") or {}
        if progress.get("step", 0) < budget:
            if self.stop.exists():
                return None
            actual = dict(config)
            latest = MODELS_DIR / f"{run}-latest.pt" if q and run else output / "latest.pt"
            if latest.exists():
                actual.pop("init", None)
                actual["resume"] = str(latest)
            if not q:
                output.mkdir(parents=True, exist_ok=True)
                (output / "STOP").unlink(missing_ok=True)
            command = command_line(actual) if q else [sys.executable, "-m", "rl.study_io", "opening", "train", *command_line(actual)[3:]]
            entry["state"] = "training"
            write_json(manifest, entry)
            self.child(command, manifest, entry, None if q else output, exp if q else None)
            run = discover_run(exp) if q else None
            output = RUNS_DIR / run if run else output
            progress = read_json(output / "status.json") or {}
        if self.stop.exists():
            return None
        if progress.get("step", 0) < budget:
            raise RuntimeError("repeat stopped below its training budget")
        source = MODELS_DIR / f"{run}-{winner['checkpoint_label']}.pt" if q else output / f"{winner['checkpoint_label']}.pt"
        snapshot = directory / "selected.pt"
        temporary = directory / "selected.tmp"
        shutil.copyfile(source, temporary)
        os.replace(temporary, snapshot)
        entry.update(state="trained", checkpoint=str(snapshot), checkpoint_sha256=sha256(snapshot))
        write_json(manifest, entry)
        return {"seed": seed, "path": str(snapshot), "sha256": entry["checkpoint_sha256"]}

    def test(self, frozen):
        winner = self.plan["winner"]
        directory = self.directory / f"seed-{frozen['seed']}"
        manifest, artifact = directory / "manifest.json", directory / "test.json"
        entry = read_json(manifest)
        seal = read_json(self.directory / "training-complete.json")
        if not seal or seal.get("selection_plan_sha256") != self.plan["plan_sha256"]:
            raise RuntimeError("test split remains closed until all repeats have a matching training seal")
        seal_training(self.directory / "training-complete.json", self.plan, seal["checkpoints"])
        if sha256(Path(frozen["path"])) != frozen["sha256"]:
            raise RuntimeError("a sealed repeat checkpoint changed")
        result = read_json(artifact)
        key = "models" if winner["kind"] == "q" else "policies"
        if not valid_test_result(result, frozen, self.plan):
            base = [sys.executable, "-m", "rl.study_io", "evaluate" if key == "models" else "opening"]
            if key == "policies":
                base.append("evaluate")
            command = base + ["--model", frozen["path"], "--output", str(artifact), "--split", "test",
                "--games", str(self.plan["test_games"]), "--replicate", "0", "--level", "2",
                "--cache-dir", str(self.root / "bayes-cache"), "--device", self.args.device, "--jobs", str(self.args.jobs)]
            flags = winner["inference"]
            command += ["--" + flag.replace("_", "-") for flag in ("no_bf16", "symmetries", "fp32_head") if flags.get(flag)]
            if key == "policies":
                command += ["--grid", str(flags["opening_lambda"])]
            entry["state"] = "testing"
            write_json(manifest, entry)
            self.child(command, manifest, entry)
            result = read_json(artifact)
        if not valid_test_result(result, frozen, self.plan):
            raise RuntimeError("test artifact does not match the sealed protocol/checkpoint")
        policy = result[key][0]
        return {"seed": frozen["seed"], "summary": policy["summary"], "paired_vs_bayes": policy["paired_vs_bayes"],
                "bayes": result["bayes"]["summary"], "artifact": str(artifact), "sha256": sha256(artifact)}

    def run(self):
        lock = CampaignLock(self.root / "runner.lock", "confirmation")
        while not self.stop.exists():
            pool, inputs, count = selection_pool(self.root)
            if pool:
                try:
                    lock.__enter__()
                    break
                except RuntimeError:
                    pass
            if not self.args.wait_for_sweeps:
                raise RuntimeError(f"waiting for all 19 completed sweeps and their campaign lock ({count}/19 complete)")
            self.status("waiting_for_sweeps", completed=count, required=19)
            print(f"confirmation waiting: {count}/19 screening cases complete", flush=True)
            for _ in range(45):
                if self.stop.exists():
                    return
                time.sleep(1)
        else:
            return
        self.locked = True
        try:
            pool, inputs, _ = selection_pool(self.root)
            winner = choose_winner(pool)
            path = self.directory / "selection-plan.json"
            payload = {"schema": 1, "winner": winner, "selection_inputs_sha256": inputs,
                "source_sha256": source_inventory(), "seeds": self.args.seeds,
                "training_clicks": self.args.train_clicks or winner["config"]["total_timesteps"],
                "test_games": self.args.test_games, "test_split": "test", "test_replicate": 0,
                "device": self.args.device, "success_criterion": "Every one of three or more fresh training seeds has positive paired win difference and paired 95% CI lower bound > 0.",
                "interpretation": "Warm-start repeatability with shared frozen initialization; not training from scratch. No policy/checkpoint retuning on test."}
            if set(self.args.seeds) & {option["config"]["seed"] for option in pool}:
                raise ValueError("confirmation seeds must be fresh relative to screening")
            self.plan = write_selection_plan(path, payload)
            self.check()
            frozen = []
            for seed in self.plan["seeds"]:
                trained = self.training(seed)
                if trained is None:
                    return
                frozen.append(trained)
            seal_training(self.directory / "training-complete.json", self.plan, frozen)
            rows = []
            for checkpoint in frozen:
                if self.stop.exists():
                    return
                rows.append(self.test(checkpoint))
            strict = (len(self.plan["seeds"]) >= 3 and self.plan["test_games"] >= 10000 and
                      self.plan["training_clicks"] == winner["config"]["total_timesteps"])
            success = strict and success_criterion(rows, self.plan["seeds"])
            summary = {"success": success, "confirmatory_budget": strict, "selection_plan_sha256": self.plan["plan_sha256"],
                "repeats": rows, "mean_win": statistics.mean(row["summary"]["win"] for row in rows),
                "training_seed_win_stdev": statistics.stdev(row["summary"]["win"] for row in rows) if len(rows) > 1 else 0,
                "conclusion": "Repeatable advantage over Bayes established under the sealed warm-start protocol." if success else
                    "Repeatable advantage over Bayes was not established under the sealed protocol."}
            write_json(self.directory / "summary.json", summary)
            text = summary["conclusion"] + "\n\nShared frozen initialization; fresh training seeds. Test policy was fixed before testing.\n\n| Seed | Wins | Efficiency | Paired win difference | 95% CI |\n|---|---:|---:|---:|---|\n"
            for row in rows:
                pair = row["paired_vs_bayes"]
                efficiency = row["summary"]["efficiency"]
                formatted = f"{efficiency:.3f}" if efficiency is not None else "n/a"
                text += f"| {row['seed']} | {row['summary']['win']:.2%} | {formatted} | {pair['win_difference']:+.2%} | {pair['win_difference_ci95']} |\n"
            text += f"\nMean win rate {summary['mean_win']:.2%}; training-seed standard deviation {summary['training_seed_win_stdev']:.2%}. Confirmatory budget: {strict}.\n"
            (self.directory / "summary.md").write_text(text, encoding="utf-8")
            self.status("complete", success=success, summary=str(self.directory / "summary.json"))
        except KeyboardInterrupt:
            self.stop.touch()
            if self.current:
                output, exp = self.current
                if exp and (run := discover_run(exp)):
                    (MODELS_DIR / f"{run}.stop").touch()
                elif not exp:
                    (output / "STOP").touch()
            raise
        finally:
            lock.__exit__(None, None, None)
            self.locked = False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study-dir", type=Path, default=STUDY)
    parser.add_argument("--wait-for-sweeps", action="store_true")
    parser.add_argument("--test-games", type=int, default=10000)
    parser.add_argument("--train-clicks", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[202, 203, 204])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be unique nonnegative integers")
    if args.test_games < 1 or args.test_games > 1_000_000 or args.jobs < 1 or (args.train_clicks is not None and args.train_clicks < 1):
        parser.error("test-games/train-clicks/jobs must be positive; test-games at most one million")
    Confirmation(args).run()


if __name__ == "__main__":
    main()
