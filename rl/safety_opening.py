"""Separate selected-outcome safety learning from conditional opening prediction.

    python -m rl.safety_opening train --init models/risk-only-latest.pt --aux zero --seed 201
    python -m rl.safety_opening evaluate --model experiments/example/latest.pt --grid 0 .001 .01 .1

Training uses only chosen clicks and their real post-click visible boards. The
safety head learns unweighted binary BCE; opening heads learn only on chosen
safe clicks. The reveal head uses BCE with a fractional target, a proper loss
for its conditional mean, and starts with a 0.03 uncovered-fraction prior.
Optional public clue/budget losses see the visible board alone.
No solver probabilities or hidden-board labels/masks enter training. The solver
is used for validation loss breakdowns and the Bayesian comparison only.

All output projections run in FP32. Play maximizes log P(safe) plus lambda times
conditional P(zero), or conditional expected uncovered fraction. Evaluation can
scan lambdas without retraining, retaining per-board paired results and hashes.
Validation is the default; opening test boards requires explicit --split test.
Create the training output directory's STOP file to save and stop cleanly.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.experiment_eval import (bayes_cache_key, bayes_cache_protocol, board_seeds,
                                paired_comparison, sha256, source_hashes, write_json)
from rl.journal import Journal, replace_with_retry
from rl.qnet import QNetwork, greedy

KIND = "safety_opening_v1"
ROOT = Path(__file__).resolve().parent.parent


def selected_labels(obs, next_obs, actions):
    """Selected safe/zero outcomes and uncovered fraction, never counterfactuals.

    The engine leaves a fatal selected mine coded hidden (0); a selected safe
    cell becomes clue 0..8 (codes 1..9). Use final_obs before SAME_STEP reset.
    """
    rows = torch.arange(len(actions), device=actions.device)
    before, after = obs[:, 0].flatten(1), next_obs[:, 0].flatten(1)
    if not bool((before[rows, actions] == 0).all()):
        raise ValueError("selected outcomes require a previously hidden on-board action")
    code = after[rows, actions]
    if not bool(((code >= 0) & (code <= 9)).all()):
        raise ValueError("invalid selected-cell post-click code")
    safe, zero = code > 0, code == 1
    uncovered = ((before == 0) & (after > 0) & (after < 10)).sum(1).float()
    safe_total = (before != 10).sum(1) - obs[:, 1, 0, 0].long()
    fraction = uncovered / safe_total.clamp(min=1)
    if not bool((safe == (uncovered > 0)).all()):
        raise ValueError("post-click board is inconsistent with the selected outcome")
    return safe.float(), zero.float(), fraction


def final_observations(next_obs, terminated, truncated, infos):
    """Replace auto-reset observations with the actual terminal observation."""
    real = next_obs.copy()
    for index in np.flatnonzero(terminated | truncated):
        real[index] = infos["final_obs"][index]
    return real


class SafetyOpening(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.q = QNetwork(config)
        self.config = {**self.q.config, "opening_lambda": 0.01, "opening_kind": "zero",
                       **(config or {}), "fp32_head": True}
        self.q.config["fp32_head"] = True
        self.zero_head = nn.Conv2d(self.config["channels"], 1, 1)
        self.reveal_head = nn.Conv2d(self.config["channels"], 1, 1)
        # Constant initial auxiliary predictions preserve the warm-start ranking.
        for head in (self.zero_head, self.reveal_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.constant_(self.reveal_head.bias, math.log(0.03 / 0.97))

    def heads(self, obs):
        x = self.q.planes(obs)
        h = self.q.body[1](self.q.body[0](x))
        for block in self.q.body[2:]:
            h = block(h, x[:, 10:11])
        with torch.autocast(h.device.type, enabled=False):
            h = h.float()
            return {"safe": self.q.head(h).flatten(1),
                    "zero": self.zero_head(h).flatten(1),
                    "reveal": self.reveal_head(h).flatten(1)}

    def forward(self, obs):
        heads = self.heads(obs)
        return F.logsigmoid(heads["safe"]) + self.config["opening_lambda"] * heads[self.config["opening_kind"]].sigmoid()


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("kind") != KIND:
        raise ValueError(f"not a {KIND} checkpoint: {path}")
    model = SafetyOpening(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def publish_checked_result(model_path, before_hash, temporary, output):
    if sha256(model_path) != before_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("checkpoint changed during evaluation; use an immutable copy")
    os.replace(temporary, output)


def initialize(path, config, device):
    if not path:
        return SafetyOpening(config).to(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") == KIND:
        model, _ = load_model(path, device)
        model.config.update(opening_lambda=config["opening_lambda"], opening_kind=config["opening_kind"])
        return model
    reward = checkpoint.get("reward", {})
    args = checkpoint.get("args", {})
    if (args.get("loss") != "bce" or args.get("gamma") != 0 or args.get("skip_coin_flips", False)
            or reward.get("mine", 0) <= 0
            or any(reward.get(key, 0) != 0 for key in ("win", "progress", "reveal", "safe", "late", "burst"))):
        raise ValueError("Q initialization must be a pure mine-outcome BCE model (gamma 0, no other rewards)")
    model = SafetyOpening({**checkpoint["config"], "opening_lambda": config["opening_lambda"],
                           "opening_kind": config["opening_kind"]}).to(device)
    model.q.load_state_dict(checkpoint["state_dict"])
    return model


def objective(model, obs, next_obs, actions, *, aux="zero", aux_weight=0.05,
              clue_weight=0.0, budget_weight=0.0):
    safe, zero, reveal = selected_labels(obs, next_obs, actions)
    heads = model.heads(obs)
    gather = lambda values: values.gather(1, actions[:, None]).squeeze(1)
    safety = F.binary_cross_entropy_with_logits(gather(heads["safe"]), safe)
    positive = safe.bool()
    zero_loss = reveal_loss = safety.new_zeros(())
    if positive.any():
        if aux in ("zero", "both"):
            zero_loss = F.binary_cross_entropy_with_logits(gather(heads["zero"])[positive], zero[positive])
        if aux in ("reveal", "both"):
            reveal_loss = F.binary_cross_entropy_with_logits(gather(heads["reveal"])[positive], reveal[positive])
    clue_loss = budget_loss = safety.new_zeros(())
    if clue_weight or budget_weight:
        from rl.public_constraints import public_constraint_loss
        clue_loss, budget_loss = public_constraint_loss(obs, heads["safe"])
    loss = safety + aux_weight * (zero_loss + reveal_loss) + clue_weight * clue_loss + budget_weight * budget_loss
    return loss, {"safety": safety.detach(), "zero": zero_loss.detach(), "reveal": reveal_loss.detach(),
                  "clue": clue_loss.detach(), "budget": budget_loss.detach()}


def provenance():
    hashes = source_hashes()
    for name in ("rl/safety_opening.py", "rl/public_constraints.py"):
        if (ROOT / name).exists():
            hashes[name] = sha256(ROOT / name)
    return hashes


def configure(device, seed, deterministic=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                        else "cpu" if device == "auto" else device)


def validation(model, args, *, checkpoint=None, step=None):
    from rl.baseline import play_bayes
    from rl.evaluate import format_row, play_q, summarize
    seeds = board_seeds(args.split, args.level, args.games, args.replicate)
    hashes = provenance()
    protocol = {"schema": 1, "split": args.split, "level": args.level, "games": args.games,
                "replicate": args.replicate, "requested_seed_start": seeds.start,
                "requested_seed_stop": seeds.stop, "source_sha256": hashes}
    cache = args.cache_dir / f"bayes-{args.split}-L{args.level}-{bayes_cache_key(protocol)}.json"
    if cache.exists():
        import json
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached["protocol"] != bayes_cache_protocol(protocol):
            raise ValueError("Bayes cache protocol mismatch")
        bayes = cached["records"]
    else:
        bayes = sorted(play_bayes(args.level, seeds, n_jobs=args.jobs), key=lambda row: row["seed"])
        write_json(cache, {"protocol": bayes_cache_protocol(protocol), "records": bayes})
    device = next(model.parameters()).device
    result = {"kind": KIND, "protocol": protocol, "checkpoint": checkpoint,
              "training_step": step, "runtime": {"torch": str(torch.__version__), "numpy": np.__version__,
              "device": str(device), "bf16": device.type == "cuda" and not args.no_bf16,
              "deterministic": torch.are_deterministic_algorithms_enabled()},
              "bayes": {"summary": summarize(bayes), "records": bayes}, "policies": [],
              "interpretation": "Lambda selection uses validation. Paired board intervals do not measure training-seed variability; search p-values are unadjusted."}
    original = model.config["opening_lambda"]
    try:
        for weight in args.grid:
            model.config["opening_lambda"] = weight
            records = sorted(play_q(model, args.level, seeds, device, n_envs=args.n_envs,
                                    breakdown=True, bf16=not args.no_bf16), key=lambda row: row["seed"])
            summary = summarize(records)
            result["policies"].append({"opening_lambda": weight, "opening_kind": model.config["opening_kind"],
                "summary": summary, "paired_vs_bayes": paired_comparison(records, bayes), "records": records})
            print(format_row(f"lambda {weight:g}", summary), flush=True)
            write_json(args.output, result)
    finally:
        model.config["opening_lambda"] = original
    return result


@dataclass
class TrainArgs:
    output_dir: Path | None = None
    init: str = ""
    resume: str = ""
    seed: int = 201
    level: int = 2
    total_timesteps: int = 500_000
    num_envs: int = 64
    channels: int = 64
    layers: int = 16
    batch_size: int = 128
    buffer_size: int = 1_000_000
    train_frequency: int = 16
    learning_starts: int = 10_000
    learning_rate: float = 3e-5
    lr_floor: float = 1e-5
    ema: float = 0.999
    aux: Literal["none", "zero", "reveal", "both"] = "zero"
    aux_weight: float = 0.05
    clue_weight: float = 0.0
    budget_weight: float = 0.0
    opening_lambda: float = 0.01
    opening_kind: Literal["zero", "reveal"] = "zero"
    start_e: float = 0.02
    end_e: float = 0.005
    exploration_fraction: float = 0.1
    eval_every: int = 100_000
    eval_games: int = 500
    checkpoint_every: int = 100_000
    augment: bool = True
    bf16: bool = True
    deterministic: bool = True
    device: str = "auto"
    jobs: int = 8


def train(args: TrainArgs):
    import gymnasium as gym
    from rl.dqn import mirror
    from rl.minesweeper_env import MinesweeperEnv, RewardConfig
    from rl.per import ReplayBuffer

    if (min(args.total_timesteps, args.num_envs, args.batch_size, args.buffer_size, args.train_frequency,
            args.learning_rate, args.lr_floor, args.checkpoint_every) <= 0
            or not 0 <= args.ema < 1 or min(args.aux_weight, args.clue_weight, args.budget_weight) < 0
            or not 0 < args.exploration_fraction <= 1 or not 0 <= args.end_e <= args.start_e <= 1
            or args.aux not in ("none", "zero", "reveal", "both") or args.opening_kind not in ("zero", "reveal")):
        raise ValueError("invalid training configuration")
    device = configure(args.device, args.seed, args.deterministic)
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False) if args.resume else None
    if checkpoint:
        for key in ("channels", "layers", "opening_lambda", "opening_kind"):
            if getattr(args, key) != checkpoint["config"][key]:
                raise ValueError(f"resume --{key.replace('_', '-')} differs from the checkpoint; start a new initialized run")
    output = (args.output_dir or (Path(args.resume).parent if args.resume else
              ROOT / "experiments" / "reward-study-20261024" / "phase2" /
              f"safety-opening-s{args.seed}-{time.strftime('%Y%m%d-%H%M%S')}" )).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "latest.pt").exists() and not args.resume:
        raise ValueError("output already contains a model; choose a new directory or --resume its latest.pt")
    config = {"channels": args.channels, "layers": args.layers,
              "opening_lambda": args.opening_lambda, "opening_kind": args.opening_kind}
    model = load_model(args.resume, device)[0] if checkpoint else initialize(args.init, config, device)
    if device.type == "cuda" and args.bf16:
        model = model.to(memory_format=torch.channels_last)
    policy = copy.deepcopy(model) if args.ema else model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    step, best_win = 0, -1.0
    if checkpoint:
        model.load_state_dict(checkpoint.get("online", checkpoint["state_dict"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        step, best_win = checkpoint["step"], checkpoint.get("best_win", -1.0)
    hashes = provenance()
    args_record = {key: str(value) if isinstance(value, Path) else value
                   for key, value in dataclasses.asdict(args).items()}
    init_meta = (checkpoint.get("initialization") if checkpoint else
                 {"path": str(Path(args.init).resolve()), "sha256": sha256(Path(args.init))} if args.init else None)
    resume_meta = {"path": str(Path(args.resume).resolve()), "sha256": sha256(Path(args.resume)),
                   "step": step} if checkpoint else None
    journal = Journal(output)
    journal.event(step, "resumed with empty replay" if checkpoint else "started", "session")
    reward = RewardConfig(win=0, progress=0, mine=1, gamma=0)
    envs = gym.vector.SyncVectorEnv([lambda i=i: MinesweeperEnv(level=args.level, seed=args.seed * 10000 + i + step,
                                                              reward=reward) for i in range(args.num_envs)],
                                   autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    obs, _ = envs.reset()
    replay, rng = ReplayBuffer(args.buffer_size), np.random.default_rng(args.seed + step)
    wins, losses = deque(maxlen=1000), {}
    start, start_step, next_status = time.time(), step, time.time()
    next_eval = (step // max(args.eval_every, 1) + 1) * max(args.eval_every, 1)
    next_save = (step // args.checkpoint_every + 1) * args.checkpoint_every
    learn_at = step + args.learning_starts
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and args.bf16)
    reason = "finished"

    def save(label):
        path, temporary = output / f"{label}.pt", output / f"{label}.tmp"
        torch.save({"kind": KIND, "config": policy.config, "state_dict": policy.state_dict(),
                    "online": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                    "best_win": best_win, "args": args_record, "source_sha256": hashes,
                    "initialization": init_meta, "resumed_from": resume_meta}, temporary)
        replace_with_retry(temporary, path)

    def status(state):
        journal.status(state=state, kind=KIND, pid=os.getpid(), step=step, total_steps=args.total_timesteps,
                       seed=args.seed, level=args.level, epsilon=epsilon, lr=lr, losses=losses,
                       win=float(np.mean(wins)) if wins else None, best_win=best_win,
                       sps=(step - start_step) / max(time.time() - start, 1e-6), stop_reason=reason)

    epsilon, lr = args.start_e, args.learning_rate
    try:
        while step < args.total_timesteps:
            if (output / "STOP").exists():
                reason = "stopped by STOP"
                break
            progress = min(1.0, step / args.total_timesteps)
            lr = args.learning_rate * (args.lr_floor / args.learning_rate) ** progress
            for group in optimizer.param_groups:
                group["lr"] = lr
            epsilon = max(args.end_e, args.start_e + (args.end_e - args.start_e) *
                          step / (args.exploration_fraction * args.total_timesteps))
            x = torch.from_numpy(obs).to(device)
            with torch.no_grad(), torch.autocast(**amp):
                actions = greedy(policy(x), x).cpu().numpy()
            for index in np.flatnonzero(rng.random(args.num_envs) < epsilon):
                actions[index] = rng.choice(np.flatnonzero(obs[index, 0].ravel() == 0))
            next_obs, _, terminated, truncated, infos = envs.step(actions)
            real = final_observations(next_obs, terminated, truncated, infos)
            safe = (real[np.arange(args.num_envs), 0].reshape(args.num_envs, -1)[np.arange(args.num_envs), actions] > 0)
            for index in range(args.num_envs):
                replay.add(obs[index].copy(), int(actions[index]), float(safe[index]), real[index].copy(), 0.0)
            for index in np.flatnonzero(terminated | truncated):
                wins.append(bool(infos["final_info"]["won"][index]))
            obs, step = next_obs, step + args.num_envs
            if step > learn_at:
                for _ in range(max(1, args.num_envs // args.train_frequency)):
                    batch = replay.sample(args.batch_size)
                    before = torch.from_numpy(batch["obs"]).to(device)
                    after = torch.from_numpy(batch["next_obs"]).to(device)
                    chosen = torch.from_numpy(batch["action"]).to(device)
                    if args.augment:
                        before, after, chosen = mirror(before, after, chosen, rng)
                    with torch.autocast(**amp):
                        loss, parts = objective(model, before, after, chosen, aux=args.aux, aux_weight=args.aux_weight,
                                                clue_weight=args.clue_weight, budget_weight=args.budget_weight)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 10)
                    optimizer.step()
                    if policy is not model:
                        with torch.no_grad():
                            torch._foreach_lerp_(list(policy.parameters()), list(model.parameters()), 1 - args.ema)
                    losses = {key: float(value) for key, value in parts.items()}
            if args.eval_games and args.eval_every > 0 and step >= next_eval:
                next_eval += args.eval_every
                status("evaluating")
                result = validation(policy, SimpleNamespace(split="validation", level=args.level, games=args.eval_games,
                    replicate=0, cache_dir=output.parent / "bayes-cache", jobs=args.jobs, grid=[policy.config["opening_lambda"]],
                    no_bf16=not args.bf16, n_envs=256, output=output / f"eval-{step}.json"),
                    checkpoint={"live_policy": True, "config": policy.config,
                                "training_args": args_record, "initialization": init_meta}, step=step)
                summary = result["policies"][0]["summary"]
                journal.log("eval", step, **summary)
                if summary["win"] > best_win:
                    best_win = summary["win"]
                    save("best")
            if step >= next_save:
                next_save += args.checkpoint_every
                save("latest")
            if time.time() >= next_status:
                status("training")
                journal.log("train", step, win=float(np.mean(wins)) if wins else None, lr=lr, **losses)
                next_status = time.time() + 5
    except KeyboardInterrupt:
        reason = "interrupted"
    except BaseException:
        reason = "failed"
        raise
    finally:
        save("latest")
        envs.close()
        journal.event(step, reason, "stop")
        status("stopped")
    return output / "latest.pt"


def main():
    import tyro
    if len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "train":
        train(tyro.cli(TrainArgs, args=__import__("sys").argv[2:]))
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("evaluate",))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", nargs="+", type=float, default=[0.0, 0.001, 0.01, 0.1])
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "experiments" / "reward-study-20261024" / "bayes-cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-envs", type=int, default=256)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()
    if any(weight < 0 or not np.isfinite(weight) for weight in args.grid):
        parser.error("lambda grid must contain finite nonnegative values")
    before = sha256(args.model)
    model, checkpoint = load_model(args.model, configure(args.device, 0))
    output = args.output
    args.output = output.with_name(output.name + f".{os.getpid()}.pending")
    validation(model, args, checkpoint={"path": str(args.model.resolve()), "sha256": before,
               "config": checkpoint["config"], "training_args": checkpoint["args"]}, step=checkpoint["step"])
    publish_checked_result(args.model, before, args.output, output)


if __name__ == "__main__":
    main()
