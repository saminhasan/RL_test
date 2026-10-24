"""Evaluate learned-risk-gated opening selection, without solver action masks.

The near-safe gate uses only the model's estimated risks. It is not a proof that
any candidate is safe. Margin zero is an explicit pure-safety control in every
regime. Positive margins use opening predictions only within a narrow learned
risk band when minimum predicted risk is below the cutoff; otherwise they use
the original log-safety plus opening score. Validation is the default.

    python -m rl.opening_policy_eval --model experiments/pilot/latest.pt --device cpu --margins 0 1e-5 1e-4 1e-3 --output experiments/gated-validation.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.experiment_eval import (bayes_cache_key, bayes_cache_protocol, board_seeds,
                                paired_comparison, sha256, write_json)


class GatedOpening(nn.Module):
    """An inference-only adapter over a SafetyOpening model's learned heads.

    Gated ties are exact lexicographic order: highest conditional zero
    probability, then highest safety logit, then first row-major cell. No tiny
    floating-point score mixture is used. ``forward`` returns scalar scores
    compatible with the existing public hidden-cell greedy selector.
    """

    def __init__(self, model, cutoff: float = 0.01, margin: float = 0.0001,
                 guess_lambda: float = 0.01):
        super().__init__()
        if (not all(math.isfinite(v) for v in (cutoff, margin, guess_lambda))
                or not 0 <= cutoff <= 1 or not 0 <= margin <= 1 or guess_lambda < 0):
            raise ValueError("cutoff/margin must be finite in [0,1] and guess_lambda finite nonnegative")
        self.model = model
        self.cutoff, self.margin, self.guess_lambda = cutoff, margin, guess_lambda

    def forward(self, obs):
        heads = self.model.heads(obs)
        safe, zero = heads["safe"].float(), heads["zero"].float().sigmoid()
        legal = (obs[:, 0] == 0).flatten(1)
        if self.margin == 0:
            return safe.masked_fill(~legal, -torch.inf)
        risk = torch.sigmoid(-safe)
        minimum = risk.masked_fill(~legal, torch.inf).amin(1)
        gated = minimum <= self.cutoff
        candidates = legal & (risk <= minimum[:, None] + self.margin)
        # Stable sort safety first, then zero probability, preserving the first
        # ordering within equal zero probabilities. Illegal cells sort last.
        safer = safe.masked_fill(~candidates, -torch.inf).argsort(dim=1, descending=True, stable=True)
        opening = zero.gather(1, safer).masked_fill(~candidates.gather(1, safer), -torch.inf)
        by_opening = opening.argsort(dim=1, descending=True, stable=True)
        chosen = safer.gather(1, by_opening[:, :1])
        gated_scores = torch.full_like(safe, -torch.inf).scatter(1, chosen, 0.0)
        guessing_scores = F.logsigmoid(safe) + self.guess_lambda * zero
        return torch.where(gated[:, None], gated_scores, guessing_scores).masked_fill(~legal, -torch.inf)


def evaluate(args):
    from rl.baseline import play_bayes
    from rl.evaluate import format_row, play_q, summarize
    from rl.safety_opening import configure, load_model, provenance

    device = configure(args.device, 0)
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    digest = sha256(args.model)
    model, checkpoint = load_model(args.model, device)
    seeds = board_seeds(args.split, args.level, args.games, args.replicate)
    hashes = {**provenance(), "rl/opening_policy_eval.py": sha256(Path(__file__))}
    protocol = {"schema": 1, "split": args.split, "level": args.level, "games": args.games,
                "replicate": args.replicate, "requested_seed_start": seeds.start,
                "requested_seed_stop": seeds.stop, "source_sha256": hashes}
    cache = args.cache_dir / f"bayes-{args.split}-L{args.level}-{bayes_cache_key(protocol)}.json"
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached["protocol"] != bayes_cache_protocol(protocol):
            raise ValueError("Bayes cache protocol mismatch")
        bayes = cached["records"]
    else:
        bayes = sorted(play_bayes(args.level, seeds, n_jobs=args.jobs), key=lambda r: r["seed"])
        write_json(cache, {"protocol": bayes_cache_protocol(protocol), "records": bayes})
    result = {
        "kind": "learned_risk_gated_opening_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": protocol,
        "checkpoint": {"path": str(args.model.resolve()), "sha256": digest,
                       "training_step": checkpoint["step"], "config": checkpoint["config"],
                       "training_args": checkpoint["args"],
                       "initialization": checkpoint.get("initialization"),
                       "training_source_sha256": checkpoint.get("source_sha256")},
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "device": str(device),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "bf16": device.type == "cuda" and not args.no_bf16,
                    "deterministic": torch.are_deterministic_algorithms_enabled(),
                    "n_envs": args.n_envs, "torch_threads": torch.get_num_threads()},
        "interpretation": "Gate uses model-estimated risks, not known-safe proofs or solver masks. Margin zero is pure safety in every regime. Use validation to choose a policy before one untouched test; paired board intervals do not measure training-seed variability. Search p-values are unadjusted.",
        "bayes": {"summary": summarize(bayes), "records": bayes}, "policies": [],
    }
    print(format_row("bayes", result["bayes"]["summary"]), flush=True)
    for margin in args.margins:
        policy = GatedOpening(model, cutoff=args.cutoff, margin=margin, guess_lambda=args.guess_lambda)
        records = sorted(play_q(policy, args.level, seeds, device, n_envs=args.n_envs,
                                breakdown=True, bf16=not args.no_bf16), key=lambda r: r["seed"])
        summary = summarize(records)
        comparison = paired_comparison(records, bayes, bootstrap_samples=args.bootstrap_samples,
                                       seed=args.statistics_seed)
        result["policies"].append({"margin": margin, "cutoff": args.cutoff,
                                    "guess_lambda": args.guess_lambda,
                                    "margin_zero_is_pure_safety": True, "opening_head": "zero",
                                    "summary": summary, "paired_vs_bayes": comparison, "records": records})
        print(format_row(f"gated margin {margin:g}", summary), flush=True)
        print(f"  paired difference {comparison['win_difference'] * 100:+.2f} points; "
              f"95% CI [{comparison['win_difference_ci95'][0] * 100:+.2f}, "
              f"{comparison['win_difference_ci95'][1] * 100:+.2f}]", flush=True)
    if sha256(args.model) != digest:
        raise RuntimeError("checkpoint changed during evaluation; no artifact published; use an immutable copy")
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margins", type=float, nargs="+", default=[0.0, 1e-5, 1e-4, 1e-3])
    parser.add_argument("--cutoff", type=float, default=0.01)
    parser.add_argument("--guess-lambda", type=float, default=0.01)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent.parent /
                        "experiments" / "reward-study-20261024" / "bayes-cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-envs", type=int, default=256)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=0, help="0 keeps the current PyTorch thread count")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--statistics-seed", type=int, default=20261024)
    args = parser.parse_args()
    if args.n_envs < 1 or args.jobs < 1 or args.cpu_threads < 0:
        parser.error("environment/worker counts must be positive and CPU thread count nonnegative")
    for margin in args.margins:
        try:
            GatedOpening(nn.Identity(), cutoff=args.cutoff, margin=margin, guess_lambda=args.guess_lambda)
        except ValueError as error:
            parser.error(str(error))
    evaluate(args)


if __name__ == "__main__":
    main()
