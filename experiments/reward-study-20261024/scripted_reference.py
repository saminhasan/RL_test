"""CPU-only solver-assisted policy ablation on study validation boards.

Known-safe clicks retain full Bayes priority and tie breaking. At forced guesses
score hidden cells by -P(mine) + lambda * heuristic_P(zero), where the heuristic
is ai_agents._zero_chance. Lambda zero exactly reproduces full Bayes, including
its zero-opening tie break and per-board RNG. This measures scripted-policy
headroom; it is not a learned agent or evidence of learned superiority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from joblib import Parallel, delayed

from ai_agents import _lowest_probability_candidates, _zero_chance, bayes_reveal
from engine import MinesweeperEngine
from rl.baseline import record
from rl.experiment_eval import board_seeds, paired_comparison, source_hashes, write_json
from rl.minesweeper_env import is_coin_flip, start_game


def play(seed: int, weight: float | None) -> dict:
    engine = MinesweeperEngine(level=2, seed=0)
    effective_seed = start_game(engine, seed)
    forced = risky = safe_bypasses = 0
    excess = 0.0
    while engine.state == engine.RUNNING:
        raw = engine.mine_probs
        probs = raw.ravel()
        hidden = np.flatnonzero(~engine._revealed_flat)
        low = float(probs[hidden].min())
        flip = is_coin_flip(raw)
        forced += low > 1e-12
        if weight is None:
            picked = low
            bayes_reveal(engine)
        else:
            if weight == 0 or low <= 1e-12:
                candidates = _lowest_probability_candidates(engine)
                chance = _zero_chance(engine, candidates)
                candidates = candidates[chance >= chance.max() - 1e-12]
            else:
                chance = _zero_chance(engine, hidden)
                scores = -probs[hidden] + weight * chance
                candidates = hidden[scores >= scores.max() - 1e-12]
                # A score tie retains the same opening preference as full Bayes.
                chance = _zero_chance(engine, candidates)
                candidates = candidates[chance >= chance.max() - 1e-12]
            cell = int(candidates[int(engine._rng.integers(candidates.size))])
            picked = float(probs[cell])
            risky += picked > low + 1e-12
            excess += max(0.0, picked - low)
            safe_bypasses += low <= 1e-12 and picked > 1e-12
            engine.reveal_count(*divmod(cell, engine.cols))
    row = record(engine, effective_seed)
    if not engine.won:
        row["lost_by"] = ("missed_safe" if low <= 1e-9 else
                          "risky_guess" if picked > low + 1e-9 else
                          "coin_flip" if flip else "unlucky")
    row.update(forced_guesses=int(forced), riskier_clicks=int(risky),
               safe_bypasses=int(safe_bypasses), accumulated_excess_mine_risk=excess)
    return row


def summary(records: list[dict]) -> dict:
    n = len(records)
    win = sum(r["won"] for r in records) / n
    wins = [r for r in records if r["won"]]
    return {"games": n, "win": win, "win_margin": 1.96 * math.sqrt(win * (1 - win) / n),
            "efficiency": float(np.mean([r["min_moves"] / r["moves"] for r in wins])) if wins else None,
            "revealed": float(np.mean([r["revealed_fraction"] for r in records])),
            "loss_counts": {kind: sum(r.get("lost_by") == kind for r in records)
                            for kind in ("missed_safe", "risky_guess", "coin_flip", "unlucky")},
            "forced_guesses": sum(r["forced_guesses"] for r in records),
            "riskier_clicks": sum(r["riskier_clicks"] for r in records),
            "safe_bypasses": sum(r["safe_bypasses"] for r in records),
            "accumulated_excess_mine_risk": sum(r["accumulated_excess_mine_risk"] for r in records)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--weights", type=float, nargs="+", default=[0, .001, .003, .01, .03, .1])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = ap.parse_args()
    if args.jobs < 1 or any(w < 0 or not math.isfinite(w) for w in args.weights):
        ap.error("jobs must be positive and weights finite/nonnegative")
    if 0.0 not in args.weights:
        ap.error("include weight zero to verify exact full-Bayes reproduction")
    seeds = board_seeds("validation", 2, args.games)
    result = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "python": platform.python_version(), "numpy": np.__version__, "device": "cpu",
              "protocol": {"split": "validation", "level": 2, "games": args.games,
                           "seed_start": seeds.start, "seed_stop": seeds.stop, "replicate": 0},
              "source_sha256": {**source_hashes(), str(Path(__file__).relative_to(ROOT)):
                                 hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "policy": "known-safe priority; at forced guesses maximize -P(mine)+lambda*neighbor-safe-product; ties prefer opening then engine RNG",
              "interpretation": "Scripted solver-assisted reference ablation, not learned success. Validation-only exploratory comparisons; p-values unadjusted across weights. Test boards remain unopened.",
              "variants": []}
    keys = ("seed", "won", "moves", "min_moves", "revealed_fraction", "lost_by")
    with Parallel(n_jobs=args.jobs, backend="loky") as parallel:
        baseline = sorted(parallel(delayed(play)(s, None) for s in seeds), key=lambda r: r["seed"])
        result["bayes"] = {"summary": summary(baseline), "records": baseline}
        print("full_bayes", json.dumps(result["bayes"]["summary"]), flush=True)
        for weight in args.weights:
            records = sorted(parallel(delayed(play)(s, weight) for s in seeds), key=lambda r: r["seed"])
            if weight == 0:
                assert [[r.get(k) for k in keys] for r in records] == [[r.get(k) for k in keys] for r in baseline]
            assert not any(r["safe_bypasses"] for r in records), "known-safe priority was violated"
            row = {"lambda": weight, "summary": summary(records),
                   "paired_vs_bayes": paired_comparison(records, baseline), "records": records}
            result["variants"].append(row)
            write_json(args.output, result)
            print("lambda", weight, "win", row["summary"]["win"], "eff", row["summary"]["efficiency"],
                  "paired", row["paired_vs_bayes"]["win_difference"],
                  "CI", row["paired_vs_bayes"]["win_difference_ci95"], flush=True)


if __name__ == "__main__":
    main()
