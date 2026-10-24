"""CPU measurement of a pure-risk network's public-board consistency.

Sample uniformly from all states in 128 Bayes games on validation replicate 1.
The solver supplies diagnostic posterior probabilities and the sampling policy;
it is never used to update the network. No test boards or GPU are accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ai_agents import bayes_reveal
from engine import MinesweeperEngine
from rl.experiment_eval import board_seeds, source_hashes, write_json
from rl.minesweeper_env import encode, start_game
from rl.qnet import load_checkpoint


def describe(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0}
    quantiles = [0, .25, .5, .75, .9, .95, .99, 1]
    return {"count": values.size, "mean": float(values.mean()),
            "quantiles": {str(q): float(v) for q, v in zip(quantiles, np.quantile(values, quantiles))}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=ROOT / "experiments/reward-study-20261024/screen/risk_only-s101-n500000-v2000/checkpoints/latest.pt")
    ap.add_argument("--expected-sha256", default="8ec3c4a583fe1508f212273824174d908831289b7416f8d5bbbfbc76c542d4ad")
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--states", type=int, default=1000)
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--sample-seed", type=int, default=20261024)
    ap.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.states < 1:
        ap.error("states must be positive")
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if args.expected_sha256 and digest != args.expected_sha256:
        raise ValueError("checkpoint hash mismatch")
    q, checkpoint = load_checkpoint(args.model, "cpu")
    reward, training = checkpoint["reward"], checkpoint["args"]
    if not (training["loss"] == "bce" and training["gamma"] == 0 and reward["mine"] > 0
            and not any(reward.get(k, 0) for k in ("win", "progress", "reveal", "safe", "late", "burst"))):
        raise ValueError("sigmoid(-logit) is a mine probability only for the pure-risk gamma0 BCE setup")
    q.eval()
    seeds = board_seeds("validation", 2, args.games, args.replicate)
    observations, probabilities, state_seeds = [], [], []
    for seed in seeds:
        engine = MinesweeperEngine(level=2, seed=0)
        effective = start_game(engine, seed)
        while engine.state == engine.RUNNING:
            obs = np.zeros((2, 9, 9), np.uint8)
            encode(engine, obs, np.empty((9, 9), np.int8))
            observations.append(obs)
            probabilities.append(engine.mine_probs.copy().ravel())
            state_seeds.append(effective)
            bayes_reveal(engine)
    available_states = len(observations)
    chosen = np.sort(np.random.default_rng(args.sample_seed).choice(
        available_states, min(args.states, available_states), replace=False))
    obs = np.stack(observations)[chosen]
    exact = np.stack(probabilities)[chosen]
    with torch.no_grad():
        logits = torch.cat([q(torch.from_numpy(obs[i:i + 128])).float()
                            for i in range(0, len(obs), 128)])
        predicted = torch.sigmoid(-logits).numpy().astype(np.float64)
    legal = obs[:, 0].reshape(-1, 81) == 0
    predicted_hidden = np.where(legal, predicted, 0)
    exact_hidden = np.nan_to_num(exact, nan=0.0)
    adjacency = np.zeros((81, 81), np.float64)
    for cell in range(81):
        r, c = divmod(cell, 9)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (dr or dc) and 0 <= rr < 9 and 0 <= cc < 9:
                    adjacency[cell, rr * 9 + cc] = 1
    hidden_neighbors = legal.astype(np.float64) @ adjacency.T
    view = obs[:, 0].reshape(-1, 81).astype(np.int32)
    clues = np.where(view > 0, view - 1, 0)
    frontier = (view > 0) & (hidden_neighbors > 0)
    residual = predicted_hidden @ adjacency.T - clues
    exact_residual = exact_hidden @ adjacency.T - clues
    global_error = predicted_hidden.sum(1) - 10
    absolute_error = np.abs(predicted - exact)
    rows = np.arange(len(obs))
    actions = np.argmax(np.where(legal, logits.numpy(), -np.inf), axis=1)
    selected_predicted, selected_exact = predicted[rows, actions], exact[rows, actions]
    minimum = np.nanmin(exact, axis=1)
    safe = legal & (exact <= 1e-9)
    mines = legal & (exact >= 1 - 1e-9)
    bins = []
    edges = [0, .0001, .001, .01, .05, .1, .2, .4, .6, .8, .95, 1.000001]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = legal & (predicted >= lo) & (predicted < hi)
        bins.append({"lower": lo, "upper": min(hi, 1), "cells": int(mask.sum()),
                     "mean_prediction": float(predicted[mask].mean()) if mask.any() else None,
                     "mean_exact_risk": float(exact[mask].mean()) if mask.any() else None})
    result = {
        "protocol": {"split": "validation", "level": 2, "replicate": args.replicate,
                     "requested_games": args.games, "seed_start": seeds.start, "seed_stop": seeds.stop,
                     "sampling_policy": "full Bayes", "available_states": available_states,
                     "sampled_states": len(obs), "state_selection": "uniform without replacement over all generated positions",
                     "sample_seed": args.sample_seed, "sampled_state_indices": chosen.tolist(),
                     "sampled_effective_board_seeds": [state_seeds[i] for i in chosen]},
        "model": {"path": str(args.model), "sha256": digest, "step": checkpoint["global_step"],
                  "config": q.config, "reward": reward},
        "runtime": {"device": "cpu", "precision": "fp32", "threads": 2,
                    "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__},
        "source_sha256": {**source_hashes(), str(Path(__file__).relative_to(ROOT)):
                           hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "public_clue_consistency": {
            "frontier_clues": int(frontier.sum()),
            "mean_signed_residual_mines": float(residual[frontier].mean()),
            "absolute_residual_mines": describe(np.abs(residual[frontier])),
            "mean_abs_normalized_residual": float((np.abs(residual[frontier]) / hidden_neighbors[frontier]).mean()),
            "normalization": "clue residual divided by number of adjacent hidden cells; only revealed clues with hidden neighbors",
            "exact_solver_max_abs_normalized_residual": float((np.abs(exact_residual[frontier]) / hidden_neighbors[frontier]).max())},
        "global_mine_budget": {"true_total": 10, "mean_predicted_total": float(predicted_hidden.sum(1).mean()),
                               "bias_mines": float(global_error.mean()), "mae_mines": float(np.abs(global_error).mean()),
                               "absolute_error": describe(np.abs(global_error)),
                               "exact_solver_max_abs_error": float(np.abs(exact_hidden.sum(1) - 10).max())},
        "legal_cell_posterior_error": {"cells": int(legal.sum()), "mae": float(absolute_error[legal].mean()),
                                       "rmse": float(np.sqrt(np.square(predicted[legal] - exact[legal]).mean())),
                                       "mean_prediction": float(predicted[legal].mean()),
                                       "mean_exact_risk": float(exact[legal].mean()), "bins": bins},
        "known_safe_predicted_mine_risk": describe(predicted[safe]),
        "known_mine_predicted_mine_risk": describe(predicted[mines]),
        "selected_actions": {"states": len(obs), "mean_predicted_mine_risk": float(selected_predicted.mean()),
                             "mean_exact_mine_risk": float(selected_exact.mean()),
                             "mean_minimum_exact_mine_risk": float(minimum.mean()),
                             "mean_excess_exact_mine_risk": float((selected_exact - minimum).mean()),
                             "states_with_known_safe": int((minimum <= 1e-9).sum()),
                             "missed_safe_choices": int(((minimum <= 1e-9) & (selected_exact > 1e-9)).sum()),
                             "riskier_choices": int((selected_exact > minimum + 1e-9).sum()),
                             "predicted_risk": describe(selected_predicted), "exact_risk": describe(selected_exact)},
        "interpretation": "Diagnostic on Bayes-distribution states, not the learned policy's state distribution or a win-rate evaluation. Exact solver is measurement-only; no training or parameter updates. Public clue/global-count residuals use visible clues and the public total mine count."}
    if digest != hashlib.sha256(args.model.read_bytes()).hexdigest():
        raise RuntimeError("checkpoint changed during diagnostic")
    write_json(args.output, result)
    import json
    print(json.dumps({key: result[key] for key in ("public_clue_consistency", "global_mine_budget", "legal_cell_posterior_error", "known_safe_predicted_mine_risk", "known_mine_predicted_mine_risk", "selected_actions")}, indent=2))


if __name__ == "__main__":
    main()
