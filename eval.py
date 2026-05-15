"""
Evaluation script. Runs all three and compares:
  - latest model (model.zip)
  - best model   (best_model.zip)
  - Bayes bot    (always click lowest mine-probability cell)

Usage:
    python eval.py
"""
from __future__ import annotations

import os
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from config import cfg
from engine import MinesweeperEngine
from minesweeper_env import make_eval_env, _click_risk, _calc_reward

N_EPISODES = 500
N_ENVS     = 16


def _eval_model(model: MaskablePPO, n_episodes: int, n_envs: int) -> dict:
    eval_env = make_eval_env(cfg, n_envs=n_envs)
    obs = eval_env.reset()

    ep_rewards = np.zeros(n_envs, dtype=np.float64)
    ep_lengths = np.zeros(n_envs, dtype=np.int32)
    wins = mine_hits = completed = 0
    total_reward = total_length = 0.0

    while completed < n_episodes:
        masks = get_action_masks(eval_env)
        actions, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(actions)

        ep_rewards += rewards
        ep_lengths += 1

        for i, done in enumerate(dones):
            if done and completed < n_episodes:
                wins         += int(infos[i].get("won", False))
                mine_hits    += int(infos[i].get("hit_mine", False))
                total_reward += float(ep_rewards[i])
                total_length += int(ep_lengths[i])
                ep_rewards[i] = 0.0
                ep_lengths[i] = 0
                completed    += 1

    eval_env.close()
    return {
        "wins": wins, "mine_hits": mine_hits,
        "total_reward": total_reward, "total_length": total_length,
        "n": n_episodes,
    }


def _eval_bayes(n_episodes: int) -> dict:
    level     = cfg.level
    base_seed = cfg.seed + 8_888_888

    wins = mine_hits = 0
    total_reward = total_length = 0.0

    for i in range(n_episodes):
        eng = MinesweeperEngine(level=level, seed=base_seed + i)
        ep_reward = ep_length = 0

        while eng.state != MinesweeperEngine.OVER:
            probs_flat  = eng.mine_probs.ravel()
            hidden_flat = np.flatnonzero(~eng.revealed.ravel())
            if hidden_flat.size == 0:
                break

            hidden_probs = probs_flat[hidden_flat]
            min_p        = float(np.min(hidden_probs))
            candidates   = hidden_flat[np.abs(hidden_probs - min_p) < 1e-9]
            flat_action  = int(candidates[0])

            p_clicked, is_ambiguous = _click_risk(eng, flat_action)
            progress  = eng.revealed_safe_cells / eng.total_safe_cells
            prev_safe = eng.revealed_safe_cells

            row, col = divmod(flat_action, eng.cols)
            eng.reveal(row, col)
            gained = eng.revealed_safe_cells - prev_safe

            ep_reward += _calc_reward(
                eng, p_clicked, is_ambiguous, gained, eng.total_safe_cells, progress
            )
            ep_length += 1

        wins      += int(eng.won)
        mine_hits += int(eng.hit_mine)
        total_reward += ep_reward
        total_length += ep_length

    return {
        "wins": wins, "mine_hits": mine_hits,
        "total_reward": total_reward, "total_length": total_length,
        "n": n_episodes,
    }


def _print_result(label: str, r: dict) -> None:
    n = r["n"]
    print(f"\n-- {label} ({cfg.level_name}) " + "-" * max(1, 44 - len(label) - len(cfg.level_name)))
    print(f"  win rate     : {r['wins']/n*100:.2f}%  ({r['wins']}/{n})")
    print(f"  loss rate    : {r['mine_hits']/n*100:.2f}%  ({r['mine_hits']}/{n})")
    print(f"  mean reward  : {r['total_reward']/n:.4f}")
    print(f"  mean ep len  : {r['total_length']/n:.1f} moves")


def main() -> None:
    latest_zip = cfg.model_path      + ".zip"
    best_zip   = cfg.best_model_path + ".zip"

    results = {}

    if os.path.exists(latest_zip):
        print(f"[eval] loading latest model from {latest_zip}")
        model = MaskablePPO.load(cfg.model_path, device="auto")
        results["latest model"] = _eval_model(model, N_EPISODES, N_ENVS)
        del model
    else:
        print(f"[eval] latest model not found: {latest_zip}")

    if os.path.exists(best_zip):
        print(f"[eval] loading best model from {best_zip}")
        model = MaskablePPO.load(cfg.best_model_path, device="auto")
        results["best model"] = _eval_model(model, N_EPISODES, N_ENVS)
        del model
    else:
        print(f"[eval] best model not found: {best_zip}")

    print(f"[eval] running Bayes bot ({N_EPISODES} episodes) ...")
    results["bayes bot"] = _eval_bayes(N_EPISODES)

    for label, r in results.items():
        _print_result(label, r)


if __name__ == "__main__":
    main()
