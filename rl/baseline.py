"""Measuring only: bayes on given boards, with the same automatic first click.

Kept free of torch on purpose: it runs in worker processes, and every worker
that imported torch would load the CUDA libraries for nothing (enough of them
at once exhausts Windows' virtual memory).
"""
from __future__ import annotations

from engine import MinesweeperEngine
from rl.minesweeper_env import is_coin_flip, start_game


def record(eng: MinesweeperEngine, seed: int) -> dict:
    return {"seed": seed, "won": bool(eng.won), "moves": eng.move_count, "min_moves": eng.min_win_moves,
            "revealed_fraction": eng.revealed_safe_cells / eng.total_safe_cells}


def play_bayes_one(level: int, seed: int) -> dict:
    from ai_agents import bayes_reveal

    eng = MinesweeperEngine(level=level, seed=0)
    s = start_game(eng, seed)
    flip = False
    while eng.state == eng.RUNNING:
        flip = is_coin_flip(eng.mine_probs)
        bayes_reveal(eng)
    rec = record(eng, s)
    if not eng.won:                 # bayes always takes one of the safest cells
        rec["lost_by"] = "coin_flip" if flip else "unlucky"
    return rec


def play_bayes(level: int, board_seeds, n_jobs: int = 8) -> list[dict]:
    """bayes on every board, in `n_jobs` worker processes that are shut down afterwards."""
    from joblib import Parallel, delayed
    from joblib.externals.loky import get_reusable_executor

    try:
        return Parallel(n_jobs=n_jobs, backend="loky")(delayed(play_bayes_one)(level, s) for s in board_seeds)
    finally:
        get_reusable_executor().shutdown(wait=True)
