"""Measuring only: how a trained Q-network plays, next to bayes, on the same boards.

Evaluation boards come from a seed range training never draws from (training
draws below 2**31). Every agent gets the same automatic first click, so they
play identical games.

    win         -- share of games won (with a 95% margin)
    efficiency  -- oracle's fewest possible moves / moves taken, over wins
    breakdown   -- why lost games were lost, judged by the exact solver:
                     missed_safe  a provably safe cell existed, it clicked something else
                     risky_guess  it had to guess, but not on one of the safest cells
                     unlucky      it took one of the safest guesses and hit a mine
                   bayes only ever loses the unlucky way.

The solver and the oracle appear here and nowhere in training.

    python -m rl.evaluate --model models/<run>-best.pt --level 4 --games 2000 --breakdown
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from engine import MinesweeperEngine
from rl.baseline import play_bayes, record
from rl.minesweeper_env import encode, start_game
from rl.qnet import greedy, load_checkpoint

EVAL_SEED0 = 10_000_000_000


def eval_board_seeds(level: int, n: int) -> range:
    start = EVAL_SEED0 + level * 10_000_000
    return range(start, start + n)


@torch.no_grad()
def play_q(q, level: int, board_seeds, device, n_envs: int = 256, breakdown: bool = False,
           canvas: tuple[int, int] | None = None) -> list[dict]:
    """Greedy play, many boards at once, drawn on `canvas` (default: the board's
    own size; use the canvas the net was trained on). breakdown=True also asks
    the solver how the fatal click of each lost game compares (measuring only)."""
    q.eval()
    seeds = iter(board_seeds)
    engines = [MinesweeperEngine(level=level, seed=0) for _ in range(n_envs)]
    rows, cols = engines[0].rows, engines[0].cols
    crows, ccols = canvas or (rows, cols)
    obs  = np.zeros((n_envs, 2, crows, ccols), np.uint8)
    view = np.empty((rows, cols), np.int8)
    seed_of = [None] * n_envs
    records = []

    def start(i: int) -> None:
        s = next(seeds, None)
        seed_of[i] = None if s is None else start_game(engines[i], s)
        if s is not None:
            encode(engines[i], obs[i], view)

    for i in range(n_envs):
        start(i)
    while any(s is not None for s in seed_of):
        active = [i for i in range(n_envs) if seed_of[i] is not None]
        x = torch.from_numpy(obs[active]).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            qv = q(x).float()
        actions = greedy(qv, x).cpu().numpy()
        for i, a in zip(active, actions):
            eng = engines[i]
            r, c = divmod(int(a), ccols)              # a canvas cell; only on-board hidden cells are picked
            if breakdown:
                p = np.nan_to_num(eng.mine_probs, nan=np.inf)
                pmin, picked = p.min(), p[r, c]
            eng.reveal_count(r, c)
            if eng.state == eng.OVER:
                rec = record(eng, seed_of[i])
                if breakdown and not eng.won:
                    rec["lost_by"] = ("missed_safe" if pmin <= 1e-9 else
                                      "risky_guess" if picked > pmin + 1e-9 else "unlucky")
                records.append(rec)
                start(i)
            else:
                encode(eng, obs[i], view)
    return records


def summarize(records: list[dict]) -> dict:
    n    = len(records)
    won  = np.array([r["won"] for r in records], bool)
    wins = [r for r in records if r["won"]]
    win  = float(won.mean()) if n else float("nan")
    out = {
        "games":      n,
        "win":        win,
        "win_margin": 1.96 * math.sqrt(win * (1 - win) / n) if n else float("nan"),
        "efficiency": float(np.mean([r["min_moves"] / r["moves"] for r in wins])) if wins else float("nan"),
        "revealed":   float(np.mean([r["revealed_fraction"] for r in records])) if n else float("nan"),
    }
    if any("lost_by" in r for r in records):
        for k in ("missed_safe", "risky_guess", "unlucky"):
            out[k] = sum(r.get("lost_by") == k for r in records) / n
    return out


def format_row(name: str, s: dict) -> str:
    row = (f"{name:26s} win {s['win']*100:6.2f}% +/-{s['win_margin']*100:4.2f}   eff {s['efficiency']:.3f}   "
           f"revealed {s['revealed']*100:5.1f}%   ({s['games']} games)")
    if "unlucky" in s:
        row += (f"\n{'':26s} lost by: missed safe {s['missed_safe']*100:4.1f}%   risky guess "
                f"{s['risky_guess']*100:4.1f}%   unlucky {s['unlucky']*100:4.1f}%")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--breakdown", action="store_true", help="classify losses with the exact solver")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q, ckpt = load_checkpoint(args.model, device)
    seeds = eval_board_seeds(args.level, args.games)
    print(format_row("bayes", summarize(play_bayes(args.level, seeds))))
    print(format_row(f"dqn (step {ckpt.get('global_step', '?')})",
                     summarize(play_q(q, args.level, seeds, device, breakdown=args.breakdown,
                                      canvas=ckpt.get("canvas")))))


if __name__ == "__main__":
    main()
