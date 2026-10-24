"""Compare bf16 and fp32 choices on fresh Bayesian-policy board positions.

Run from the repo root with .venv/Scripts/python.exe
experiments/reward-study-20261024/diagnose_precision.py --output <result.json>.
This is a choice/calibration diagnostic, not an independent win-rate evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ai_agents import bayes_reveal
from engine import MinesweeperEngine
from rl.minesweeper_env import encode, start_game
from rl.qnet import greedy, load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/easy-exp2-c64x16-20261024-040042-best.pt")
    ap.add_argument("--seed-start", type=int, default=30_000_000_000)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--max-states", type=int, default=512)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    model = Path(args.model)
    if not model.is_absolute():
        model = ROOT / model
    q, checkpoint = load_checkpoint(model, "cuda")
    q.eval()
    observations, risks = [], []
    for seed in range(args.seed_start, args.seed_start + args.games):
        engine = MinesweeperEngine(level=2, seed=0)
        start_game(engine, seed)
        while engine.state == engine.RUNNING and len(observations) < args.max_states:
            obs = np.zeros((2, 9, 9), np.uint8)
            encode(engine, obs, np.empty((9, 9), np.int8))
            observations.append(obs)
            risks.append(engine.mine_probs.copy().ravel())
            bayes_reveal(engine)
        if len(observations) >= args.max_states:
            break
    array, probabilities = np.stack(observations), np.stack(risks)
    x = torch.from_numpy(array).cuda()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bf16 = q(x).float()
        fp32 = q(x).float()
        picks = {"bf16": greedy(bf16, x).cpu().numpy(), "fp32": greedy(fp32, x).cpu().numpy()}
        legal = (x[:, 0] == 0).flatten(1)
        masked = bf16.masked_fill(~legal, float("-inf"))
        ties = (masked == masked.max(1, keepdim=True).values).sum(1).cpu().numpy()
        values = [(-10_000 + 10_100 * torch.sigmoid(z)).cpu().numpy() for z in (bf16, fp32)]
    index, lowest = np.arange(len(array)), np.nanmin(probabilities, axis=1)
    result = {
        "model": str(model),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "checkpoint_step": checkpoint["global_step"],
        "seed_start": args.seed_start,
        "games": args.games,
        "max_states": args.max_states,
        "states": len(array),
        "sampling_policy": "bayes",
        "level": 2,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "bf16_top_tie_fraction": float((ties > 1).mean()),
        "mean_top_ties": float(ties.mean()),
        "fp32_changes_choice": float((picks["bf16"] != picks["fp32"]).mean()),
        "mean_abs_reward_difference_legal": float(np.abs(values[0] - values[1])[array[:, 0].reshape(-1, 81) == 0].mean()),
        "safe_available_fraction": float((lowest < 1e-9).mean()),
        "choices": {},
        "note": "Reward reconstruction assumes this checkpoint's mine=10000, reveal=100, gamma=0 setup.",
    }
    for name, actions in picks.items():
        selected = probabilities[index, actions]
        result["choices"][name] = {
            "missed_safe_choice": int(((lowest < 1e-9) & (selected > 1e-9)).sum()),
            "riskier_choice": int((selected > lowest + 1e-9).sum()),
            "mean_mine_risk": float(selected.mean()),
        }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
