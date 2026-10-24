"""Audit every chosen click with exact probabilities, for measurement only.

The neural policy acts on visible observations. Solver values are read only
after its actions have been selected and never fed back into action selection.
This catches unnecessary guesses that survive and are absent from a fatal-only
loss breakdown. Immediate excess risk on visited states is a diagnostic, not a
proof of regret relative to the optimal eventual-win policy.

    python -m rl.policy_audit --model models/frozen.pt --split validation --games 1000 --output experiments/audit.json
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from rl.experiment_eval import board_seeds, sha256, source_hashes, write_json


def decision_risk(probabilities: np.ndarray, action: int, tolerance: float = 1e-9) -> dict:
    """Pure, per-decision diagnostics; NaN means a revealed/noncandidate cell."""
    flat = np.asarray(probabilities, dtype=float).ravel()
    if action < 0 or action >= flat.size or not np.isfinite(flat[action]):
        raise ValueError("selected action must have a finite hidden-cell mine probability")
    hidden = flat[np.isfinite(flat)]
    if not hidden.size or (hidden < -tolerance).any() or (hidden > 1 + tolerance).any():
        raise ValueError("invalid mine probabilities")
    chosen, safest = float(flat[action]), float(hidden.min())
    safe_available = safest <= tolerance
    forced = not safe_available
    excess = max(0.0, chosen - safest)
    return {
        "selected_mine_probability": chosen,
        "minimum_mine_probability": safest,
        "excess_mine_probability": excess,
        "safe_available": safe_available,
        "safe_available_choice_error": safe_available and chosen > tolerance,
        "forced_guess": forced,
        "riskier_than_safest": chosen > safest + tolerance,
        "riskier_forced_guess": forced and chosen > safest + tolerance,
        "coin_flip": forced and (float(hidden.max()) - safest < tolerance or safest >= 0.5 - tolerance),
        "known_mine_chosen": chosen >= 1 - tolerance,
    }


def summarize_audit(records: list[dict]) -> dict:
    """Per-board and per-decision denominators prevent misleading totals."""
    if not records:
        raise ValueError("at least one board is required")
    moves = [d for r in records for d in r["decisions"]]
    n, m = len(records), len(moves)
    fields = ("safe_available", "safe_available_choice_error", "forced_guess", "riskier_than_safest",
              "riskier_forced_guess", "coin_flip", "known_mine_chosen")
    totals = {name: sum(bool(d[name]) for d in moves) for name in fields}
    excess = sum(d["excess_mine_probability"] for d in moves)
    return {
        "games": n, "policy_decisions": m,
        "win": sum(bool(r["won"]) for r in records) / n,
        "counts": totals,
        "rates_per_decision": {key: value / m if m else 0.0 for key, value in totals.items()},
        "mean_counts_per_board": {key: value / n for key, value in totals.items()},
        "games_with_safe_available_choice_error": sum(any(d["safe_available_choice_error"] for d in r["decisions"])
                                                       for r in records),
        "games_with_riskier_forced_guess": sum(any(d["riskier_forced_guess"] for d in r["decisions"])
                                                for r in records),
        "mean_selected_risk_per_decision": sum(d["selected_mine_probability"] for d in moves) / m if m else 0.0,
        "mean_minimum_risk_per_decision": sum(d["minimum_mine_probability"] for d in moves) / m if m else 0.0,
        "total_excess_mine_probability": excess,
        "mean_excess_mine_probability_per_decision": excess / m if m else 0.0,
        "mean_excess_mine_probability_per_board": excess / n,
        "mean_forced_guess_progress": float(np.mean([d["progress"] for d in moves if d["forced_guess"]]))
        if any(d["forced_guess"] for d in moves) else None,
    }


def audit_model(q, seeds, level: int, device, *, n_envs: int = 128, canvas=None, bf16: bool = True) -> list[dict]:
    import torch

    from engine import MinesweeperEngine
    from rl.baseline import record
    from rl.minesweeper_env import encode, start_game
    from rl.qnet import greedy

    q.eval()
    seed_iter = iter(seeds)
    engines = [MinesweeperEngine(level=level, seed=0) for _ in range(n_envs)]
    rows, cols = engines[0].rows, engines[0].cols
    crows, ccols = canvas or (rows, cols)
    obs = np.zeros((n_envs, 2, crows, ccols), np.uint8)
    view = np.empty((rows, cols), np.int8)
    seed_of = [None] * n_envs
    decisions = [[] for _ in range(n_envs)]
    records = []

    def start(i):
        seed = next(seed_iter, None)
        seed_of[i] = None if seed is None else start_game(engines[i], seed)
        decisions[i] = []
        if seed is not None:
            encode(engines[i], obs[i], view)

    for i in range(n_envs):
        start(i)
    with torch.inference_mode():
        while any(s is not None for s in seed_of):
            active = [i for i in range(n_envs) if seed_of[i] is not None]
            x = torch.from_numpy(obs[active]).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and bf16):
                actions = greedy(q(x).float(), x).cpu().numpy()
            for i, action in zip(active, actions):
                eng = engines[i]
                row, col = divmod(int(action), ccols)
                # The model has already chosen. The solver is now an observer.
                d = decision_risk(eng.mine_probs, row * cols + col)
                d.update(move=eng.move_count + 1, row=row, col=col,
                         progress=eng.revealed_safe_cells / eng.total_safe_cells)
                decisions[i].append(d)
                eng.reveal_count(row, col)
                if eng.state == eng.OVER:
                    records.append({**record(eng, seed_of[i]), "decisions": decisions[i]})
                    start(i)
                else:
                    encode(eng, obs[i], view)
    return sorted(records, key=lambda r: r["seed"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="an immutable checkpoint")
    ap.add_argument("--split", choices=("validation", "test"), default="validation")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--fp32-head", action="store_true")
    ap.add_argument("--symmetries", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.n_envs < 1:
        ap.error("--n-envs must be positive")
    seeds = board_seeds(args.split, args.level, args.games, args.replicate)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    from rl.evaluate import summarize
    from rl.qnet import MODELS_DIR, SymmetryEnsemble, load_checkpoint

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    path = Path(args.model)
    if not path.exists():
        path = MODELS_DIR / (path.name if path.suffix == ".pt" else f"{path.name}.pt")
    digest = sha256(path)
    q, ckpt = load_checkpoint(path, device)
    if args.fp32_head:
        q.config["fp32_head"] = True
    inference_config = dict(q.config)
    if args.symmetries:
        q = SymmetryEnsemble(q, probabilities=(ckpt.get("args") or {}).get("loss") == "bce")
    records = audit_model(q, seeds, args.level, device, n_envs=args.n_envs,
                          canvas=ckpt.get("canvas"), bf16=not args.no_bf16)
    if digest != sha256(path):
        raise RuntimeError("checkpoint changed during audit; use an immutable copy")
    result = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(path.resolve()), "model_sha256": digest,
        "training_step": ckpt.get("global_step"), "training_args": ckpt.get("args"),
        "inference_config": inference_config,
        "split": args.split, "level": args.level, "games": args.games, "replicate": args.replicate,
        "requested_seed_start": seeds.start, "requested_seed_stop": seeds.stop,
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "device": str(device),
                    "bf16": device.type == "cuda" and not args.no_bf16, "symmetries": args.symmetries,
                    "fp32_head_override": args.fp32_head, "n_envs": args.n_envs},
        "source_sha256": {**source_hashes(), "rl/policy_audit.py": sha256(Path(__file__))},
        "interpretation": "Solver used only after action selection. Automatic first clicks excluded. Immediate excess risk on visited states is a diagnostic, not optimal eventual-win regret. Test boards must remain untouched during tuning.",
        "performance": summarize(records), "audit": summarize_audit(records), "records": records,
    }
    write_json(args.output, result)
    s = result["audit"]
    print(f"{s['games']} boards, {s['policy_decisions']} decisions, win {s['win']*100:.2f}%; "
          f"safe-available errors {s['counts']['safe_available_choice_error']}, "
          f"riskier forced guesses {s['counts']['riskier_forced_guess']}, "
          f"excess mine probability/board {s['mean_excess_mine_probability_per_board']:.5f}", flush=True)


if __name__ == "__main__":
    main()
