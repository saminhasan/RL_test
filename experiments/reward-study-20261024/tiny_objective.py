"""Enumerate a toy reward policy exactly, without a network or GPU.

The immediate objective is -P(mine) + E[new safe cells]/(ratio*S).
Report its actual eventual win probability, with tied actions chosen uniformly.
This tests the objective, not whether a neural network can learn it. TinyBoard's
Bayes uses exact P(zero); live bayes_reveal uses a product approximation for its
zero-opening tie break, so these toy scores are not live-agent benchmarks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rl.exact_tiny import WIN, TinyBoard


def immediate_win(tb: TinyBoard, nodes: dict, roots: dict, ratio: Fraction) -> Fraction:
    values = {}
    for key, node in sorted(nodes.items(), key=lambda item: -item[1].revealed.bit_count()):
        total, before = len(node.layouts), node.revealed.bit_count()
        scores, wins = {}, {}
        for cell, (lost, kids) in node.actions.items():
            gained = sum(mass * ((tb.safe_total if child == WIN else child[0].bit_count()) - before)
                         for child, mass in kids.items())
            scores[cell] = -Fraction(lost, total) + Fraction(gained, total * tb.safe_total) / ratio
            wins[cell] = sum(Fraction(mass, total) * (1 if child == WIN else values[child])
                             for child, mass in kids.items())
        best = max(scores.values())
        chosen = [cell for cell, score in scores.items() if score == best]
        values[key] = sum(wins[cell] for cell in chosen) / len(chosen)
    total = sum(roots.values())
    return sum(Fraction(mass, total) * values[key] for key, mass in roots.items())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--mines", type=int, default=2)
    ap.add_argument("--ratios", nargs="+", default=["10", "100", "1000", "1000000"])
    ap.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = ap.parse_args()
    ratios = [Fraction(text) for text in args.ratios]
    if any(r <= 0 for r in ratios):
        ap.error("ratios must be positive")
    tb = TinyBoard(args.rows, args.cols, args.mines)
    firsts = []
    for first in range(tb.n):
        nodes, roots = tb.build(first)
        tb.solve(nodes)
        objectives = {str(r): immediate_win(tb, nodes, roots, r) for r in ratios}
        firsts.append({"first_cell": first, "nodes": len(nodes),
                       "nonwinning_first_layouts": sum(roots.values()),
                       "perfect_win": tb.start_value(nodes, roots, "perfect"),
                       "bayes_exact_zero_win": tb.start_value(nodes, roots, "bayes"),
                       "immediate_win": {r: float(p) for r, p in objectives.items()},
                       "immediate_win_fraction": {r: str(p) for r, p in objectives.items()}})
    mean = lambda field: sum(row[field] for row in firsts) / tb.n
    average = {str(r): sum(Fraction(row["immediate_win_fraction"][str(r)]) for row in firsts) / tb.n
               for r in ratios}
    files = [Path(__file__), ROOT / "rl/exact_tiny.py"]
    result = {"board": {"rows": args.rows, "cols": args.cols, "mines": args.mines},
              "device": "cpu", "python": platform.python_version(),
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              "first_click": "safe; immediate wins redealt; uniformly average all first cells",
              "immediate_score": "-P(mine) + E[newly revealed safe cells]/(ratio*S)",
              "ties": "uniform among exactly tied rational expected immediate scores",
              "perfect_win": mean("perfect_win"),
              "bayes_exact_zero_win": mean("bayes_exact_zero_win"),
              "immediate_win": {r: float(p) for r, p in average.items()},
              "immediate_win_fraction": {r: str(p) for r, p in average.items()},
              "first_cells": firsts,
              "caveat": "Toy objective diagnostic only. Tiny Bayes uses exact zero probabilities; live Bayes uses a product approximation. No neural training or easy-board superiority is demonstrated."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("perfect_win", "bayes_exact_zero_win", "immediate_win")}, indent=2))


if __name__ == "__main__":
    main()
