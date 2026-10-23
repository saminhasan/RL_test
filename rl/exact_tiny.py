"""Measuring only: perfect play on tiny boards, worked out exactly as a graph.

Every visible board is a node. Clicking a covered cell forks: to a loss with
the chance that the cell is a mine, otherwise to one of the boards the click
can reveal, each with its own chance (every mine layout that fits the visible
board is equally likely). Working back from the won boards gives, at every
node, the win chance of

    perfect play   the best click everywhere
    bayes          the lowest mine chance, ties broken towards the likeliest
                   zero and then at random, as in ai_agents.bayes_reveal
    a Q-network    its greedy click (optional, --model)

The first click is the automatic, safe one used in training: mines go anywhere
except there, and layouts where that click already wins are dealt again (as in
rl.minesweeper_env.start_game). Results are averaged over every first cell.

Safe clicks never change the perfect-play win chance (you would click the cell
eventually anyway, and knowing its number can only help), so a node with a
safe cell is never a decision. The decisions are the nodes without one: forced
guesses. --dot writes the graph contracted to those decisions.

    python -m rl.exact_tiny --rows 4 --cols 4 --mines 3
    python -m rl.exact_tiny --rows 3 --cols 3 --mines 2 --first 0 --dot graph.dot --model models/<run>-best.pt
"""
from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

WIN, LOSS = "win", "loss"


@dataclass
class Node:
    revealed: int                       # bitmask of uncovered cells
    numbers: tuple                      # the numbers on them, in cell order
    layouts: list                       # mine layouts (bitmasks) that fit this board
    actions: dict = field(default_factory=dict)   # cell -> (mine layouts, {child key: layouts})
    p_zero: dict = field(default_factory=dict)    # cell -> chance it shows a 0
    value: dict = field(default_factory=dict)     # policy -> win chance from here
    choice: dict = field(default_factory=dict)    # policy -> the cells it may click


class TinyBoard:
    def __init__(self, rows: int, cols: int, mines: int) -> None:
        self.rows, self.cols, self.mines = rows, cols, mines
        self.n = rows * cols
        self.nb = []
        for c in range(self.n):
            r, k = divmod(c, cols)
            mask = 0
            for dr, dk in itertools.product((-1, 0, 1), repeat=2):
                rr, kk = r + dr, k + dk
                if (dr or dk) and 0 <= rr < rows and 0 <= kk < cols:
                    mask |= 1 << (rr * cols + kk)
            self.nb.append(mask)
        self.safe_total = self.n - mines

    def count(self, layout: int, c: int) -> int:
        return bin(layout & self.nb[c]).count("1")

    def reveal(self, revealed: int, layout: int, c: int) -> int:
        """The uncovered cells after clicking safe cell c: zeros open their neighbours."""
        stack = [c]
        while stack:
            x = stack.pop()
            if revealed >> x & 1:
                continue
            revealed |= 1 << x
            if self.count(layout, x) == 0:
                stack.extend(y for y in range(self.n) if self.nb[x] >> y & 1 and not revealed >> y & 1)
        return revealed

    def key(self, revealed: int, layout: int) -> tuple:
        return revealed, tuple(self.count(layout, c) for c in range(self.n) if revealed >> c & 1)

    def won(self, revealed: int) -> bool:
        return bin(revealed).count("1") == self.safe_total

    # -- the graph ---------------------------------------------------------------

    def build(self, first: int) -> tuple[dict, dict]:
        """All boards reachable after the automatic first click on `first`.
        Returns (nodes by key, {root key: number of layouts})."""
        others = [c for c in range(self.n) if c != first]
        roots: dict = defaultdict(list)
        for mines in itertools.combinations(others, self.mines):
            layout = sum(1 << c for c in mines)
            rev = self.reveal(0, layout, first)
            if not self.won(rev):                       # a first click that wins at once is dealt again
                roots[self.key(rev, layout)].append(layout)
        nodes: dict = {}
        todo = []
        for k, layouts in roots.items():
            nodes[k] = Node(k[0], k[1], layouts)
            todo.append(k)
        while todo:
            node = nodes[todo.pop()]
            for c in range(self.n):
                if node.revealed >> c & 1:
                    continue
                lost, kids, zeros = 0, defaultdict(list), 0
                for layout in node.layouts:
                    if layout >> c & 1:
                        lost += 1
                        continue
                    zeros += self.count(layout, c) == 0
                    rev = self.reveal(node.revealed, layout, c)
                    kids[WIN if self.won(rev) else self.key(rev, layout)].append(layout)
                node.p_zero[c] = zeros / len(node.layouts)
                node.actions[c] = (lost, {k: len(v) for k, v in kids.items()})
                for k, v in kids.items():
                    if k != WIN and k not in nodes:
                        nodes[k] = Node(k[0], k[1], v)
                        todo.append(k)
        return nodes, {k: len(v) for k, v in roots.items()}

    def solve(self, nodes: dict, net_choice: dict | None = None) -> None:
        """Win chances from every node under each policy (children always have more cells uncovered)."""
        order = sorted(nodes.values(), key=lambda nd: -bin(nd.revealed).count("1"))
        policies = ["perfect", "bayes"] + (["net"] if net_choice is not None else [])
        for nd in order:
            total = len(nd.layouts)
            q = {}
            for c, (lost, kids) in nd.actions.items():
                for pol in policies:
                    q.setdefault(pol, {})[c] = sum(m / total * (1.0 if k == WIN else nodes[k].value[pol])
                                                   for k, m in kids.items())
            risk = {c: lost / total for c, (lost, _) in nd.actions.items()}
            # perfect: the best click
            best = max(q["perfect"].values())
            nd.choice["perfect"] = [c for c, v in q["perfect"].items() if v >= best - 1e-12]
            nd.value["perfect"] = best
            # bayes: lowest mine chance, then likeliest zero, then any of those at random
            low = min(risk.values())
            cand = [c for c in risk if risk[c] <= low + 1e-12]
            z = max(nd.p_zero[c] for c in cand)
            cand = [c for c in cand if nd.p_zero[c] >= z - 1e-12]
            nd.choice["bayes"] = cand
            nd.value["bayes"] = float(np.mean([q["bayes"][c] for c in cand]))
            if net_choice is not None:
                c = net_choice[(nd.revealed, nd.numbers)]
                nd.choice["net"] = [c]
                nd.value["net"] = q["net"][c]
            nd.q = q["perfect"]
            nd.risk = risk

    def start_value(self, nodes: dict, roots: dict, policy: str) -> float:
        total = sum(roots.values())
        return sum(m / total * nodes[k].value[policy] for k, m in roots.items())

    def visits(self, nodes: dict, roots: dict, policy: str) -> dict:
        """The chance of reaching each node when playing `policy` (ties split evenly)."""
        p = defaultdict(float)
        total = sum(roots.values())
        for k, m in roots.items():
            p[k] += m / total
        for nd in sorted(nodes.values(), key=lambda nd: bin(nd.revealed).count("1")):
            here = p[(nd.revealed, nd.numbers)]
            if not here:
                continue
            cells = nd.choice[policy]
            for c in cells:
                _, kids = nd.actions[c]
                for k, m in kids.items():
                    if k != WIN:
                        p[k] += here / len(cells) * m / len(nd.layouts)
        return p

    # -- the Q-network's clicks ------------------------------------------------------

    def net_choices(self, nodes: dict, model: str) -> dict:
        import torch

        from rl.qnet import greedy, load_checkpoint

        q, _ = load_checkpoint(model)
        keys = list(nodes)
        obs = np.zeros((len(keys), 2, self.rows, self.cols), np.uint8)
        for i, (rev, numbers) in enumerate(keys):
            it = iter(numbers)
            for c in range(self.n):
                if rev >> c & 1:
                    obs[i, 0].flat[c] = next(it) + 1           # clue k -> k + 1, hidden stays 0
            obs[i, 1] = self.mines
        with torch.no_grad():
            x = torch.from_numpy(obs)
            picks = greedy(q(x).float(), x).numpy()
        return {k: int(a) for k, a in zip(keys, picks)}

    # -- drawing ----------------------------------------------------------------------

    def board_text(self, nd: Node, mark: int | None = None) -> str:
        it = iter(nd.numbers)
        rows = []
        for r in range(self.rows):
            row = ""
            for k in range(self.cols):
                c = r * self.cols + k
                row += (str(next(it)) if nd.revealed >> c & 1 else ("x" if c == mark else "#"))
            rows.append(row)
        return "\\n".join(rows)

    def dot(self, nodes: dict, roots: dict, path: str) -> None:
        """The graph contracted to decisions: each forced guess, the guesses perfect play and bayes
        make, and where each leads (the next forced guess, a win, or a loss), with chances."""
        reach = self.visits(nodes, roots, "perfect")
        bayes_reach = self.visits(nodes, roots, "bayes")

        def next_decisions(k, share, out):
            """Follow safe clicks (perfect play's) from node k to the next forced guesses."""
            if k == WIN:
                out[WIN] += share
                return
            nd = nodes[k]
            if min(nd.risk.values()) > 1e-12:
                out[k] += share
                return
            c = next(c for c in nd.choice["perfect"] if nd.risk[c] <= 1e-12)   # a safe click: always a best one
            for kk, m in nd.actions[c][1].items():
                next_decisions(kk, share * m / len(nd.layouts), out)

        def colour(v):
            r, g = int(255 * (1 - v)), int(200 * v)
            return f"#{r:02x}{g:02x}50"

        decisions = [k for k, nd in nodes.items() if min(nd.risk.values()) > 1e-12
                     and (reach[k] > 1e-12 or bayes_reach[k] > 1e-12)]
        lines = ["digraph G {", '  rankdir=LR; node [fontname="Consolas", fontsize=10, shape=box, style="rounded,filled"];',
                 '  win [label="WIN", fillcolor="#7bd389", shape=doublecircle];',
                 '  loss [label="LOSS", fillcolor="#f28b82", shape=doublecircle];',
                 '  start [label="first click", shape=plaintext, style=""];']
        ids = {k: f"n{i}" for i, k in enumerate(decisions)}

        def ref(k):
            return "win" if k == WIN else ids[k]
        start_out = defaultdict(float)
        total = sum(roots.values())
        for k, m in roots.items():
            next_decisions(k, m / total, start_out)
        for k, nd in ((k, nodes[k]) for k in decisions):
            p, b = nd.value["perfect"], nd.value["bayes"]
            note = f"perfect {p:.2f}  bayes {b:.2f}" + ("  <- bayes loses here" if b < p - 1e-9 else "")
            lines.append(f'  {ids[k]} [label="{self.board_text(nd)}\\n{note}", fillcolor="{colour(p)}"];')
        for k, share in start_out.items():
            lines.append(f'  start -> {ref(k)} [label="{share:.2f}"];')
        for k in decisions:
            nd = nodes[k]
            best = nd.choice["perfect"][0]
            for c, style in [(best, "perfect"), *[(c, "bayes") for c in nd.choice["bayes"][:1] if c != best]]:
                lost, kids = nd.actions[c]
                hub = f"{ids[k]}_{c}"
                r, kk = divmod(c, self.cols)
                colr = "#1a73e8" if style == "perfect" else "#e8710a"
                lines.append(f'  {hub} [shape=point, width=0.08, color="{colr}"];')
                lines.append(f'  {ids[k]} -> {hub} [label="{style}: ({r},{kk})  mine {nd.risk[c]:.2f}", color="{colr}", fontcolor="{colr}"];')
                lines.append(f'  {hub} -> loss [label="{lost / len(nd.layouts):.2f}", color="#f28b82"];')
                out = defaultdict(float)
                for k2, m in kids.items():
                    next_decisions(k2, m / len(nd.layouts), out)
                for k2, share in out.items():
                    lines.append(f'  {hub} -> {ref(k2)} [label="{share:.2f}", color="{colr}"];')
        lines.append("}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--mines", type=int, default=3)
    ap.add_argument("--first", type=int, default=-1, help="first click cell (default: average over all)")
    ap.add_argument("--model", default="", help="also play a Q-network checkpoint")
    ap.add_argument("--dot", default="", help="write the decision graph (needs --first)")
    args = ap.parse_args()

    tb = TinyBoard(args.rows, args.cols, args.mines)
    firsts = [args.first] if args.first >= 0 else range(tb.n)
    sums = defaultdict(float)
    stats = defaultdict(float)
    for first in firsts:
        nodes, roots = tb.build(first)
        tb.solve(nodes, tb.net_choices(nodes, args.model) if args.model else None)
        reach = tb.visits(nodes, roots, "perfect")
        bayes_reach = tb.visits(nodes, roots, "bayes")
        for pol in ("perfect", "bayes", "net") if args.model else ("perfect", "bayes"):
            sums[pol] += tb.start_value(nodes, roots, pol) / len(firsts)
        dec = [nd for nd in nodes.values() if min(nd.risk.values()) > 1e-12]
        stats["nodes"] += len(nodes) / len(firsts)
        stats["decisions"] += len(dec) / len(firsts)
        stats["bayes_wrong"] += sum(nd.value["bayes"] < nd.value["perfect"] - 1e-9 for nd in dec) / len(firsts)
        stats["guesses_perfect"] += sum(reach[(nd.revealed, nd.numbers)] for nd in dec) / len(firsts)
        stats["guesses_bayes"] += sum(bayes_reach[(nd.revealed, nd.numbers)] for nd in dec) / len(firsts)
        ties = [len(nd.choice["perfect"]) for nd in dec]
        stats["one_best"] += (sum(t == 1 for t in ties) / max(1, len(ties))) / len(firsts)
        gaps = []
        for nd in dec:
            vals = sorted(nd.q.values(), reverse=True)
            if len(vals) > 1:
                gaps.append(vals[0] - vals[1])
        stats["gap_median"] += float(np.median(gaps)) / len(firsts) if gaps else 0.0
        if args.dot and args.first >= 0:
            tb.dot(nodes, roots, args.dot)
    print(f"{args.rows}x{args.cols}, {args.mines} mines, first click "
          + (f"cell {args.first}" if args.first >= 0 else "averaged over every cell"))
    print(f"  boards in the graph: {stats['nodes']:.0f}, of them forced guesses: {stats['decisions']:.0f}"
          f" (bayes picks a worse guess than perfect play at {stats['bayes_wrong']:.0f})")
    print(f"  forced guesses per game: perfect play {stats['guesses_perfect']:.2f}, bayes {stats['guesses_bayes']:.2f}")
    print(f"  at a forced guess: one best cell {stats['one_best']*100:.0f}% of the time;"
          f" median gap between the best and the second-best guess {stats['gap_median']*100:.1f} points")
    print("  win chance: " + ", ".join(f"{k} {v*100:.2f}%" for k, v in sums.items()))


if __name__ == "__main__":
    main()
