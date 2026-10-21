"""Engine and solver checks.

The solver is checked against brute force: on small boards, enumerate every
mine placement consistent with what the player can see and count how often
each hidden cell holds a mine. That is the exact answer by definition.

Run:  python tests/test_engine.py      (or pytest)
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import MinesweeperEngine  # noqa: E402


def brute_force_probs(eng: MinesweeperEngine) -> np.ndarray:
    """Exact mine probabilities by enumerating every consistent placement."""
    rows, cols = eng.rows, eng.cols
    revealed   = eng._revealed_flat.copy()
    counts     = eng._neighbor_counts_flat.astype(int)
    exploded   = -1 if eng.exploded_cell is None else eng.exploded_cell[0] * cols + eng.exploded_cell[1]
    hidden     = np.flatnonzero(~revealed)
    to_place   = eng.mine_count - (1 if exploded >= 0 else 0)

    clues = [f for f in np.flatnonzero(revealed) if f != exploded]
    nbrs  = [set(eng._neighbor_indices[f, : eng._neighbor_lengths[f]]) for f in range(rows * cols)]
    adj   = np.array([[h in nbrs[c] for h in hidden] for c in clues], dtype=np.int64).reshape(len(clues), len(hidden))
    rhs   = np.array([counts[c] - (exploded in nbrs[c]) for c in clues], dtype=np.int64)

    combos = np.array(list(itertools.combinations(range(len(hidden)), to_place)), dtype=np.int64)
    placed = np.zeros((len(combos), len(hidden)), dtype=np.int64)
    if to_place:
        np.put_along_axis(placed, combos, 1, axis=1)
    ok = (placed @ adj.T == rhs).all(axis=1)

    out = np.full(rows * cols, np.nan)
    out[hidden] = placed[ok].mean(axis=0)
    return out.reshape(rows, cols)


def _random_states(rng: np.random.Generator, rows: int, cols: int, mines: int, games: int):
    """Yield engines at every step of random games on a custom board."""
    for _ in range(games):
        layout = np.zeros(rows * cols, dtype=bool)
        layout[rng.choice(rows * cols, size=mines, replace=False)] = True
        eng = MinesweeperEngine.from_state(layout.reshape(rows, cols), np.zeros((rows, cols), dtype=bool))
        while eng.state == eng.RUNNING:
            hidden = np.flatnonzero(~eng._revealed_flat)
            safe   = hidden[~eng._mines_flat[hidden]]
            # mostly safe moves to get deep, sometimes any cell to reach losses
            pool = hidden if rng.random() < 0.15 else safe
            eng.reveal_count(*divmod(int(rng.choice(pool)), cols))
            yield eng


def test_solver_matches_brute_force() -> None:
    rng = np.random.default_rng(1)
    checked = 0
    for rows, cols, mines in [(4, 5, 4), (5, 5, 5), (4, 6, 6)]:
        for eng in _random_states(rng, rows, cols, mines, games=40):
            got  = eng.get_mine_probabilities()
            want = brute_force_probs(eng)
            assert np.array_equal(np.isnan(got), np.isnan(want))
            assert np.allclose(got[~np.isnan(got)], want[~np.isnan(want)], atol=1e-12), (got, want)
            checked += 1
    assert checked > 100


def test_probabilities_sum_to_mine_count() -> None:
    # With a fixed mine total, the expected number of hidden mines is exact.
    rng = np.random.default_rng(2)
    for level in (3, 4):
        for g in range(10):
            eng = MinesweeperEngine(level=level, seed=int(rng.integers(2**31)))
            while eng.state == eng.RUNNING:
                p      = eng.mine_probs
                hidden = np.flatnonzero(~eng._revealed_flat)
                assert abs(np.nansum(p) - (eng.mine_count - eng.hit_mine)) < 1e-9
                hp = p.ravel()[hidden]
                eng.reveal_count(*divmod(int(rng.choice(hidden[hp == hp.min()])), eng.cols))


def test_first_click_is_never_a_mine() -> None:
    for seed in range(300):
        eng = MinesweeperEngine(level=4, seed=seed)
        eng.reveal_count(seed % 16, seed % 30)
        assert not eng.hit_mine
        assert int(eng._mines_flat.sum()) == eng.mine_count


def test_same_seed_same_board() -> None:
    a = MinesweeperEngine(level=3, seed=123)
    b = MinesweeperEngine(level=3, seed=123)
    a.reveal_count(5, 5)
    b.reveal_count(5, 5)
    assert np.array_equal(a._mines, b._mines)
    a.reset()  # no seed: replays the same board
    a.reveal_count(5, 5)
    assert np.array_equal(a._mines, b._mines)


def test_game_flow_and_views() -> None:
    eng = MinesweeperEngine(level=2, seed=5)
    assert eng.state == eng.RUNNING and not eng.started
    assert np.allclose(eng.mine_probs[~eng.revealed], 10 / 81)

    n = eng.reveal_count(4, 4)
    assert n >= 1 and eng.started and eng.move_count == 1
    assert eng.reveal_count(4, 4) == 0 and eng.move_count == 1  # already revealed: no-op

    view = eng.get_public_view()
    assert view.dtype == np.int8
    assert np.array_equal(view >= 0, eng.revealed)
    assert np.array_equal(view[eng.revealed], eng._neighbor_counts[eng.revealed])

    # play safe cells only until the board is cleared
    while eng.state == eng.RUNNING:
        eng.reveal_count(*divmod(int(eng._safe_flat_cells[0]), eng.cols))
    assert eng.won and not eng.hit_mine
    assert eng.revealed_safe_cells == eng.total_safe_cells
    assert eng._safe_pool_size == 0


def test_loss_and_from_state() -> None:
    mines    = np.zeros((4, 4), dtype=bool)
    mines[0, 0] = mines[3, 3] = True
    revealed = np.zeros((4, 4), dtype=bool)
    revealed[0, 0] = True
    eng = MinesweeperEngine.from_state(mines, revealed)
    assert eng.hit_mine and eng.state == eng.OVER and eng.exploded_cell == (0, 0)
    view = eng.get_public_view()
    assert view[0, 0] == 9 and view[3, 3] == 9
    assert eng.get_public_view(reveal_mines_on_loss=False)[3, 3] == -1

    revealed[:] = ~mines
    eng = MinesweeperEngine.from_state(mines, revealed)
    assert eng.won and eng.state == eng.OVER


def _fewest_clicks_by_search(mines: np.ndarray) -> int:
    """Fewest clicks to clear a board with known mines, by breadth-first search
    over every reachable set of revealed cells. Only feasible on tiny boards."""
    rows, cols = mines.shape
    eng   = MinesweeperEngine.from_state(mines, np.zeros_like(mines))
    nbrs  = [eng._neighbor_indices[f, : eng._neighbor_lengths[f]] for f in range(rows * cols)]
    zero  = eng._neighbor_counts_flat == 0
    safe  = [f for f in range(rows * cols) if not mines.flat[f]]
    goal  = sum(1 << f for f in safe)

    def click(state: int, f: int) -> int:
        todo = [f]
        state |= 1 << f
        while todo:
            cur = todo.pop()
            if zero[cur]:
                for g in nbrs[cur]:
                    if not state >> int(g) & 1:
                        state |= 1 << int(g)
                        todo.append(int(g))
        return state

    frontier, seen, depth = {0}, {0}, 0
    while goal not in frontier:
        depth += 1
        nxt = set()
        for s in frontier:
            for f in safe:
                if not s >> f & 1:
                    t = click(s, f)
                    if t not in seen:
                        seen.add(t)
                        nxt.add(t)
        frontier = nxt
    return depth


def test_min_clicks_matches_search() -> None:
    rng = np.random.default_rng(3)
    for rows, cols, mines in [(3, 4, 2), (4, 4, 3), (4, 5, 4), (5, 4, 2)]:
        for _ in range(40):
            layout = np.zeros(rows * cols, dtype=bool)
            layout[rng.choice(rows * cols, size=mines, replace=False)] = True
            layout = layout.reshape(rows, cols)
            eng = MinesweeperEngine.from_state(layout, np.zeros_like(layout))
            assert eng.min_clicks_left() == _fewest_clicks_by_search(layout)


def test_oracle_wins_in_min_moves() -> None:
    from ai_agents import oracle_reveal

    for level in (1, 2, 3, 4):
        for seed in range(40):
            eng = MinesweeperEngine(level=level, seed=seed)
            assert eng.min_win_moves is None
            oracle_reveal(eng)
            target = eng.min_win_moves
            assert target == eng.move_count + eng.min_clicks_left()
            while eng.state == eng.RUNNING:
                oracle_reveal(eng)
            assert eng.won and eng.move_count == target and eng.min_clicks_left() == 0


def test_bayes_starts_in_a_corner() -> None:
    from ai_agents import bayes_reveal

    for agent in (bayes_reveal,):
        for level in (1, 4):
            for seed in range(10):
                eng     = MinesweeperEngine(level=level, seed=seed)
                clicks  = []
                reveal  = eng.reveal_count
                eng.reveal_count = lambda r, c: clicks.append((r, c)) or reveal(r, c)
                agent(eng)
                corners = {(0, 0), (0, eng.cols - 1), (eng.rows - 1, 0), (eng.rows - 1, eng.cols - 1)}
                assert clicks and clicks[0] in corners


def test_off_board_raises() -> None:
    eng = MinesweeperEngine(level=1)
    try:
        eng.reveal_count(6, 0)
    except IndexError:
        return
    raise AssertionError("expected IndexError")


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
