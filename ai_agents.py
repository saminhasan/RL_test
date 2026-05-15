"""
AI agents and benchmarking for MinesweeperEngine.

Agents are standalone functions that take an engine and return a reveal result.
Engine stays pure (game mechanics + Bayes probs only).

Agents:
  random_safe  -- reveal a random known-safe cell
  random_any   -- reveal a random hidden cell (may hit a mine)
  bayes        -- reveal lowest Bayesian mine-probability cell
  oracle       -- never hits a mine; picks largest cascade (oracle mine-avoidance + cascade)
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from engine import MinesweeperEngine

AGENT_NAMES = ("random_any", "random_safe", "bayes", "oracle")

def _random_hidden_flat(engine: MinesweeperEngine) -> int | None:
    if engine.covered_count <= 0:
        return None
    if engine.covered_count * 4 <= engine._flat_size:
        hidden = np.flatnonzero(~engine._revealed_flat)
        if hidden.size == 0:
            return None
        return int(hidden[int(engine._rng.integers(hidden.size))])
    while True:
        choice = int(engine._rng.integers(engine._flat_size))
        if not engine._revealed_flat[choice]:
            return choice


def _random_safe_flat(engine: MinesweeperEngine) -> int | None:
    if engine._safe_pool_size <= 0:
        return None
    idx = int(engine._rng.integers(engine._safe_pool_size))
    return int(engine._safe_flat_cells[idx])


def _reveal_flat(engine: MinesweeperEngine, flat: int | None) -> int:
    """Reveal a flat-indexed cell. Returns safe-cells uncovered (0 on miss/None)."""
    if flat is None or engine.state == engine.OVER:
        return 0
    row, col = divmod(int(flat), engine.cols)
    return engine.reveal_count(row, col)


def random_reveal_any(engine: MinesweeperEngine) -> int:
    """Reveal a random hidden cell (may hit a mine)."""
    if engine.state == engine.OVER:
        return 0
    return _reveal_flat(engine, _random_hidden_flat(engine))


def random_reveal_safe(engine: MinesweeperEngine) -> int:
    """Reveal a random known-safe cell. Falls back to any hidden cell before first move."""
    if engine.state == engine.OVER:
        return 0
    if engine.started:
        flat = _random_safe_flat(engine)
    else:
        flat = _random_hidden_flat(engine)
    return _reveal_flat(engine, flat)

# ---------------------------------------------------------------------------
# Bayes agent
# ---------------------------------------------------------------------------

def _lowest_probability_candidates(engine: MinesweeperEngine) -> np.ndarray:
    hidden = np.flatnonzero(~engine._revealed_flat)
    if hidden.size == 0:
        return hidden
    probs  = engine._mine_probs.ravel()
    hp     = probs[hidden]
    lowest = float(hp.min())
    return hidden[hp == lowest]


def bayes_reveal(engine: MinesweeperEngine) -> int:
    """Reveal the hidden cell with lowest Bayesian mine probability."""
    if engine.state == engine.OVER:
        return 0
    choices = _lowest_probability_candidates(engine)
    if choices.size == 0:
        return 0
    choice = int(choices[int(engine._rng.integers(len(choices)))])
    return _reveal_flat(engine, choice)


# ---------------------------------------------------------------------------
# Oracle agent (oracle mine-avoidance + cascade maximisation)
# ---------------------------------------------------------------------------
def _cascade_size(engine, f: int) -> int:
    """BFS through _neighbor_counts_flat to predict cascade cells if f is revealed."""
    counts   = engine._neighbor_counts_flat
    revealed = engine._revealed_flat

    if int(counts[f]) != 0:
        return 1

    nidx  = engine._neighbor_indices
    nlen  = engine._neighbor_lengths
    fsize = engine._flat_size

    visited = np.zeros(fsize, dtype=bool)
    queue   = np.empty(fsize, dtype=np.int32)
    head = tail = 0
    queue[tail] = f; tail += 1
    visited[f] = True
    size = 0

    while head < tail:
        cur   = int(queue[head]); head += 1
        size += 1
        if int(counts[cur]) != 0:
            continue
        for k in range(int(nlen[cur])):
            n = int(nidx[cur, k])
            if not visited[n] and not revealed[n]:
                visited[n] = True
                queue[tail] = n; tail += 1

    return size


def _pick_largest_cascade(engine, cells: np.ndarray) -> int:
    """Return the cell in `cells` with the largest predicted cascade."""
    csizes = np.array([_cascade_size(engine, int(f)) for f in cells], dtype=np.int64)
    best   = int(csizes.max())
    top    = cells[csizes == best]
    return int(top[int(engine._rng.integers(top.size))])


def oracle_choice(engine) -> int | None:
    """
    Select best hidden cell (flat index) to reveal.

    Never hits a mine. Among safe candidates, always picks the cell that
    cascades the most cells in one move.
    """
    if engine.state == engine.OVER:
        return None

    hidden = np.flatnonzero(~engine._revealed_flat)
    if hidden.size == 0:
        return None

    if not engine.started:
        return 0

    probs = engine.mine_probs.ravel()
    hp    = probs[hidden]

    good = ~np.isnan(hp)
    if not good.any():
        safe = hidden[~engine._mines_flat[hidden]]
        if safe.size == 0:
            return None
        return _pick_largest_cascade(engine, safe)

    hidden = hidden[good]
    hp     = hp[good]
    pmin   = float(hp.min())
    cand   = hidden[np.abs(hp - pmin) <= 1e-9]

    safe_cand = cand[~engine._mines_flat[cand]]

    if safe_cand.size == 0:
        # All minimum-p candidates are mines — escape to any safe hidden cell
        all_safe = np.flatnonzero(~engine._revealed_flat & ~engine._mines_flat)
        if all_safe.size == 0:
            return None
        return _pick_largest_cascade(engine, all_safe)

    if safe_cand.size == 1:
        return int(safe_cand[0])

    return _pick_largest_cascade(engine, safe_cand)


def oracle_reveal(engine: MinesweeperEngine) -> int:
    """Never hits a mine; always picks the largest cascade among safe cells."""
    return int(_reveal_flat(engine, oracle_choice(engine)))


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _normalize_level(level: int | str) -> int:
    if isinstance(level, int):
        if level in MinesweeperEngine.LEVELS:
            return level
        raise ValueError(f"unknown level id {level}")
    normalized = level.strip().lower()
    if normalized.isdigit():
        return _normalize_level(int(normalized))
    for level_id, level_name in MinesweeperEngine.LEVELS.items():
        if normalized == level_name:
            return level_id
    choices = ", ".join(f"{k}:{v}" for k, v in MinesweeperEngine.LEVELS.items())
    raise ValueError(f"unknown level {level!r}. Choose one of: {choices}")


def _run_agent_game(
    level: int,
    agent_name: str,
    seed: int,
) -> dict[str, float | int | str | bool]:
    engine = MinesweeperEngine(level=level, seed=seed)

    if agent_name == "random_any":
        reveal = lambda: random_reveal_any(engine)
    elif agent_name == "random_safe":
        reveal = lambda: random_reveal_safe(engine)
    elif agent_name == "bayes":
        reveal = lambda: bayes_reveal(engine)
    elif agent_name == "oracle":
        reveal = lambda: oracle_reveal(engine)
    else:
        raise ValueError(f"unknown agent {agent_name!r}")

    first_changed = _reveal_flat(engine, 0)
    reveal_events = 1
    total_changed = first_changed
    max_changed   = first_changed


    while engine.state != engine.OVER:
        changed_count = int(reveal())
        total_changed += changed_count
        max_changed    = max(max_changed, changed_count)
        reveal_events += 1

    safe_revealed = int(engine.revealed_safe_cells)
    return {
        "level":           level,
        "level_name":      MinesweeperEngine.LEVELS[level],
        "agent":           agent_name,
        "seed":            seed,
        "won":             bool(engine.won),
        "hit_mine":        bool(engine.hit_mine),
        "moves":           int(engine.move_count),
        "reveal_events":   reveal_events,
        "safe_revealed":   safe_revealed,
        "safe_total":      int(engine.total_safe_cells),
        "safe_fraction":   safe_revealed / engine.total_safe_cells if engine.total_safe_cells else 0.0,
        "total_changed":   total_changed,
        "max_changed":     max_changed,
        "avg_changed_per_move": total_changed / reveal_events if reveal_events else 0.0,
    }


def benchmark_agents(
    levels: tuple[int | str, ...] | list[int | str] = (1,),
    num_games: int = 1_000,
    seed: int = 42,
    n_jobs: int = -1,
    backend: str = "loky",
    agent_names: tuple[str, ...] = AGENT_NAMES,
) -> list[dict[str, float | int | str | bool]]:
    """
    Run num_games per agent/level in parallel and print aggregate stats.
    Returns one result dict per agent/game.
    """
    if num_games <= 0:
        raise ValueError("num_games must be > 0")

    from joblib import Parallel, delayed
    from tqdm import tqdm

    level_ids = [_normalize_level(level) for level in levels]
    tasks = [
        (level, agent, seed + level * 10_000_000 + i)
        for level in level_ids
        for agent in agent_names
        for i in range(num_games)
    ]

    start = perf_counter()
    results = [
        r for r in Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_run_agent_game)(level, agent, gseed)
            for level, agent, gseed in tqdm(tasks, desc="agent games", unit="game")
        )
        if r is not None
    ]
    elapsed = perf_counter() - start

    print(f"benchmark_agents: games={num_games} per agent/level, elapsed={elapsed:.2f}s")
    print(f"  total games : {len(results)}")
    print(f"  games/sec   : {len(results)/elapsed if elapsed > 0 else float('inf'):.2f}")

    for level in level_ids:
        level_name = MinesweeperEngine.LEVELS[level]
        print(f"\nlevel {level} ({level_name})")
        for agent in agent_names:
            rows = [r for r in results if r["level"] == level and r["agent"] == agent]
            wins        = np.array([r["won"]       for r in rows], dtype=np.float64)
            losses      = 1.0 - wins
            hit_mines   = np.array([r["hit_mine"]  for r in rows], dtype=np.float64)
            moves       = np.array([r["moves"]     for r in rows], dtype=np.float64)
            won_moves   = moves[wins.astype(bool)]
            lost_moves  = moves[losses.astype(bool)]
            safe_frac   = np.array([r["safe_fraction"]        for r in rows], dtype=np.float64)
            avg_changed = np.array([r["avg_changed_per_move"] for r in rows], dtype=np.float64)
            max_changed = np.array([r["max_changed"]          for r in rows], dtype=np.float64)

            won_moves_text = (
                f"avg={won_moves.mean():.2f}, median={np.median(won_moves):.2f}, "
                f"min={won_moves.min():.0f}, max={won_moves.max():.0f}"
                if len(won_moves) else "n/a"
            )
            lost_moves_text = (
                f"avg={lost_moves.mean():.2f}, median={np.median(lost_moves):.2f}, "
                f"min={lost_moves.min():.0f}, max={lost_moves.max():.0f}"
                if len(lost_moves) else "n/a"
            )

            print(
                f"  {agent}:\n"
                f"    wins          : {int(wins.sum())}/{len(rows)} ({wins.mean()*100:.2f}%)\n"
                f"    losses        : {int(losses.sum())}/{len(rows)} ({losses.mean()*100:.2f}%)\n"
                f"    mine hits     : {int(hit_mines.sum())}/{len(rows)} ({hit_mines.mean()*100:.2f}%)\n"
                f"    moves         : avg={moves.mean():.2f}, median={np.median(moves):.2f}, "
                f"std={moves.std():.2f}, min={moves.min():.0f}, max={moves.max():.0f}\n"
                f"    won moves     : {won_moves_text}\n"
                f"    lost moves    : {lost_moves_text}\n"
                f"    safe revealed : avg={safe_frac.mean()*100:.2f}% of safe cells\n"
                f"    cells/move    : avg={avg_changed.mean():.2f}, median={np.median(avg_changed):.2f}\n"
                f"    max cascade   : avg={max_changed.mean():.2f}, median={np.median(max_changed):.2f}"
            )

    return results


def benchmark_engine(
    levels: list[int] | None = None,
    games_per_level: int = 500,
    seed: int = 2026,
) -> None:
    """Simple random-policy throughput benchmark for the engine itself."""
    from tqdm import tqdm

    if levels is None:
        levels = list(MinesweeperEngine.LEVELS.keys())

    print("benchmark_engine:")
    for level in levels:
        total_moves = 0
        wins = 0
        start = perf_counter()

        for i in tqdm(
            range(games_per_level),
            total=games_per_level,
            desc=f"level {level} ({MinesweeperEngine.LEVELS[level]})",
            unit="games",
        ):
            engine = MinesweeperEngine(level=level, seed=seed + level * 1_000_000 + i)
            moves = 0
            while engine.state != MinesweeperEngine.OVER:
                random_reveal_any(engine)
                moves += 1
            total_moves += moves
            wins += int(engine.won)

        elapsed = perf_counter() - start
        level_name = MinesweeperEngine.LEVELS[level]
        print(
            f"level: {level}({level_name})\n",
            f"  games : {games_per_level}\n",
            f"  wins : {wins}\n",
            f"  win rate : {wins/games_per_level*100:.2f}%\n",
            f"  avg moves : {total_moves/games_per_level:.2f}\n",
            f"  elapsed time : {elapsed:.2f}s\n",
            f"  games/seconds : {games_per_level/elapsed if elapsed > 0 else float('inf'):.2f}\n",
        )


if __name__ == "__main__":
    benchmark_agents(
        levels=[1, 2, 3, 4],
        num_games=1_000,
        seed=42,
        n_jobs=6,
        backend="loky",
        agent_names=AGENT_NAMES,
    )
