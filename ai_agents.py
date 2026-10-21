"""
AI agents and benchmarking for MinesweeperEngine.

Agents are standalone functions that take an engine and return a reveal result.
Engine stays pure (game mechanics, Bayes probs, fewest-moves count).

Agents:
  random_any    -- reveal a random hidden cell (may hit a mine)
  random_safe   -- reveal a random known-safe cell (cheats: reads the mines)
  bayes         -- lowest Bayesian mine probability; ties go to the cell likeliest to be a zero
  oracle        -- sees the mines; wins in the fewest possible moves
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from engine import MinesweeperEngine


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
# Bayes agent (the visible board only)
# ---------------------------------------------------------------------------

def _lowest_probability_candidates(engine: MinesweeperEngine) -> np.ndarray:
    hidden = np.flatnonzero(~engine._revealed_flat)
    if hidden.size == 0:
        return hidden
    probs  = engine.mine_probs.ravel()
    hp     = probs[hidden]
    lowest = float(hp.min())
    # tolerance: equal probabilities from different fronts can differ in the last bit
    return hidden[hp <= lowest + 1e-12]


def _zero_chance(engine: MinesweeperEngine, cells: np.ndarray) -> np.ndarray:
    """Rough chance each cell is a zero: none of its hidden neighbours is a mine.

    Multiplies (1 - p) over the neighbours as if they were independent. Exact
    when the neighbours are already known (p = 0 or 1), which is the usual case
    for the safe cells this is used on.
    """
    probs = np.nan_to_num(engine.mine_probs.ravel(), nan=0.0)  # revealed: not a mine
    safe  = np.append(1.0 - probs, 1.0)                        # padding index -1 -> 1.0
    return safe[engine._neighbor_indices[cells]].prod(axis=1)


def bayes_reveal(engine: MinesweeperEngine) -> int:
    """Reveal the hidden cell with the lowest Bayesian mine probability.

    Among the equally safest cells it picks the one likeliest to be a zero: a
    zero opens a whole patch in one click, while a number next to an unopened
    patch is a wasted click (the patch would uncover it anyway). Before the
    first click every cell is equally safe, so it starts in a corner, the cell
    likeliest to be a zero.
    """
    if engine.state == engine.OVER:
        return 0
    choices = _lowest_probability_candidates(engine)
    if choices.size == 0:
        return 0
    chance = _zero_chance(engine, choices)
    best   = choices[chance >= chance.max() - 1e-12]
    return _reveal_flat(engine, int(best[int(engine._rng.integers(best.size))]))


# ---------------------------------------------------------------------------
# Oracle agent (sees the mines, fewest possible moves)
# ---------------------------------------------------------------------------

def oracle_choice(engine: MinesweeperEngine) -> int | None:
    """
    Select the next cell for a fewest-moves win.

    Knows where the mines are. Clicks one cell of each opening (a patch of
    zeros, which uncovers itself and its border), then every number cell no
    opening uncovered. That is exactly engine.min_clicks_left() clicks, the
    minimum. The first click is blind like every other agent's: mines do not
    exist until it lands.
    """
    if engine.state == engine.OVER:
        return None
    if not engine.started:
        return _random_hidden_flat(engine)

    hidden_safe = ~engine._revealed_flat & ~engine._mines_flat
    openings    = np.flatnonzero(hidden_safe & (engine._neighbor_counts_flat == 0))
    cells       = openings if openings.size else np.flatnonzero(hidden_safe)
    if cells.size == 0:
        return None
    return int(cells[int(engine._rng.integers(cells.size))])


def oracle_reveal(engine: MinesweeperEngine) -> int:
    """Never hits a mine; wins in the fewest possible moves."""
    return int(_reveal_flat(engine, oracle_choice(engine)))


AGENTS = {
    "random_any":   random_reveal_any,
    "random_safe":  random_reveal_safe,
    "bayes":        bayes_reveal,
    "oracle":       oracle_reveal,
}
AGENT_NAMES : tuple[str, ...] = tuple(AGENTS)


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
) -> dict[str, float | int | str | bool | None]:
    if agent_name not in AGENTS:
        raise ValueError(f"unknown agent {agent_name!r}")
    agent  = AGENTS[agent_name]
    engine = MinesweeperEngine(level=level, seed=seed)

    first_changed = int(agent(engine))
    reveal_events = 1
    total_changed = first_changed
    max_changed   = first_changed

    while engine.state != engine.OVER:
        changed_count = int(agent(engine))
        total_changed += changed_count
        max_changed    = max(max_changed, changed_count)
        reveal_events += 1

    safe_revealed = int(engine.revealed_safe_cells)
    moves         = int(engine.move_count)
    min_moves     = engine.min_win_moves
    return {
        "level":           level,
        "level_name":      MinesweeperEngine.LEVELS[level],
        "agent":           agent_name,
        "seed":            seed,
        "won":             bool(engine.won),
        "hit_mine":        bool(engine.hit_mine),
        "moves":           moves,
        "min_moves":       min_moves,
        # fewest possible moves / moves taken: 1.0 is a perfect win
        "efficiency":      min_moves / moves if engine.won and min_moves is not None else None,
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
) -> list[dict[str, float | int | str | bool | None]]:
    """
    Run num_games per agent/level in parallel and print aggregate stats.
    Returns one result dict per agent/game.

    Agents get the same seeds. Agents with the same first click (all but
    bayes, which starts in a corner) therefore play the same boards.
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
            won_rows    = [r for r in rows if r["won"]]
            efficiency  = np.array([r["efficiency"] for r in won_rows], dtype=np.float64)
            extra_moves = np.array([r["moves"] - r["min_moves"] for r in won_rows], dtype=np.float64)

            def _fmt(a: np.ndarray) -> str:
                if len(a) == 0:
                    return "n/a"
                return (f"avg={a.mean():.2f}  std={a.std():.2f}"
                        f"  min={a.min():.2f}  max={a.max():.2f}")

            print(
                f"  {agent}:\n"
                f"    wins          : {int(wins.sum())}/{len(rows)} ({wins.mean()*100:.2f}%)\n"
                f"    losses        : {int(losses.sum())}/{len(rows)} ({losses.mean()*100:.2f}%)\n"
                f"    mine hits     : {int(hit_mines.sum())}/{len(rows)} ({hit_mines.mean()*100:.2f}%)\n"
                f"    moves         : {_fmt(moves)}\n"
                f"    won moves     : {_fmt(won_moves)}\n"
                f"    efficiency    : {_fmt(efficiency)}  (wins: fewest possible / moves)\n"
                f"    extra moves   : {_fmt(extra_moves)}  (wins: moves - fewest possible)\n"
                f"    lost moves    : {_fmt(lost_moves)}\n"
                f"    safe revealed : {_fmt(safe_frac * 100)} %\n"
                f"    cells/move    : {_fmt(avg_changed)}\n"
                f"    max cascade   : {_fmt(max_changed)}"
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
            f"level: {level}({level_name})\n"
            f"  games        : {games_per_level}\n"
            f"  wins         : {wins}\n"
            f"  win rate     : {wins/games_per_level*100:.2f}%\n"
            f"  avg moves    : {total_moves/games_per_level:.2f}\n"
            f"  elapsed time : {elapsed:.2f}s\n"
            f"  games/sec    : {games_per_level/elapsed if elapsed > 0 else float('inf'):.2f}"
        )


if __name__ == "__main__":
    benchmark_agents(
        levels=[1],
        num_games=10_000,
        seed=42,
        n_jobs=-1,
        backend="loky",
        # backend="threading",
        agent_names=AGENT_NAMES,
    )
