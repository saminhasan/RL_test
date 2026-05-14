from __future__ import annotations

from time import perf_counter

import numpy as np

from bayes import mine_probabilities_for_engine


SOLVER_NAMES = ("random_safe", "bayes", "oracle")

class MinesweeperEngine:
    """NumPy-based reveal-only Minesweeper engine (no flags, no chord)."""

    _NEIGHBORS_CACHE: dict[tuple[int, int], tuple[tuple[tuple[int, int], ...], ...]] = {}
    _FLAT_NEIGHBORS_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    DIFFICULTIES = {
        "easy": (9, 9, 10), # 12.35% mines
        "medium": (16, 16, 40), # 15.62% mines
        "hard": (16, 30, 99), # 20.62% mines
    }
    LEVELS = {1: "easy", 2: "medium", 3: "hard"}
    RUNNING = 1
    OVER = 0

    def __init__(self, level: int = 1, seed: int = 42, headless: bool = False) -> None:
        rows, cols, mine_count = self.DIFFICULTIES[self.LEVELS[level]]

        self.rows = rows
        self.cols = cols
        self.mine_count = mine_count
        self.total_safe_cells = rows * cols - mine_count
        self.headless = headless
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self._mines = np.zeros((rows, cols), dtype=bool)
        self._neighbor_counts = np.zeros((rows, cols), dtype=np.uint8)
        self.revealed = np.zeros((rows, cols), dtype=bool)
        self._flat_size = rows * cols
        self._mines_flat = self._mines.ravel()
        self._neighbor_counts_flat = self._neighbor_counts.ravel()
        self._revealed_flat = self.revealed.ravel()
        self.covered_count = self._flat_size
        self._safe_flat_cells = np.empty(self.total_safe_cells, dtype=np.int32)
        self._safe_flat_pos = np.full(self._flat_size, -1, dtype=np.int32)
        self._safe_pool_size = 0
        self._flood_queue = np.empty(self._flat_size, dtype=np.int32)
        self._changed_buffer = None if headless else np.zeros((rows, cols), dtype=bool)

        cache_key = (rows, cols)
        cached_neighbors = self._NEIGHBORS_CACHE.get(cache_key)
        if cached_neighbors is None:
            cached_neighbors = tuple(
                tuple(self._build_neighbors(i, j))
                for i in range(rows)
                for j in range(cols)
            )
            self._NEIGHBORS_CACHE[cache_key] = cached_neighbors
        self._neighbors = cached_neighbors

        flat_cached_neighbors = self._FLAT_NEIGHBORS_CACHE.get(cache_key)
        if flat_cached_neighbors is None:
            neighbor_indices = np.full((self._flat_size, 8), -1, dtype=np.int32)
            neighbor_lengths = np.zeros(self._flat_size, dtype=np.int8)
            for flat, neighbors in enumerate(cached_neighbors):
                count = len(neighbors)
                neighbor_lengths[flat] = count
                for idx, (row, col) in enumerate(neighbors):
                    neighbor_indices[flat, idx] = row * cols + col
            neighbor_indices.setflags(write=False)
            neighbor_lengths.setflags(write=False)
            flat_cached_neighbors = (neighbor_indices, neighbor_lengths)
            self._FLAT_NEIGHBORS_CACHE[cache_key] = flat_cached_neighbors
        self._neighbor_indices, self._neighbor_lengths = flat_cached_neighbors

        self.state = self.RUNNING
        self.started = False
        self.revealed_safe_cells = 0
        self.exploded_cell: tuple[int, int] | None = None
        self.hit_mine = False
        self.won = False
        self.move_count = 0
        self.max_moves = self.rows * self.cols - self.mine_count

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)

        self._mines.fill(False)
        self._neighbor_counts.fill(0)
        self.revealed.fill(False)
        self.covered_count = self._flat_size
        self._safe_flat_pos.fill(-1)
        self._safe_pool_size = 0
        if self._changed_buffer is not None:
            self._changed_buffer.fill(False)

        self.state = self.RUNNING
        self.started = False
        self.revealed_safe_cells = 0
        self.exploded_cell = None
        self.hit_mine = False
        self.won = False
        self.move_count = 0

    def reveal(self, row: int, col: int) -> np.ndarray:
        """Reveal a cell and return the changed mask. Kept for UI/debug use."""
        changed = self._empty_changed()
        flat = self._flat(row, col)
        if self.state == self.OVER or self._revealed_flat[flat]:
            return changed
        self._reveal_flat_core(flat, changed)
        self.move_count += 1
        return changed

    def reveal_count(self, row: int, col: int) -> int:
        """Reveal a cell and return how many safe cells were uncovered."""
        flat = self._flat(row, col)
        if self.state == self.OVER or self._revealed_flat[flat]:
            return 0
        revealed_count = self._reveal_flat_core(flat, None)
        self.move_count += 1
        return revealed_count

    def _empty_changed(self) -> np.ndarray:
        if self._changed_buffer is None:
            return np.zeros_like(self.revealed, dtype=bool)
        self._changed_buffer.fill(False)
        return self._changed_buffer

    def _reveal_flat_core(self, flat: int, changed: np.ndarray | None) -> int:
        if self.state == self.OVER:
            return 0

        if not self.started:
            self._place_mines_flat(first_flat=int(flat))
            self.started = True

        revealed = self._revealed_flat
        mines = self._mines_flat
        flat = int(flat)

        if revealed[flat]:
            return 0

        changed_flat = changed.ravel() if changed is not None else None

        if mines[flat]:
            row, col = divmod(flat, self.cols)
            if changed_flat is not None:
                changed_flat[flat] = True
            revealed[flat] = True
            self.covered_count -= 1
            self.state = self.OVER
            self.exploded_cell = (row, col)
            self.hit_mine = True
            self.won = False
            return 0

        neighbor_counts = self._neighbor_counts_flat
        neighbor_indices = self._neighbor_indices
        neighbor_lengths = self._neighbor_lengths
        queue = self._flood_queue

        head = 0
        tail = 1
        queue[0] = flat

        revealed[flat] = True
        if changed_flat is not None:
            changed_flat[flat] = True

        revealed_count = 0

        while head < tail:
            current = int(queue[head])
            head += 1

            revealed_count += 1
            self.covered_count -= 1
            self._remove_from_pool(current)

            if neighbor_counts[current] != 0:
                continue

            for idx in range(int(neighbor_lengths[current])):
                neighbor = int(neighbor_indices[current, idx])
                if not revealed[neighbor] and not mines[neighbor]:
                    revealed[neighbor] = True
                    if changed_flat is not None:
                        changed_flat[neighbor] = True
                    queue[tail] = neighbor
                    tail += 1

        if revealed_count:
            self.revealed_safe_cells += revealed_count

        if self.revealed_safe_cells == self.total_safe_cells:
            self.state = self.OVER
            self.won = True

        return revealed_count

    def get_public_view(self, reveal_mines_on_loss: bool = True, out: np.ndarray | None = None) -> np.ndarray:
        """
        Returns an int8 board where:
        -1 = hidden
         0..8 = revealed neighbor count
         9 = mine (shown only after loss when reveal_mines_on_loss is True)
        """
        if out is None:
            view = np.empty((self.rows, self.cols), dtype=np.int8)
        else:
            view = out
        view.fill(-1)

        view_flat = view.ravel()
        safe_revealed = self._revealed_flat & ~self._mines_flat
        view_flat[safe_revealed] = self._neighbor_counts_flat[safe_revealed]

        if self.state == self.OVER and self.hit_mine and reveal_mines_on_loss:
            view_flat[self._mines_flat] = 9

        return view

    def get_mine_mask(self) -> np.ndarray:
        """Debug/helper accessor. Returns a copy of the mine layout."""
        return self._mines.copy()

    def get_neighbor_counts(self) -> np.ndarray:
        """Debug/helper accessor. Returns a copy of neighbor counts."""
        return self._neighbor_counts.copy()

    def get_mine_probabilities(self) -> np.ndarray:
        """
        Bayesian estimate of mine probability per cell from visible state only.

        Returns a float64 board where:
        - revealed cells are np.nan
        - hidden cells are probabilities in [0, 1]
        """
        return mine_probabilities_for_engine(
            revealed=self.revealed,
            neighbor_counts=self._neighbor_counts,
            mine_count=self.mine_count,
            started=self.started,
            hit_mine=self.hit_mine,
            exploded_cell=self.exploded_cell,
            rows=self.rows,
            cols=self.cols,
            neighbors=self._neighbors,
        )

    def _build_neighbors(self, i: int, j: int) -> list[tuple[int, int]]:
        r0 = i - 1 if i else 0
        r1 = i + 1 if i + 1 < self.rows else self.rows - 1
        c0 = j - 1 if j else 0
        c1 = j + 1 if j + 1 < self.cols else self.cols - 1

        return [
            (ni, nj)
            for ni in range(r0, r1 + 1)
            for nj in range(c0, c1 + 1)
            if ni != i or nj != j
        ]

    def _flat(self, row: int, col: int) -> int:
        return row * self.cols + col

    def get_neighbors(self, row: int, col: int) -> tuple[tuple[int, int], ...]:
        return self._neighbors[self._flat(row, col)]

    def _place_mines_flat(self, first_flat: int) -> None:
        self._mines_flat.fill(False)

        # Sample from [0, flat_size - 1) then shift values past first_flat.
        # This avoids building a full candidate mask every episode.
        chosen = self._rng.choice(self._flat_size - 1, size=self.mine_count, replace=False)
        chosen = chosen + (chosen >= int(first_flat))
        self._mines_flat[chosen] = True

        self._neighbor_counts[:] = self._compute_neighbor_counts(self._mines)
        self._init_safe_cells()

    def _remove_from_pool(self, flat: int) -> None:
        pos = int(self._safe_flat_pos[flat])
        if pos < 0:
            return

        last_index = self._safe_pool_size - 1
        last_flat = int(self._safe_flat_cells[last_index])
        if pos != last_index:
            self._safe_flat_cells[pos] = last_flat
            self._safe_flat_pos[last_flat] = pos
        self._safe_flat_pos[flat] = -1
        self._safe_pool_size -= 1

    def _init_safe_cells(self) -> None:
        safe = np.flatnonzero(~self._mines_flat)
        count = int(safe.size)
        self._safe_flat_cells[:count] = safe
        self._safe_flat_pos.fill(-1)
        self._safe_flat_pos[safe] = np.arange(count, dtype=np.int32)
        self._safe_pool_size = count

    def _random_hidden_flat(self) -> int | None:
        if self.covered_count <= 0:
            return None

        if self.covered_count * 4 <= self._flat_size:
            hidden = np.flatnonzero(~self._revealed_flat)
            if hidden.size == 0:
                return None
            return int(hidden[int(self._rng.integers(hidden.size))])

        while True:
            choice = int(self._rng.integers(self._flat_size))
            if not self._revealed_flat[choice]:
                return choice

    def _random_safe_flat(self) -> int | None:
        if self._safe_pool_size <= 0:
            return None
        idx = int(self._rng.integers(self._safe_pool_size))
        return int(self._safe_flat_cells[idx])

    def _lowest_probability_candidates(self) -> np.ndarray:
        hidden_flat = np.flatnonzero(~self._revealed_flat)
        if hidden_flat.size == 0:
            return hidden_flat

        probabilities = self.get_mine_probabilities().ravel()
        hidden_probabilities = probabilities[hidden_flat]
        lowest = float(hidden_probabilities.min())
        return hidden_flat[hidden_probabilities == lowest]

    def _compute_neighbor_counts(self, mines: np.ndarray) -> np.ndarray:
        m = mines.view(np.uint8)   # mines should be bool
        c = np.zeros_like(m, dtype=np.uint8)

        c[1:, 1:]   += m[:-1, :-1]
        c[1:, :]    += m[:-1, :]
        c[1:, :-1]  += m[:-1, 1:]

        c[:, 1:]    += m[:, :-1]
        c[:, :-1]   += m[:, 1:]

        c[:-1, 1:]  += m[1:, :-1]
        c[:-1, :]   += m[1:, :]
        c[:-1, :-1] += m[1:, 1:]

        return c

    def reveal_all_safe(self) -> np.ndarray:
        changed = self._empty_changed()
        if self.state == self.OVER:
            return changed

        if not self.started:
            self._place_mines_flat(first_flat=0)
            self.started = True

        np.logical_and(~self.revealed, ~self._mines, out=changed)
        if changed.any():
            self.revealed |= changed
            changed_count = int(changed.sum())
            self.revealed_safe_cells += changed_count
            self.covered_count -= changed_count
            self._safe_flat_pos.fill(-1)
            self._safe_pool_size = 0

        if self.revealed_safe_cells == self.total_safe_cells:
            self.state = self.OVER
            self.won = True

        return changed

    def _reveal_random_choice(self, choice: int | None, changed: bool | None) -> np.ndarray | int:
        use_changed = (not self.headless) if changed is None else changed
        if choice is None:
            return self._empty_changed() if use_changed else 0

        row, col = divmod(int(choice), self.cols)
        if use_changed:
            return self.reveal(row, col)
        return self.reveal_count(row, col)

    def random_reveal(self, safe: bool = False, changed: bool | None = None) -> np.ndarray | int:
        """
        Reveal a random hidden cell.

        safe=False chooses from all hidden cells.
        safe=True chooses from known safe cells after the first move.

        changed=None returns a changed mask for normal/UI engines and a count for
        headless engines. Pass changed=True/False to force either behavior.
        """
        if self.state == self.OVER:
            use_changed = (not self.headless) if changed is None else changed
            return self._empty_changed() if use_changed else 0

        if safe and self.started:
            choice = self._random_safe_flat()
        else:
            choice = self._random_hidden_flat()
        return self._reveal_random_choice(choice, changed)

    def bayes_reveal(self, changed: bool | None = None) -> np.ndarray | int:
        """Reveal a hidden cell with the lowest Bayesian mine probability."""
        use_changed = (not self.headless) if changed is None else changed
        if self.state == self.OVER:
            return self._empty_changed() if use_changed else 0

        choices = self._lowest_probability_candidates()
        if choices.size == 0:
            return self._empty_changed() if use_changed else 0

        choice = int(choices[int(self._rng.integers(len(choices)))])
        return self._reveal_random_choice(choice, use_changed)

    def oracle_reveal(self, changed: bool | None = None) -> np.ndarray | int:
        """
        Reveal using Bayesian probabilities, but cheat by avoiding known mines.

        If the Bayes-selected lowest-probability candidates include safe cells,
        choose among those safe cells. If all tied candidates are mines, choose
        a random hidden safe cell using the engine's private mine mask.
        """
        use_changed = (not self.headless) if changed is None else changed
        if self.state == self.OVER:
            return self._empty_changed() if use_changed else 0

        candidates = self._lowest_probability_candidates()
        if candidates.size == 0:
            return self._empty_changed() if use_changed else 0

        safe_candidates = candidates[~self._mines_flat[candidates]]
        if safe_candidates.size:
            choice = int(safe_candidates[int(self._rng.integers(safe_candidates.size))])
        else:
            hidden_safe = np.flatnonzero(~self._revealed_flat & ~self._mines_flat)
            if len(hidden_safe) == 0:
                return self._empty_changed() if use_changed else 0
            choice = int(hidden_safe[int(self._rng.integers(len(hidden_safe)))])

        return self._reveal_random_choice(choice, use_changed)


    @classmethod
    def benchmark_engine(
        cls,
        levels: list[int] | None = None,
        games_per_level: int = 500,
        seed: int = 2026,
        include_probabilities: bool = False,
    ) -> None:
        """Simple random-policy benchmark over N games for each level."""
        if games_per_level <= 0:
            raise ValueError("games_per_level must be > 0")

        from tqdm import tqdm

        print("benchmark_engine:")
        if levels is None:
            levels = list(cls.LEVELS.keys())

        for level in levels:
            total_moves = 0
            wins = 0
            start = perf_counter()

            for i in tqdm(
                range(games_per_level),
                total=games_per_level,
                desc=f"level {level} ({cls.LEVELS[level]})",
                unit="games",
            ):
                engine = cls(level=level, seed=seed + level * 1_000_000 + i, headless=True)
                moves = 0

                while engine.state != cls.OVER:
                    engine.random_reveal(changed=False)
                    if include_probabilities:
                        engine.get_mine_probabilities()
                    moves += 1

                total_moves += moves
                wins += int(engine.won)

            elapsed = perf_counter() - start
            games_per_sec = games_per_level / elapsed if elapsed > 0 else float("inf")
            avg_moves = total_moves / games_per_level
            win_rate = wins / games_per_level
            level_name = cls.LEVELS[level]

            print(
                f"level: {level}({level_name})\n",
                f"  games : {games_per_level}\n",
                f"  wins : {wins}\n",
                f"  win rate : {win_rate*100.0:.2f}%\n",
                f"  avg moves : {avg_moves:.2f}\n",
                f"  elapsed time :{elapsed:.2f}s\n",
                f"  games/seconds : {games_per_sec:.2f}\n",
            )

def _normalize_benchmark_level(level: int | str) -> int:
    if isinstance(level, int):
        if level in MinesweeperEngine.LEVELS:
            return level
        raise ValueError(f"unknown level id {level}")

    normalized = level.strip().lower()
    if normalized.isdigit():
        return _normalize_benchmark_level(int(normalized))

    for level_id, level_name in MinesweeperEngine.LEVELS.items():
        if normalized == level_name:
            return level_id

    choices = ", ".join(f"{idx}:{name}" for idx, name in MinesweeperEngine.LEVELS.items())
    raise ValueError(f"unknown level {level!r}. Choose one of {choices}")


def _run_solver_game(level: int, solver_name: str, seed: int) -> dict[str, float | int | str | bool]:
    engine = MinesweeperEngine(level=level, seed=seed, headless=True)

    if solver_name == "random_safe":
        reveal = lambda: engine.random_reveal(safe=True, changed=False)
    elif solver_name == "bayes":
        reveal = lambda: engine.bayes_reveal(changed=False)
    elif solver_name == "oracle":
        reveal = lambda: engine.oracle_reveal(changed=False)
    else:
        raise ValueError(f"unknown solver {solver_name!r}")

    reveal_events = 0
    total_changed = 0
    max_changed = 0

    while engine.state != engine.OVER:
        changed_count = int(reveal())
        total_changed += changed_count
        max_changed = max(max_changed, changed_count)
        reveal_events += 1

    safe_revealed = int(engine.revealed_safe_cells)
    return {
        "level": level,
        "level_name": MinesweeperEngine.LEVELS[level],
        "solver": solver_name,
        "seed": seed,
        "won": bool(engine.won),
        "hit_mine": bool(engine.hit_mine),
        "moves": int(engine.move_count),
        "reveal_events": reveal_events,
        "safe_revealed": safe_revealed,
        "safe_total": int(engine.total_safe_cells),
        "safe_fraction": safe_revealed / engine.total_safe_cells if engine.total_safe_cells else 0.0,
        "total_changed": total_changed,
        "max_changed": max_changed,
        "avg_changed_per_move": total_changed / reveal_events if reveal_events else 0.0,
    }


def benchmark_solvers(
    levels: tuple[int | str, ...] | list[int | str] = (1,),
    num_games: int = 1_000,
    seed: int = 42,
    n_jobs: int = -1,
    backend: str = "loky",
    solver_names: tuple[str, ...] = SOLVER_NAMES,
) -> list[dict[str, float | int | str | bool]]:
    """
    Compare random-safe, Bayesian, and oracle reveal policies in parallel.

    Returns one result dict per solver/game and prints aggregate stats useful
    for reward shaping.
    """
    if num_games <= 0:
        raise ValueError("num_games must be > 0")

    from joblib import Parallel, delayed
    from tqdm import tqdm

    level_ids = [_normalize_benchmark_level(level) for level in levels]
    tasks = []
    for level in level_ids:
        for solver_name in solver_names:
            for game_idx in range(num_games):
                game_seed = seed + level * 10_000_000 + game_idx
                tasks.append((level, solver_name, game_seed))

    start = perf_counter()
    results: list[dict[str, float | int | str | bool]] = [
        r for r in Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_run_solver_game)(level, solver_name, game_seed)
            for level, solver_name, game_seed in tqdm(tasks, desc="solver games", unit="game")
        ) if r is not None
    ]
    elapsed = perf_counter() - start

    print(f"benchmark_solvers: games={num_games} per solver/level, elapsed={elapsed:.2f}s")
    print(f"  total games : {len(results)}")
    print(f"  games/sec   : {len(results) / elapsed if elapsed > 0 else float('inf'):.2f}")

    for level in level_ids:
        level_name = MinesweeperEngine.LEVELS[level]
        print(f"\nlevel {level} ({level_name})")
        for solver_name in solver_names:
            rows = [
                row for row in results
                if row is not None and row.get("level") == level and row.get("solver") == solver_name
            ]
            wins = np.array([bool(row["won"]) for row in rows], dtype=np.float64)
            losses = 1.0 - wins
            hit_mines = np.array([bool(row["hit_mine"]) for row in rows], dtype=np.float64)
            moves = np.array([float(row["moves"]) for row in rows], dtype=np.float64)
            won_moves = moves[wins.astype(bool)]
            lost_moves = moves[losses.astype(bool)]
            safe_fraction = np.array([float(row["safe_fraction"]) for row in rows], dtype=np.float64)
            avg_changed = np.array([float(row["avg_changed_per_move"]) for row in rows], dtype=np.float64)
            max_changed = np.array([float(row["max_changed"]) for row in rows], dtype=np.float64)

            won_moves_text = (
                f"avg={won_moves.mean():.2f}, median={np.median(won_moves):.2f}, "
                f"min={won_moves.min():.0f}, max={won_moves.max():.0f}"
                if len(won_moves)
                else "n/a"
            )
            lost_moves_text = (
                f"avg={lost_moves.mean():.2f}, median={np.median(lost_moves):.2f}, "
                f"min={lost_moves.min():.0f}, max={lost_moves.max():.0f}"
                if len(lost_moves)
                else "n/a"
            )

            print(
                f"  {solver_name}:\n"
                f"    wins          : {int(wins.sum())}/{len(rows)} ({wins.mean() * 100.0:.2f}%)\n"
                f"    losses        : {int(losses.sum())}/{len(rows)} ({losses.mean() * 100.0:.2f}%)\n"
                f"    mine hits     : {int(hit_mines.sum())}/{len(rows)} ({hit_mines.mean() * 100.0:.2f}%)\n"
                f"    moves         : avg={moves.mean():.2f}, median={np.median(moves):.2f}, "
                f"std={moves.std():.2f}, min={moves.min():.0f}, max={moves.max():.0f}\n"
                f"    won moves     : {won_moves_text}\n"
                f"    lost moves    : {lost_moves_text}\n"
                f"    safe revealed : avg={safe_fraction.mean() * 100.0:.2f}% of safe cells\n"
                f"    cells/move    : avg={avg_changed.mean():.2f}, median={np.median(avg_changed):.2f}\n"
                f"    max cascade   : avg={max_changed.mean():.2f}, median={np.median(max_changed):.2f}"
            )

    return results

if __name__ == "__main__":
    benchmark_solvers(levels=[2], num_games=10_000, seed=42, n_jobs=16 , backend="loky", solver_names=("bayes", "oracle", "random_safe"))
