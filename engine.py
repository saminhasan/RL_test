import numpy as np
from bayes import mine_probabilities_for_engine


class MinesweeperEngine:
    """NumPy-based reveal-only Minesweeper engine (no flags, no chord)."""

    _NEIGHBORS_CACHE: dict[tuple[int, int], tuple[tuple[tuple[int, int], ...], ...]] = {}
    _FLAT_NEIGHBORS_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    DIFFICULTIES = {
        "test":   (6,  6,  6),   # 16.67% mines
        "easy":   (9,  9, 10),   # 12.35% mines
        "medium": (16, 16, 40),  # 15.62% mines
        "hard":   (16, 30, 99),  # 20.62% mines
    }
    LEVELS = dict(enumerate(DIFFICULTIES, 1))

    # Per-level (min_moves_ref, max_moves_ref) from benchmark_optimality_distribution.
    # min = p5 of oracle wins, max = p95 of anti_oracle wins (random first click).
    # Update by running: benchmark_optimality_distribution(levels=[1,2,3,4], num_games=5000)
    LEVEL_WIN_BOUNDS: dict[int, tuple[int, int]] = {
        1: (5, 29),   # test   6×6    — oracle p5=5, anti_oracle p95=29
        2: (6, 25),   # easy   9×9    — placeholder, run benchmark to update
        3: (15, 60),  # medium 16×16  — placeholder, run benchmark to update
        4: (30, 110), # hard   16×30  — placeholder, run benchmark to update
    }
    RUNNING  = 1
    OVER     = 0

    def __init__(self, level: int = 1, seed: int = 42) -> None:
        rows, cols, mine_count = self.DIFFICULTIES[self.LEVELS[level]]
        self.level             = level
        self.rows              = rows
        self.cols              = cols
        self.mine_count        = mine_count
        self.total_safe_cells  = rows * cols - mine_count
        self._seed             = seed
        self._rng              = np.random.default_rng(seed)

        self._mines               = np.zeros((rows, cols), dtype=bool)
        self._neighbor_counts     = np.zeros((rows, cols), dtype=np.uint8)
        self.revealed             = np.zeros((rows, cols), dtype=bool)
        self._flat_size           = rows * cols
        self._mines_flat          = self._mines.ravel()
        self._neighbor_counts_flat = self._neighbor_counts.ravel()
        self._revealed_flat       = self.revealed.ravel()
        self.covered_count        = self._flat_size
        self._safe_flat_cells     = np.empty(self.total_safe_cells, dtype=np.int32)
        self._safe_flat_pos       = np.full(self._flat_size, -1, dtype=np.int32)
        self._safe_pool_size      = 0
        self._flood_queue         = np.empty(self._flat_size, dtype=np.int32)

        (self._neighbors,
         self._neighbor_indices,
         self._neighbor_lengths) = self._ensure_neighbor_cache(rows, cols)

        self.state               = self.RUNNING
        self.started             = False
        self.revealed_safe_cells = 0
        self.exploded_cell: tuple[int, int] | None = None
        self.hit_mine            = False
        self.won                 = False
        self.move_count          = 0
        self._mine_probs: np.ndarray = self.get_mine_probabilities()
        self.min_win_moves: int | None = None
        self.max_win_moves: int | None = None

    @property
    def mine_probs(self) -> np.ndarray:
        """Cached Bayesian mine probabilities. NaN for revealed cells, float in [0,1] for hidden."""
        return self._mine_probs

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)

        self._mines.fill(False)
        self._neighbor_counts.fill(0)
        self.revealed.fill(False)
        self.covered_count   = self._flat_size
        self._safe_flat_pos.fill(-1)
        self._safe_pool_size = 0

        self.state               = self.RUNNING
        self.started             = False
        self.revealed_safe_cells = 0
        self.exploded_cell       = None
        self.hit_mine            = False
        self.won                 = False
        self.move_count          = 0
        self._mine_probs         = self.get_mine_probabilities()
        self.min_win_moves       = None
        self.max_win_moves       = None

    @classmethod
    def from_state(
        cls,
        mines: np.ndarray,
        revealed: np.ndarray,
        seed: int = 0,
    ) -> "MinesweeperEngine":
        """Construct engine from an existing mid-game state (mines already placed).

        mines    : (rows, cols) bool — True where mines are
        revealed : (rows, cols) bool — True where cells have been revealed
        seed     : RNG seed used by agents for random decisions

        Game state (running / won / hit-mine) is inferred from the arrays.
        Revealed mine → hit_mine=True.  All safe cells revealed → won=True.
        """
        if mines.ndim != 2 or mines.shape != revealed.shape:
            raise ValueError("mines and revealed must be 2-D arrays with the same shape")

        rows, cols = int(mines.shape[0]), int(mines.shape[1])
        mine_count = int(mines.sum())

        eng = object.__new__(cls)
        eng.rows              = rows
        eng.cols              = cols
        eng.mine_count        = mine_count
        eng.total_safe_cells  = rows * cols - mine_count
        eng._seed             = seed
        eng._rng              = np.random.default_rng(seed)

        eng._mines               = mines.astype(bool, copy=True)
        eng._neighbor_counts     = cls._compute_neighbor_counts(eng._mines)
        eng.revealed             = revealed.astype(bool, copy=True)
        eng._flat_size           = rows * cols
        eng._mines_flat          = eng._mines.ravel()
        eng._neighbor_counts_flat = eng._neighbor_counts.ravel()
        eng._revealed_flat       = eng.revealed.ravel()
        eng.covered_count        = int((~eng.revealed).sum())
        eng._safe_flat_cells     = np.empty(eng.total_safe_cells, dtype=np.int32)
        eng._safe_flat_pos       = np.full(eng._flat_size, -1, dtype=np.int32)
        eng._safe_pool_size      = 0
        eng._flood_queue         = np.empty(eng._flat_size, dtype=np.int32)

        (eng._neighbors,
         eng._neighbor_indices,
         eng._neighbor_lengths) = cls._ensure_neighbor_cache(rows, cols)

        eng.started    = True
        eng.move_count = 0

        mine_revealed = eng._mines_flat & eng._revealed_flat
        if mine_revealed.any():
            exploded_flat            = int(np.flatnonzero(mine_revealed)[0])
            eng.exploded_cell        = divmod(exploded_flat, cols)
            eng.hit_mine             = True
            eng.won                  = False
            eng.state                = cls.OVER
            eng.revealed_safe_cells  = int((eng._revealed_flat & ~eng._mines_flat).sum())
        else:
            eng.exploded_cell        = None
            eng.hit_mine             = False
            eng.revealed_safe_cells  = int(eng._revealed_flat.sum())
            if eng.revealed_safe_cells == eng.total_safe_cells:
                eng.won   = True
                eng.state = cls.OVER
            else:
                eng.won   = False
                eng.state = cls.RUNNING

        unrevealed_safe = np.flatnonzero(~eng._mines_flat & ~eng._revealed_flat)
        count = int(unrevealed_safe.size)
        eng._safe_flat_cells[:count]        = unrevealed_safe
        eng._safe_flat_pos[unrevealed_safe] = np.arange(count, dtype=np.int32)
        eng._safe_pool_size                 = count

        eng._mine_probs = eng.get_mine_probabilities()
        return eng

    def reveal(self, row: int, col: int) -> None:
        flat = self._flat(row, col)
        if self.state == self.OVER or self._revealed_flat[flat]:
            return
        self._reveal_flat_core(flat)
        self.move_count += 1

    def reveal_count(self, row: int, col: int) -> int:
        flat = self._flat(row, col)
        if self.state == self.OVER or self._revealed_flat[flat]:
            return 0
        revealed_count = self._reveal_flat_core(flat)
        self.move_count += 1
        return revealed_count

    def _reveal_flat_core(self, flat: int) -> int:
        if self.state == self.OVER:
            return 0

        if not self.started:
            self._place_mines_flat(first_flat=flat)
            self.started = True

        revealed = self._revealed_flat
        mines    = self._mines_flat

        if revealed[flat]:
            return 0

        if mines[flat]:
            row, col = divmod(flat, self.cols)
            revealed[flat]     = True
            self.covered_count -= 1
            self.state         = self.OVER
            self.exploded_cell = (row, col)
            self.hit_mine      = True
            self.won           = False
            self._mine_probs   = self.get_mine_probabilities()
            return 0

        neighbor_counts  = self._neighbor_counts_flat
        neighbor_indices = self._neighbor_indices
        neighbor_lengths = self._neighbor_lengths
        queue            = self._flood_queue

        head           = 0
        tail           = 1
        queue[0]       = flat
        revealed[flat] = True
        revealed_count = 0

        while head < tail:
            current = int(queue[head])
            head   += 1

            revealed_count    += 1
            self.covered_count -= 1
            self._remove_from_pool(current)

            if neighbor_counts[current] != 0:
                continue

            for idx in range(int(neighbor_lengths[current])):
                neighbor = int(neighbor_indices[current, idx])
                if not revealed[neighbor] and not mines[neighbor]:
                    revealed[neighbor] = True
                    queue[tail]        = neighbor
                    tail              += 1

        if revealed_count:
            self.revealed_safe_cells += revealed_count

        if self.revealed_safe_cells == self.total_safe_cells:
            self.state = self.OVER
            self.won   = True

        self._mine_probs = self.get_mine_probabilities()
        return revealed_count

    def get_public_view(self, reveal_mines_on_loss: bool = True, out: np.ndarray | None = None) -> np.ndarray:
        """
        Returns an int8 board where:
        -1    = hidden
        0..8  = revealed neighbor count
        9     = mine (shown only after loss when reveal_mines_on_loss is True)
        """
        view = np.empty((self.rows, self.cols), dtype=np.int8) if out is None else out
        view.fill(-1)

        view_flat     = view.ravel()
        safe_revealed = self._revealed_flat & ~self._mines_flat
        view_flat[safe_revealed] = self._neighbor_counts_flat[safe_revealed]

        if self.state == self.OVER and self.hit_mine and reveal_mines_on_loss:
            view_flat[self._mines_flat] = 9

        return view

    def get_mine_probabilities(self) -> np.ndarray:
        """Bayesian estimate of mine probability per cell from visible state only.

        Returns float64 board: revealed cells are np.nan; hidden cells are in [0, 1].
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

    @classmethod
    def _ensure_neighbor_cache(
        cls, rows: int, cols: int
    ) -> tuple[tuple, np.ndarray, np.ndarray]:
        """Return (neighbors, neighbor_indices, neighbor_lengths), building cache on first call."""
        cache_key = (rows, cols)

        neighbors = cls._NEIGHBORS_CACHE.get(cache_key)
        if neighbors is None:
            neighbors = tuple(
                tuple(cls._build_neighbors(i, j, rows, cols))
                for i in range(rows)
                for j in range(cols)
            )
            cls._NEIGHBORS_CACHE[cache_key] = neighbors

        flat_cached = cls._FLAT_NEIGHBORS_CACHE.get(cache_key)
        if flat_cached is None:
            flat_size = rows * cols
            idx_arr   = np.full((flat_size, 8), -1, dtype=np.int32)
            len_arr   = np.zeros(flat_size, dtype=np.int8)
            for flat, neighs in enumerate(neighbors):
                n = len(neighs)
                len_arr[flat] = n
                for k, (r, c) in enumerate(neighs):
                    idx_arr[flat, k] = r * cols + c
            idx_arr.setflags(write=False)
            len_arr.setflags(write=False)
            flat_cached = (idx_arr, len_arr)
            cls._FLAT_NEIGHBORS_CACHE[cache_key] = flat_cached

        return neighbors, flat_cached[0], flat_cached[1]

    @staticmethod
    def _build_neighbors(i: int, j: int, rows: int, cols: int) -> list[tuple[int, int]]:
        r0, r1 = max(i - 1, 0), min(i + 1, rows - 1)
        c0, c1 = max(j - 1, 0), min(j + 1, cols - 1)
        return [
            (ni, nj)
            for ni in range(r0, r1 + 1)
            for nj in range(c0, c1 + 1)
            if ni != i or nj != j
        ]

    def _flat(self, row: int, col: int) -> int:
        return row * self.cols + col

    def _place_mines_flat(self, first_flat: int) -> None:
        self._mines_flat.fill(False)
        chosen = self._rng.choice(self._flat_size - 1, size=self.mine_count, replace=False)
        chosen = chosen + (chosen >= first_flat)
        self._mines_flat[chosen] = True
        self._neighbor_counts[:] = self._compute_neighbor_counts(self._mines)
        self._init_safe_cells()

    def _remove_from_pool(self, flat: int) -> None:
        pos = int(self._safe_flat_pos[flat])
        if pos < 0:
            return
        last_index = self._safe_pool_size - 1
        last_flat  = int(self._safe_flat_cells[last_index])
        if pos != last_index:
            self._safe_flat_cells[pos]     = last_flat
            self._safe_flat_pos[last_flat] = pos
        self._safe_flat_pos[flat] = -1
        self._safe_pool_size     -= 1

    def _init_safe_cells(self) -> None:
        safe  = np.flatnonzero(~self._mines_flat)
        count = int(safe.size)
        self._safe_flat_cells[:count] = safe
        self._safe_flat_pos.fill(-1)
        self._safe_flat_pos[safe]     = np.arange(count, dtype=np.int32)
        self._safe_pool_size          = count

    @staticmethod
    def _compute_neighbor_counts(mines: np.ndarray) -> np.ndarray:
        m = mines.view(np.uint8)
        c = np.zeros_like(m, dtype=np.uint8)

        c[1:,   1:] += m[:-1, :-1]
        c[1:,    :] += m[:-1,   :]
        c[1:,  :-1] += m[:-1,  1:]
        c[:,    1:] += m[:,   :-1]
        c[:,   :-1] += m[:,    1:]
        c[:-1,  1:] += m[1:,  :-1]
        c[:-1,   :] += m[1:,    :]
        c[:-1, :-1] += m[1:,   1:]

        return c
