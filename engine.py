"""Reveal-only Minesweeper engine (no flags, no chord).

The game logic and the exact Bayesian solver are C++ (cpp/msw/), loaded through
cppyy by _native.py. This class keeps the old Python API on top: the board
arrays are zero-copy numpy views into the C++ engine's memory.
"""
from __future__ import annotations

import numpy as np

from _native import msw, view

_U64 = (1 << 64) - 1


class MinesweeperEngine:
    """Reveal-only Minesweeper engine (no flags, no chord)."""

    DIFFICULTIES = {
        "test":   (6,  6,  6),   # 16.67% mines
        "easy":   (9,  9, 10),   # 12.35% mines
        "medium": (16, 16, 40),  # 15.62% mines
        "hard":   (16, 30, 99),  # 20.62% mines
    }
    LEVELS = dict(enumerate(DIFFICULTIES, 1))
    RUNNING  = 1
    OVER     = 0

    def __init__(self, level: int = 1, seed: int = 42) -> None:
        rows, cols, mine_count = self.DIFFICULTIES[self.LEVELS[level]]
        self.level = level
        self._setup(rows, cols, mine_count, seed)

    def _setup(self, rows: int, cols: int, mine_count: int, seed: int) -> None:
        self.rows             = rows
        self.cols             = cols
        self.mine_count       = mine_count
        self.total_safe_cells = rows * cols - mine_count
        self._flat_size       = rows * cols
        self._seed            = seed
        # Mine placement uses the C++ engine's own RNG; this one is for agents.
        self._rng             = np.random.default_rng(seed)

        self._c = c = msw.Engine(rows, cols, mine_count, int(seed) & _U64)
        n = self._flat_size

        self._mines_flat           = view(c, c.mines_addr(), np.bool_, n)
        self._revealed_flat        = view(c, c.revealed_addr(), np.bool_, n)
        self._neighbor_counts_flat = view(c, c.neighbor_counts_addr(), np.uint8, n)
        self._safe_flat_cells      = view(c, c.safe_cells_addr(), np.int32, n)
        self._mines                = self._mines_flat.reshape(rows, cols)
        self.revealed              = self._revealed_flat.reshape(rows, cols)
        self._neighbor_counts      = self._neighbor_counts_flat.reshape(rows, cols)

        self._neighbor_indices = view(c, c.neighbor_idx_addr(), np.int32, n * 8).reshape(n, 8)
        self._neighbor_lengths = view(c, c.neighbor_len_addr(), np.int8, n)
        self._neighbor_indices.setflags(write=False)
        self._neighbor_lengths.setflags(write=False)

        self._mine_probs: np.ndarray | None = None
        self._mine_probs_version = -1

    # -- state (lives in C++) ----------------------------------------------

    @property
    def state(self) -> int:
        return self._c.state

    @property
    def started(self) -> bool:
        return bool(self._c.started)

    @property
    def hit_mine(self) -> bool:
        return bool(self._c.hit_mine)

    @property
    def won(self) -> bool:
        return bool(self._c.won)

    @property
    def move_count(self) -> int:
        return self._c.move_count

    @property
    def covered_count(self) -> int:
        return self._c.covered_count

    @property
    def revealed_safe_cells(self) -> int:
        return self._c.revealed_safe_cells

    @property
    def _safe_pool_size(self) -> int:
        return self._c.safe_pool_size

    @property
    def exploded_cell(self) -> tuple[int, int] | None:
        flat = self._c.exploded_flat
        return None if flat < 0 else divmod(flat, self.cols)

    @property
    def min_win_moves(self) -> int | None:
        """Fewest moves that win this board, counting the first click. None before it.

        Mines are placed on the first click, so this is the best any player could
        do from that click on, even one who could see the mines.
        """
        n = self._c.min_win_moves
        return None if n < 0 else n

    def min_clicks_left(self) -> int:
        """Fewest clicks that clear every hidden safe cell, knowing where the mines are.

        One click per opening (connected patch of hidden zeros, which uncovers
        itself and its border) plus one per hidden number touching no hidden zero.
        """
        return self._c.min_clicks_left()

    @property
    def mine_probs(self) -> np.ndarray:
        """Bayesian mine probabilities, computed on first access after each change.

        NaN for revealed cells, float in [0, 1] for hidden. Callers that never
        touch this property never pay for the solve.
        """
        version = self._c.version
        if self._mine_probs is None or self._mine_probs_version != version:
            self._mine_probs         = self.get_mine_probabilities()
            self._mine_probs_version = version
        return self._mine_probs

    # -- game ---------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self._c.reset(int(self._seed) & _U64)
        self._mine_probs = None

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
        mines    = np.asarray(mines)
        revealed = np.asarray(revealed)
        if mines.ndim != 2 or mines.shape != revealed.shape:
            raise ValueError("mines and revealed must be 2-D arrays with the same shape")

        rows, cols = int(mines.shape[0]), int(mines.shape[1])
        m = np.ascontiguousarray(mines.astype(bool), dtype=np.uint8)
        r = np.ascontiguousarray(revealed.astype(bool), dtype=np.uint8)

        eng = object.__new__(cls)
        eng.level = None
        eng._setup(rows, cols, int(m.sum()), seed)
        eng._c.load_state(m.ctypes.data, r.ctypes.data)
        return eng

    def reveal(self, row: int, col: int) -> None:
        self.reveal_count(row, col)

    def reveal_count(self, row: int, col: int) -> int:
        """Reveal a cell. Returns the number of safe cells uncovered (0 on a mine or a no-op)."""
        n = self._c.reveal_count(int(row), int(col))
        if n < 0:
            raise IndexError(f"cell ({row}, {col}) is off the {self.rows}x{self.cols} board")
        return n

    def get_public_view(self, reveal_mines_on_loss: bool = True, out: np.ndarray | None = None) -> np.ndarray:
        """
        Returns an int8 board where:
        -1    = hidden
        0..8  = revealed neighbor count
        9     = mine (shown only after loss when reveal_mines_on_loss is True)
        """
        if out is None:
            out = np.empty((self.rows, self.cols), dtype=np.int8)
        if out.dtype == np.int8 and out.flags.c_contiguous and out.size == self._flat_size:
            self._c.public_view(out.ctypes.data, bool(reveal_mines_on_loss))
        else:
            tmp = np.empty(self._flat_size, dtype=np.int8)
            self._c.public_view(tmp.ctypes.data, bool(reveal_mines_on_loss))
            out[...] = tmp.reshape(out.shape)
        return out

    def get_mine_probabilities(self) -> np.ndarray:
        """Bayesian estimate of mine probability per cell from visible state only.

        Returns float64 board: revealed cells are np.nan; hidden cells are in [0, 1].
        """
        out = np.empty((self.rows, self.cols), dtype=np.float64)
        self._c.mine_probabilities(out.ctypes.data)
        return out
