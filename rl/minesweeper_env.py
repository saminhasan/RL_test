"""Minesweeper as a gymnasium environment, for reinforcement learning.

Observation: uint8 array (2, canvas_rows, canvas_cols)
  [0] the visible board: 0 = hidden, 1..9 = a revealed clue 0..8, 10 = off the
      board (only when the board is smaller than the canvas)
  [1] the board's mine total, repeated in every cell. Together with the number
      of hidden cells it gives the mine density, the one global fact a player
      knows. (Repeating it keeps the observation a single small integer array,
      which the replay buffer stores as cheaply as possible.)

The canvas defaults to the board's own size. Training on several board sizes
in one run (easy -> medium -> hard) draws every board on the same canvas, the
hard board's 16 x 30, with the unused cells marked off the board; the board
size can then change between games (set_level) while the replay buffer keeps
one observation shape.

Action: Discrete(canvas_rows * canvas_cols), the cell to click. Hidden cells
are the useful ones; info["action_mask"] marks them. Clicking a revealed or
off-board cell does nothing but still costs a click (agents should mask; this
keeps every action valid). A game is cut off (truncated) after rows * cols
clicks, which no real game needs.

Every episode starts after one automatic first click: the same cell for a
given board seed, whoever plays, so every agent faces the same game.

The agent never sees the mines, and the reward (RewardConfig) uses neither the
solver nor the oracle. info["min_moves"] (the oracle's fewest possible moves
for this board) is there for measuring only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from engine import MinesweeperEngine

HIDDEN    = 0        # observation plane 0 codes
OFF_BOARD = 10


def board_shape(level: int) -> tuple[int, int]:
    rows, cols, _ = MinesweeperEngine.DIFFICULTIES[MinesweeperEngine.LEVELS[level]]
    return rows, cols


@dataclass(frozen=True)
class RewardConfig:
    """Reward for one click. S = the board's number of safe cells; scaling by S
    keeps the numbers the same on every board size.

        progress  + progress * (cells this click uncovered) / S
        click     - click / S                   on every click
        win       + win                         when the last safe cell is uncovered
        mine      - mine                        when a mine is hit (the game ends)

    Why these terms, with defaults progress 0.5, click 0.25, win 1, mine 0.5:

    * Winning matters most. A won game returns at least 1 + 0.5 - 0.25 = 1.25;
      a lost game at most 0.5 - 0.5 = 0.
    * A signal on every safe click, not only at the end: progress pays for what
      each click uncovers, so a big opening pays more right away.
    * It can't be gamed by clicking more. Over a whole won game, progress always
      adds up to exactly `progress` (each safe cell is uncovered once), however
      many clicks it took. A flat bonus per safe click would instead teach the
      agent to take many small clicks.
    * Fewer moves is better: each click costs click / S, so a wasted click (a
      number that an opening would have uncovered anyway) lowers the return.
    * Never worth dying to stop paying click costs: a safe click uncovers at
      least one cell, so it nets at least (progress - click) / S > 0.
    * Nothing hidden is used: no mine positions, no solver, no oracle count. The
      automatic first click is not the agent's move and earns nothing.
    """

    progress: float = 0.5
    click: float = 0.25
    win: float = 1.0
    mine: float = 0.5

    def __post_init__(self) -> None:
        if not self.progress > self.click >= 0:
            raise ValueError("need progress > click >= 0, or dying could beat a safe click")


def first_click(board_seed: int, n_cells: int) -> int:
    """The automatic first click for a board seed: the same cell for every agent."""
    return int(np.random.default_rng([int(board_seed), 1]).integers(n_cells))


def start_game(engine: MinesweeperEngine, board_seed: int) -> int:
    """Deal the board for `board_seed` and make the automatic first click.
    A first click that clears the whole board is redealt. Returns the seed used."""
    n_cells = engine.rows * engine.cols
    while True:
        engine.reset(int(board_seed))
        engine.reveal_count(*divmod(first_click(board_seed, n_cells), engine.cols))
        if engine.state == engine.RUNNING:
            return int(board_seed)
        board_seed = int(board_seed) + 1_000_003


def encode(engine: MinesweeperEngine, out: np.ndarray, view: np.ndarray) -> np.ndarray:
    """Write the observation for `engine` into `out` (2, canvas_rows, canvas_cols) uint8;
    `view` is scratch space of the board's own shape."""
    engine.get_public_view(reveal_mines_on_loss=False, out=view)
    rows, cols = view.shape
    if out.shape[1:] != view.shape:
        out[0].fill(OFF_BOARD)
    np.add(view, 1, out=out[0, :rows, :cols], casting="unsafe")   # -1 hidden -> 0, clue k -> k + 1
    out[1] = engine.mine_count
    return out


class MinesweeperEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, level: int = 4, reward: RewardConfig | None = None, seed: int | None = None,
                 canvas: tuple[int, int] | None = None, render_mode: str | None = None) -> None:
        self.canvas = tuple(canvas) if canvas else board_shape(level)
        self.reward_config = reward or RewardConfig()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(0, 255, (2, *self.canvas), np.uint8)
        self.action_space = spaces.Discrete(self.canvas[0] * self.canvas[1])
        self._boards = np.random.default_rng(seed)
        self._obs = np.zeros((2, *self.canvas), np.uint8)
        self.board_seed = -1
        self._clicks = 0
        self._pending_level = None
        self._use_level(level)

    def _use_level(self, level: int) -> None:
        rows, cols = board_shape(level)
        if rows > self.canvas[0] or cols > self.canvas[1]:
            raise ValueError(f"level {level} ({rows}x{cols}) does not fit the {self.canvas} canvas")
        self.level, self.rows, self.cols = level, rows, cols
        self.engine = MinesweeperEngine(level=level, seed=0)
        self._view = np.empty((rows, cols), np.int8)
        self._on_board = np.zeros(self.canvas, bool)
        self._on_board[:rows, :cols] = True

    def set_level(self, level: int) -> None:
        """Switch board size from the next game on (the current game finishes first)."""
        self._pending_level = level

    # -- gymnasium API ------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._boards = np.random.default_rng(seed)
        if self._pending_level is not None and self._pending_level != self.level:
            self._use_level(self._pending_level)
        self._pending_level = None
        board_seed = (options or {}).get("board_seed")
        if board_seed is None:
            board_seed = int(self._boards.integers(2**31))
        self.board_seed = start_game(self.engine, board_seed)
        self._clicks = 0
        return encode(self.engine, self._obs, self._view).copy(), self._info()

    def step(self, action):
        eng = self.engine
        row, col = divmod(int(action), self.canvas[1])
        cfg, safe = self.reward_config, eng.total_safe_cells
        self._clicks += 1
        truncated = self._clicks >= self.rows * self.cols
        if row >= self.rows or col >= self.cols or eng.revealed[row, col]:   # wasted click: nothing happens
            return encode(eng, self._obs, self._view).copy(), -cfg.click / safe, False, truncated, self._info()
        uncovered = eng.reveal_count(row, col)
        reward = -cfg.click / safe
        if eng.hit_mine:
            reward -= cfg.mine
        else:
            reward += cfg.progress * uncovered / safe
            if eng.won:
                reward += cfg.win
        terminated = eng.state == eng.OVER
        return encode(eng, self._obs, self._view).copy(), float(reward), terminated, truncated and not terminated, self._info()

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        chars = np.array(list("#012345678*"))
        board = self.engine.get_public_view(reveal_mines_on_loss=True)
        return "\n".join("".join(chars[np.where(row == 9, 10, row + 1)]) for row in board)

    # ------------------------------------------------------------------------

    def _info(self) -> dict:
        eng = self.engine
        mask = self._on_board.copy()
        mask[:self.rows, :self.cols] &= ~eng.revealed
        return {
            "action_mask":       mask.ravel(),
            "level":             self.level,
            "board_seed":        self.board_seed,
            "won":               eng.won,
            "hit_mine":          eng.hit_mine,
            "moves":             eng.move_count,
            "revealed_fraction": eng.revealed_safe_cells / eng.total_safe_cells,
            "min_moves":         eng.min_win_moves,   # oracle count: for measuring only
        }

    def reward_dict(self) -> dict:
        return asdict(self.reward_config)
