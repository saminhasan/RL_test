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
    """Reward for one click: the goal, plus a progress signal that does not
    change which way of playing is best.

    The goal, paid when a game ends:
        win                            for clearing the board
        credit * progress * cleared    for the share of the safe cells uncovered (1 for a win)
        - mine                         for hitting a mine
    credit 0 makes winning the only goal; credit > 0 also counts how much of the
    board a lost game cleared (partial credit).

    The mine penalty on its own (mine 1, win 0, progress 0) with gamma 0 makes
    the value of a click minus the chance that the cell is a mine, learned only
    from how the agent's own clicks turned out: the greedy agent then clicks the
    cell least likely to be a mine, which is how bayes plays. Raising gamma
    towards 1 turns the same reward into minus the chance of losing the game.

    Paid on every click, potential-based shaping (Ng, Harada & Russell 1999):
        gamma * Phi(after) - Phi(before),
        Phi = progress * (share of the safe cells uncovered), and 0 once the game is over

    Click by click:
    * a safe click pays for the cells it uncovers, about progress * cells / S
      (S = the board's safe cells), so every click teaches something;
    * hitting a mine takes back the unbanked part of it, (1 - credit) * progress * cleared;
    * winning keeps all of it and adds `win`.

    Summed with the discount over a game, the shaping cancels out: the return is
    gamma^(clicks - 1) * (win * won + credit * progress * cleared at the end),
    minus Phi of the first board (where the automatic first click left off). So:
    * the best way to play is the one that is best for the goal above, exactly
      as without shaping (shaping speeds up learning and changes nothing else);
    * with credit 0, dying late is no better than dying early;
    * the discount gamma makes a quicker win worth a little more (about 0.1% per
      click at 0.999), so fewer clicks breaks ties without ever being worth a
      risk; and no reward can be had by giving up;
    * nothing hidden is used: no mine positions, no solver, no oracle count.

    gamma must be the discount the agent learns with (the trainer passes its own).
    """

    win: float = 1.0
    progress: float = 1.0
    credit: float = 0.0
    mine: float = 0.0
    gamma: float = 0.999

    def __post_init__(self) -> None:
        if not (min(self.win, self.progress, self.mine) >= 0 and self.win + self.mine > 0
                and 0 <= self.credit <= 1 and 0 <= self.gamma <= 1):
            raise ValueError("need win, progress, mine >= 0 with win or mine > 0, 0 <= credit <= 1, 0 <= gamma <= 1")

    def potential(self, engine: MinesweeperEngine) -> float:
        if engine.state == engine.OVER:
            return 0.0
        return self.progress * engine.revealed_safe_cells / engine.total_safe_cells

    def goal(self, engine: MinesweeperEngine) -> float:
        """The goal's payment once a game is over (0 while it runs)."""
        if engine.state != engine.OVER:
            return 0.0
        return (self.win * engine.won - self.mine * engine.hit_mine
                + self.credit * self.progress * engine.revealed_safe_cells / engine.total_safe_cells)


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
        cfg = self.reward_config
        before = cfg.potential(eng)
        self._clicks += 1
        truncated = self._clicks >= self.rows * self.cols
        if not (row >= self.rows or col >= self.cols or eng.revealed[row, col]):   # else a wasted click: nothing happens
            eng.reveal_count(row, col)
        terminated = eng.state == eng.OVER
        reward = cfg.goal(eng) + cfg.gamma * cfg.potential(eng) - before
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
