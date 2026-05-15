"""
Gymnasium Minesweeper environment + vectorized variant + SB3 adapter.

Observation : (15, rows, cols)  float32  — fixed 15 channels, see OBS_CHANNELS
Action      : Discrete(rows * cols)   flat cell index
Reward      : Bayes-aware dense shaping, values in [-2, 1]:
               safe step : log1p(gained)/log1p(total_safe) * (1 + 0.5*(1-p_click))  in [0, 1]
               win       : +1.0
               mine hit  : 0.0 if ambiguous tie at p_min,
                           else -clip(p*(1+0.5*progress), 0, 1) - 1.0              in [-2, -1]

Observation channels
--------------------
 0  hidden_mask          1 = not yet revealed
 1  action_mask          1 = valid click  (= ch0 now; diverges if flagging added)
 2  one_hot_count_0      revealed AND neighbor_count == 0
 3  one_hot_count_1      revealed AND neighbor_count == 1
 ...
10  one_hot_count_8      revealed AND neighbor_count == 8
11  bayes_prob           mine probability [0,1] for hidden; 0 for revealed
                         (uniform prior before first click; 0 for revealed cells)
12  mine_density_prior   mine_count / total_cells  (constant)
13  covered_mine_density mine_count / covered_count  (rises as safe cells revealed)
14  safe_progress        revealed_safe / total_safe
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

from engine import MinesweeperEngine

N_OBS_CHANNELS = 15


# ──────────────────────────────────────────────────────────────────────────────
# Shared obs builder (used by single env and batched env)
# ──────────────────────────────────────────────────────────────────────────────

def _build_obs(buf: np.ndarray, eng: MinesweeperEngine) -> None:
    """Fill 15-channel float32 obs buffer in-place. buf shape: (15, rows, cols)."""
    total_cells = eng.rows * eng.cols
    mine_prior = eng.mine_count / total_cells

    hidden = ~eng.revealed
    buf[0] = hidden
    buf[1] = hidden  # action_mask = hidden_mask (same without flagging)

    # one-hot neighbor counts: only non-zero for revealed cells
    buf[2:11] = 0.0
    counts = eng._neighbor_counts
    revealed = eng.revealed
    for k in range(9):
        buf[2 + k] = revealed & (counts == k)

    # Bayes mine probability — cached by engine after every reveal/reset
    buf[11] = eng.mine_probs               # float64→float32 copy; does not touch cache
    np.nan_to_num(buf[11], nan=0.0, copy=False)

    buf[12] = mine_prior

    covered = int(eng.covered_count)
    buf[13] = (eng.mine_count / covered) if covered > 0 else 0.0
    buf[14] = eng.revealed_safe_cells / eng.total_safe_cells


def _click_risk(eng: MinesweeperEngine, flat_action: int) -> tuple[float, bool]:
    """
    Returns (p_clicked, is_ambiguous) from eng.mine_probs (engine-cached, no recompute).
    is_ambiguous: clicked cell tied for lowest prob among hidden cells — unavoidable loss.
    """
    probs_flat = eng.mine_probs.ravel()
    p_clicked = float(np.nan_to_num(probs_flat[flat_action], nan=0.0))
    hidden_probs = probs_flat[~eng.revealed.ravel()]
    valid = hidden_probs[~np.isnan(hidden_probs)]
    if len(valid) == 0:
        return p_clicked, False
    p_min = float(np.min(valid))
    n_at_min = int(np.sum(np.abs(valid - p_min) < 1e-6))
    return p_clicked, (abs(p_clicked - p_min) < 1e-6 and n_at_min > 1)


def _calc_reward(
    eng: MinesweeperEngine,
    p_clicked: float,
    is_ambiguous: bool,
    gained: int,
    total_safe: int,
    progress: float,
) -> float:
    """
    Return values in [-2, 1].

    Mine hit:
      - ambiguous (clicked lowest-prob cell in a tie): 0.0 — unavoidable
      - culpable: -clip(p_clicked * (1 + 0.5*progress), 0, 1) - 1.0   range [-2, -1]
        riskier click = larger penalty; late-game hit = up to 50% more severe

    Win: +1.0

    Safe step in [0, 1]:
      log1p(gained)/log1p(total_safe) — logarithmic cascade in [0,1]
      multiplied by (1 + 0.5*(1-p_clicked)) — Bayes-safe clicks get up to 1.5x
      capped at 1.0 to stay bounded
    """
    if eng.hit_mine:
        if is_ambiguous:
            return 0.0
        return -float(np.clip(p_clicked * (1.0 + 0.5 * progress), 0.0, 1.0)) - 1.0
    if eng.won:
        return 1.0
    cascade = float(np.log1p(gained) / np.log1p(total_safe))
    return min(cascade * (1.0 + 0.5 * (1.0 - p_clicked)), 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Single environment
# ──────────────────────────────────────────────────────────────────────────────

class MinesweeperEnv(gym.Env):
    """Single Gymnasium Minesweeper environment. Headless only."""

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        level: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        rows, cols, mine_count = MinesweeperEngine.DIFFICULTIES[
            MinesweeperEngine.LEVELS[level]
        ]
        self.level = level
        self.rows, self.cols = rows, cols
        self.mine_count = mine_count
        self.total_safe = rows * cols - mine_count
        self.n_cells = rows * cols

        self.action_space = spaces.Discrete(self.n_cells)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_OBS_CHANNELS, rows, cols), dtype=np.float32
        )

        self._engine: Optional[MinesweeperEngine] = None
        self._ep_count = 0
        self._base_seed = seed
        self._obs_buf = np.zeros((N_OBS_CHANNELS, rows, cols), dtype=np.float32)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        ep_seed = self._next_seed()

        if self._engine is None:
            self._engine = MinesweeperEngine(level=self.level, seed=ep_seed)
        else:
            self._engine.reset(seed=ep_seed)
        self._ep_count += 1

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self._engine is not None, "call reset() before step()"
        row, col = divmod(int(action), self.cols)

        if bool(self._engine.revealed[row, col]):
            return self._get_obs(), 0.0, False, False, self._get_info()

        p_clicked, is_ambiguous = _click_risk(self._engine, int(action))
        progress = self._engine.revealed_safe_cells / self.total_safe

        prev_safe = self._engine.revealed_safe_cells
        self._engine.reveal(row, col)
        gained = self._engine.revealed_safe_cells - prev_safe
        terminated = self._engine.state == MinesweeperEngine.OVER
        reward = _calc_reward(self._engine, p_clicked, is_ambiguous, gained, self.total_safe, progress)

        return self._get_obs(), reward, terminated, False, self._get_info()

    def action_masks(self) -> np.ndarray:
        """(n_cells,) bool — True = valid action."""
        assert self._engine is not None
        return ~self._engine.revealed.ravel()

    def render(self) -> None:
        pass

    def close(self) -> None:
        self._engine = None

    def _next_seed(self) -> int:
        if self._base_seed is None:
            return int(self.np_random.integers(0, 2**31))
        return self._base_seed + self._ep_count

    def _get_obs(self) -> np.ndarray:
        assert self._engine is not None
        _build_obs(self._obs_buf, self._engine)
        return self._obs_buf.copy()

    def _get_info(self) -> Dict[str, Any]:
        eng = self._engine
        if eng is None:
            return {}
        return {
            "won": bool(eng.won),
            "hit_mine": bool(eng.hit_mine),
            "safe_revealed": int(eng.revealed_safe_cells),
            "safe_total": self.total_safe,
            "moves": int(eng.move_count),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Vectorized environment (single-process batched)
# ──────────────────────────────────────────────────────────────────────────────

class VectorizedMinesweeperEnv:
    """
    Single-process batched Minesweeper envs.

    Runs N engines in one Python process — no subprocess/IPC overhead.
    Returns pre-allocated numpy arrays for zero-copy torch.from_numpy() → GPU.
    Auto-resets individual envs on episode termination.

    Parameters
    ----------
    num_envs    : parallel environments
    level       : 1=test  2=easy  3=medium  4=hard
    base_seed   : env i seeds with base_seed + i; increments by num_envs per episode

    Interface
    ---------
    obs, info               = env.reset()           obs: (N, 15, H, W) float32
    obs, rew, done, _, info = env.step(actions)     actions: (N,) int
    masks                   = env.action_masks()    (N, n_cells) bool
    """

    def __init__(
        self,
        num_envs: int = 64,
        level: int = 1,
        base_seed: int = 42,
    ) -> None:
        self.num_envs = num_envs
        self.level = level

        rows, cols, mine_count = MinesweeperEngine.DIFFICULTIES[
            MinesweeperEngine.LEVELS[level]
        ]
        self.rows, self.cols = rows, cols
        self.mine_count = mine_count
        self.total_safe = rows * cols - mine_count
        self.n_cells = rows * cols

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_OBS_CHANNELS, rows, cols), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_cells)

        self._obs_buf  = np.zeros((num_envs, N_OBS_CHANNELS, rows, cols), dtype=np.float32)
        self._rew_buf  = np.zeros(num_envs, dtype=np.float32)
        self._done_buf = np.zeros(num_envs, dtype=bool)
        self._mask_buf = np.ones((num_envs, self.n_cells), dtype=bool)
        self._seeds    = np.arange(num_envs, dtype=np.int64) + base_seed
        self._engines: List[MinesweeperEngine] = []

    def reset(
        self, *, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if seed is not None:
            self._seeds[:] = np.arange(self.num_envs, dtype=np.int64) + seed

        if not self._engines:
            self._engines = [
                MinesweeperEngine(level=self.level, seed=int(s))
                for s in self._seeds
            ]
        else:
            for i, eng in enumerate(self._engines):
                eng.reset(seed=int(self._seeds[i]))

        self._seeds += self.num_envs

        for i, eng in enumerate(self._engines):
            _build_obs(self._obs_buf[i], eng)
            np.logical_not(eng.revealed.ravel(), out=self._mask_buf[i])

        return self._obs_buf.copy(), self._collect_info()

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        won_buf   = np.zeros(self.num_envs, dtype=bool)
        mine_buf  = np.zeros(self.num_envs, dtype=bool)
        safe_buf  = np.zeros(self.num_envs, dtype=np.int32)
        moves_buf = np.zeros(self.num_envs, dtype=np.int32)

        for i, (eng, action) in enumerate(zip(self._engines, actions)):
            row, col = divmod(int(action), self.cols)

            if bool(eng.revealed[row, col]):
                self._rew_buf[i] = 0.0
                self._done_buf[i] = False
            else:
                p_clicked, is_ambiguous = _click_risk(eng, int(action))
                progress = eng.revealed_safe_cells / self.total_safe

                prev_safe = eng.revealed_safe_cells
                eng.reveal(row, col)
                gained = eng.revealed_safe_cells - prev_safe
                self._done_buf[i] = eng.state == MinesweeperEngine.OVER
                self._rew_buf[i] = _calc_reward(eng, p_clicked, is_ambiguous, gained, self.total_safe, progress)

            won_buf[i]   = eng.won
            mine_buf[i]  = eng.hit_mine
            safe_buf[i]  = eng.revealed_safe_cells
            moves_buf[i] = eng.move_count

            if self._done_buf[i]:
                self._seeds[i] += self.num_envs
                eng.reset(seed=int(self._seeds[i]))

            _build_obs(self._obs_buf[i], eng)
            np.logical_not(eng.revealed.ravel(), out=self._mask_buf[i])

        info: Dict[str, np.ndarray] = {
            "won":           won_buf,
            "hit_mine":      mine_buf,
            "safe_revealed": safe_buf,
            "safe_total":    np.full(self.num_envs, self.total_safe, dtype=np.int32),
            "moves":         moves_buf,
        }

        return (
            self._obs_buf.copy(),
            self._rew_buf.copy(),
            self._done_buf.copy(),
            np.zeros(self.num_envs, dtype=bool),
            info,
        )

    def action_masks(self) -> np.ndarray:
        """(num_envs, n_cells) bool — True = valid action."""
        return self._mask_buf.copy()

    def close(self) -> None:
        self._engines.clear()

    def _collect_info(self) -> Dict[str, np.ndarray]:
        return {
            "won":           np.array([e.won for e in self._engines],                dtype=bool),
            "hit_mine":      np.array([e.hit_mine for e in self._engines],            dtype=bool),
            "safe_revealed": np.array([e.revealed_safe_cells for e in self._engines], dtype=np.int32),
            "safe_total":    np.full(self.num_envs, self.total_safe,                  dtype=np.int32),
            "moves":         np.array([e.move_count for e in self._engines],          dtype=np.int32),
        }


# ──────────────────────────────────────────────────────────────────────────────
# SB3 VecEnv adapter (wraps VectorizedMinesweeperEnv for MaskablePPO)
# ──────────────────────────────────────────────────────────────────────────────

class SB3VecEnvAdapter(VecEnv):
    """
    Wraps VectorizedMinesweeperEnv to satisfy SB3's VecEnv interface.
    Use via make_train_env(config) when config.vec_backend == "batched".

    Wires env_method("action_masks") so that sb3-contrib's
    get_action_masks() works correctly with MaskablePPO.
    """

    def __init__(self, venv: VectorizedMinesweeperEnv) -> None:
        self.venv = venv
        super().__init__(venv.num_envs, venv.observation_space, venv.action_space)
        self._actions: Optional[np.ndarray] = None

    def reset(self) -> VecEnvObs:
        obs, _ = self.venv.reset()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        assert self._actions is not None, "call step_async() before step_wait()"
        obs, rews, dones, _, infos_dict = self.venv.step(self._actions)
        infos = [
            {k: (v[i].item() if hasattr(v[i], "item") else v[i]) for k, v in infos_dict.items()}
            for i in range(self.num_envs)
        ]
        return obs, rews, dones, infos

    def close(self) -> None:
        self.venv.close()

    def env_method(
        self, method_name: str, *method_args, indices=None, **method_kwargs
    ) -> List[Any]:
        if method_name == "action_masks":
            masks = self.venv.action_masks()  # (N, n_cells)
            return [masks[i] for i in range(self.num_envs)]
        raise NotImplementedError(f"env_method '{method_name}' not supported by SB3VecEnvAdapter")

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        val = getattr(self.venv, attr_name, None)
        n = (1 if isinstance(indices, int) else len(list(indices))) if indices is not None else self.num_envs
        return [val] * n

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self.venv, attr_name, value)

    def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]:
        n = (1 if isinstance(indices, int) else len(list(indices))) if indices is not None else self.num_envs
        return [False] * n

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        return [None] * self.num_envs

    def render(self, mode: str = "human") -> None:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Factory functions
# ──────────────────────────────────────────────────────────────────────────────

def make_train_env(config) -> VecEnv:
    """Build SB3-compatible training VecEnv from TrainConfig."""
    if config.vec_backend == "batched":
        venv = VectorizedMinesweeperEnv(
            num_envs=config.num_envs,
            level=config.level,
            base_seed=config.seed,
        )
        return SB3VecEnvAdapter(venv)

    # "dummy" — SB3 native, handles action_masks() via env_method automatically
    def _make():
        return MinesweeperEnv(level=config.level)
    return DummyVecEnv([_make] * config.num_envs)


def make_eval_env(config, n_envs: int = 4) -> VecEnv:
    """Build small deterministic eval VecEnv. Always uses DummyVecEnv."""
    def _make():
        return MinesweeperEnv(level=config.level, seed=config.seed + 8_888_888)
    return DummyVecEnv([_make] * n_envs)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("-- single env ----------------------------------------")
    env = MinesweeperEnv(level=2, seed=0)
    obs, _ = env.reset()
    print(f"obs  : {obs.shape}  dtype={obs.dtype}")
    print(f"mask : {env.action_masks().shape}")

    done = False
    total_r = 0.0
    while not done:
        valid = np.flatnonzero(env.action_masks())
        obs, r, done, _, info = env.step(int(np.random.choice(valid)))
        total_r += r
    print(f"won={info['won']}  reward={total_r:.3f}")
    env.close()

    print("\n-- vectorized env (16 envs, 200 steps) --------------")
    venv = VectorizedMinesweeperEnv(num_envs=16, level=2, base_seed=0)
    obs, _ = venv.reset()
    print(f"obs  : {obs.shape}")

    t0 = time.perf_counter()
    for _ in range(200):
        masks = venv.action_masks()
        actions = np.array([int(np.random.choice(np.flatnonzero(m))) for m in masks])
        obs, rew, done, _, info = venv.step(actions)
    dt = time.perf_counter() - t0
    print(f"steps/sec : {16 * 200 / dt:,.0f}")
    venv.close()
