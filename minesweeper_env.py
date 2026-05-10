from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from engine import MinesweeperEngine


BOARD_CHANNELS = 14


@dataclass(frozen=True)
class MinesweeperEnvConfig:
    level: int | str = "test"
    seed: int = 42
    n_envs: int = 1
    use_subproc_env: bool = True


class EnvConfigLike(Protocol):
    level: int | str
    seed: int
    n_envs: int
    use_subproc_env: bool


def level_to_id(level: int | str) -> int:
    if isinstance(level, int):
        if level in MinesweeperEngine.LEVELS:
            return level
        raise ValueError(f"unknown level id {level}")

    normalized = level.strip().lower()
    if normalized.isdigit():
        return level_to_id(int(normalized))

    for level_id, level_name in MinesweeperEngine.LEVELS.items():
        if normalized == level_name:
            return level_id

    choices = ", ".join(f"{idx}:{name}" for idx, name in MinesweeperEngine.LEVELS.items())
    raise ValueError(f"unknown level {level!r}. Choose one of {choices}")


def level_name(level: int | str) -> str:
    return MinesweeperEngine.LEVELS[level_to_id(level)]


def encode_board(public_view: np.ndarray) -> np.ndarray:
    """
    Encode an engine public board as model-ready channels.

    public_view values:
      -1 = hidden
       0..8 = revealed neighbor count

    Channels:
      0      hidden cells
      1..9   revealed numbers 0..8
      10     any revealed cell
      11     constant bias plane
      12     normalized row coordinate
      13     normalized column coordinate
    """
    encoded = np.zeros((BOARD_CHANNELS, *public_view.shape), dtype=np.float32)
    hidden = public_view < 0
    encoded[0] = hidden.astype(np.float32)

    for value in range(9):
        encoded[value + 1] = (public_view == value).astype(np.float32)

    encoded[10] = (~hidden).astype(np.float32)
    encoded[11].fill(1.0)

    rows, cols = public_view.shape
    if rows > 1:
        encoded[12] = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None]
    if cols > 1:
        encoded[13] = np.linspace(0.0, 1.0, cols, dtype=np.float32)[None, :]
    return encoded


class MinesweeperMaskEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, level: int | str = "test", seed: int = 42) -> None:
        super().__init__()
        self.level_id = level_to_id(level)
        self.base_seed = seed
        self.episode_index = 0
        self.engine = MinesweeperEngine(level=self.level_id, seed=seed, headless=True)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(BOARD_CHANNELS, self.engine.rows, self.engine.cols),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.engine.rows * self.engine.cols)
        self._obs_buffer = np.zeros(self.observation_space.shape, dtype=np.float32)
        self._obs_work = np.empty(self.engine.rows * self.engine.cols, dtype=bool)
        self._action_mask = np.empty(self.engine.rows * self.engine.cols, dtype=bool)
        self._init_static_obs_planes()

    def _init_static_obs_planes(self) -> None:
        self._obs_buffer[11].fill(1.0)
        if self.engine.rows > 1:
            self._obs_buffer[12] = np.linspace(0.0, 1.0, self.engine.rows, dtype=np.float32)[:, None]
        if self.engine.cols > 1:
            self._obs_buffer[13] = np.linspace(0.0, 1.0, self.engine.cols, dtype=np.float32)[None, :]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.base_seed = seed
            self.episode_index = 0

        episode_seed = self.base_seed + self.episode_index
        self.episode_index += 1
        self.engine.reset(seed=episode_seed)
        return self._obs(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        row, col = divmod(int(action), self.engine.cols)
        invalid = self.engine.revealed[row, col]
        revealed_count = self.engine.reveal_count(row, col)
        terminated = self.engine.state == MinesweeperEngine.OVER
        reward = self._reward(revealed_count, invalid)
        info = self._info(invalid=bool(invalid))
        return self._obs(), reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        np.logical_not(self.engine._revealed_flat, out=self._action_mask)
        return self._action_mask

    def _obs(self) -> np.ndarray:
        obs = self._obs_buffer
        obs[:11].fill(0.0)

        revealed = self.engine._revealed_flat
        counts = self.engine._neighbor_counts_flat
        hidden_plane = obs[0].ravel()
        revealed_plane = obs[10].ravel()

        np.logical_not(revealed, out=self._obs_work)
        hidden_plane[:] = self._obs_work
        revealed_plane[:] = revealed

        for value in range(9):
            np.equal(counts, value, out=self._obs_work)
            np.logical_and(self._obs_work, revealed, out=self._obs_work)
            obs[value + 1].ravel()[:] = self._obs_work

        return obs

    def _info(self, invalid: bool = False) -> dict[str, Any]:
        return {
            "level": level_name(self.level_id),
            "level_id": self.level_id,
            "won": self.engine.won,
            "hit_mine": self.engine.hit_mine,
            "moves": self.engine.move_count,
            "invalid": invalid,
            "revealed_safe_cells": self.engine.revealed_safe_cells,
            "total_safe_cells": self.engine.total_safe_cells,
        }

    def _reward(self, revealed_count: int, invalid: bool) -> float:
        if invalid:
            return -0.2

        if self.engine.hit_mine:
            return -1.0

        progress = revealed_count / self.engine.total_safe_cells

        if self.engine.won:
            return 2.0 + progress

        return 0.05 * progress


def _env_config(
    config_or_level: EnvConfigLike | MinesweeperEnvConfig | int | str = "test",
    seed: int = 42,
    n_envs: int = 1,
    use_subproc_env: bool = True,
) -> MinesweeperEnvConfig:
    if isinstance(config_or_level, (int, str)):
        return MinesweeperEnvConfig(
            level=config_or_level,
            seed=seed,
            n_envs=n_envs,
            use_subproc_env=use_subproc_env,
        )

    return MinesweeperEnvConfig(
        level=config_or_level.level,
        seed=config_or_level.seed,
        n_envs=config_or_level.n_envs,
        use_subproc_env=config_or_level.use_subproc_env,
    )


def make_env(
    config: EnvConfigLike | MinesweeperEnvConfig | int | str = "test",
    seed: int = 42,
    seed_offset: int = 0,
) -> MinesweeperMaskEnv:
    config = _env_config(config, seed=seed)
    return MinesweeperMaskEnv(level=config.level, seed=config.seed + seed_offset)


def make_env_fn(config: EnvConfigLike | MinesweeperEnvConfig, rank: int):
    def _init() -> MinesweeperMaskEnv:
        return make_env(config, seed_offset=rank * 1_000_000)

    return _init


def make_train_env(
    config: EnvConfigLike | MinesweeperEnvConfig | int | str = "test",
    seed: int = 42,
    n_envs: int = 1,
    use_subproc_env: bool = True,
) -> VecEnv:
    config = _env_config(
        config,
        seed=seed,
        n_envs=n_envs,
        use_subproc_env=use_subproc_env,
    )
    env_fns = [make_env_fn(config, rank) for rank in range(config.n_envs)]
    if config.n_envs <= 1:
        return DummyVecEnv(env_fns)
    if config.use_subproc_env:
        return SubprocVecEnv(env_fns, start_method="spawn")
    return DummyVecEnv(env_fns)
