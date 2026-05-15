"""
Training configuration. Edit cfg at the bottom, then run train.py.
All path properties are derived from level — do not set manually.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from engine import MinesweeperEngine


@dataclass
class TrainConfig:
    # ── environment ───────────────────────────────────────────────────────────
    level: int = 2              # 1=test  2=easy  3=medium  4=hard
    vec_backend: str = "dummy"  # "dummy" = DummyVecEnv
                                # "batched" = SB3VecEnvAdapter (faster on hard)
    num_envs: int = 16          # parallel envs — PPO collects n_steps * num_envs per update
    # ── PPO ──────────────────────────────────────────────────────────────────
    learning_rate: float = 3e-4
    n_steps: int = 512          # rollout steps per env before each update
    batch_size: int = 256       # minibatch size (must divide n_steps * num_envs)
    n_epochs: int = 4           # gradient epochs per rollout
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01      # entropy bonus — keeps exploration alive
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True

    # ── training ─────────────────────────────────────────────────────────────
    total_timesteps: int = 1_000_000
    seed: int = 42

    # ── model ─────────────────────────────────────────────────────────────────
    features_dim: int = 512     # CNN output feature vector size

    # ── eval & checkpointing ──────────────────────────────────────────────────
    eval_freq: int = 10_000     # evaluate every N env steps
    eval_episodes: int = 200

    # ── derived paths (read-only) ─────────────────────────────────────────────
    @property
    def level_name(self) -> str:
        return MinesweeperEngine.LEVELS[self.level]

    @property
    def save_dir(self) -> str:
        return f"models/rl/{self.level_name}"

    @property
    def model_path(self) -> str:
        return f"{self.save_dir}/model"

    @property
    def best_model_path(self) -> str:
        return f"{self.save_dir}/best_model"

    @property
    def tb_log_dir(self) -> str:
        return f"{self.save_dir}/tensorboard"

    @property
    def train_state_path(self) -> str:
        return f"{self.save_dir}/train_state.json"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["level_name"] = self.level_name
        d["save_dir"] = self.save_dir
        return d


# ── edit this instance before running train.py ────────────────────────────────
cfg = TrainConfig(
    level=1,
    vec_backend="dummy",
    num_envs=32,

    learning_rate=2e-4,
    n_steps=512,
    batch_size=256,
    n_epochs=8,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    normalize_advantage=True,

    total_timesteps=5_000_000,
    seed=42,

    features_dim=512,
    eval_freq=10_000,
    eval_episodes=1000,
)
