"""
Model creation, saving, and loading for Minesweeper PPO training.

CNN architecture:
    Input : (N, 15, rows, cols)  float32

    Local stream — 3 conv blocks, each: Conv2d 3×3 same-pad → ReLU → SE
        Conv2d(15  → 64,  3×3, pad=1) → ReLU → SE(64)
        Conv2d(64  → 128, 3×3, pad=1) → ReLU → SE(128)
        Conv2d(128 → 128, 3×3, pad=1) → ReLU → SE(128)
        → (N, 128, H, W)

    SE block (Squeeze-and-Excite):
        AdaptiveAvgPool2d(1) → Linear(C → C//8) → ReLU → Linear(C//8 → C) → Sigmoid
        Channel-wise multiply: re-weights feature maps by their global importance.

    Global branch (parallel to flatten):
        AdaptiveAvgPool2d(1) → Flatten → Linear(128 → 64) → ReLU
        Captures board-wide spatial averages; complements per-cell local features.

    Head:
        cat(Flatten(local), global) → Linear(n_flat + 64 → features_dim) → ReLU
        → actor + critic heads added by SB3 (MaskablePPO)

Optimizer (Adam) state is stored inside model.zip automatically by PyTorch/SB3.
Resuming with load_or_create() restores weights + Adam moments.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnv

from minesweeper_env import N_OBS_CHANNELS


class SEBlock(nn.Module):
    """Squeeze-and-Excite: learn per-channel importance from global average."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.excite(self.squeeze(x))           # (B, C)
        return x * w.unsqueeze(-1).unsqueeze(-1)   # scale each channel


class MinesweeperCNN(BaseFeaturesExtractor):
    """
    CNN feature extractor for the 15-channel Minesweeper observation.

    Local stream  : 3× [Conv 3×3 same-pad → ReLU → SE]  →  (B, 128, H, W)
    Global branch : AdaptiveAvgPool2d(1) → Linear(128→64) → ReLU  →  (B, 64)
    Head          : cat(flatten, global) → Linear → ReLU  →  (B, features_dim)
    """

    GLOBAL_DIM: int = 64

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512) -> None:
        super().__init__(observation_space, features_dim)

        n_in = observation_space.shape[0]

        self.conv = nn.Sequential(
            nn.Conv2d(n_in, 64,  kernel_size=3, padding=1), nn.ReLU(), SEBlock(64),
            nn.Conv2d(64,  128, kernel_size=3, padding=1), nn.ReLU(), SEBlock(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), SEBlock(128),
        )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, self.GLOBAL_DIM),
            nn.ReLU(),
        )

        with torch.no_grad():
            sample  = torch.zeros(1, *observation_space.shape)
            n_flat  = self.conv(sample).flatten(1).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flat + self.GLOBAL_DIM, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(obs)
        local    = conv_out.flatten(1)
        global_  = self.global_pool(conv_out)
        return self.linear(torch.cat([local, global_], dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# Build / save / load
# ──────────────────────────────────────────────────────────────────────────────

def build_model(config, env: VecEnv) -> MaskablePPO:
    """Create a fresh MaskablePPO with MinesweeperCNN feature extractor."""
    policy_kwargs = {
        "features_extractor_class": MinesweeperCNN,
        "features_extractor_kwargs": {"features_dim": config.features_dim},
    }

    return MaskablePPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        normalize_advantage=config.normalize_advantage,
        tensorboard_log=config.tb_log_dir,
        seed=config.seed,
        device="auto",
        verbose=1,
        policy_kwargs=policy_kwargs,
    )


def _default_train_state(config) -> Dict[str, Any]:
    return {
        "total_timesteps_done": 0,
        "episodes_done": 0,
        "best_win_rate": -1.0,
        "best_model_timestep": 0,
        "last_saved": None,
        "level": config.level_name,
        "config_snapshot": config.to_dict(),
        "recent_metrics": {
            "timesteps":       [],
            "win_rates":       [],
            "mean_rewards":    [],
            "mean_ep_lengths": [],
        },
    }


def _load_train_state(config) -> Dict[str, Any]:
    path = config.train_state_path
    if not os.path.exists(path):
        return _default_train_state(config)
    with open(path) as f:
        state = json.load(f)
    for key, val in _default_train_state(config).items():
        state.setdefault(key, val)
    return state


def save_checkpoint(
    model: MaskablePPO,
    config,
    train_state: Dict[str, Any],
) -> None:
    """Save model weights (+ Adam optimizer state) and train_state.json."""
    os.makedirs(config.save_dir, exist_ok=True)
    model.save(config.model_path)
    train_state["total_timesteps_done"] = int(model.num_timesteps)
    train_state["last_saved"] = datetime.now().isoformat()
    train_state["config_snapshot"] = config.to_dict()
    with open(config.train_state_path, "w") as f:
        json.dump(train_state, f, indent=2)
    print(f"[model] saved -> {config.save_dir}/  ({model.num_timesteps:,} steps)")


def load_or_create(
    config, env: VecEnv
) -> Tuple[MaskablePPO, Dict[str, Any]]:
    """
    Return (model, train_state).

    Existing model.zip -> load weights + optimizer state, restore timestep counter.
    No model.zip      -> build fresh.
    Updated hyperparams in config (lr, clip_range, ent_coef, etc.) apply on resume.
    """
    model_zip = config.model_path + ".zip"

    if os.path.exists(model_zip):
        train_state = _load_train_state(config)
        model = MaskablePPO.load(
            config.model_path,
            env=env,
            device="auto",
            # updated hyperparams apply immediately on resume
            learning_rate=config.learning_rate,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            n_epochs=config.n_epochs,
            max_grad_norm=config.max_grad_norm,
        )
        model.num_timesteps = train_state["total_timesteps_done"]
        model._episode_num  = train_state.get("episodes_done", 0)
        print(
            f"[model] resumed - {train_state['total_timesteps_done']:,} steps done  "
            f"best win rate: {train_state['best_win_rate']:.3f}"
        )
    else:
        model = build_model(config, env)
        train_state = _default_train_state(config)
        print(f"[model] fresh model — {config.level_name} level")

    return model, train_state
