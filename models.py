from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnv

from engine import MinesweeperEngine
from minesweeper_env import BOARD_CHANNELS, level_name, level_to_id


RL_MODEL_ROOT = Path("models") / "rl"
PPO_MODEL_ROOT = RL_MODEL_ROOT / "ppo"
DQN_MODEL_ROOT = RL_MODEL_ROOT / "dqn"


@dataclass(frozen=True)
class LevelNetworkConfig:
    name: str
    channels: int
    plain_conv_layers: int
    block_dilations: tuple[int, ...]
    global_context: bool = False

    @property
    def base_channels(self) -> int:
        return self.channels

    @property
    def conv_layers(self) -> int:
        return self.plain_conv_layers + 2 * len(self.block_dilations)

    @property
    def residual_blocks(self) -> int:
        return len(self.block_dilations)

    @property
    def body_out_channels(self) -> int:
        return self.channels * (2 if self.global_context else 1)

    @property
    def policy_layers(self) -> tuple[int, ...]:
        return ()

    @property
    def value_layers(self) -> tuple[int, ...]:
        return ()


@dataclass(frozen=True)
class MaskablePPOConfig:
    level: int | str = "test"
    seed: int = 42
    target_win_rate_pct: float = 68.0
    eval_games: int = 512
    eval_every_timesteps: int = 8192
    max_timesteps: int | None = None
    learning_rate: float = 1.0e-3
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.03
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_envs: int = 1
    use_subproc_env: bool = True
    freeze_features_extractor: bool = False
    load_best_checkpoint: bool = True
    rollback_drop_pct: float = 10.0
    rollback_patience: int = 3
    device: str = "auto"

    @property
    def level_id(self) -> int:
        return level_to_id(self.level)

    @property
    def level_name(self) -> str:
        return level_name(self.level)


LEVEL_NETWORKS: dict[str, LevelNetworkConfig] = {
    "test": LevelNetworkConfig(
        name="tiny_cnn",
        channels=32,
        plain_conv_layers=3,
        block_dilations=(),
        global_context=False,
    ),
    "easy": LevelNetworkConfig(
        name="small_rescnn",
        channels=64,
        plain_conv_layers=1,
        block_dilations=(1, 1, 1),
        global_context=False,
    ),
    "medium": LevelNetworkConfig(
        name="dilated_rescnn_global",
        channels=96,
        plain_conv_layers=1,
        block_dilations=(1, 1, 2, 2, 4, 1, 1),
        global_context=True,
    ),
    "hard": LevelNetworkConfig(
        name="rescnn_global_context",
        channels=96,
        plain_conv_layers=1,
        block_dilations=(1, 1, 2, 4, 1, 1),
        global_context=True,
    ),
}


def network_config_for(level: int | str) -> LevelNetworkConfig:
    return LEVEL_NETWORKS[level_name(level)]


def _norm_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_norm_groups(out_channels), out_channels),
            nn.SiLU(),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(_norm_groups(channels), channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(_norm_groups(channels), channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class SpatialPolicyValueNet(nn.Module):
    def __init__(self, observation_space: spaces.Box, config: LevelNetworkConfig) -> None:
        super().__init__()
        in_channels, rows, cols = observation_space.shape
        self.rows = int(rows)
        self.cols = int(cols)
        self.action_dim = self.rows * self.cols
        self.config = config

        trunk: list[nn.Module] = []
        current_channels = int(in_channels)
        for _ in range(config.plain_conv_layers):
            trunk.append(ConvNormAct(current_channels, config.channels))
            current_channels = config.channels

        for dilation in config.block_dilations:
            trunk.append(ResidualBlock(config.channels, dilation=dilation))

        self.trunk = nn.Sequential(*trunk)
        if config.global_context:
            self.global_context = nn.AdaptiveAvgPool2d(1)
        else:
            self.global_context = None

        head_channels = config.body_out_channels
        self.policy_head = nn.Conv2d(head_channels, 1, kernel_size=1)
        self.risk_head = nn.Conv2d(head_channels, 1, kernel_size=1)
        self.value_head = nn.Conv2d(head_channels, 1, kernel_size=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def trunk_parameters(self):
        yield from self.trunk.parameters()
        if self.global_context is not None:
            yield from self.global_context.parameters()

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.trunk(obs.float())
        if self.global_context is None:
            return x

        context = self.global_context(x).expand(-1, -1, x.shape[2], x.shape[3])
        return torch.cat((x, context), dim=1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features(obs)
        policy_logits = self.policy_head(features).flatten(1)
        risk_logits = self.risk_head(features).flatten(1)
        values = self.value_head(features).mean(dim=(2, 3))
        return policy_logits, values, risk_logits


class MinesweeperMaskablePolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        network_config: LevelNetworkConfig,
        *args,
        **kwargs,
    ) -> None:
        self.network_config = network_config
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            net_arch=[],
            activation_fn=nn.SiLU,
            ortho_init=False,
            **kwargs,
        )

    def _build(self, lr_schedule) -> None:
        assert isinstance(self.observation_space, spaces.Box)
        self.policy_value_net = SpatialPolicyValueNet(self.observation_space, self.network_config)
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def forward_heads(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.policy_value_net(obs)

    def action_logits(self, obs: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward_heads(obs)
        return logits

    def risk_logits(self, obs: torch.Tensor) -> torch.Tensor:
        _, _, risk_logits = self.forward_heads(obs)
        return risk_logits

    def _distribution_from_logits(self, logits: torch.Tensor, action_masks=None):
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values, _ = self.forward_heads(obs)
        distribution = self._distribution_from_logits(logits, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions.reshape((-1, *self.action_space.shape)), values, log_prob

    def _predict(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> torch.Tensor:
        return self.get_distribution(observation, action_masks).get_actions(deterministic=deterministic)

    def get_distribution(self, obs: torch.Tensor, action_masks: np.ndarray | None = None):
        logits = self.action_logits(obs)
        return self._distribution_from_logits(logits, action_masks)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        logits, values, _ = self.forward_heads(obs)
        distribution = self._distribution_from_logits(logits, action_masks)
        log_prob = distribution.log_prob(actions.long())
        return values, log_prob, distribution.entropy()

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        _, values, _ = self.forward_heads(obs)
        return values


def model_dir(level: int | str, root: Path | str = PPO_MODEL_ROOT) -> Path:
    return Path(root) / level_name(level)


def latest_model_path(level: int | str, root: Path | str = PPO_MODEL_ROOT) -> Path:
    return model_dir(level, root) / "maskable_ppo.zip"


def best_model_path(level: int | str, root: Path | str = PPO_MODEL_ROOT) -> Path:
    return model_dir(level, root) / "maskable_ppo_best.zip"


def metadata_path(level: int | str, root: Path | str = PPO_MODEL_ROOT) -> Path:
    return model_dir(level, root) / "maskable_ppo.json"


def ppo_tensorboard_dir(level: int | str, root: Path | str = PPO_MODEL_ROOT) -> Path:
    return model_dir(level, root) / "tensorboard"


def dqn_model_dir(level: int | str, root: Path | str = DQN_MODEL_ROOT) -> Path:
    return Path(root) / level_name(level)


def dqn_tensorboard_dir(level: int | str, root: Path | str = DQN_MODEL_ROOT) -> Path:
    return dqn_model_dir(level, root) / "tensorboard"


def policy_kwargs_for(level: int | str) -> dict[str, Any]:
    return {
        "network_config": network_config_for(level),
        "normalize_images": False,
    }


def save_metadata(
    config: MaskablePPOConfig,
    timesteps: int,
    best_win_rate: float,
    root: Path | str = PPO_MODEL_ROOT,
) -> None:
    path = metadata_path(config.level, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "level_name": config.level_name,
                "timesteps": timesteps,
                "best_win_rate": best_win_rate,
                "network": asdict(network_config_for(config.level)),
                "tensorboard": str(ppo_tensorboard_dir(config.level, root)),
            },
            indent=2,
        )
    )


def load_metadata(config: MaskablePPOConfig, root: Path | str = PPO_MODEL_ROOT) -> tuple[int, float]:
    path = metadata_path(config.level, root)
    if not path.exists():
        return 0, 0.0
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return 0, 0.0
    return int(data.get("timesteps", 0)), float(data.get("best_win_rate", 0.0))


def create_maskable_ppo(
    config: MaskablePPOConfig,
    env: VecEnv,
    root: Path | str = PPO_MODEL_ROOT,
) -> MaskablePPO:
    tb_dir = ppo_tensorboard_dir(config.level, root)
    tb_dir.mkdir(parents=True, exist_ok=True)
    return MaskablePPO(
        MinesweeperMaskablePolicy,
        env,
        policy_kwargs=policy_kwargs_for(config.level),
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
        seed=config.seed,
        verbose=0,
        device=config.device,
        tensorboard_log=str(tb_dir),
    )


def apply_training_config(model: MaskablePPO, config: MaskablePPOConfig) -> None:
    model.learning_rate = config.learning_rate
    model.lr_schedule = lambda _: config.learning_rate
    model.n_steps = config.n_steps
    model.batch_size = config.batch_size
    model.n_epochs = config.n_epochs
    model.gamma = config.gamma
    model.gae_lambda = config.gae_lambda
    model.clip_range = lambda _: config.clip_range
    model.ent_coef = config.ent_coef
    model.vf_coef = config.vf_coef
    model.max_grad_norm = config.max_grad_norm
    for group in model.policy.optimizer.param_groups:
        group["lr"] = config.learning_rate


def apply_freeze_config(model: MaskablePPO, config: MaskablePPOConfig) -> None:
    network = getattr(model.policy, "policy_value_net", None)
    if network is not None and hasattr(network, "trunk_parameters"):
        params = network.trunk_parameters()
    else:
        params = model.policy.features_extractor.parameters()

    for param in params:
        param.requires_grad_(not config.freeze_features_extractor)


def rebuild_model_from_checkpoint(
    checkpoint_path: Path,
    config: MaskablePPOConfig,
    env: VecEnv,
    timesteps: int,
    root: Path | str = PPO_MODEL_ROOT,
) -> MaskablePPO:
    loaded_model = MaskablePPO.load(checkpoint_path, env=env, device=config.device)
    model = create_maskable_ppo(config, env, root)
    model.policy.load_state_dict(loaded_model.policy.state_dict())
    model.num_timesteps = timesteps
    apply_training_config(model, config)
    apply_freeze_config(model, config)
    return model


def load_or_create_maskable_ppo(
    config: MaskablePPOConfig,
    env: VecEnv,
    root: Path | str = PPO_MODEL_ROOT,
) -> tuple[MaskablePPO, int, float]:
    latest_path = latest_model_path(config.level, root)
    best_path = best_model_path(config.level, root)
    load_path = best_path if config.load_best_checkpoint and best_path.exists() else latest_path
    timesteps, best_win_rate = load_metadata(config, root)

    if load_path.exists():
        try:
            model = rebuild_model_from_checkpoint(load_path, config, env, timesteps, root)
            print(f"loaded MaskablePPO policy weights: {load_path}")
            print("rebuilt rollout buffer with current config")
            return model, timesteps, best_win_rate
        except Exception as exc:
            print(f"checkpoint architecture mismatch ({exc}); starting fresh")
            timesteps = 0
            best_win_rate = 0.0

    model = create_maskable_ppo(config, env, root)
    apply_freeze_config(model, config)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(latest_path)
    save_metadata(config, timesteps=0, best_win_rate=0.0, root=root)
    print(f"created MaskablePPO checkpoint: {latest_path}")
    return model, 0, 0.0


def count_policy_parameters(model: MaskablePPO) -> tuple[int, int]:
    total = sum(param.numel() for param in model.policy.parameters())
    trainable = sum(param.numel() for param in model.policy.parameters() if param.requires_grad)
    return total, trainable


def print_model_architecture(level: int | str) -> None:
    rows, cols, mines = MinesweeperEngine.DIFFICULTIES[level_name(level)]
    net = network_config_for(level)
    total_params = sum(
        p.numel()
        for p in SpatialPolicyValueNet(
            spaces.Box(low=0.0, high=1.0, shape=(BOARD_CHANNELS, rows, cols), dtype=np.float32),
            net,
        ).parameters()
    )
    print("MaskablePPO model:")
    print(f"  level              : {level_name(level)}")
    print(f"  board              : {rows}x{cols}, mines={mines}")
    print(f"  architecture       : {net.name}")
    print(f"  input channels     : {BOARD_CHANNELS}")
    print(f"  action logits      : {rows * cols} ({rows}x{cols})")
    print(f"  trunk channels     : {net.channels}")
    print(f"  plain conv layers  : {net.plain_conv_layers}")
    print(f"  residual blocks    : {net.residual_blocks}")
    print(f"  block dilations    : {net.block_dilations or 'none'}")
    print(f"  global context     : {net.global_context} (avg pool broadcast, no FC)")
    print(f"  policy head        : conv 1x1 -> 1")
    print(f"  risk head          : conv 1x1 -> 1")
    print(f"  value head         : conv 1x1 -> spatial mean -> scalar")
    print(f"  model params       : {total_params:,}")


# Backward-friendly aliases for older scripts/snippets.
MinesweeperCNN = SpatialPolicyValueNet
create_model = create_maskable_ppo
load_or_create_model = load_or_create_maskable_ppo
