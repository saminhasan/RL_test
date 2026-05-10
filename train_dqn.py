from __future__ import annotations

import json
import pprint
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine import MinesweeperEngine
from minesweeper_env import MinesweeperMaskEnv, level_name, level_to_id, make_env
from models import DQN_MODEL_ROOT, LEVEL_NETWORKS, SpatialPolicyValueNet, dqn_model_dir, dqn_tensorboard_dir, network_config_for
from torch.utils.tensorboard import SummaryWriter

@dataclass(frozen=True)
class DQNConfig:
    level: int | str = "test"
    seed: int = 42
    target_win_rate_pct: float = 69.0
    eval_games: int = 512
    eval_every_steps: int = 8192
    max_steps: int | None = None

    learning_rate: float = 1.0e-4
    gamma: float = 0.99
    batch_size: int = 256
    replay_size: int = 200_000
    learning_starts: int = 10_000
    train_every_steps: int = 4
    gradient_steps: int = 2
    target_update_steps: int = 10_000
    max_grad_norm: float = 1.0
    double_dqn: bool = True

    epsilon_start: float = 1.0
    epsilon_final: float = 0.10
    epsilon_decay_steps: int = 2_000_000

    n_envs: int = 8
    use_subproc_env: bool = False
    load_best_checkpoint: bool = True
    device: str = "auto"

    @property
    def level_id(self) -> int:
        return level_to_id(self.level)

    @property
    def level_name(self) -> str:
        return level_name(self.level)


TRAINING_CONFIG = DQNConfig(
    level="test",
    seed=42,
    target_win_rate_pct=72.0,
    eval_games=1024,
    eval_every_steps=8192,
    max_steps=None,

    learning_rate=1.0e-4,
    gamma=0.99,
    batch_size=256,
    replay_size=500_000,
    learning_starts=20_000,
    train_every_steps=8,
    gradient_steps=4,
    target_update_steps=10_000,
    max_grad_norm=1.0,

    epsilon_start=0.9,
    epsilon_final=0.09,
    epsilon_decay_steps=1_800_000,

    n_envs=4,
    use_subproc_env=True,
    load_best_checkpoint=True,
    device="auto",
)
ASK_FOR_LEVEL = True
ROOT = DQN_MODEL_ROOT


class QNet(nn.Module):
    def __init__(self, env: MinesweeperMaskEnv, level: int | str) -> None:
        super().__init__()
        self.net = SpatialPolicyValueNet(env.observation_space, network_config_for(level))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        q_values, _, _ = self.net(obs.float())
        return q_values


def dqn_dir(level: int | str) -> Path:
    return dqn_model_dir(level, ROOT)


def latest_path(level: int | str) -> Path:
    return dqn_dir(level) / "dqn_latest.pt"


def best_path(level: int | str) -> Path:
    return dqn_dir(level) / "dqn_best.pt"


def metadata_path(level: int | str) -> Path:
    return dqn_dir(level) / "metadata.json"


def save_checkpoint(path: Path, model: QNet, optimizer: torch.optim.Optimizer, config: DQNConfig, steps: int, best_win_rate: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "steps": steps,
            "best_win_rate": best_win_rate,
        },
        path,
    )
    metadata_path(config.level).write_text(
        json.dumps(
            {
                "config": asdict(config),
                "level_name": config.level_name,
                "steps": steps,
                "best_win_rate": best_win_rate,
                "network": asdict(network_config_for(config.level)),
            },
            indent=2,
        )
    )


def load_checkpoint(model: QNet, optimizer: torch.optim.Optimizer, config: DQNConfig, device: torch.device) -> tuple[int, float]:
    load_path = best_path(config.level) if config.load_best_checkpoint and best_path(config.level).exists() else latest_path(config.level)
    if not load_path.exists():
        return 0, 0.0
    try:
        ckpt = torch.load(load_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        print(f"loaded DQN checkpoint: {load_path}")
        return int(ckpt.get("steps", 0)), float(ckpt.get("best_win_rate", 0.0))
    except Exception as exc:
        print(f"checkpoint load failed ({exc}); starting fresh")
        return 0, 0.0


class ReplayBuffer:
    def __init__(self, size: int, obs_shape: tuple[int, ...], action_dim: int) -> None:
        self.size = int(size)
        self.i = 0
        self.full = False
        self.obs = np.empty((size, *obs_shape), dtype=np.float32)
        self.next_obs = np.empty((size, *obs_shape), dtype=np.float32)
        self.actions = np.empty(size, dtype=np.int64)
        self.rewards = np.empty(size, dtype=np.float32)
        self.dones = np.empty(size, dtype=np.float32)
        self.next_masks = np.empty((size, action_dim), dtype=bool)

    def __len__(self) -> int:
        return self.size if self.full else self.i

    def add(self, obs, action: int, reward: float, next_obs, done: bool, next_mask) -> None:
        self.obs[self.i] = obs
        self.next_obs[self.i] = next_obs
        self.actions[self.i] = int(action)
        self.rewards[self.i] = float(reward)
        self.dones[self.i] = float(done)
        self.next_masks[self.i] = next_mask
        self.i += 1
        if self.i == self.size:
            self.i = 0
            self.full = True

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        idx = rng.integers(len(self), size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
            self.next_masks[idx],
        )


def device_for(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def epsilon_at(step: int, config: DQNConfig) -> float:
    t = min(1.0, step / max(config.epsilon_decay_steps, 1))
    return config.epsilon_start + t * (config.epsilon_final - config.epsilon_start)


def masked_argmax(q_values: np.ndarray, mask: np.ndarray) -> int:
    q = q_values.copy()
    q[~mask] = -np.inf
    return int(q.argmax())


def choose_action(model: QNet, obs: np.ndarray, mask: np.ndarray, epsilon: float, rng: np.random.Generator, device: torch.device) -> int:
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return 0
    if rng.random() < epsilon:
        return int(legal[int(rng.integers(legal.size))])
    with torch.no_grad():
        obs_t = torch.as_tensor(obs[None], device=device)
        q = model(obs_t).detach().cpu().numpy()[0]
    return masked_argmax(q, mask)


def train_step(
    model: QNet,
    target_model: QNet,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    config: DQNConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    obs, actions, rewards, next_obs, dones, next_masks = replay.sample(config.batch_size, rng)

    obs_t = torch.as_tensor(obs, device=device)
    actions_t = torch.as_tensor(actions, device=device).view(-1, 1)
    rewards_t = torch.as_tensor(rewards, device=device)
    next_obs_t = torch.as_tensor(next_obs, device=device)
    dones_t = torch.as_tensor(dones, device=device)
    next_masks_t = torch.as_tensor(next_masks, device=device)

    q = model(obs_t).gather(1, actions_t).squeeze(1)

    with torch.no_grad():
        if config.double_dqn:
            next_q_online = model(next_obs_t).masked_fill(~next_masks_t, -1.0e9)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q = target_model(next_obs_t).gather(1, next_actions).squeeze(1)
        else:
            next_q = target_model(next_obs_t)
            next_q = next_q.masked_fill(~next_masks_t, -1.0e9).max(dim=1).values

        target = rewards_t + config.gamma * (1.0 - dones_t) * next_q

    loss = F.smooth_l1_loss(q, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()
    return float(loss.item())


def evaluate_model(model: QNet, config: DQNConfig, games: int, device: torch.device) -> dict[str, float]:
    model.eval()
    wins = losses = moves = won_moves = lost_moves = 0
    returns = 0.0

    for game in range(games):
        env = make_env(config, seed_offset=50_000_000 + game)
        obs, _ = env.reset()
        done = False
        episode_return = 0.0

        while not done:
            mask = env.action_masks().copy()
            with torch.no_grad():
                q = model(torch.as_tensor(obs[None], device=device)).detach().cpu().numpy()[0]
            action = masked_argmax(q, mask)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            done = terminated or truncated

        won = bool(info["won"])
        episode_moves = int(info["moves"])
        wins += int(won)
        losses += int(not won)
        moves += episode_moves
        won_moves += episode_moves if won else 0
        lost_moves += episode_moves if not won else 0
        returns += episode_return

    model.train()
    return {
        "win_rate": wins / games if games else 0.0,
        "loss_rate": losses / games if games else 0.0,
        "avg_moves": moves / games if games else 0.0,
        "avg_won_moves": won_moves / wins if wins else 0.0,
        "avg_lost_moves": lost_moves / losses if losses else 0.0,
        "wins": float(wins),
        "losses": float(losses),
        "avg_return": returns / games if games else 0.0,
    }


def pick_level(default_level: int | str) -> str:
    choices = ", ".join(f"{idx}:{name}" for idx, name in MinesweeperEngine.LEVELS.items())
    print(f"available levels: {choices}")
    answer = input(f"select level [{level_name(default_level)}]: ").strip()
    if not answer:
        return level_name(default_level)
    return level_name(level_to_id(answer))


def build_config() -> DQNConfig:
    level = pick_level(TRAINING_CONFIG.level) if ASK_FOR_LEVEL else level_name(TRAINING_CONFIG.level)
    return DQNConfig(**{**asdict(TRAINING_CONFIG), "level": level})


def print_training_setup(config: DQNConfig, env: MinesweeperMaskEnv, model: QNet) -> None:
    net = LEVEL_NETWORKS[config.level_name]
    total = sum(p.numel() for p in model.parameters())
    print("DQN setup:")
    print(f"  level              : {config.level_name}")
    print(f"  board              : {env.engine.rows}x{env.engine.cols}")
    print(f"  input shape         : {env.observation_space.shape}")
    print(f"  action count        : {env.action_space.n}")
    print(f"  model dir           : {dqn_dir(config.level)}")
    print(f"  tensorboard         : {dqn_tensorboard_dir(config.level, ROOT)}")
    print(f"  architecture        : {net.name}")
    print(f"  trunk channels      : {net.channels}")
    print(f"  residual blocks     : {net.residual_blocks}")
    print(f"  global context      : {net.global_context}")
    print(f"  params              : {total:,}")
    print(f"  replay size         : {config.replay_size:,}")
    print(f"  double dqn          : {config.double_dqn}")
    print(f"  n_envs              : {config.n_envs}")
    print(f"  subproc envs        : False (manual env list)")
    print(f"  device              : {config.device}")


def make_writer(config: DQNConfig) -> SummaryWriter:
    run_dir = dqn_tensorboard_dir(config.level, ROOT) / "train"
    run_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(run_dir))

def main(config: DQNConfig | None = None) -> None:
    config = build_config() if config is None else config
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    device = device_for(config.device)
    probe_env = make_env(config)
    envs = [make_env(config, seed_offset=i * 1_000_000) for i in range(config.n_envs)]
    obs = []
    masks = []
    for env in envs:
        o, _ = env.reset()
        obs.append(o.copy())
        masks.append(env.action_masks().copy())

    model = QNet(probe_env, config.level).to(device)
    target_model = QNet(probe_env, config.level).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    steps, best_win_rate = load_checkpoint(model, optimizer, config, device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    replay = ReplayBuffer(config.replay_size, probe_env.observation_space.shape, probe_env.action_space.n)
    print_training_setup(config, probe_env, model)
    print(f"Starting DQN training with config:\n{pprint.pformat(asdict(config))}")
    writer = make_writer(config)
    writer.add_text("config", json.dumps(asdict(config), indent=2), 0)
    target = config.target_win_rate_pct / 100.0
    last_loss = 0.0
    model.train()

    try:
        while config.max_steps is None or steps < config.max_steps:
            epsilon = epsilon_at(steps, config)

            for env_i, env in enumerate(envs):
                action = choose_action(model, obs[env_i], masks[env_i], epsilon, rng, device)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_mask = env.action_masks().copy()

                replay.add(obs[env_i], action, reward, next_obs, done, next_mask)
                steps += 1

                if done:
                    next_obs, _ = env.reset()
                    next_mask = env.action_masks().copy()

                obs[env_i] = next_obs.copy()
                masks[env_i] = next_mask

                if len(replay) >= config.learning_starts and steps % config.train_every_steps == 0:
                    for _ in range(config.gradient_steps):
                        last_loss = train_step(model, target_model, optimizer, replay, config, rng, device)

                    writer.add_scalar("train/loss", last_loss, steps)
                    writer.add_scalar("train/epsilon", epsilon, steps)
                    writer.add_scalar("train/replay_size", len(replay), steps)

                if steps % config.target_update_steps == 0:
                    target_model.load_state_dict(model.state_dict())

                if steps % config.eval_every_steps == 0:
                    stats = evaluate_model(model, config, config.eval_games, device)
                    win_rate = stats["win_rate"]

                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                        save_checkpoint(best_path(config.level), model, optimizer, config, steps, best_win_rate)

                    writer.add_scalar("eval/win_rate_pct", win_rate * 100.0, steps)
                    writer.add_scalar("eval/loss_rate_pct", stats["loss_rate"] * 100.0, steps)
                    writer.add_scalar("eval/best_win_rate_pct", best_win_rate * 100.0, steps)
                    writer.add_scalar("eval/avg_moves", stats["avg_moves"], steps)
                    writer.add_scalar("eval/avg_won_moves", stats["avg_won_moves"], steps)
                    writer.add_scalar("eval/avg_lost_moves", stats["avg_lost_moves"], steps)
                    writer.add_scalar("eval/avg_return", stats["avg_return"], steps)
                    writer.add_scalar("train/epsilon", epsilon, steps)
                    writer.add_scalar("train/replay_size", len(replay), steps)
                    writer.flush()

                    save_checkpoint(latest_path(config.level), model, optimizer, config, steps, best_win_rate)
                    print(
                        f"steps={steps:09d} | eps={epsilon:.3f} loss={last_loss:.4f} "
                        f"win={win_rate * 100.0:.2f}% loss_rate={stats['loss_rate'] * 100.0:.2f}% "
                        f"best={best_win_rate * 100.0:.2f}% avg_moves={stats['avg_moves']:.2f} "
                        f"won_moves={stats['avg_won_moves']:.2f} lost_moves={stats['avg_lost_moves']:.2f} "
                        f"return={stats['avg_return']:.3f} replay={len(replay):d}"
                    )
                    if win_rate >= target:
                        print(f"target reached: {win_rate * 100.0:.2f}% >= {config.target_win_rate_pct:.2f}%")
                        writer.flush()
                        return
    except KeyboardInterrupt:
        print("\nCtrl+C received. Saving latest DQN checkpoint before exit...")
    finally:
        save_checkpoint(latest_path(config.level), model, optimizer, config, steps, best_win_rate)
        writer.flush()
        writer.close()
        print(f"saved latest checkpoint: {latest_path(config.level)}")
        for env in envs:
            env.close()
        probe_env.close()

if __name__ == "__main__":
    main()
