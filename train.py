"""
Training script for Minesweeper PPO (MaskablePPO).

Edit config.py to change hyperparams / level, then run:
    python train.py

Ctrl-C at any time -> saves checkpoint and exits cleanly.
Training resumes automatically if a checkpoint exists for the configured level.
"""
from __future__ import annotations

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from config import cfg
from minesweeper_env import make_eval_env, make_train_env
from model import load_or_create, save_checkpoint


# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------

def evaluate(
    model: MaskablePPO, eval_env: VecEnv, n_episodes: int
) -> tuple[float, float, float]:
    """
    Run n_episodes deterministically. Returns (win_rate, mean_reward, mean_ep_len).
    """
    obs = eval_env.reset()
    n = eval_env.num_envs
    ep_rewards = np.zeros(n, dtype=np.float64)
    ep_lengths = np.zeros(n, dtype=np.int32)

    wins = 0
    total_reward = 0.0
    total_length = 0
    completed = 0

    while completed < n_episodes:
        masks = get_action_masks(eval_env)
        actions, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(actions)

        ep_rewards += rewards
        ep_lengths += 1

        for i, done in enumerate(dones):
            if done and completed < n_episodes:
                wins         += int(infos[i].get("won", False))
                total_reward += float(ep_rewards[i])
                total_length += int(ep_lengths[i])
                ep_rewards[i] = 0.0
                ep_lengths[i] = 0
                completed    += 1

    return wins / n_episodes, total_reward / n_episodes, total_length / n_episodes


# ------------------------------------------------------------------------------
# Callback
# ------------------------------------------------------------------------------

class WinRateCallback(BaseCallback):
    """
    Evaluates win rate every eval_freq steps.
    Logs to TensorBoard. Saves best_model.zip on improvement.
    Tracks metrics history in train_state for resume continuity.
    """

    def __init__(self, eval_env: VecEnv, config, train_state: dict, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.eval_env    = eval_env
        self.config      = config
        self.train_state = train_state
        self.best_win_rate = float(train_state.get("best_win_rate", -1.0))

    def _on_step(self) -> bool:
        if self.num_timesteps % self.config.eval_freq < self.training_env.num_envs:
            win_rate, mean_reward, mean_ep_len = evaluate(
                self.model, self.eval_env, self.config.eval_episodes
            )

            self.logger.record("eval/win_rate",       win_rate)
            self.logger.record("eval/mean_reward",    mean_reward)
            self.logger.record("eval/mean_ep_length", mean_ep_len)
            self.logger.dump(self.num_timesteps)

            m = self.train_state["recent_metrics"]
            m["timesteps"].append(self.num_timesteps)
            m["win_rates"].append(win_rate)
            m["mean_rewards"].append(mean_reward)
            m["mean_ep_lengths"].append(mean_ep_len)

            if self.verbose:
                print(
                    f"  step {self.num_timesteps:>10,}  "
                    f"win={win_rate:.3f}  rew={mean_reward:.3f}  ep_len={mean_ep_len:.1f}"
                )

            if win_rate > self.best_win_rate:
                self.best_win_rate = win_rate
                self.train_state["best_win_rate"]       = win_rate
                self.train_state["best_model_timestep"] = self.num_timesteps
                self.model.save(self.config.best_model_path)
                if self.verbose:
                    print(f"  ^ new best model saved  win_rate={win_rate:.3f}")

        return True

    def _on_training_end(self) -> None:
        self.train_state["episodes_done"] = int(self.model._episode_num)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main() -> None:
    train_env = make_train_env(cfg)
    eval_env  = make_eval_env(cfg, n_envs=min(cfg.num_envs, 8))

    model, train_state = load_or_create(cfg, train_env)

    remaining = cfg.total_timesteps - model.num_timesteps
    if remaining <= 0:
        print(f"[train] already at {model.num_timesteps:,} / {cfg.total_timesteps:,} steps. done.")
        train_env.close()
        eval_env.close()
        return

    callback = WinRateCallback(eval_env, cfg, train_state, verbose=1)

    print(
        f"[train] {cfg.level_name} | backend={cfg.vec_backend} | "
        f"envs={cfg.num_envs} | remaining={remaining:,} steps"
    )

    try:
        model.learn(
            total_timesteps=remaining,
            reset_num_timesteps=False,  # keeps internal counters for correct resume
            callback=callback,
            tb_log_name="ppo",
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[train] interrupted by user")
    finally:
        save_checkpoint(model, cfg, train_state)
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
    from eval import main as eval_main
    eval_main()  # run final evaluation after training completes
