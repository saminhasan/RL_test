"""Checks for the reinforcement learning code (rl/).

Run:  python tests/test_rl.py      (or pytest)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from engine import MinesweeperEngine  # noqa: E402
from rl.minesweeper_env import MinesweeperEnv, RewardConfig, first_click, start_game  # noqa: E402
from rl.qnet import QNetwork, greedy, load_checkpoint, save_checkpoint  # noqa: E402


def test_env_passes_gymnasium_checker() -> None:
    from gymnasium.utils.env_checker import check_env

    check_env(MinesweeperEnv(level=2, seed=0), skip_render_check=True)


def test_first_click_is_automatic_and_the_same_per_board_seed() -> None:
    env = MinesweeperEnv(level=4, seed=0)
    for seed in range(20):
        obs, info = env.reset(options={"board_seed": seed})
        again = MinesweeperEngine(level=4, seed=0)
        start_game(again, seed)
        assert env.engine.started and env.engine.move_count == 1
        assert np.array_equal(env.engine.revealed, again.revealed)
        assert np.array_equal(info["action_mask"], (obs[0] == 0).ravel())
        assert obs[0].ravel()[first_click(info["board_seed"], 480)] != 0   # the clicked cell is revealed


def test_reward_adds_up_as_designed() -> None:
    cfg = RewardConfig()
    env = MinesweeperEnv(level=2, reward=cfg, seed=3)
    rng = np.random.default_rng(0)
    seen_win = seen_loss = False
    for _ in range(400):
        obs, info = env.reset()
        safe = env.engine.total_safe_cells
        start_revealed = env.engine.revealed_safe_cells
        total, clicks, done = 0.0, 0, False
        while not done:
            hidden = np.flatnonzero(info["action_mask"])
            safe_cells = hidden[~env.engine._mines_flat[hidden]]
            a = rng.choice(safe_cells if rng.random() < 0.97 else hidden)   # mostly safe, so some games are won
            obs, r, done, _, info = env.step(a)
            total += r
            clicks += 1
        progress = cfg.progress * (env.engine.revealed_safe_cells - start_revealed) / safe
        if info["won"]:
            seen_win = True
            assert np.isclose(total, cfg.win + progress - cfg.click * clicks / safe)
        else:
            seen_loss = True
            assert np.isclose(total, progress - cfg.mine - cfg.click * clicks / safe)
    assert seen_win and seen_loss


def test_reward_config_rejects_settings_that_reward_dying() -> None:
    try:
        RewardConfig(progress=0.1, click=0.2)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_qnet_only_picks_hidden_cells_and_checkpoints_round_trip() -> None:
    q = QNetwork({"channels": 8, "layers": 3})
    env = MinesweeperEnv(level=3, seed=1)
    obs = np.stack([env.reset()[0] for _ in range(6)])
    x = torch.from_numpy(obs)
    out = q(x)
    assert out.shape == (6, 256)
    picks = greedy(out, x).numpy()
    assert all(obs[i, 0].ravel()[a] == 0 for i, a in enumerate(picks))

    from rl.qnet import MODELS_DIR
    path = save_checkpoint(MODELS_DIR / "unit-test-qnet.pt", q, {"note": "unit test"})
    try:
        again, ckpt = load_checkpoint(path)
        assert ckpt["note"] == "unit test" and again.config == q.config
        assert torch.allclose(again(x), out)
    finally:
        path.unlink()


def test_short_training_run_completes_and_saves() -> None:
    import shutil

    from rl.dqn import RUNS_DIR, Args, train
    from rl.qnet import MODELS_DIR

    # also covers uniform replay with one-step returns, the plain Double DQN setting
    plain = Args(exp_name="unit-test-plain", curriculum="1", total_timesteps=1200, num_envs=8, channels=8, layers=2,
                 learning_starts=300, eval_every=100_000, eval_games=20, checkpoint_every=100_000, cuda=False,
                 n_step=1, per_alpha=0.0)
    _, ck = load_checkpoint(train(plain))
    for p in MODELS_DIR.glob(f"{ck['run_name']}*"):
        p.unlink()
    shutil.rmtree(RUNS_DIR / ck["run_name"], ignore_errors=True)

    args = Args(exp_name="unit-test", curriculum="1:1500,2", total_timesteps=3000, num_envs=8, channels=8,
                layers=2, learning_starts=500, eval_every=2000, eval_games=40, checkpoint_every=1000, cuda=False)
    latest = train(args)
    try:
        q, ckpt = load_checkpoint(latest)
        assert ckpt["global_step"] >= 3000 and ckpt["eval_history"] and ckpt["stage"] == 1
        assert ckpt["canvas"] == (9, 9)
    finally:
        for p in MODELS_DIR.glob(f"{ckpt['run_name']}*"):
            p.unlink()
        shutil.rmtree(RUNS_DIR / ckpt["run_name"], ignore_errors=True)


def test_small_board_on_a_bigger_canvas() -> None:
    env = MinesweeperEnv(level=2, seed=0, canvas=(16, 30))
    obs, info = env.reset()
    assert obs.shape == (2, 16, 30)
    assert (obs[0, 9:, :] == 10).all() and (obs[0, :, 9:] == 10).all()   # off the 9x9 board
    mask = info["action_mask"].reshape(16, 30)
    assert not mask[9:, :].any() and not mask[:, 9:].any()
    assert np.array_equal(mask[:9, :9], obs[0, :9, :9] == 0)
    obs2, r, term, trunc, _ = env.step(15 * 30 + 29)                     # an off-board click: a wasted click
    assert np.array_equal(obs2, obs) and r < 0 and not term

    env.set_level(4)                                                       # switches at the next game
    assert env.level == 2
    obs, info = env.reset()
    assert env.level == 4 and not (obs[0] == 10).any()


def test_n_step_returns() -> None:
    from rl.dqn import NStep

    ns = NStep(n=3, gamma=0.5, num_envs=1)
    assert ns.push(0, "s0", 0, 1.0, "s1", False, False) == []
    assert ns.push(0, "s1", 1, 2.0, "s2", False, False) == []
    out = ns.push(0, "s2", 2, 4.0, "s3", False, False)
    assert out == [("s0", 0, 1.0 + 0.5 * 2.0 + 0.25 * 4.0, "s3", 0.125)]
    out = ns.push(0, "s3", 3, 8.0, "s4", True, False)                     # the game ends: flush, no bootstrap
    assert out == [("s1", 1, 2.0 + 0.5 * 4.0 + 0.25 * 8.0, "s4", 0.0),
                   ("s2", 2, 4.0 + 0.5 * 8.0, "s4", 0.0),
                   ("s3", 3, 8.0, "s4", 0.0)]


def test_prioritized_replay_prefers_high_priority() -> None:
    from rl.per import PrioritizedReplayBuffer

    rb = PrioritizedReplayBuffer(64, alpha=1.0)
    for i in range(64):
        rb.add(np.zeros((2, 3, 3), np.uint8), i, 0.0, np.zeros((2, 3, 3), np.uint8), 0.99)
    rb.update_priorities(list(range(64)), [100.0 if i == 7 else 1.0 for i in range(64)])
    batch = rb.sample(256, beta=1.0)
    picks = np.array(batch["indexes"])
    assert (picks == 7).mean() > 0.4                                       # 100 / (100 + 63) of the mass
    assert batch["weights"][picks == 7].max() < batch["weights"][picks != 7].min()   # and down-weighted


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
