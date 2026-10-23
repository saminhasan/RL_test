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


def test_reward_shaping_cancels_out_over_a_game() -> None:
    """The discounted return is gamma^(clicks - 1) * (win * won + credit * progress * cleared)
    minus Phi of the first board: with credit 0, however much of the board a lost game uncovered."""
    for credit in (0.0, 0.5):
        cfg = RewardConfig(win=1.0, progress=1.0, credit=credit, gamma=0.99)
        env = MinesweeperEnv(level=2, reward=cfg, seed=3)
        rng = np.random.default_rng(0)
        seen_win = seen_late_loss = False
        for _ in range(300):
            obs, info = env.reset()
            safe = env.engine.total_safe_cells
            phi0 = env.engine.revealed_safe_cells / safe
            ret, clicks, done = 0.0, 0, False
            while not done:
                hidden = np.flatnonzero(info["action_mask"])
                safe_cells = hidden[~env.engine._mines_flat[hidden]]
                before = env.engine.revealed_safe_cells
                a = rng.choice(safe_cells if rng.random() < 0.97 else hidden)   # mostly safe, so some games are won
                obs, r, done, _, info = env.step(a)
                if not done:                                                   # a safe click pays for what it uncovers
                    gained = (env.engine.revealed_safe_cells - before) / safe
                    assert np.isclose(r, cfg.gamma * gained - (1 - cfg.gamma) * before / safe)
                ret += cfg.gamma ** clicks * r
                clicks += 1
            goal = float(info["won"]) + credit * info["revealed_fraction"]
            assert np.isclose(ret, cfg.gamma ** (clicks - 1) * goal - phi0)
            seen_win |= info["won"]
            seen_late_loss |= not info["won"] and info["revealed_fraction"] > 0.5
        assert seen_win and seen_late_loss


def test_mine_penalty_alone_scores_each_click() -> None:
    """mine 1, win 0, progress 0, gamma 0: every click earns -1 if it was a mine, else 0."""
    env = MinesweeperEnv(level=2, reward=RewardConfig(win=0.0, progress=0.0, mine=1.0, gamma=0.0), seed=5)
    rng = np.random.default_rng(1)
    for _ in range(50):
        obs, info = env.reset()
        done = False
        while not done:
            a = rng.choice(np.flatnonzero(info["action_mask"]))
            was_mine = bool(env.engine._mines_flat[a])
            obs, r, done, _, info = env.step(a)
            assert r == (-1.0 if was_mine else 0.0)


def test_reveal_reward_pays_for_uncovered_cells() -> None:
    env = MinesweeperEnv(level=2, reward=RewardConfig(win=0.0, progress=0.0, mine=10.0, reveal=1.0, gamma=0.0), seed=7)
    rng = np.random.default_rng(3)
    for _ in range(50):
        obs, info = env.reset()
        done = False
        while not done:
            a = rng.choice(np.flatnonzero(info["action_mask"]))
            was_mine, before = bool(env.engine._mines_flat[a]), env.engine.revealed_safe_cells
            obs, r, done, _, info = env.step(a)
            gained = (env.engine.revealed_safe_cells - before) / env.engine.total_safe_cells
            assert np.isclose(r, -10.0 if was_mine else gained) and (was_mine or gained > 0)


def test_late_reward_favours_the_end_of_the_game() -> None:
    cfg = RewardConfig(win=0.0, progress=0.0, mine=100.0, safe=-0.01, late=10.0, late_power=10.0, gamma=0.0)
    env = MinesweeperEnv(level=2, reward=cfg, seed=11)
    rng = np.random.default_rng(5)
    seen_win = False
    for _ in range(200):
        obs, info = env.reset()
        safe = env.engine.total_safe_cells
        g0, total, done = env.engine.revealed_safe_cells / safe, 0.0, False
        clicks = 0
        while not done:
            hidden = np.flatnonzero(info["action_mask"])
            safe_cells = hidden[~env.engine._mines_flat[hidden]]
            a = rng.choice(safe_cells if rng.random() < 0.98 else hidden)
            was_mine, before = bool(env.engine._mines_flat[a]), env.engine.revealed_safe_cells / safe
            obs, r, done, _, info = env.step(a)
            after = env.engine.revealed_safe_cells / safe
            assert np.isclose(r, -100.0 if was_mine else 10 * (after ** 10 - before ** 10) - 0.01)
            total += r
            clicks += 1
        if info["won"]:
            seen_win = True
            assert np.isclose(total, 10 * (1 - g0 ** 10) - 0.01 * clicks) and total <= 10
    assert seen_win


def test_burst_coin_flip_and_wall() -> None:
    from rl.minesweeper_env import is_coin_flip

    cfg = RewardConfig(win=0.0, progress=0.0, mine=100.0, burst=0.5, burst_rate=5.0, gamma=0.0)
    assert cfg.max_click() == 0.5
    env = MinesweeperEnv(level=2, reward=cfg, seed=13)
    rng = np.random.default_rng(2)
    for _ in range(30):
        obs, info = env.reset()
        done = False
        while not done:
            a = rng.choice(np.flatnonzero(info["action_mask"]))
            was_mine, before = bool(env.engine._mines_flat[a]), env.engine.revealed_safe_cells
            obs, r, done, _, info = env.step(a)
            k = env.engine.revealed_safe_cells - before
            want = -100.0 if was_mine else 0.5 * np.expm1(5 * k / env.engine.total_safe_cells) / np.expm1(5)
            assert np.isclose(r, want)
    nan = float("nan")
    assert is_coin_flip(np.array([[nan, 0.5], [0.5, nan]]))             # a 50/50
    assert is_coin_flip(np.array([[0.3, 0.3], [0.3, nan]]))             # all the same
    assert not is_coin_flip(np.array([[0.0, 0.5], [0.5, nan]]))         # a safe cell exists
    assert not is_coin_flip(np.array([[0.2, 0.5], [0.5, nan]]))         # a better guess exists
    # with skipping, a coin-flip click costs nothing and says so
    env = MinesweeperEnv(level=2, reward=RewardConfig(win=0.0, progress=0.0, mine=10.0, gamma=0.0), seed=1,
                         skip_coin_flips=True)
    seen = False
    for _ in range(300):
        obs, info = env.reset()
        done = False
        while not done:
            flip = is_coin_flip(env.engine.mine_probs)
            hidden = np.flatnonzero(info["action_mask"])
            safe_cells = hidden[~env.engine._mines_flat[hidden]]    # mostly safe clicks, so games reach the end
            obs, r, done, _, info = env.step(rng.choice(safe_cells if rng.random() < 0.97 else hidden))
            assert info["coin_flip"] == flip and (r == 0.0 if flip else True)
            seen |= flip
    assert seen
    from rl.dqn import Args, train
    try:
        train(Args(exp_name="unit-wall", curriculum="1", total_timesteps=64, num_envs=8, channels=8, layers=2,
                   reward_win=0, reward_progress=0, reward_mine=10, reward_late=1, wall=100, cuda=False))
    except ValueError as e:
        assert "--wall" in str(e)
    else:
        raise AssertionError("expected the wall rule to refuse")
    finally:
        import shutil

        from rl.dqn import RUNS_DIR
        for d in RUNS_DIR.glob("unit-wall-*"):
            shutil.rmtree(d, ignore_errors=True)


def test_reward_second_stage_switches_on() -> None:
    import shutil

    from rl import journal
    from rl.dqn import RUNS_DIR, Args, train
    from rl.qnet import MODELS_DIR

    a = Args(exp_name="unit-stage", curriculum="1", total_timesteps=1600, num_envs=8, channels=8, layers=2, loss="bce",
             reward_win=0, reward_progress=0, reward_mine=10, reward_reveal=1, reveal_from=800, stage2_mine=20, gamma=0,
             learning_starts=200, eval_every=100_000, eval_games=20, checkpoint_every=100_000, cuda=False, n_step=1,
             per_alpha=0.0)
    _, ck = load_checkpoint(train(a))
    b = Args(**{**vars(a), "exp_name": "unit-stage-b", "reveal_from": 0, "stage2_mine": 0.0, "reward_reveal": 0.0,
                "reward_safe": 0.01, "stage2_from": 800, "stage2": "mine=100,safe=-0.01,late=10,late_power=10",
                "skip_coin_flips": True, "lr_mode": "restart", "lr_floor": 1e-5, "lr_decay_clicks": 400})
    _, ckb = load_checkpoint(train(b))
    assert ckb["reward"]["mine"] == 100.0 and ckb["reward"]["late"] == 10.0 and ckb["reward"]["safe"] == -0.01
    for p in MODELS_DIR.glob(f"{ckb['run_name']}*"):
        p.unlink()
    shutil.rmtree(RUNS_DIR / ckb["run_name"], ignore_errors=True)
    try:
        assert ck["reward"]["reveal"] == 1.0 and ck["reward"]["mine"] == 20.0
        _, ck1 = load_checkpoint(MODELS_DIR / f"{ck['run_name']}-stage1.pt")
        assert ck1["reward"]["mine"] == 10.0 and ck1["reward"]["reveal"] == 0.0 and ck1["global_step"] >= 800
        assert any(e["text"].startswith("stage 2") for e in journal.read(RUNS_DIR / ck["run_name"])["events"])
    finally:
        for p in [*MODELS_DIR.glob(f"{ck['run_name']}*"), *MODELS_DIR.glob("bayes-L1-*.json")]:
            p.unlink()
        shutil.rmtree(RUNS_DIR / ck["run_name"], ignore_errors=True)


def test_wasted_click_changes_nothing_and_never_pays() -> None:
    env = MinesweeperEnv(level=2, seed=0)
    obs, info = env.reset()
    revealed = int(np.flatnonzero(~info["action_mask"])[0])
    obs2, r, term, trunc, _ = env.step(revealed)
    assert np.array_equal(obs2, obs) and r < 0 and not term


def test_reward_config_rejects_bad_settings() -> None:
    for bad in (dict(win=0.0), dict(progress=-1.0), dict(credit=1.5), dict(mine=-1.0), dict(gamma=-0.1), dict(gamma=1.5)):
        try:
            RewardConfig(**bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


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

        from rl import journal                                             # what the dashboard reads
        status = journal.read_status(RUNS_DIR / ckpt["run_name"])
        assert status["state"] == "stopped" and status["step"] == ckpt["global_step"]
        rows = journal.read(RUNS_DIR / ckpt["run_name"])
        assert len(rows["eval"]) == len(ckpt["eval_history"])
        assert [e["tag"] for e in rows["events"]] == ["session", "level", "stop"]
    finally:
        for p in [*MODELS_DIR.glob(f"{ckpt['run_name']}*"), *MODELS_DIR.glob("bayes-L1-*.json"), MODELS_DIR / "bayes-L2-40.json"]:
            p.unlink()
        shutil.rmtree(RUNS_DIR / ckpt["run_name"], ignore_errors=True)


def test_journal_keeps_the_newer_copy_of_replayed_steps() -> None:
    import tempfile
    from pathlib import Path

    from rl.journal import Journal, read

    with tempfile.TemporaryDirectory() as d:
        j = Journal(Path(d))
        j.event(0, "started", "session")
        for step in (10, 20, 30):
            j.log("train", step, win=0.1)
        j.log("eval", 25, win=0.2)
        j.event(20, "resumed", "session")                                  # a crash: back to the step-20 checkpoint
        j.log("train", 30, win=0.3)
        rows = read(Path(d))
        assert [(r["step"], r["win"]) for r in rows["train"]] == [(10, 0.1), (20, 0.1), (30, 0.3)]
        assert rows["eval"] == [] and [e["text"] for e in rows["events"]] == ["started", "resumed"]


def test_cross_entropy_and_pooling_train_and_resume() -> None:
    import shutil

    from rl.dqn import RUNS_DIR, Args, train
    from rl.qnet import MODELS_DIR

    a = Args(exp_name="unit-bce", curriculum="1", total_timesteps=1200, num_envs=8, channels=8, layers=5, pool_every=2,
             loss="bce", reward_win=0, reward_progress=0, reward_mine=1, gamma=0, learning_starts=300,
             eval_every=100_000, eval_games=20, checkpoint_every=100_000, cuda=False, n_step=1, per_alpha=0.0)
    first = train(a)
    q, ck = load_checkpoint(first)
    assert q.config["pool_every"] == 2 and any(".pool." in k for k in ck["state_dict"])
    # phase 2: a new run from it, playing for the win with look-ahead
    b = Args(**{**vars(a), "exp_name": "unit-bce-win", "init": str(first), "reward_win": 1.0, "reward_mine": 0.0,
                "gamma": 0.99, "pool_every": 0})
    second = train(b)
    q2, ck2 = load_checkpoint(second)
    assert q2.config["pool_every"] == 2 and ck2["global_step"] >= 1200
    for c in (ck, ck2):
        for p in [*MODELS_DIR.glob(f"{c['run_name']}*"), *MODELS_DIR.glob("bayes-L1-*.json")]:
            p.unlink()
        shutil.rmtree(RUNS_DIR / c["run_name"], ignore_errors=True)


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


def test_mirrored_boards_keep_the_clicked_cell() -> None:
    from rl.dqn import mirror

    for level in (3, 4):                                                   # square and not
        env = MinesweeperEnv(level=level, seed=2)
        obs = np.stack([env.reset()[0] for _ in range(16)])
        x = torch.from_numpy(obs)
        cols = obs.shape[-1]
        a = torch.tensor([int(np.flatnonzero(o[0].ravel() != 0)[0]) for o in obs])   # a revealed cell, to track
        for seed in range(12):
            mo, mn, ma = mirror(x, x, a, np.random.default_rng(seed))
            assert mo.shape[-2:] == x.shape[-2:]
            for i in range(16):                                            # the same clue sits under the moved cell
                assert mo[i, 0].flatten()[ma[i]] == x[i, 0].flatten()[a[i]]
            assert sorted(mo[:, 0].flatten().tolist()) == sorted(x[:, 0].flatten().tolist())


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
