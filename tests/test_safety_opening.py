"""Selected-outcome and CPU integration checks for the phase-two prototype."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from rl.safety_opening import (KIND, SafetyOpening, TrainArgs, final_observations,
                               initialize, load_model, objective, publish_checked_result,
                               selected_labels, train)


def synthetic_batch():
    before = torch.zeros((3, 2, 3, 3), dtype=torch.uint8)
    before[:, 1] = 2
    after = before.clone()
    after[1, 0].flatten()[:3] = torch.tensor([1, 2, 3], dtype=torch.uint8)
    after[2, 0].flatten()[8] = 2
    return before, after, torch.tensor([0, 0, 8])


def test_selected_labels_distinguish_mine_zero_number_and_flood():
    before, after, chosen = synthetic_batch()
    safe, zero, reveal = selected_labels(before, after, chosen)
    assert torch.equal(safe, torch.tensor([0., 1., 1.]))
    assert torch.equal(zero, torch.tensor([0., 1., 0.]))
    assert torch.allclose(reveal, torch.tensor([0., 3 / 7, 1 / 7]))
    before[0, 0, 0, 0] = 2
    try:
        selected_labels(before, after, chosen)
    except ValueError:
        pass
    else:
        raise AssertionError("a previously visible action must be rejected")


def test_warm_start_preserves_binary_safety_and_has_fp32_heads():
    from rl.qnet import QNetwork

    torch.manual_seed(6)
    base = QNetwork({"channels": 4, "layers": 2})
    before, _, _ = synthetic_batch()
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "risk.pt"
        torch.save({"config": base.config, "state_dict": base.state_dict(),
                    "args": {"loss": "bce", "gamma": 0},
                    "reward": {"mine": 1, "win": 0, "progress": 0, "reveal": 0}}, path)
        model = initialize(str(path), {"opening_lambda": 0., "opening_kind": "zero"}, "cpu")
        assert torch.allclose(model(before), F.logsigmoid(base(before)))
        assert torch.allclose(model.heads(before)["reveal"].sigmoid(), torch.full((3, 9), 0.03))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert all(value.dtype == torch.float32 for value in model.heads(before).values())
        torch.save({"kind": KIND, "config": model.config, "state_dict": model.state_dict()}, path)
        reloaded, _ = load_model(path)
        assert torch.allclose(reloaded(before), model(before))
        invalid = {"config": base.config, "state_dict": base.state_dict(),
                   "args": {"loss": "bce", "gamma": 0, "skip_coin_flips": True},
                   "reward": {"mine": 1, "win": 0, "progress": 0, "reveal": 0}}
        torch.save(invalid, path)
        try:
            initialize(str(path), {"opening_lambda": 0., "opening_kind": "zero"}, "cpu")
        except ValueError:
            pass
        else:
            raise AssertionError("solver-filtered initialization must be rejected")


def test_auxiliary_labels_only_train_on_selected_safe_clicks():
    model = SafetyOpening({"channels": 4, "layers": 2})
    before, after, chosen = synthetic_batch()
    loss, pieces = objective(model, before, after, chosen, aux="both")
    assert torch.isfinite(loss) and pieces["zero"] > 0 and pieces["reveal"] > 0
    loss.backward()
    assert model.zero_head.weight.grad is not None
    model.zero_grad(set_to_none=True)
    loss, pieces = objective(model, before[:1], after[:1], chosen[:1], aux="both")
    assert pieces["zero"] == 0 and pieces["reveal"] == 0
    loss.backward()
    assert model.zero_head.weight.grad is None
    assert model.q.head.weight.grad is not None


def test_real_terminal_labels_use_final_observation_before_auto_reset():
    import gymnasium as gym
    from rl.minesweeper_env import MinesweeperEnv

    envs = gym.vector.SyncVectorEnv([lambda: MinesweeperEnv(level=1, seed=3)],
                                   autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    try:
        before, _ = envs.reset()
        engine = envs.envs[0].engine
        # Hidden truth is used here only to force the terminal integration case.
        chosen = np.array([np.flatnonzero(engine._mines_flat & ~engine._revealed_flat)[0]])
        after, _, ended, truncated, infos = envs.step(chosen)
        assert ended[0]
        real = final_observations(after, ended, truncated, infos)
        safe, zero, reveal = selected_labels(torch.from_numpy(before), torch.from_numpy(real),
                                             torch.from_numpy(chosen))
        assert safe[0] == zero[0] == reveal[0] == 0
        assert real[0, 0].ravel()[chosen[0]] == 0
    finally:
        envs.close()


def test_short_cpu_training_saves_and_resume_round_trips():
    import json
    import subprocess

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "trial"
        args = TrainArgs(output_dir=output, seed=201, level=1, total_timesteps=64, num_envs=4,
                         channels=4, layers=2, batch_size=8, buffer_size=128,
                         learning_starts=8, train_frequency=4, checkpoint_every=32,
                         eval_games=0, device="cpu", bf16=False, aux="both")
        path = train(args)
        model, checkpoint = load_model(path)
        assert checkpoint["step"] == 64 and checkpoint["kind"] == KIND
        assert checkpoint["args"]["seed"] == 201
        assert checkpoint["args"]["output_dir"] == str(output)
        assert checkpoint["source_sha256"]["rl/safety_opening.py"]
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        assert not torch.equal(checkpoint["state_dict"]["q.head.bias"], checkpoint["online"]["q.head.bias"])
        assert any(value["step"] > 0 for value in checkpoint["optimizer"]["state"].values())
        # A no-update resume must retain separate online/acting weights and Adam.
        args.resume = str(path)
        path = train(args)
        reloaded = load_model(path)[1]
        for key, value in checkpoint["state_dict"].items():
            assert torch.equal(value, reloaded["state_dict"][key])
            assert torch.equal(checkpoint["online"][key], reloaded["online"][key])
        for key, value in checkpoint["optimizer"]["state"].items():
            for field, tensor in value.items():
                assert torch.equal(tensor, reloaded["optimizer"]["state"][key][field])
        args.resume, args.total_timesteps = str(path), 96
        path = train(args)
        assert load_model(path)[1]["step"] == 96
        args.opening_lambda = .03
        try:
            train(args)
        except ValueError as error:
            assert "opening-lambda" in str(error)
        else:
            raise AssertionError("changed resume policy must be rejected")
        args.opening_lambda = .01
        # Exercise the real standalone loader/grid/cache/statistics/JSON path.
        artifact = output / "grid.json"
        result = subprocess.run([sys.executable, "-m", "rl.safety_opening", "evaluate",
            "--model", str(path), "--output", str(artifact), "--grid", "0", "0.01",
            "--device", "cpu", "--games", "3", "--n-envs", "3", "--jobs", "1",
            "--cache-dir", str(output / "bayes-cache")], cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        saved = json.loads(artifact.read_text())
        assert len(saved["policies"]) == 2
        assert all(len(policy["records"]) == 3 for policy in saved["policies"])
        assert saved["checkpoint"]["training_args"]["output_dir"] == str(output)


def test_changed_checkpoint_does_not_publish_evaluation_artifact():
    from rl.experiment_eval import sha256

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model, pending, output = root / "model.pt", root / "pending.json", root / "result.json"
        model.write_bytes(b"old")
        before = sha256(model)
        pending.write_text("{}")
        model.write_bytes(b"changed")
        try:
            publish_checked_result(model, before, pending, output)
        except RuntimeError:
            pass
        else:
            raise AssertionError("mutable checkpoint result must not be published")
        assert not pending.exists() and not output.exists()
        pending.write_text("{}")
        publish_checked_result(model, sha256(model), pending, output)
        assert output.exists()


if __name__ == "__main__":
    torch.set_num_threads(1)
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"{name}: passed")
