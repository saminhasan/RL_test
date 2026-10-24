"""Checks for reward wiring, precision, and study evaluation isolation."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from rl.dqn import Args, reward_schedule
from rl.evaluate import eval_board_seeds
from rl.qnet import QNetwork, SymmetryEnsemble


def test_burst_cli_is_applied_and_survives_stage_switch():
    at, first, second = reward_schedule(Args(reward_burst=3, burst_rate=7, stage2_from=100,
                                            stage2="mine=100,burst=2"))
    assert at == 100
    assert first.burst == 3 and second.burst == 2
    assert first.burst_rate == second.burst_rate == 7


def test_bce_rejects_unbounded_return_terms_in_both_stages():
    for args in (Args(loss="bce", reward_progress=0, stage2="progress=1"),
                 Args(loss="bce", reward_progress=0, reward_safe=1, gamma=1),
                 Args(loss="bce", reward_progress=0, stage2="safe=-0.1", gamma=0.99)):
        try:
            reward_schedule(args)
        except ValueError:
            pass
        else:
            raise AssertionError("BCE accepted rewards outside its configured return bounds")
    _, _, immediate = reward_schedule(Args(loss="bce", reward_progress=0, reward_safe=-0.1, gamma=0))
    assert immediate.safe == -0.1


def test_second_stage_cannot_change_the_learners_discount():
    try:
        reward_schedule(Args(stage2="gamma=0.5", gamma=0.99))
    except ValueError as error:
        assert "same discount" in str(error)
    else:
        raise AssertionError("the reward discount diverged from the learner discount")


def test_fp32_head_keeps_small_score_differences_under_autocast():
    obs = torch.zeros((2, 2, 9, 9), dtype=torch.uint8)
    obs[:, 1] = 10
    q = QNetwork({"channels": 4, "layers": 2})
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert q(obs).dtype == torch.bfloat16
        q.config["fp32_head"] = True
        assert q(obs).dtype == torch.float32


def test_symmetry_scores_return_to_the_original_cells():
    class Biased(torch.nn.Module):
        def forward(self, obs):
            return (obs[:, 0].float() + torch.arange(obs.shape[-1])[None, None, :]).flatten(1)

    for rows, cols in ((3, 3), (3, 5)):
        obs = torch.arange(2 * rows * cols, dtype=torch.uint8).reshape(1, 2, rows, cols)
        scores = SymmetryEnsemble(Biased())(obs)
        assert torch.allclose(scores.reshape(1, rows, cols), obs[:, 0].float() + (cols - 1) / 2)
        # Averaging probabilities must agree with independently mapped predictions.
        class Pointwise(torch.nn.Module):
            def forward(self, x):
                return x[:, 0].float().flatten(1) / 20 - 1
        model = Pointwise()
        assert torch.allclose(SymmetryEnsemble(model, probabilities=True)(obs), model(obs), atol=1e-6)


def test_validation_seed_base_preserves_original_evaluations():
    assert eval_board_seeds(2, 3).start == 10_020_000_000
    assert eval_board_seeds(2, 3, seed0=1_000_000_000_000).start == 1_000_020_000_000


if __name__ == "__main__":
    torch.set_num_threads(2)
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: passed")
