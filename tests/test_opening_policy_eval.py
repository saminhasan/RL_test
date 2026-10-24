"""CPU-only learned-gate action selection checks; no engine or GPU execution."""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.opening_policy_eval import GatedOpening


class FixedHeads(nn.Module):
    def __init__(self, risks, zero_probabilities):
        super().__init__()
        self.register_buffer("safe", -torch.logit(torch.tensor(risks, dtype=torch.float32)))
        self.register_buffer("zero", torch.logit(torch.tensor(zero_probabilities, dtype=torch.float32)))

    def heads(self, obs):
        return {"safe": self.safe, "zero": self.zero}


def observation(batch, rows, cols):
    return torch.zeros((batch, 2, rows, cols), dtype=torch.uint8)


def test_gate_chooses_opening_only_within_the_learned_risk_band():
    model = FixedHeads([[0.001, 0.00105, 0.005, 0.5]], [[0.1, 0.8, 0.999, 0.999]])
    obs = observation(1, 2, 2)
    score = GatedOpening(model, cutoff=0.01, margin=0.0001)(obs)
    assert score.argmax(1).item() == 1
    assert torch.isneginf(score[0, 2:]).all()


def test_margin_zero_is_exact_pure_safety_even_in_guess_regime():
    model = FixedHeads([[0.001, 0.00101, 0.005, 0.5], [0.1, 0.11, 0.3, 0.4]],
                       [[0.1, 0.9, 0.99, 0.9], [0.01, 0.99, 0.99, 0.99]])
    obs = observation(2, 2, 2)
    score = GatedOpening(model, margin=0.0, guess_lambda=1.0)(obs)
    assert torch.equal(score.argmax(1), model.safe.argmax(1))
    assert torch.equal(score, model.safe)


def test_guess_branch_uses_original_composite_score():
    model = FixedHeads([[0.1, 0.11, 0.5, 0.8]], [[0.01, 0.99, 0.01, 0.01]])
    obs = observation(1, 2, 2)
    score = GatedOpening(model, cutoff=0.01, margin=1e-4, guess_lambda=1.0)(obs)
    expected = torch.nn.functional.logsigmoid(model.safe) + model.zero.sigmoid()
    assert torch.equal(score, expected)
    assert score.argmax(1).item() == 1


def test_gated_ties_favor_safety_then_first_row_major_cell():
    model = FixedHeads([[0.001, 0.00095, 0.00095, 0.5]], [[0.7, 0.7, 0.7, 0.99]])
    obs = observation(1, 2, 2)
    assert GatedOpening(model, margin=1e-4)(obs).argmax(1).item() == 1


def test_legal_masks_and_square_batches_are_respected():
    model = FixedHeads([[0.00001, 0.001, 0.00105, 0.5], [0.00001, 0.00001, 0.2, 0.3]],
                       [[0.99, 0.1, 0.9, 0.9], [0.99, 0.99, 0.1, 0.1]])
    obs = observation(2, 2, 2)
    obs[0, 0].flatten()[0] = 1      # visible clue zero
    obs[1, 0].flatten()[:2] = 10    # off-board padding
    score = GatedOpening(model, margin=1e-4)(obs)
    assert score.shape == (2, 4)
    assert torch.equal(score.argmax(1), torch.tensor([2, 2]))
    assert torch.isneginf(score[0, 0]) and torch.isneginf(score[1, :2]).all()


def test_invalid_policy_parameters_are_rejected():
    for kwargs in ({"cutoff": -1}, {"cutoff": float("nan")}, {"margin": -1},
                   {"margin": float("inf")}, {"guess_lambda": -0.1}):
        try:
            GatedOpening(nn.Identity(), **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("expected invalid policy parameters to fail")


if __name__ == "__main__":
    torch.set_num_threads(1)
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"{name}: passed")
