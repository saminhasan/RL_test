"""Synthetic CPU checks for public-observation consistency and its gradients."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.public_constraints import public_constraint_loss


def _obs(view, mines):
    view = torch.tensor(view, dtype=torch.uint8)
    return torch.stack((view, torch.full_like(view, mines)))[None]


def test_feasible_probability_map_has_zero_public_loss():
    # One central clue of 1, eight unknown neighbours, one public mine total.
    obs = _obs([[0, 0, 0], [0, 2, 0], [0, 0, 0]], 1)
    logits = torch.full((1, 9), torch.logit(torch.tensor(0.875)), requires_grad=True)
    clue, budget = public_constraint_loss(obs, logits)
    assert clue.item() < 1e-12 and budget.item() < 1e-12
    (clue + budget).backward()
    assert torch.isfinite(logits.grad).all()


def test_zero_clue_pushes_its_hidden_neighbours_toward_safety():
    obs = _obs([[0, 0, 0], [0, 1, 0], [0, 0, 0]], 0)
    logits = torch.zeros((1, 3, 3), requires_grad=True)
    clue, _ = public_constraint_loss(obs, logits)
    assert torch.isclose(clue, torch.tensor(0.25))
    clue.backward()
    hidden = obs[:, 0] == 0
    assert (logits.grad[hidden] < 0).all()  # gradient descent raises safety logits
    assert logits.grad[0, 1, 1] == 0       # the visible cell is not a variable


def test_global_budget_remains_active_without_public_clues():
    obs = _obs([[0, 0], [0, 0]], 1)
    logits = torch.zeros((1, 4), requires_grad=True)
    clue, budget = public_constraint_loss(obs, logits)
    assert clue.item() == 0
    assert budget.item() == 1             # expected 2 mines, public total 1
    budget.backward()
    assert (logits.grad < 0).all()         # move expected mines down
    logits = torch.full((1, 4), 5.0, requires_grad=True)
    _, budget = public_constraint_loss(obs, logits)
    budget.backward()
    assert (logits.grad > 0).all()         # too few predicted mines: move them up


def test_visible_and_padded_cells_never_consume_mine_budget():
    obs = _obs([[0, 1, 10], [0, 10, 10]], 1)
    logits = torch.zeros((1, 1, 2, 3), requires_grad=True)
    _, budget = public_constraint_loss(obs, logits)
    assert budget.item() == 0             # two hidden cells * 0.5
    # Perturb a hidden probability so nonzero budget gradients reveal masking.
    with torch.no_grad():
        logits[0, 0, 0, 0] = 2
    _, budget = public_constraint_loss(obs, logits)
    budget.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[0, 0][obs[0, 0] != 0] == 0).all()
    assert (logits.grad[0, 0][obs[0, 0] == 0] != 0).all()


def test_losses_are_averaged_per_board_and_remain_fp32():
    obs = _obs([[0, 0, 0], [0, 1, 0], [0, 0, 0]], 0)
    logits = torch.zeros((1, 9), dtype=torch.bfloat16, requires_grad=True)
    single = public_constraint_loss(obs, logits)
    repeated = public_constraint_loss(obs.repeat(3, 1, 1, 1), logits.repeat(3, 1))
    assert all(a.dtype == torch.float32 and torch.allclose(a, b) for a, b in zip(single, repeated))
    sum(single).backward()
    assert torch.isfinite(logits.grad).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: passed")
