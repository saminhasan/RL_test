"""Differentiable consistency with public clues and the public mine budget.

These are constraints on expected mine counts, not labels for individual hidden
cells. The true posterior satisfies them, but many other probability maps also
do. Use as an explicitly weighted auxiliary to calibrated selected-click BCE.
Inputs contain only the public board encoding used by MinesweeperEnv.
"""
from __future__ import annotations


def public_constraint_loss(obs, safe_logits):
    """Return unweighted ``(clue_loss, budget_loss)`` scalar FP32 tensors.

    ``obs`` has shape (B, 2, H, W), with 0=hidden, 1..9=clue 0..8,
    10=off-board in plane 0 and the public mine total in plane 1.
    ``safe_logits`` has shape (B, H*W), (B, H, W), or (B, 1, H, W).

    For each revealed clue with hidden neighbours, square its expected-count
    residual divided by the number of hidden neighbours. Average over active
    clues within each board, then over boards, so clue-rich boards do not get
    larger total weight. Boards without active clues contribute zero locally.
    The budget loss is the squared total-mine residual divided by
    max(public mine total, 1)^2, averaged over boards. It remains active on a
    board without revealed clues. Suggested initial weights for an experiment
    are 1 for clues and 0.1 for budget; zero-weight controls are essential.

    Revealed and off-board cells contribute no mine probability. All operations
    on predictions use FP32, even under mixed precision, with gradients intact.
    The module does not read game internals, a solver, or a hidden layout.
    """
    import torch
    import torch.nn.functional as F

    if obs.ndim != 4 or obs.shape[1] != 2 or min(obs.shape[0], *obs.shape[2:]) < 1:
        raise ValueError("obs must have nonempty shape (B, 2, H, W)")
    batch, _, rows, cols = obs.shape
    allowed = ((batch, rows * cols), (batch, rows, cols), (batch, 1, rows, cols))
    if tuple(safe_logits.shape) not in allowed:
        raise ValueError("safe_logits must have shape (B,H*W), (B,H,W), or (B,1,H,W)")
    with torch.autocast(safe_logits.device.type, enabled=False):
        view = obs[:, :1].to(device=safe_logits.device)
        hidden = (view == 0).float()
        # sigmoid(-z) avoids cancellation from 1 - sigmoid(z) near safety=1.
        mine_probability = torch.sigmoid(-safe_logits.float().reshape(batch, 1, rows, cols)) * hidden
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=safe_logits.device)
        kernel[0, 0, 1, 1] = 0
        neighbour_mines = F.conv2d(mine_probability, kernel, padding=1)
        neighbour_hidden = F.conv2d(hidden, kernel, padding=1)
        active = ((view >= 1) & (view <= 9) & (neighbour_hidden > 0)).float()
        target = (view.float() - 1).clamp(min=0)
        local_residual = (neighbour_mines - target) / neighbour_hidden.clamp(min=1)
        active_per_board = active.sum((1, 2, 3)).clamp(min=1)
        clue_loss = ((local_residual.square() * active).sum((1, 2, 3)) / active_per_board).mean()
        mine_total = obs[:, 1, 0, 0].to(device=safe_logits.device, dtype=torch.float32)
        budget_residual = (mine_probability.sum((1, 2, 3)) - mine_total) / mine_total.clamp(min=1)
        budget_loss = budget_residual.square().mean()
    return clue_loss, budget_loss
