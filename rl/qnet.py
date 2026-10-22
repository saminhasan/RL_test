"""The Q-network: one value per cell, "how good is clicking here".

Fully convolutional (3x3 filters only), so the same weights play any board
size, and deep enough to chain clue logic across a 16x30 board: with the
default 16 layers each cell's value can depend on cells 16 steps away. Every
layer after the first adds onto its input (a residual shortcut), starting close
to pass-through, so deep stacks train stably.

With pool_every > 0, every pool_every-th residual layer also sees the whole
board: the mean and the maximum of each channel over the board's cells, turned
into one bias per channel (global pooling, as in KataGo, Wu 2019). Convolutions
alone only see a few cells around each cell; how many mines are left and where
the unknown cells are is a whole-board question. The pooling starts switched off
(zero weights), so a net with it begins exactly like one without.

Input planes, built from the uint8 observation on the GPU:
  hidden, clue 0, ..., clue 8   (one-hot of the visible board)
  on board                      (0 off the board and in the zero padding at the edge)
  mine density                  (mines / hidden cells, the same in every on-board cell)

Checkpoints hold the config with the weights, so a net is rebuilt from its file.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.journal import replace_with_retry

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_CONFIG = {"channels": 64, "layers": 16, "kernel": 3, "pool_every": 0}
N_PLANES = 12


class _Residual(nn.Module):
    def __init__(self, ch: int, k: int, pool: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, k, padding=k // 2)
        self.pool = nn.Linear(2 * ch, ch) if pool else None
        with torch.no_grad():
            self.conv.weight.mul_(0.2)
            self.conv.bias.zero_()
            if self.pool is not None:
                self.pool.weight.zero_()
                self.pool.bias.zero_()

    def forward(self, h: torch.Tensor, on_board: torch.Tensor) -> torch.Tensor:
        y = self.conv(h)
        if self.pool is not None:                                     # h >= 0 (after a ReLU), so masking keeps the max
            seen = h * on_board
            mean = seen.sum((2, 3)) / on_board.sum((2, 3)).clamp(min=1)
            y = y + self.pool(torch.cat([mean, seen.amax((2, 3))], 1))[:, :, None, None]
        return torch.relu(h + y)


class QNetwork(nn.Module):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        ch, k = self.config["channels"], self.config["kernel"]
        every = self.config["pool_every"]
        self.body = nn.Sequential(
            nn.Conv2d(N_PLANES, ch, k, padding=k // 2), nn.ReLU(),
            *[_Residual(ch, k, pool=every > 0 and (i + 1) % every == 0) for i in range(self.config["layers"] - 1)],
        )
        self.head = nn.Conv2d(ch, 1, 1)

    @staticmethod
    def planes(obs: torch.Tensor) -> torch.Tensor:
        view   = obs[:, 0].long()                                       # 0 hidden, 1..9 clue 0..8, 10 off board
        onehot = F.one_hot(view, 11)[..., :10].permute(0, 3, 1, 2).float()
        on_board = (view != 10).float()[:, None]
        hidden = (view == 0).flatten(1).sum(1).clamp(min=1).float()
        density = (obs[:, 1, 0, 0].float() / hidden)[:, None, None, None] * on_board
        return torch.cat([onehot, on_board, density], dim=1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (B, 2, rows, cols) uint8 -> Q (B, rows * cols). Mask illegal cells yourself."""
        x = self.planes(obs)
        on_board = x[:, 10:11]
        h = self.body[1](self.body[0](x))
        for block in self.body[2:]:
            h = block(h, on_board)
        return self.head(h).flatten(1)


def legal(obs: torch.Tensor) -> torch.Tensor:
    """(B, rows * cols) bool: hidden cells, the only legal clicks."""
    return (obs[:, 0] == 0).flatten(1)


def greedy(q: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Best legal cell per board."""
    return q.masked_fill(~legal(obs), float("-inf")).argmax(dim=1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(path: Path, q: QNetwork, extra: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"config": q.config, "state_dict": q.state_dict(), **(extra or {})}, tmp)
    replace_with_retry(tmp, path)                                       # never a half-written checkpoint
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[QNetwork, dict]:
    p = Path(path)
    if not p.exists():
        p = MODELS_DIR / (p.name if p.suffix == ".pt" else f"{p.name}.pt")
    ckpt = torch.load(p, map_location=device, weights_only=False)
    q = QNetwork(ckpt["config"]).to(device)
    q.load_state_dict(ckpt["state_dict"])
    return q, ckpt
