"""The Q-network: one value per cell, "how good is clicking here".

Fully convolutional (3x3 filters only), so the same weights play any board
size, and deep enough to chain clue logic across a 16x30 board: with the
default 16 layers each cell's value can depend on cells 16 steps away. Every
layer after the first adds onto its input (a residual shortcut), starting close
to pass-through, so deep stacks train stably.

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

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_CONFIG = {"channels": 64, "layers": 16, "kernel": 3}
N_PLANES = 12


class _Residual(nn.Module):
    def __init__(self, ch: int, k: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, k, padding=k // 2)
        with torch.no_grad():
            self.conv.weight.mul_(0.2)
            self.conv.bias.zero_()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.relu(h + self.conv(h))


class QNetwork(nn.Module):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        ch, k = self.config["channels"], self.config["kernel"]
        self.body = nn.Sequential(
            nn.Conv2d(N_PLANES, ch, k, padding=k // 2), nn.ReLU(),
            *[_Residual(ch, k) for _ in range(self.config["layers"] - 1)],
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
        return self.head(self.body(self.planes(obs))).flatten(1)


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
    tmp.replace(path)                                                   # never a half-written checkpoint
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[QNetwork, dict]:
    p = Path(path)
    if not p.exists():
        p = MODELS_DIR / (p.name if p.suffix == ".pt" else f"{p.name}.pt")
    ckpt = torch.load(p, map_location=device, weights_only=False)
    q = QNetwork(ckpt["config"]).to(device)
    q.load_state_dict(ckpt["state_dict"])
    return q, ckpt
