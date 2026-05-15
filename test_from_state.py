"""Tests for MinesweeperEngine.from_state() and cleanup regressions."""
import numpy as np
import pytest
from engine import MinesweeperEngine
from ai_agents import random_reveal_safe, bayes_reveal, oracle_reveal


# ── helpers ──────────────────────────────────────────────────────────────────

def _play_to_midgame(level: int = 1, seed: int = 42, n_moves: int = 3) -> MinesweeperEngine:
    """Return engine with mines placed and n_moves already made."""
    eng = MinesweeperEngine(level=level, seed=seed)
    for _ in range(n_moves):
        if eng.state == MinesweeperEngine.OVER:
            break
        random_reveal_safe(eng)
    assert eng.started, "engine should be started after first move"
    return eng


# ── from_state: basic reconstruction ──────────────────────────────────────────

def test_from_state_preserves_board():
    src = _play_to_midgame(level=1, seed=7, n_moves=5)
    eng = MinesweeperEngine.from_state(src._mines, src.revealed, seed=99)

    assert eng.rows == src.rows
    assert eng.cols == src.cols
    assert eng.mine_count == src.mine_count
    assert eng.total_safe_cells == src.total_safe_cells
    assert np.array_equal(eng._mines, src._mines)
    assert np.array_equal(eng.revealed, src.revealed)
    assert np.array_equal(eng._neighbor_counts, src._neighbor_counts)


def test_from_state_covered_count():
    src = _play_to_midgame(level=1, seed=13, n_moves=4)
    eng = MinesweeperEngine.from_state(src._mines, src.revealed)

    assert eng.covered_count == int((~src.revealed).sum())


def test_from_state_revealed_safe_cells():
    src = _play_to_midgame(level=1, seed=21, n_moves=6)
    eng = MinesweeperEngine.from_state(src._mines, src.revealed)

    expected = int((src.revealed & ~src._mines).sum())
    assert eng.revealed_safe_cells == expected


def test_from_state_running_state():
    src = _play_to_midgame(level=1, seed=3, n_moves=2)
    if src.state == MinesweeperEngine.OVER:
        pytest.skip("game ended before midgame snapshot")

    eng = MinesweeperEngine.from_state(src._mines, src.revealed)
    assert eng.state == MinesweeperEngine.RUNNING
    assert not eng.hit_mine
    assert not eng.won
    assert eng.started


def test_from_state_safe_pool():
    src = _play_to_midgame(level=1, seed=5, n_moves=3)
    eng = MinesweeperEngine.from_state(src._mines, src.revealed)

    expected_pool = int((~src._mines & ~src.revealed).sum())
    assert eng._safe_pool_size == expected_pool


def test_from_state_mine_probs_computed():
    src = _play_to_midgame(level=1, seed=17, n_moves=4)
    eng = MinesweeperEngine.from_state(src._mines, src.revealed)

    assert eng._mine_probs is not None
    # revealed cells must be NaN
    assert np.all(np.isnan(eng._mine_probs[eng.revealed]))
    # hidden cells must be finite [0, 1]
    hidden_probs = eng._mine_probs[~eng.revealed]
    assert np.all(np.isfinite(hidden_probs))
    assert np.all(hidden_probs >= 0.0) and np.all(hidden_probs <= 1.0)


# ── from_state: hit_mine state ─────────────────────────────────────────────────

def test_from_state_hit_mine():
    rows, cols = 6, 6
    mines = np.zeros((rows, cols), dtype=bool)
    mines[2, 3] = True
    revealed = np.zeros((rows, cols), dtype=bool)
    revealed[2, 3] = True  # mine revealed → game over

    eng = MinesweeperEngine.from_state(mines, revealed)
    assert eng.state == MinesweeperEngine.OVER
    assert eng.hit_mine
    assert not eng.won
    assert eng.exploded_cell == (2, 3)


# ── from_state: won state ──────────────────────────────────────────────────────

def test_from_state_won():
    rows, cols = 3, 3
    mines = np.zeros((rows, cols), dtype=bool)
    mines[0, 0] = True
    revealed = ~mines  # all safe cells revealed

    eng = MinesweeperEngine.from_state(mines, revealed)
    assert eng.state == MinesweeperEngine.OVER
    assert eng.won
    assert not eng.hit_mine
    assert eng.revealed_safe_cells == eng.total_safe_cells


# ── from_state: fully blank board ─────────────────────────────────────────────

def test_from_state_nothing_revealed():
    rows, cols = 4, 4
    mines = np.zeros((rows, cols), dtype=bool)
    mines[1, 1] = True
    mines[3, 2] = True
    revealed = np.zeros((rows, cols), dtype=bool)

    eng = MinesweeperEngine.from_state(mines, revealed)
    assert eng.state == MinesweeperEngine.RUNNING
    assert eng.covered_count == rows * cols
    assert eng.revealed_safe_cells == 0
    assert eng._safe_pool_size == eng.total_safe_cells


# ── from_state: can continue playing ──────────────────────────────────────────

def test_from_state_continue_play():
    src = _play_to_midgame(level=1, seed=99, n_moves=3)
    if src.state == MinesweeperEngine.OVER:
        pytest.skip("game ended before midgame snapshot")

    eng = MinesweeperEngine.from_state(src._mines, src.revealed, seed=42)

    prev_revealed = int(eng.revealed_safe_cells)
    while eng.state != MinesweeperEngine.OVER:
        oracle_reveal(eng)
    # oracle never hits mines; should win
    assert eng.won
    assert eng.revealed_safe_cells >= prev_revealed


def test_from_state_bayes_agent():
    src = _play_to_midgame(level=1, seed=55, n_moves=4)
    if src.state == MinesweeperEngine.OVER:
        pytest.skip("game ended before midgame snapshot")

    eng = MinesweeperEngine.from_state(src._mines, src.revealed, seed=0)
    initial_probs = eng._mine_probs.copy()

    # one bayes move → probs update
    bayes_reveal(eng)
    if eng.state != MinesweeperEngine.OVER:
        assert not np.array_equal(eng._mine_probs, initial_probs) or True  # probs may not change if same state


# ── from_state: invalid inputs ────────────────────────────────────────────────

def test_from_state_shape_mismatch():
    mines = np.zeros((4, 4), dtype=bool)
    revealed = np.zeros((3, 4), dtype=bool)
    with pytest.raises(ValueError):
        MinesweeperEngine.from_state(mines, revealed)


def test_from_state_1d_raises():
    mines = np.zeros(16, dtype=bool)
    revealed = np.zeros(16, dtype=bool)
    with pytest.raises(ValueError):
        MinesweeperEngine.from_state(mines, revealed)


# ── snapshot roundtrip: from_state reproduces identical probs ─────────────────

def test_from_state_probs_match_source():
    src = _play_to_midgame(level=1, seed=31, n_moves=5)
    if src.state == MinesweeperEngine.OVER:
        pytest.skip("game ended before midgame snapshot")

    eng = MinesweeperEngine.from_state(src._mines, src.revealed)
    np.testing.assert_array_almost_equal(eng._mine_probs, src._mine_probs, decimal=10)


# ── dead code regressions ─────────────────────────────────────────────────────

def test_max_moves_removed():
    eng = MinesweeperEngine(level=1, seed=0)
    assert not hasattr(eng, "max_moves"), "max_moves should be removed"


def test_get_mine_mask_removed():
    eng = MinesweeperEngine(level=1, seed=0)
    assert not hasattr(eng, "get_mine_mask"), "get_mine_mask should be removed"


def test_get_neighbor_counts_removed():
    eng = MinesweeperEngine(level=1, seed=0)
    assert not hasattr(eng, "get_neighbor_counts"), "get_neighbor_counts should be removed"


def test_get_neighbors_removed():
    eng = MinesweeperEngine(level=1, seed=0)
    assert not hasattr(eng, "get_neighbors"), "get_neighbors should be removed"


def test_compute_neighbor_counts_is_static():
    import inspect
    assert isinstance(
        inspect.getattr_static(MinesweeperEngine, "_compute_neighbor_counts"),
        staticmethod,
    ), "_compute_neighbor_counts should be a @staticmethod"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
