from __future__ import annotations

import unittest

import numpy as np

from bayes import mine_probabilities_for_engine as bayes_probabilities
from engine import MinesweeperEngine
from graph_solver import mine_probabilities_for_engine as graph_probabilities


def solver_args(engine: MinesweeperEngine) -> dict:
    return dict(
        revealed=engine.revealed.copy(),
        neighbor_counts=engine.get_neighbor_counts(),
        mine_count=engine.mine_count,
        started=engine.started,
        hit_mine=engine.hit_mine,
        exploded_cell=engine.exploded_cell,
        rows=engine.rows,
        cols=engine.cols,
        neighbors=engine._neighbors,
    )


def assert_solver_match(testcase: unittest.TestCase, engine: MinesweeperEngine) -> None:
    args = solver_args(engine)
    old = bayes_probabilities(**args)
    new = graph_probabilities(**args)

    testcase.assertEqual(old.shape, new.shape)
    testcase.assertEqual(old.dtype, new.dtype)
    np.testing.assert_allclose(old, new, rtol=0.0, atol=1e-12, equal_nan=True)


class GraphSolverMatchesBayes(unittest.TestCase):
    def test_not_started_uniform_matches(self) -> None:
        for level in (0, 1):
            engine = MinesweeperEngine(level=level, seed=123)
            assert_solver_match(self, engine)

    def test_test_level_safe_sequences_match(self) -> None:
        for seed in range(30):
            engine = MinesweeperEngine(level=0, seed=10_000 + seed)
            assert_solver_match(self, engine)

            engine.reveal(2, 2)
            assert_solver_match(self, engine)

            for _ in range(12):
                if engine.state == engine.OVER:
                    break
                engine.random_reveal(safe=True)
                assert_solver_match(self, engine)

    def test_test_level_random_sequences_match_even_after_loss(self) -> None:
        for seed in range(20):
            engine = MinesweeperEngine(level=0, seed=20_000 + seed)

            for _ in range(16):
                assert_solver_match(self, engine)
                if engine.state == engine.OVER:
                    break
                engine.random_reveal()

            assert_solver_match(self, engine)

    def test_easy_level_first_few_safe_moves_match(self) -> None:
        for seed in range(5):
            engine = MinesweeperEngine(level=1, seed=30_000 + seed)
            engine.reveal(4, 4)
            assert_solver_match(self, engine)

            for _ in range(4):
                if engine.state == engine.OVER:
                    break
                engine.random_reveal(safe=True)
                assert_solver_match(self, engine)


if __name__ == "__main__":
    unittest.main(verbosity=2)
