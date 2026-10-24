"""Fast synthetic checks for experiment split isolation and paired statistics."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.experiment_eval import SOURCE_FILES, bayes_cache_key, bayes_cache_protocol, board_seeds, paired_comparison


def _records(wins):
    return [{"seed": 100 + i, "won": won} for i, won in enumerate(wins)]


def test_board_blocks_are_disjoint_and_repeatable():
    blocks = [board_seeds(split, level, 1000, replicate)
              for split in ("validation", "test") for level in (1, 2, 3, 4) for replicate in (0, 1)]
    assert all(r.start > 20_000_000_000 for r in blocks)
    for i, left in enumerate(blocks):
        assert len(left) == 1000
        for right in blocks[i + 1:]:
            assert left.stop <= right.start or right.stop <= left.start
    assert list(board_seeds("test", 2, 10)) == list(board_seeds("test", 2, 100))[:10]


def test_invalid_board_block_rejected():
    for kwargs in ({"split": "train"}, {"level": 5}, {"games": 0}, {"games": 1_000_001},
                   {"replicate": -1}, {"replicate": 1024}):
        try:
            board_seeds(**{"split": "validation", "level": 2, "games": 100, **kwargs})
        except ValueError:
            pass
        else:
            raise AssertionError("expected invalid protocol to fail")


def test_paired_counts_and_exact_test():
    model = _records([True] * 5 + [True, False])
    bayes = _records([False] * 5 + [True, False])
    result = paired_comparison(model[::-1], bayes, bootstrap_samples=1000)
    assert result["model_only_wins"] == 5 and result["bayes_only_wins"] == 0
    assert result["both_win"] == 1 and result["both_lose"] == 1
    assert result["win_difference"] == 5 / 7
    assert result["mcnemar_exact_two_sided_p"] == 0.0625
    assert result["model_better_exact_one_sided_p"] == 0.03125
    assert result == paired_comparison(model, bayes[::-1], bootstrap_samples=1000)


def test_identical_policies_have_zero_difference():
    records = _records([True, False, True, True])
    result = paired_comparison(records, records, bootstrap_samples=1000)
    assert result["win_difference"] == 0
    assert result["win_difference_ci95"] == [0.0, 0.0]
    assert result["mcnemar_exact_two_sided_p"] == 1.0


def test_missing_or_duplicate_boards_are_not_silently_dropped():
    model = _records([True, False])
    for bayes in (model[:1], [model[0], model[0]], []):
        try:
            paired_comparison(model, bayes, bootstrap_samples=1000)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ambiguous pairing to fail")


def test_bayes_cache_depends_only_on_its_sources_and_boards():
    protocol = {"schema": 1, "split": "validation", "level": 2, "games": 100,
                "replicate": 0, "requested_seed_start": 1234, "requested_seed_stop": 1334,
                "source_sha256": {name: "old" for name in SOURCE_FILES}}
    original = bayes_cache_key(protocol)
    changed = {**protocol, "source_sha256": dict(protocol["source_sha256"]),
               "runtime": {"symmetries": True, "fp32_head_override": True}}
    for name in ("rl/qnet.py", "rl/dqn.py", "rl/per.py", "rl/journal.py", "rl/experiment_eval.py"):
        changed["source_sha256"][name] = "new"
    assert bayes_cache_key(changed) == original
    assert "rl/qnet.py" not in bayes_cache_protocol(protocol)["source_sha256"]
    changed["source_sha256"]["engine.py"] = "new"
    assert bayes_cache_key(changed) != original
    assert bayes_cache_key({**protocol, "requested_seed_start": 5678}) != original


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: passed")
