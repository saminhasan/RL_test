"""Synthetic confirmation-protocol tests; no training or board evaluation."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.confirm_study import (choose_winner, seal_training, success_criterion,
                              valid_test_result, write_selection_plan)
from rl.experiment_eval import board_seeds, sha256


def test_selection_uses_validation_win_then_efficiency():
    a = {"identity": "a", "split": "validation", "summary": {"win": .88, "efficiency": .90}}
    b = {"identity": "b", "split": "validation", "summary": {"win": .89, "efficiency": .80}}
    c = {"identity": "c", "split": "validation", "summary": {"win": .89, "efficiency": .85}}
    assert choose_winner([a, b, c]) == c
    try:
        choose_winner([a, {**c, "split": "test"}])
    except ValueError:
        pass
    else:
        raise AssertionError("test results must not participate in selection")


def test_selection_plan_is_immutable_and_sealed():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "selection-plan.json"
        plan = {"seeds": [202, 203, 204], "lambda": .01, "split": "test", "games": 10000}
        assert write_selection_plan(path, plan) == write_selection_plan(path, plan)
        for changed in ({**plan, "lambda": .03}, {**plan, "games": 1000}):
            try:
                write_selection_plan(path, changed)
            except ValueError:
                pass
            else:
                raise AssertionError("post-selection changes must fail")


def test_every_repeat_must_have_positive_paired_interval():
    seeds = [202, 203, 204]
    rows = [{"seed": seed, "paired_vs_bayes": {"win_difference": .02,
             "win_difference_ci95": [.005, .035]}} for seed in seeds]
    assert success_criterion(rows, seeds)
    assert not success_criterion(rows[:2], seeds)
    assert not success_criterion(rows[:1], seeds[:1])
    rows[-1]["paired_vs_bayes"]["win_difference_ci95"][0] = 0
    assert not success_criterion(rows, seeds)
    rows[-1]["paired_vs_bayes"]["win_difference_ci95"][0] = -.001
    assert not success_criterion(rows, seeds)


def test_test_split_seal_requires_every_trained_checkpoint():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = {"seeds": [202, 203, 204], "plan_sha256": "fixed-plan"}
        checkpoints = []
        for seed in plan["seeds"]:
            path = root / f"{seed}.pt"
            path.write_bytes(str(seed).encode())
            checkpoints.append({"seed": seed, "path": str(path), "sha256": sha256(path)})
        try:
            seal_training(root / "seal.json", plan, checkpoints[:2])
        except ValueError:
            pass
        else:
            raise AssertionError("test split must stay closed before all repeats finish")
        assert not (root / "seal.json").exists()
        seal_training(root / "seal.json", plan, checkpoints)
        Path(checkpoints[-1]["path"]).write_bytes(b"changed")
        try:
            seal_training(root / "seal.json", plan, checkpoints)
        except ValueError:
            pass
        else:
            raise AssertionError("mutated checkpoint must invalidate the test seal")


def test_test_result_must_match_split_checkpoint_sources_and_fixed_policy():
    import copy

    seeds = board_seeds("test", 2, 2)
    plan = {"test_games": 2, "source_sha256": {"source": "correct"},
            "winner": {"kind": "q", "inference": {"symmetries": False, "fp32_head": True}}}
    frozen = {"sha256": "model"}
    result = {"protocol": {"split": "test", "level": 2, "replicate": 0, "games": 2,
              "requested_seed_start": seeds.start, "requested_seed_stop": seeds.stop,
              "source_sha256": {"source": "correct"}},
              "runtime": {"symmetries": False, "fp32_head_override": True},
              "models": [{"sha256": "model", "records": [{}, {}]}]}
    assert valid_test_result(result, frozen, plan)
    for section, field, value in (("protocol", "split", "validation"),
                                   ("protocol", "source_sha256", {"source": "changed"}),
                                   ("runtime", "fp32_head_override", False)):
        changed = copy.deepcopy(result)
        changed[section][field] = value
        assert not valid_test_result(changed, frozen, plan)
    assert not valid_test_result(result, {"sha256": "other-model"}, plan)


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"{name}: passed")
