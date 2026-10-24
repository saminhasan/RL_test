"""Synthetic queue protocol checks; does not launch children."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.auxiliary_sweep import candidates, child_command, complete_result, policy_grid, training_config


def test_six_cases_and_policy_grids():
    cases = candidates()
    assert len(cases) == 6
    assert cases["safety_control"]["aux"] == "none"
    assert cases["zero_representation"]["opening_lambda"] == 0
    assert cases["zero_public_strong"]["clue_weight"] == 10
    assert policy_grid(cases["public_rules"]) == [0]
    assert policy_grid(cases["zero_representation"]) == [0, .001, .003, .01, .03, .1]


def test_exact_train_arguments_and_smoke_intervals():
    args = SimpleNamespace(init=Path("init file.pt"), seed=201, clicks=500_000,
                           validation_games=2000, jobs=4, device="cpu")
    config = training_config(args, "zero_public", Path("output folder"))
    assert config["num_envs"] == 64 and config["eval_games"] == 500
    command = child_command(config)
    assert command[1:4] == ["-m", "rl.safety_opening", "train"]
    assert command[command.index("--output-dir") + 1] == "output folder"
    assert command[command.index("--device") + 1] == "cpu"
    args.clicks, args.validation_games = 128, 3
    config = training_config(args, "safety_control", Path("output"))
    assert config["eval_every"] == config["checkpoint_every"] == 128
    assert config["eval_games"] == 3


def test_partial_lambda_artifact_is_not_counted_complete():
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "result.json"
        result = {"protocol": {"split": "validation", "games": 3},
                  "policies": [{"opening_lambda": 0}]}
        path.write_text(json.dumps(result))
        assert not complete_result(path, [0, .01], 3)
        result["policies"].append({"opening_lambda": .01})
        path.write_text(json.dumps(result))
        assert complete_result(path, [0, .01], 3)
        assert not complete_result(path, [0, .01], 4)
        assert not complete_result(path, [0, .01], 3, checkpoint_hash="actual")
        result["checkpoint"] = {"sha256": "actual"}
        result["protocol"]["source_sha256"] = {"source": "correct"}
        path.write_text(json.dumps(result))
        assert complete_result(path, [0, .01], 3, checkpoint_hash="actual", sources={"source": "correct"})
        assert not complete_result(path, [0, .01], 3, checkpoint_hash="changed", sources={"source": "correct"})
        assert not complete_result(path, [0, .01], 3, checkpoint_hash="actual", sources={"source": "changed"})


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"{name}: passed")
