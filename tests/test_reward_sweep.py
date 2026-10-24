"""Fast protocol checks; no training, engine, CUDA or evaluation is launched."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.reward_sweep import (CampaignLock, VALIDATION_SEED0, campaign_exp_name,
                             command_line, discover_run, fingerprint,
                             inference_variants, open_manifest, screen_candidates,
                             selected_names, training_config)
from rl.experiment_eval import board_seeds


def test_candidate_protocol_and_validation_split():
    assert len(screen_candidates()) == 13
    assert len(inference_variants()) == 5
    for name in screen_candidates():
        config = training_config(name)
        assert config["seed"] == 101 and config["total_timesteps"] == 500_000
        assert config["reward_mine"] == 1 and config["reward_progress"] == 0
        assert config["wall"] == 0 and config["deterministic"]
        assert config["loss"] == "bce" and config["buffer_size"] == 1_000_000
        assert config["n_step"] in (1, 3)
    assert training_config("old_precision_control")["fp32_head"] is False
    assert training_config("precise_ratio100")["fp32_head"] is True
    assert training_config("risk_only")["reward_reveal"] == 0
    assert training_config("future_win3")["gamma"] == 1
    assert VALIDATION_SEED0 + 2 * 10_000_000 == board_seeds("validation", 2, 10).start


def test_command_is_argument_list_with_explicit_booleans():
    command = command_line({"init": "D:/folder with spaces/model.pt", "fp32_head": False,
                            "deterministic": True, "reward_reveal": 0.01}, "python.exe")
    assert command[:3] == ["python.exe", "-m", "rl.dqn"]
    assert command[command.index("--init") + 1] == "D:/folder with spaces/model.pt"
    assert "--no-fp32-head" in command and "--deterministic" in command
    assert "False" not in command and "True" not in command


def test_selection_rejects_unknown_or_duplicate_variants():
    assert selected_names(screen_candidates(), ["risk_only"]) == ["risk_only"]
    for names in (["typo"], ["risk_only", "risk_only"]):
        try:
            selected_names(screen_candidates(), names)
        except ValueError:
            pass
        else:
            raise AssertionError("expected invalid selection to fail")


def test_manifest_cannot_mix_protocols():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "manifest.json"
        protocol = {"config": {"gamma": 0}, "source_sha256": {"dqn": "abc"}}
        first = open_manifest(path, protocol)
        assert open_manifest(path, protocol) == first
        for changed in ({**protocol, "config": {"gamma": 1}},
                        {**protocol, "source_sha256": {"dqn": "def"}}):
            try:
                open_manifest(path, changed)
            except ValueError:
                pass
            else:
                raise AssertionError("expected protocol mismatch to fail")


def test_run_discovery_only_finds_its_exact_study_prefix():
    exp = training_config("risk_only", clicks=1024)["exp_name"]
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        (directory / "easy-exp2-c64x16-20261024-040042").mkdir()
        (directory / (exp + "0-c64x16-20261024-123456")).mkdir()
        assert discover_run(exp, directory) is None
        name = exp + "-c64x16-20261024-123456"
        (directory / name).mkdir()
        assert discover_run(exp, directory) == name
        (directory / (exp + "-c64x16-20261024-123457")).mkdir()
        try:
            discover_run(exp, directory)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ambiguous runs to fail")


def test_campaign_prefix_changes_with_directory_source_and_init():
    config = training_config("risk_only")
    original = campaign_exp_name(config, Path("campaign-a"), {"dqn": "abc"}, "init-a")
    assert original == campaign_exp_name(config, Path("campaign-a"), {"dqn": "abc"}, "init-a")
    assert original != campaign_exp_name(config, Path("campaign-b"), {"dqn": "abc"}, "init-a")
    assert original != campaign_exp_name(config, Path("campaign-a"), {"dqn": "def"}, "init-a")
    assert original != campaign_exp_name(config, Path("campaign-a"), {"dqn": "abc"}, "init-b")


def test_campaign_lock_blocks_active_owner_and_recovers_stale():
    import json

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "runner.lock"
        with CampaignLock(path, "screen"):
            try:
                with CampaignLock(path, "inference"):
                    raise AssertionError("second runner unexpectedly acquired the lock")
            except RuntimeError as error:
                assert "already active" in str(error) and "PID" in str(error)
        assert json.loads(path.read_text())["released"] is True
        with CampaignLock(path, "inference"):
            assert json.loads(path.read_text())["phase"] == "inference"
        path.write_text(json.dumps({"pid": 2_147_483_647, "process_identity": [2_147_483_647, 1, 2],
                                    "phase": "screen", "released": False}))
        with CampaignLock(path, "screen"):
            assert json.loads(path.read_text())["pid"] == os.getpid()


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"{name}: passed")
