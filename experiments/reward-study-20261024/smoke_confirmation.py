"""Exercise the confirmation training path without opening any test boards."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rl.confirm_study import Confirmation, source_inventory, write_selection_plan
from rl.study_io import install_retry


def main():
    install_retry()
    study = Path(__file__).resolve().parent
    manifest = json.loads((study / "screen/old_precision_control-s101-n500000-v2000/manifest.json").read_text())
    config = {**manifest["protocol"]["config"], "eval_games": 20, "learning_starts": 64}
    winner = {"kind": "q", "name": "training-path-smoke", "config": config,
              "checkpoint_label": "latest", "init_sha256": manifest["protocol"]["init_sha256"],
              "inference": {"no_bf16": False, "symmetries": False, "fp32_head": False}}
    driver = Confirmation(SimpleNamespace(study_dir=study / "confirmation-training-smoke", device="cuda", jobs=2))
    payload = {"smoke_only": True, "source_sha256": source_inventory(), "winner": winner,
               "seeds": [900], "training_clicks": 128, "test_games": 10000,
               "test_split": "closed; this script calls training only"}
    driver.plan = write_selection_plan(driver.directory / "selection-plan.json", payload)
    frozen = driver.training(900)
    print(json.dumps({"training_path_completed": frozen, "test_boards_opened": 0}))


if __name__ == "__main__":
    main()
