"""Refresh the local study report without loading torch or interrupting training."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parent


def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def pct(value):
    return f"{100 * value:.2f}%" if value is not None else "pending"


def main():
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    current = load(STUDY / "status.json") or {}
    auxiliary_current = load(STUDY / "auxiliary-status.json") or {}
    progress = current.get("progress") or {}
    inference = []
    for path in sorted(STUDY.glob("baseline-validation-*.json")):
        result = load(path)
        if result and result.get("models"):
            model = result["models"][0]
            inference.append({"name": path.stem.removeprefix("baseline-validation-"),
                              "win": model["summary"]["win"],
                              "efficiency": model["summary"]["efficiency"],
                              "paired": model["paired_vs_bayes"], "result": str(path.relative_to(STUDY))})
    candidates = []
    for path in sorted(STUDY.glob("screen/*/manifest.json")):
        manifest = load(path)
        if not manifest:
            continue
        item = {"name": manifest["protocol"]["name"], "state": manifest["state"],
                "run": manifest.get("run"), "models": []}
        result_path = manifest.get("result")
        result = load(Path(result_path)) if result_path else None
        if result:
            for model in result["models"]:
                item["models"].append({"checkpoint": Path(model["path"]).stem,
                                       "step": model["training_step"],
                                       "win": model["summary"]["win"],
                                       "efficiency": model["summary"]["efficiency"],
                                       "paired": model["paired_vs_bayes"]})
            item["result"] = str(Path(result_path).relative_to(STUDY))
        candidates.append(item)
    auxiliary = []
    for path in sorted(STUDY.glob("auxiliary/*/manifest.json")):
        manifest = load(path)
        if not manifest:
            continue
        item = {"name": manifest["protocol"]["name"], "state": manifest["state"], "models": []}
        for label in ("best", "latest"):
            artifact = path.parent / f"{label}-validation.json"
            result = load(artifact)
            if not result or not result.get("policies"):
                continue
            winner = max(result["policies"], key=lambda p: (p["summary"]["win"], p["summary"]["efficiency"]))
            item["models"].append({"checkpoint": label, "lambda": winner["opening_lambda"],
                "win": winner["summary"]["win"], "efficiency": winner["summary"]["efficiency"],
                "paired": winner["paired_vs_bayes"], "result": str(artifact.relative_to(STUDY)),
                "policy_grid": [{"lambda": p["opening_lambda"], "win": p["summary"]["win"],
                                 "efficiency": p["summary"]["efficiency"]} for p in result["policies"]]})
        auxiliary.append(item)
    summary = {"updated_utc": now, "current": {"variant": current.get("variant"),
               "state": current.get("state"), "step": progress.get("step"),
               "total_steps": progress.get("total_steps")},
               "inference": inference, "candidates": candidates,
               "auxiliary_current": auxiliary_current, "auxiliary": auxiliary,
               "test_split_opened": False,
               "success": False,
               "interpretation": "Validation is for selection; no repeatable win-rate advantage is established."}
    lines = ["# Reward study results", "", f"Updated {now}.", "",
             "Baseline code checkpoint: `13e3083`, tag `reward-study-baseline-20261024`.", "",
             "**No repeatable win-rate advantage over Bayes has been established.** "
             "All numbers below are validation results; the final test split is still unused.", ""]
    if current:
        lines += [f"Current: **{current.get('variant')}**, {current.get('state')}, "
                  f"{progress.get('step', 0):,} / {progress.get('total_steps', 0):,} clicks.", ""]
    if auxiliary_current.get("state") == "waiting_for_screen":
        lines += ["The six-case safety/opening batch is queued behind the reward screen.", ""]
    if inference:
        lines += ["## Frozen model: inference ablations", "",
                  "Same 2,000 validation boards; Bayes wins 88.60% with efficiency 0.936.", "",
                  "| Inference | Wins | Efficiency | Paired win gap vs Bayes | 95% interval |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for item in sorted(inference, key=lambda x: -x["win"]):
            paired = item["paired"]
            low, high = paired["win_difference_ci95"]
            lines.append(f"| [{item['name']}]({item['result']}) | {pct(item['win'])} | "
                         f"{item['efficiency']:.3f} | {paired['win_difference'] * 100:+.2f} points | "
                         f"[{low * 100:+.2f}, {high * 100:+.2f}] |")
        lines += [""]
    lines += ["## Equal-budget training screen", "",
              "13 recipes, each warm-started from the same frozen model and trained for "
              "500,000 additional clicks with seed 101. Best checkpoints are selected on "
              "500 validation boards, then best and latest are measured on the same "
              "2,000-board validation set. These comparisons are exploratory.", "",
              "| Recipe | State | Checkpoint | Wins | Efficiency | Gap vs Bayes |",
              "| --- | --- | --- | ---: | ---: | ---: |"]
    for item in candidates:
        if not item["models"]:
            lines.append(f"| {item['name']} | {item['state']} | pending | pending | pending | pending |")
        for model in item["models"]:
            lines.append(f"| [{item['name']}]({item['result']}) | {item['state']} | "
                         f"{model['checkpoint']} at {model['step']:,} | {pct(model['win'])} | "
                         f"{model['efficiency']:.3f} | {model['paired']['win_difference'] * 100:+.2f} points |")
    if auxiliary:
        lines += ["", "## Separate safety/opening heads and public consistency", "",
                  "Each checkpoint's displayed lambda is selected from its validation grid. "
                  "This is additional validation selection, not a held-out success claim. "
                  "Complete grids remain in each linked result.", "",
                  "| Recipe | State | Checkpoint | Selected lambda | Wins | Efficiency | Gap vs Bayes |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
        for item in auxiliary:
            if not item["models"]:
                lines.append(f"| {item['name']} | {item['state']} | pending | pending | pending | pending | pending |")
            for model in item["models"]:
                lines.append(f"| [{item['name']}]({model['result']}) | {item['state']} | "
                             f"{model['checkpoint']} | {model['lambda']:g} | {pct(model['win'])} | "
                             f"{model['efficiency']:.3f} | {model['paired']['win_difference'] * 100:+.2f} points |")
    lines += ["", "## Interpretation", "",
              "Changing the immediate reward ratio alone cannot train future-win predictions "
              "when gamma is zero. The screen also includes terminal win/loss objectives with "
              "gamma near one. Higher precision alone did not close the measured gap.", "",
              "A successful recipe must be frozen, repeated with fresh training seeds, and "
              "confirmed on the reserved test boards. Validation p-values are not evidence "
              "of repeatable success after searching many variants.", "",
              "See [protocol and commands](README.md). Create `STOP` in this directory "
              "to stop the study cleanly; this does not target the original easy-exp2 run.", ""]
    (STUDY / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (STUDY / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"current": summary["current"], "completed": sum(c["state"] == "complete" for c in candidates),
                      "inference": [{k: r[k] for k in ("name", "win", "efficiency")} for r in inference]}))


if __name__ == "__main__":
    main()
