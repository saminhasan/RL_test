"""Run frozen study modules with retries for Windows atomic-write sharing locks.

    python -m rl.study_io reward --phase screen
    python -m rl.study_io auxiliary --wait-for-screen
    python -m rl.study_io confirm --wait-for-sweeps

Only PermissionError during JSON publication is retried. This wrapper leaves
the archived training/evaluation source files, objectives and seeds unchanged.
Its activation and recovered errors are retained separately for provenance.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

TARGETS = {"reward": "rl.reward_sweep", "auxiliary": "rl.auxiliary_sweep",
           "confirm": "rl.confirm_study", "evaluate": "rl.experiment_eval",
           "opening": "rl.safety_opening", "gated": "rl.opening_policy_eval"}


def retry_writer(writer, *, attempts=40, pause=time.sleep, observer=None):
    """Retry the same data; genuine failures still propagate after a bounded wait."""
    def wrapped(path, data):
        for attempt in range(attempts):
            try:
                return writer(path, data)
            except PermissionError as error:
                if attempt + 1 == attempts:
                    raise
                if observer is not None:
                    observer(path, attempt + 1, error)
                pause(0.1 if attempt < 10 else 0.25)
    wrapped._study_io_retry = True
    return wrapped


def install_retry(*, observer=None):
    evaluation = importlib.import_module("rl.experiment_eval")
    original = evaluation.write_json
    if getattr(original, "_study_io_retry", False):
        return original
    wrapped = retry_writer(original, observer=observer)
    # Several frozen runners imported this function by name. Preserve those
    # modules' source bytes while replacing their exact writer alias in memory.
    for module in list(sys.modules.values()):
        if module is not None and getattr(module, "__name__", "").startswith("rl."):
            if getattr(module, "write_json", None) is original:
                module.write_json = wrapped
    return wrapped


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in TARGETS:
        raise SystemExit("usage: python -m rl.study_io {" + ",".join(TARGETS) + "} [original module arguments]")
    target, arguments = sys.argv[1], sys.argv[2:]
    root = Path(__file__).resolve().parent.parent
    log = root / "experiments" / "reward-study-20261024" / "io-supervision.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)

    def event(kind, **fields):
        try:
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"kind": kind, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "pid": os.getpid(), "target": target, **fields}) + "\n")
        except OSError:
            pass  # Diagnostic logging must not prevent a recovered publication.

    module = importlib.import_module(TARGETS[target])
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    event("start", wrapper_sha256=digest, argv=arguments,
          scope="JSON PermissionError retries only; frozen algorithm source unchanged")
    install_retry(observer=lambda path, attempt, error: event("retry", path=str(path), attempt=attempt,
                                                             error=str(error)))
    sys.argv = [TARGETS[target], *arguments]
    try:
        module.main()
    except BaseException as error:
        event("failed", error=repr(error))
        raise
    else:
        event("complete")


if __name__ == "__main__":
    main()
