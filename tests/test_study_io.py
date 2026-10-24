"""Bounded publication retries do not change data or hide permanent failures."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rl.study_io import retry_writer


def test_transient_sharing_errors_retry_the_same_object():
    data, calls, waits = {"win": 0.9}, [], []
    def writer(path, value):
        assert value is data
        calls.append(path)
        if len(calls) < 3:
            raise PermissionError("sharing violation")
        return "saved"
    assert retry_writer(writer, pause=waits.append)("result.json", data) == "saved"
    assert calls == ["result.json"] * 3 and waits == [0.1, 0.1]


def test_permanent_or_unrelated_errors_propagate():
    for error in (PermissionError("permanent"), ValueError("bad data")):
        calls = []
        def writer(path, value):
            calls.append(path)
            raise error
        try:
            retry_writer(writer, attempts=3, pause=lambda _: None)("result.json", {})
        except type(error):
            assert len(calls) == (3 if isinstance(error, PermissionError) else 1)
        else:
            raise AssertionError("write failure was hidden")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: passed")
