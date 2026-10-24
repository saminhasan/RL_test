"""Synthetic, CPU-only checks of all-decision risk diagnostics."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.policy_audit import decision_risk, summarize_audit


def test_safe_available_error_is_seen_even_if_it_survives():
    d = decision_risk(np.array([[np.nan, 0.0, 0.2]]), 2)
    assert d["safe_available_choice_error"] and d["riskier_than_safest"]
    assert not d["forced_guess"] and not d["riskier_forced_guess"]
    assert d["excess_mine_probability"] == 0.2


def test_forced_guess_and_coin_flip_are_distinct():
    d = decision_risk(np.array([0.1, 0.3, np.nan]), 1)
    assert d["forced_guess"] and d["riskier_forced_guess"] and not d["coin_flip"]
    d = decision_risk(np.array([0.5, 0.5]), 0)
    assert d["forced_guess"] and d["coin_flip"] and not d["riskier_than_safest"]
    d = decision_risk(np.array([0.2, 0.2]), 1)
    assert d["coin_flip"]


def test_revealed_or_offboard_action_rejected():
    for action in (0, -1, 2):
        try:
            decision_risk(np.array([np.nan, 0.1]), action)
        except ValueError:
            pass
        else:
            raise AssertionError("expected invalid selected action to fail")


def test_per_board_and_per_decision_denominators():
    bad = {**decision_risk(np.array([0.0, 0.2]), 1), "progress": 0.5}
    good = {**decision_risk(np.array([0.0, 0.2]), 0), "progress": 0.6}
    guess = {**decision_risk(np.array([0.3, 0.3]), 0), "progress": 0.9}
    s = summarize_audit([{"won": True, "decisions": [bad, good]}, {"won": False, "decisions": [guess]}])
    assert s["policy_decisions"] == 3 and s["games"] == 2
    assert s["counts"]["safe_available_choice_error"] == 1
    assert s["rates_per_decision"]["safe_available_choice_error"] == 1 / 3
    assert s["mean_counts_per_board"]["safe_available_choice_error"] == 0.5
    assert s["games_with_safe_available_choice_error"] == 1
    assert np.isclose(s["mean_excess_mine_probability_per_board"], 0.1)
    assert s["mean_forced_guess_progress"] == 0.9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: passed")
