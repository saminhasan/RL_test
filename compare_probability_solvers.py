from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import multiprocessing as mp
from statistics import mean, median
from time import perf_counter
from typing import Callable, Iterator

import numpy as np

import bayes
import graph_solver
from engine import MinesweeperEngine


# No argparse by design: tweak these and run the file.
LEVEL = "hard"
NUM_STATES = 30
SEED = 44_000
WARMUP_STATES = 8
REPEATS = 1
COMPARE_TOL = 1e-9
CALL_TIMEOUT_SECONDS = 8.0

# Hard-board exact solvers can get wildly different costs depending on state.
# These are safe-click counts used only for generating benchmark snapshots.
SAFE_CLICK_BUCKETS = (
    ("early", 1, 12),
    ("middle", 13, 55),
    ("late", 56, 120),
)

RUN_BAYES_WITH_CYTHON_DISABLED = True


CYTHON_ATTRS = (
    "_cy_build_rule_specs",
    "_cy_has_compatible_any",
    "_cy_pick_smallest_rule",
    "_cy_propagate_cascades",
    "_cy_simplify_rule_specs",
)


@dataclass(frozen=True)
class SolverState:
    name: str
    phase: str
    rows: int
    cols: int
    mine_count: int
    started: bool
    hit_mine: bool
    exploded_cell: tuple[int, int] | None
    revealed: np.ndarray
    neighbor_counts: np.ndarray
    neighbors: tuple[tuple[tuple[int, int], ...], ...]
    move_count: int
    revealed_safe_cells: int
    total_safe_cells: int

    @property
    def revealed_fraction(self) -> float:
        return self.revealed_safe_cells / max(1, self.total_safe_cells)

    def solver_args(self) -> dict:
        return {
            "revealed": self.revealed,
            "neighbor_counts": self.neighbor_counts,
            "mine_count": self.mine_count,
            "started": self.started,
            "hit_mine": self.hit_mine,
            "exploded_cell": self.exploded_cell,
            "rows": self.rows,
            "cols": self.cols,
            "neighbors": self.neighbors,
        }


@dataclass
class DiffStats:
    states: int = 0
    mismatched_states: int = 0
    cells_over_tol: int = 0
    max_abs_diff: float = 0.0
    mean_abs_diffs: list[float] | None = None

    def __post_init__(self) -> None:
        if self.mean_abs_diffs is None:
            self.mean_abs_diffs = []


@dataclass
class SolverRun:
    times_by_phase: dict[str, list[float]]
    outputs: dict[int, np.ndarray]
    timeouts: list[int]
    errors: dict[int, str]


def _solve_in_worker(solver_name: str, args: dict) -> tuple[float, np.ndarray]:
    t0 = perf_counter()
    if solver_name == "bayes":
        out = bayes.mine_probabilities_for_engine(**args)
    elif solver_name == "bayes_no_cython":
        with bayes_cython_enabled(False):
            out = bayes.mine_probabilities_for_engine(**args)
    elif solver_name == "graph":
        out = graph_solver.mine_probabilities_for_engine(**args)
    else:
        raise ValueError(f"unknown solver {solver_name!r}")
    return perf_counter() - t0, out


@contextmanager
def bayes_cython_enabled(enabled: bool) -> Iterator[None]:
    old_values = {name: getattr(bayes, name) for name in CYTHON_ATTRS}
    if not enabled:
        for name in CYTHON_ATTRS:
            setattr(bayes, name, None)
    try:
        yield
    finally:
        for name, value in old_values.items():
            setattr(bayes, name, value)


def level_to_index(level: str) -> int:
    for idx, name in MinesweeperEngine.LEVELS.items():
        if name == level:
            return idx
    raise ValueError(f"unknown level {level!r}; choose one of {tuple(MinesweeperEngine.LEVELS.values())}")


def snapshot_engine(engine: MinesweeperEngine, phase: str, name: str) -> SolverState:
    return SolverState(
        name=name,
        phase=phase,
        rows=engine.rows,
        cols=engine.cols,
        mine_count=engine.mine_count,
        started=engine.started,
        hit_mine=engine.hit_mine,
        exploded_cell=engine.exploded_cell,
        revealed=engine.revealed.copy(),
        neighbor_counts=engine.get_neighbor_counts(),
        neighbors=engine._neighbors,
        move_count=engine.move_count,
        revealed_safe_cells=engine.revealed_safe_cells,
        total_safe_cells=engine.total_safe_cells,
    )


def generate_states(num_states: int = NUM_STATES, level: str = LEVEL, seed: int = SEED) -> list[SolverState]:
    rng = np.random.default_rng(seed)
    level_idx = level_to_index(level)
    states: list[SolverState] = []

    for state_id in range(num_states):
        phase, min_clicks, max_clicks = SAFE_CLICK_BUCKETS[state_id % len(SAFE_CLICK_BUCKETS)]
        target_clicks = int(rng.integers(min_clicks, max_clicks + 1))
        engine = MinesweeperEngine(level=level_idx, seed=seed + state_id, headless=True)

        # Center first click avoids giving either solver control over state generation.
        engine.reveal_count(engine.rows // 2, engine.cols // 2)

        for _ in range(target_clicks - 1):
            if engine.state == engine.OVER:
                break
            engine.random_reveal(safe=True)

        states.append(snapshot_engine(engine, phase, f"{phase}-{state_id:04d}"))

    return states


def bayes_solver(args: dict) -> np.ndarray:
    with bayes_cython_enabled(True):
        return bayes.mine_probabilities_for_engine(**args)


def bayes_no_cython_solver(args: dict) -> np.ndarray:
    with bayes_cython_enabled(False):
        return bayes.mine_probabilities_for_engine(**args)


def graph_solver_call(args: dict) -> np.ndarray:
    return graph_solver.mine_probabilities_for_engine(**args)


def finite_diff(a: np.ndarray, b: np.ndarray, tol: float) -> tuple[bool, int, float, float]:
    nan_match = np.array_equal(np.isnan(a), np.isnan(b))
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.any():
        diff = np.abs(a[finite] - b[finite])
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        over_tol = int(np.count_nonzero(diff > tol))
    else:
        max_diff = 0.0
        mean_diff = 0.0
        over_tol = 0
    return nan_match and over_tol == 0, over_tol, max_diff, mean_diff


def compare_outputs(
    states: list[SolverState],
    runs: dict[str, SolverRun],
    baseline_name: str = "bayes",
    tol: float = COMPARE_TOL,
) -> dict[str, DiffStats]:
    baseline_outputs = runs[baseline_name].outputs
    stats = {name: DiffStats() for name in runs if name != baseline_name}

    for state_idx, _state in enumerate(states):
        base = baseline_outputs.get(state_idx)
        if base is None:
            continue
        for name, run in runs.items():
            if name == baseline_name:
                continue
            out = run.outputs.get(state_idx)
            if out is None:
                continue
            ok, cells_over_tol, max_diff, mean_diff = finite_diff(base, out, tol)
            item = stats[name]
            item.states += 1
            item.cells_over_tol += cells_over_tol
            item.max_abs_diff = max(item.max_abs_diff, max_diff)
            item.mean_abs_diffs.append(mean_diff)
            if not ok:
                item.mismatched_states += 1

    return stats


def run_solver_with_timeouts(
    states: list[SolverState],
    solver_name: str,
    repeats: int = REPEATS,
    timeout_seconds: float = CALL_TIMEOUT_SECONDS,
) -> SolverRun:
    times_by_phase: dict[str, list[float]] = {"all": []}
    outputs: dict[int, np.ndarray] = {}
    timeouts: list[int] = []
    errors: dict[int, str] = {}
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=1)

    try:
        for repeat_idx in range(repeats):
            for state_idx, state in enumerate(states):
                async_result = pool.apply_async(_solve_in_worker, (solver_name, state.solver_args()))
                try:
                    elapsed, out = async_result.get(timeout=timeout_seconds)
                except mp.TimeoutError:
                    timeouts.append(state_idx)
                    pool.terminate()
                    pool.join()
                    pool = ctx.Pool(processes=1)
                    continue
                except Exception as exc:
                    errors[state_idx] = repr(exc)
                    continue

                if repeat_idx == 0:
                    outputs[state_idx] = out
                times_by_phase.setdefault(state.phase, []).append(elapsed)
                times_by_phase["all"].append(elapsed)
    finally:
        pool.close()
        pool.join()

    return SolverRun(times_by_phase=times_by_phase, outputs=outputs, timeouts=timeouts, errors=errors)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def format_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:9.3f} ms"


def print_timing_table(results: dict[str, SolverRun]) -> None:
    phases = ["all"] + [phase for phase, _, _ in SAFE_CLICK_BUCKETS]
    print("\ntiming:")
    for solver_name, run in results.items():
        by_phase = run.times_by_phase
        print(f"  {solver_name}:")
        for phase in phases:
            values = by_phase.get(phase, [])
            if not values:
                continue
            total = sum(values)
            print(
                f"    {phase:6s} n={len(values):4d} "
                f"total={total:8.3f}s avg={format_ms(mean(values))} "
                f"median={format_ms(median(values))} p90={format_ms(percentile(values, 90))} "
                f"p99={format_ms(percentile(values, 99))}"
            )
        if run.timeouts:
            print(f"    timeouts={len(run.timeouts)} at {CALL_TIMEOUT_SECONDS:.1f}s each")
        if run.errors:
            print(f"    errors={len(run.errors)}")

    base_run = results.get("bayes")
    base = base_run.times_by_phase.get("all", []) if base_run else []
    if base:
        base_avg = mean(base)
        print("\nspeed ratios vs bayes avg:")
        for solver_name, run in results.items():
            values = run.times_by_phase.get("all", [])
            if not values:
                continue
            avg = mean(values)
            ratio = avg / base_avg if base_avg else float("nan")
            print(f"  {solver_name:16s}: {ratio:7.3f}x bayes time")


def print_diff_stats(stats: dict[str, DiffStats]) -> None:
    print("\ncorrectness vs bayes:")
    for name, item in stats.items():
        mean_abs = mean(item.mean_abs_diffs) if item.mean_abs_diffs else 0.0
        print(
            f"  {name:16s}: states={item.states} mismatched={item.mismatched_states} "
            f"cells_over_tol={item.cells_over_tol} max_abs={item.max_abs_diff:.3e} "
            f"mean_abs={mean_abs:.3e}"
        )


def print_cython_status() -> None:
    print("cython status in bayes.py:")
    for name in CYTHON_ATTRS:
        value = getattr(bayes, name)
        print(f"  {name:26s}: {'yes' if value is not None else 'no'}")
    print("graph_solver.py uses cython : no")


def print_state_summary(states: list[SolverState]) -> None:
    print(f"\nstates: {len(states)}")
    for phase, _, _ in SAFE_CLICK_BUCKETS:
        phase_states = [s for s in states if s.phase == phase]
        if not phase_states:
            continue
        moves = [s.move_count for s in phase_states]
        revealed = [100.0 * s.revealed_fraction for s in phase_states]
        print(
            f"  {phase:6s}: n={len(phase_states):4d} "
            f"moves avg={mean(moves):6.2f} median={median(moves):6.2f} "
            f"safe revealed avg={mean(revealed):6.2f}%"
        )


def run_comparison(num_states: int = NUM_STATES, level: str = LEVEL, seed: int = SEED) -> None:
    print(f"solver comparison level={level} num_states={num_states} repeats={REPEATS}")
    print_cython_status()

    states = generate_states(num_states=num_states, level=level, seed=seed)
    print_state_summary(states)

    solver_names = ["bayes", "graph"]
    if RUN_BAYES_WITH_CYTHON_DISABLED:
        solver_names.append("bayes_no_cython")

    runs = {name: run_solver_with_timeouts(states, name) for name in solver_names}

    diff_stats = compare_outputs(states, runs)
    print_diff_stats(diff_stats)

    print_timing_table(runs)


def main() -> None:
    run_comparison()


if __name__ == "__main__":
    main()
