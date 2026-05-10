from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


RuleSpec = Tuple[Tuple[int, ...], int]
Neighbors = Tuple[Tuple[Tuple[int, int], ...], ...]


@dataclass
class ComponentResult:
    vars: Tuple[int, ...]
    counts: Dict[int, int]
    mine_sums: Dict[int, List[int]]


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)

    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :] |= mask[:-1, :]
    out[1:, :-1] |= mask[:-1, 1:]

    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]

    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :] |= mask[1:, :]
    out[:-1, :-1] |= mask[1:, 1:]

    return out


@lru_cache(maxsize=32)
def _neighbors_flat(rows: int, cols: int, neighbors: Neighbors) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(r * cols + c for r, c in ns) for ns in neighbors)


def _fallback(probabilities: np.ndarray, mask: np.ndarray, mines: int) -> np.ndarray:
    n = int(mask.sum())
    if n:
        probabilities[mask] = float(np.clip(mines / n, 0.0, 1.0))
    return probabilities


def _build_rules(
    revealed: np.ndarray,
    hidden_mask: np.ndarray,
    neighbor_counts: np.ndarray,
    rows: int,
    cols: int,
    neighbors: Neighbors,
    known_mine_flat: Optional[int],
) -> Tuple[List[RuleSpec], bool, np.ndarray]:
    hidden_flat = np.flatnonzero(hidden_mask.ravel())
    tag_map = np.full(rows * cols, -1, dtype=np.int32)
    tag_map[hidden_flat] = np.arange(hidden_flat.size, dtype=np.int32)

    hidden_flat_mask = hidden_mask.ravel()
    revealed_flat = revealed.ravel()
    counts_flat = neighbor_counts.ravel()
    neigh_flat = _neighbors_flat(rows, cols, neighbors)

    rule_map: Dict[Tuple[int, ...], int] = {}
    frontier = np.flatnonzero((_dilate8(hidden_mask) & revealed).ravel())

    for clue_flat in frontier:
        clue_flat = int(clue_flat)
        if known_mine_flat is not None and clue_flat == known_mine_flat:
            continue

        rhs = int(counts_flat[clue_flat])
        vars_idx: List[int] = []

        for n_flat in neigh_flat[clue_flat]:
            if hidden_flat_mask[n_flat]:
                vars_idx.append(int(tag_map[n_flat]))
            elif known_mine_flat is not None and n_flat == known_mine_flat:
                rhs -= 1

        if vars_idx:
            rhs = max(0, min(rhs, len(vars_idx)))
            key = tuple(vars_idx)
            old = rule_map.get(key)
            if old is None:
                rule_map[key] = rhs
            elif old != rhs:
                return [], True, hidden_flat
        elif rhs != 0:
            return [], True, hidden_flat

    return list(rule_map.items()), False, hidden_flat


def _simplify_rules(rule_specs: List[RuleSpec]) -> Tuple[List[RuleSpec], Dict[int, int], bool]:
    fixed: Dict[int, int] = {}
    rule_map: Dict[Tuple[int, ...], int] = {}

    for vars_idx, rhs in rule_specs:
        key = tuple(sorted(vars_idx))
        old = rule_map.get(key)
        if old is None:
            rule_map[key] = int(rhs)
        elif old != rhs:
            return [], {}, True

    while True:
        changed = False
        normalized: Dict[Tuple[int, ...], int] = {}

        for vars_idx, rhs0 in rule_map.items():
            rhs = int(rhs0)
            remaining: List[int] = []
            fixed_mines = 0

            for v in vars_idx:
                val = fixed.get(v)
                if val is None:
                    remaining.append(v)
                else:
                    fixed_mines += val

            rhs -= fixed_mines
            n = len(remaining)

            if rhs < 0 or rhs > n:
                return [], {}, True
            if n == 0:
                if rhs != 0:
                    return [], {}, True
                continue
            if rhs == 0 or rhs == n:
                val = 1 if rhs == n else 0
                for v in remaining:
                    old = fixed.get(v)
                    if old is None:
                        fixed[v] = val
                        changed = True
                    elif old != val:
                        return [], {}, True
                continue

            key = tuple(remaining)
            old = normalized.get(key)
            if old is None:
                normalized[key] = rhs
            elif old != rhs:
                return [], {}, True

        rule_map = normalized
        keys = sorted(rule_map.keys(), key=len)

        inferred: Dict[Tuple[int, ...], int] = {}
        key_sets = [set(k) for k in keys]

        for i, small in enumerate(keys):
            small_set = key_sets[i]
            rhs_small = rule_map[small]
            len_small = len(small)

            for j in range(i + 1, len(keys)):
                big = keys[j]
                if len_small >= len(big):
                    continue

                big_set = key_sets[j]
                if not small_set.issubset(big_set):
                    continue

                diff = tuple(v for v in big if v not in small_set)
                rhs_diff = rule_map[big] - rhs_small

                if rhs_diff < 0 or rhs_diff > len(diff):
                    return [], {}, True
                if not diff:
                    if rhs_diff != 0:
                        return [], {}, True
                    continue

                old = rule_map.get(diff)
                if old is not None:
                    if old != rhs_diff:
                        return [], {}, True
                    continue

                old = inferred.get(diff)
                if old is None:
                    inferred[diff] = rhs_diff
                    changed = True
                elif old != rhs_diff:
                    return [], {}, True

        if inferred:
            rule_map.update(inferred)

        if not changed:
            break

    return list(rule_map.items()), fixed, False


def _remap_rules(rule_specs: List[RuleSpec], unresolved_tags: np.ndarray, hidden_count: int) -> Tuple[List[RuleSpec], np.ndarray]:
    if len(unresolved_tags) == hidden_count:
        return rule_specs, unresolved_tags

    remap = np.full(hidden_count, -1, dtype=np.int32)
    remap[unresolved_tags] = np.arange(unresolved_tags.size, dtype=np.int32)

    out: List[RuleSpec] = []
    for vars_idx, rhs in rule_specs:
        out.append((tuple(int(remap[v]) for v in vars_idx), int(rhs)))
    return out, unresolved_tags


def _split_components(rule_specs: List[RuleSpec], var_count: int) -> Tuple[List[Tuple[Tuple[int, ...], List[RuleSpec]]], List[int]]:
    var_to_rules: List[List[int]] = [[] for _ in range(var_count)]
    constrained = set()

    for ri, (vars_idx, _) in enumerate(rule_specs):
        for v in vars_idx:
            constrained.add(v)
            var_to_rules[v].append(ri)

    seen = set()
    components: List[Tuple[Tuple[int, ...], List[RuleSpec]]] = []

    for start in sorted(constrained):
        if start in seen:
            continue

        q = deque([start])
        seen.add(start)
        comp_vars = set([start])
        comp_rule_ids = set()

        while q:
            v = q.popleft()
            for ri in var_to_rules[v]:
                if ri in comp_rule_ids:
                    continue
                comp_rule_ids.add(ri)
                for u in rule_specs[ri][0]:
                    if u not in seen:
                        seen.add(u)
                        comp_vars.add(u)
                        q.append(u)

        comp_vars_tuple = tuple(sorted(comp_vars))
        comp_rules = [rule_specs[ri] for ri in sorted(comp_rule_ids)]
        components.append((comp_vars_tuple, comp_rules))

    other = [v for v in range(var_count) if v not in constrained]
    return components, other


def _solve_component(comp_vars: Tuple[int, ...], comp_rules: List[RuleSpec], mine_limit: int) -> ComponentResult:
    glob_to_local = {g: i for i, g in enumerate(comp_vars)}
    rules = [(tuple(glob_to_local[v] for v in vars_idx), int(rhs)) for vars_idx, rhs in comp_rules]

    n = len(comp_vars)
    rule_sizes = [len(vs) for vs, _ in rules]
    rule_mines = [0] * len(rules)
    rule_assigned = [0] * len(rules)
    var_to_rules: List[List[int]] = [[] for _ in range(n)]

    for ri, (vars_idx, _) in enumerate(rules):
        for v in vars_idx:
            var_to_rules[v].append(ri)

    order = sorted(range(n), key=lambda v: (-len(var_to_rules[v]), v))
    assigned = [-1] * n
    counts: Dict[int, int] = defaultdict(int)
    mine_sums: Dict[int, List[int]] = {}

    def valid_after(touched: Iterable[int]) -> bool:
        for ri in touched:
            rhs = rules[ri][1]
            mines = rule_mines[ri]
            left = rule_sizes[ri] - rule_assigned[ri]
            if mines > rhs or mines + left < rhs:
                return False
        return True

    def rec(pos: int, mines_so_far: int) -> None:
        if mines_so_far > mine_limit:
            return
        if pos == n:
            for ri, (_, rhs) in enumerate(rules):
                if rule_mines[ri] != rhs:
                    return
            counts[mines_so_far] += 1
            sums = mine_sums.get(mines_so_far)
            if sums is None:
                sums = [0] * n
                mine_sums[mines_so_far] = sums
            for li, val in enumerate(assigned):
                if val == 1:
                    sums[li] += 1
            return

        v = order[pos]
        touched = var_to_rules[v]

        for val in (0, 1):
            assigned[v] = val
            for ri in touched:
                rule_assigned[ri] += 1
                rule_mines[ri] += val

            if valid_after(touched):
                rec(pos + 1, mines_so_far + val)

            for ri in touched:
                rule_assigned[ri] -= 1
                rule_mines[ri] -= val
            assigned[v] = -1

    rec(0, 0)

    if not counts:
        raise ValueError("component has no valid assignments")

    return ComponentResult(comp_vars, dict(counts), mine_sums)


def _conv(a: Dict[int, int], b: Dict[int, int], limit: int) -> Dict[int, int]:
    out: Dict[int, int] = defaultdict(int)
    for ka, va in a.items():
        for kb, vb in b.items():
            k = ka + kb
            if k <= limit:
                out[k] += va * vb
    return dict(out)


def _context_count(prefix: Dict[int, int], suffix: Dict[int, int], target: int) -> int:
    total = 0
    for k, v in prefix.items():
        total += v * suffix.get(target - k, 0)
    return total


def _solve_graph(rule_specs: List[RuleSpec], var_count: int, total_mines: int) -> Dict[Optional[int], float]:
    comp_specs, other_vars = _split_components(rule_specs, var_count)
    comps = [_solve_component(vars_idx, rules, total_mines) for vars_idx, rules in comp_specs]

    other_count = len(other_vars)
    other_poly = {k: math.comb(other_count, k) for k in range(min(other_count, total_mines) + 1)}
    units = [c.counts for c in comps] + [other_poly]

    prefix: List[Dict[int, int]] = [{0: 1}]
    for poly in units:
        prefix.append(_conv(prefix[-1], poly, total_mines))

    suffix: List[Dict[int, int]] = [{} for _ in range(len(units) + 1)]
    suffix[-1] = {0: 1}
    for i in range(len(units) - 1, -1, -1):
        suffix[i] = _conv(units[i], suffix[i + 1], total_mines)

    total_configs = prefix[-1].get(total_mines, 0)
    if total_configs == 0:
        raise ValueError("global mine count has no valid assignments")

    out: Dict[Optional[int], float] = {}

    for ci, comp in enumerate(comps):
        numer = [0] * len(comp.vars)
        for k, count in comp.counts.items():
            context = _context_count(prefix[ci], suffix[ci + 1], total_mines - k)
            if context == 0:
                continue
            sums = comp.mine_sums[k]
            for li, s in enumerate(sums):
                numer[li] += s * context

        for li, g in enumerate(comp.vars):
            out[g] = numer[li] / total_configs

    if other_count:
        oi = len(comps)
        mines_numer = 0
        for k, count in other_poly.items():
            context = _context_count(prefix[oi], suffix[oi + 1], total_mines - k)
            mines_numer += k * count * context
        p_other = mines_numer / total_configs / other_count
        for v in other_vars:
            out[v] = p_other
        out[None] = p_other
    else:
        out[None] = 0.0

    return out


def mine_probabilities_for_engine(
    revealed: np.ndarray,
    neighbor_counts: np.ndarray,
    mine_count: int,
    started: bool,
    hit_mine: bool,
    exploded_cell: Optional[Tuple[int, int]],
    rows: int,
    cols: int,
    neighbors: Neighbors,
) -> np.ndarray:
    """
    Exact factor-graph Minesweeper probability solver.

    Drop-in replacement for bayes.mine_probabilities_for_engine().
    Revealed cells are np.nan. Hidden cells get mine probabilities in [0, 1].
    """
    probabilities = np.full((rows, cols), np.nan, dtype=np.float64)
    hidden_mask = ~revealed

    if not hidden_mask.any():
        return probabilities

    hidden_count = int(hidden_mask.sum())
    if not started:
        probabilities[hidden_mask] = mine_count / hidden_count
        return probabilities

    known_mines = 1 if hit_mine and exploded_cell is not None else 0
    mines_remaining = max(0, int(mine_count) - known_mines)
    known_mine_flat = None if exploded_cell is None else exploded_cell[0] * cols + exploded_cell[1]

    rule_specs, inconsistent, hidden_flat = _build_rules(
        revealed,
        hidden_mask,
        neighbor_counts,
        rows,
        cols,
        neighbors,
        known_mine_flat if hit_mine else None,
    )

    if inconsistent:
        return _fallback(probabilities, hidden_mask, mines_remaining)
    if not rule_specs:
        return _fallback(probabilities, hidden_mask, mines_remaining)

    rule_specs, fixed, inconsistent = _simplify_rules(rule_specs)
    if inconsistent:
        return _fallback(probabilities, hidden_mask, mines_remaining)

    flat_probs = probabilities.ravel()
    unresolved_mask = np.ones(hidden_count, dtype=bool)
    fixed_mines = 0

    for tag, val in fixed.items():
        flat_probs[int(hidden_flat[tag])] = float(val)
        unresolved_mask[tag] = False
        fixed_mines += int(val)

    mines_remaining = max(0, mines_remaining - fixed_mines)
    unresolved_tags = np.flatnonzero(unresolved_mask)
    unresolved_count = int(unresolved_tags.size)

    if unresolved_count == 0:
        return probabilities
    if not rule_specs:
        base = float(np.clip(mines_remaining / unresolved_count, 0.0, 1.0))
        flat_probs[hidden_flat[unresolved_tags]] = base
        return probabilities

    remapped_specs, solver_tag_to_old_tag = _remap_rules(rule_specs, unresolved_tags, hidden_count)
    solver_tag_to_flat = hidden_flat[solver_tag_to_old_tag]

    try:
        solved = _solve_graph(remapped_specs, unresolved_count, mines_remaining)
    except Exception:
        base = float(np.clip(mines_remaining / unresolved_count, 0.0, 1.0))
        flat_probs[solver_tag_to_flat] = base
        return probabilities

    default_p = float(solved.get(None, 0.0))
    flat_probs[solver_tag_to_flat] = default_p

    for tag, p in solved.items():
        if tag is None:
            continue
        t = int(tag)
        if 0 <= t < unresolved_count:
            flat_probs[int(solver_tag_to_flat[t])] = float(p)

    return probabilities
