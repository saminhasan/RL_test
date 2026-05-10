# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False

import numpy as np
cimport numpy as np


def build_rule_specs(
    np.ndarray[np.uint8_t, ndim=1] revealed_flat,
    np.ndarray[np.uint8_t, ndim=1] hidden_mask_flat,
    np.ndarray[np.uint8_t, ndim=1] neighbor_counts_flat,
    np.ndarray[np.int32_t, ndim=2] neighbors_idx,
    np.ndarray[np.int32_t, ndim=1] neighbors_len,
    int known_mine_flat,
    np.ndarray[np.int32_t, ndim=1] tag_map,
):
    cdef Py_ssize_t n_cells = revealed_flat.shape[0]
    cdef Py_ssize_t clue_flat
    cdef Py_ssize_t k
    cdef int n_flat
    cdef int rhs
    cdef int n_vars
    cdef int n_neigh
    cdef list vars_idx
    cdef tuple key
    cdef dict rule_map = {}
    cdef object prev_rhs

    for clue_flat in range(n_cells):
        if revealed_flat[clue_flat] == 0:
            continue
        if known_mine_flat >= 0 and clue_flat == known_mine_flat:
            continue

        rhs = neighbor_counts_flat[clue_flat]
        vars_idx = []
        n_neigh = neighbors_len[clue_flat]

        for k in range(n_neigh):
            n_flat = neighbors_idx[clue_flat, k]
            if hidden_mask_flat[n_flat] != 0:
                vars_idx.append(<int>tag_map[n_flat])
            elif known_mine_flat >= 0 and n_flat == known_mine_flat:
                rhs -= 1

        n_vars = len(vars_idx)
        if n_vars != 0:
            if rhs < 0:
                rhs = 0
            elif rhs > n_vars:
                rhs = n_vars

            key = tuple(vars_idx)
            prev_rhs = rule_map.get(key)
            if prev_rhs is None:
                rule_map[key] = rhs
            elif <int>prev_rhs != rhs:
                return [], True
        elif rhs != 0:
            return [], True

    return list(rule_map.items()), False


def has_compatible_any(permus, permu):
    cdef object p
    for p in permus:
        if p.compatible(permu):
            return True
    return False


def pick_smallest_rule(free):
    cdef object rule
    cdef object permus
    cdef object best_rule = None
    cdef Py_ssize_t best_len = 9223372036854775807
    cdef Py_ssize_t n

    for rule, permus in free.items():
        n = len(permus)
        if n < best_len:
            best_len = n
            best_rule = rule
    return best_rule


def propagate_cascades(free, fixed, overlapping_rules, allowed_permus, rule, permu):
    cdef list cascades = [(rule, permu)]
    cdef object curr_rule
    cdef object curr_permu
    cdef object related_rule
    cdef object linked_permus
    cdef object allowed
    cdef int n

    while cascades:
        curr_rule, curr_permu = cascades.pop()
        linked_permus = free.get(curr_rule)
        if linked_permus is None:
            continue

        fixed.add(curr_permu)
        del free[curr_rule]

        for related_rule in overlapping_rules(curr_rule):
            linked_permus = free.get(related_rule)
            if linked_permus is None:
                continue

            allowed = allowed_permus(curr_permu, related_rule)
            linked_permus.intersection_update(allowed)

            n = len(linked_permus)
            if n == 0:
                return False
            if n == 1:
                cascades.append((related_rule, next(iter(linked_permus))))

    return True


def simplify_rule_specs(rule_specs):
    cdef dict fixed_values = {}
    cdef dict rule_map = {}
    cdef dict normalized
    cdef dict inferred
    cdef tuple vars_idx
    cdef tuple key
    cdef tuple small
    cdef tuple big
    cdef tuple diff
    cdef object rhs_obj
    cdef object prev_rhs_obj
    cdef object fixed_val_obj
    cdef int rhs
    cdef int prev_rhs
    cdef int fixed_mines
    cdef int n_vars
    cdef int forced_val
    cdef int rhs_small
    cdef int len_small
    cdef int rhs_diff
    cdef bint changed
    cdef list remaining
    cdef list keys
    cdef list key_sets
    cdef int i
    cdef int j

    for vars_idx, rhs_obj in rule_specs:
        rhs = int(rhs_obj)
        key = tuple(sorted(vars_idx))
        prev_rhs_obj = rule_map.get(key)
        if prev_rhs_obj is None:
            rule_map[key] = rhs
        elif int(prev_rhs_obj) != rhs:
            return [], {}, True

    while True:
        changed = False
        normalized = {}

        for vars_idx, rhs_obj in rule_map.items():
            rhs = int(rhs_obj)
            if not vars_idx:
                if rhs != 0:
                    return [], {}, True
                continue

            remaining = []
            fixed_mines = 0
            for i in range(len(vars_idx)):
                fixed_val_obj = fixed_values.get(vars_idx[i])
                if fixed_val_obj is None:
                    remaining.append(vars_idx[i])
                else:
                    fixed_mines += int(fixed_val_obj)

            rhs -= fixed_mines
            n_vars = len(remaining)
            if rhs < 0 or rhs > n_vars:
                return [], {}, True

            if n_vars == 0:
                if rhs != 0:
                    return [], {}, True
                continue

            if rhs == 0 or rhs == n_vars:
                forced_val = 1 if rhs == n_vars else 0
                for i in range(n_vars):
                    prev_rhs_obj = fixed_values.get(remaining[i])
                    if prev_rhs_obj is None:
                        fixed_values[remaining[i]] = forced_val
                        changed = True
                    elif int(prev_rhs_obj) != forced_val:
                        return [], {}, True
                continue

            key = tuple(remaining)
            prev_rhs_obj = normalized.get(key)
            if prev_rhs_obj is None:
                normalized[key] = rhs
            elif int(prev_rhs_obj) != rhs:
                return [], {}, True

        rule_map = normalized

        keys = sorted(rule_map.keys(), key=len)
        if keys:
            key_sets = [set(k) for k in keys]
            inferred = {}

            for i in range(len(keys)):
                small = keys[i]
                rhs_small = int(rule_map[small])
                len_small = len(small)

                for j in range(i + 1, len(keys)):
                    big = keys[j]
                    if len_small >= len(big):
                        continue

                    if not key_sets[i].issubset(key_sets[j]):
                        continue

                    diff = tuple(v for v in big if v not in key_sets[i])
                    rhs_diff = int(rule_map[big]) - rhs_small

                    if rhs_diff < 0 or rhs_diff > len(diff):
                        return [], {}, True

                    if not diff:
                        if rhs_diff != 0:
                            return [], {}, True
                        continue

                    prev_rhs_obj = rule_map.get(diff)
                    if prev_rhs_obj is not None:
                        if int(prev_rhs_obj) != rhs_diff:
                            return [], {}, True
                        continue

                    prev_rhs_obj = inferred.get(diff)
                    if prev_rhs_obj is None:
                        inferred[diff] = rhs_diff
                        changed = True
                    elif int(prev_rhs_obj) != rhs_diff:
                        return [], {}, True

            if inferred:
                rule_map.update(inferred)

        if not changed:
            break

    return list(rule_map.items()), fixed_values, False
