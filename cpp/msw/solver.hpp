// Exact Bayesian mine probabilities for reveal-only Minesweeper.
//
// C++ port of the old bayes.py, which was built on mrgriscom's "minesweepr"
// solver. Pipeline:
//
//   1. every revealed clue becomes a rule: "these hidden cells hold N mines"
//   2. cheap simplification to a fixed point: forced cells (N == 0 or
//      N == size) and subset differences
//   3. supercells: cells that appear in exactly the same rules are merged
//   4. logical reduction: subtract sub-rules from super-rules
//   5. per-rule mine permutations, cross-elimination between overlapping
//      rules, and splitting of rules whose permutations factor
//   6. independent fronts (connected groups of rules), each enumerated
//      exhaustively and tallied by mine count
//   7. fronts are weighted against each other and against the cells no clue
//      touches, under the fixed total mine count
//
// The result is exact, not an estimate.
//
// No exceptions anywhere in this file: under cppyy's JIT on Windows any C++
// throw aborts the process, even one that is caught. Inconsistent input is
// reported by returning false, and the caller falls back to a uniform fill,
// exactly like the Python version did.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <queue>
#include <utility>
#include <vector>

namespace msw {

using Cells = std::vector<int>;      // sorted, unique ids
using Perm  = std::vector<uint8_t>;  // mines per supercell, aligned with a rule's Cells

namespace detail {

// A rule never spans more than one clue's neighbourhood, so it has at most 8
// supercells and at most C(8, 4) = 70 permutations.
constexpr int kMaxRuleCells = 8;
constexpr int kMaxPerms     = 128;

inline double choose(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k > n - k) k = n - k;
    double r = 1.0;
    for (int i = 1; i <= k; ++i) r = r * double(n - k + i) / double(i);
    return r;
}

inline double clamped_probability(int mines, int cells) {
    if (cells <= 0) return 0.0;
    const double p = double(mines) / double(cells);
    return p < 0.0 ? 0.0 : (p > 1.0 ? 1.0 : p);
}

// Set of up to 128 permutation indices.
struct Bits {
    uint64_t w[2] = {0, 0};
    void set(int i) { w[i >> 6] |= uint64_t(1) << (i & 63); }
    void clear(int i) { w[i >> 6] &= ~(uint64_t(1) << (i & 63)); }
    bool any() const { return (w[0] | w[1]) != 0; }
    int count() const { return __builtin_popcountll(w[0]) + __builtin_popcountll(w[1]); }
    int first() const { return w[0] ? __builtin_ctzll(w[0]) : 64 + __builtin_ctzll(w[1]); }
    void operator&=(const Bits& o) { w[0] &= o.w[0]; w[1] &= o.w[1]; }
};

// Positions of the cells two sorted cell lists have in common.
inline void shared_positions(const Cells& a, const Cells& b, std::vector<int>& pa, std::vector<int>& pb) {
    pa.clear();
    pb.clear();
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] < b[j]) ++i;
        else if (b[j] < a[i]) ++j;
        else { pa.push_back(int(i++)); pb.push_back(int(j++)); }
    }
}

// Pack a permutation's counts at the given positions into one integer.
inline uint64_t perm_key(const Perm& p, const std::vector<int>& pos) {
    uint64_t k = 0;
    for (int i : pos) k = (k << 8) | p[i];
    return k;
}

// ---------------------------------------------------------------------------
// Step 2: simplification over plain cells.
//
// rule_map: sorted vars -> mines. fixed[v]: -1 unknown, 0 safe, 1 mine.

inline bool simplify(std::map<Cells, int>& rule_map, std::vector<int8_t>& fixed) {
    Cells remaining, diff;
    for (;;) {
        bool changed = false;
        std::map<Cells, int> normalized;
        for (const auto& kv : rule_map) {
            int rhs = kv.second;
            remaining.clear();
            for (int v : kv.first) {
                if (fixed[v] < 0) remaining.push_back(v);
                else rhs -= fixed[v];
            }
            const int n = int(remaining.size());
            if (rhs < 0 || rhs > n) return false;
            if (n == 0) continue;
            if (rhs == 0 || rhs == n) {
                const int8_t val = rhs == n ? 1 : 0;
                for (int v : remaining) {
                    if (fixed[v] < 0) { fixed[v] = val; changed = true; }
                    else if (fixed[v] != val) return false;
                }
                continue;
            }
            auto ins = normalized.emplace(remaining, rhs);
            if (!ins.second && ins.first->second != rhs) return false;
        }
        rule_map.swap(normalized);

        // small within big  =>  (big minus small) holds big - small mines
        //
        // Sorted by size, ties in map order. (Not std::stable_sort: it pulls
        // in std::exception RTTI, which cling on Windows cannot link.)
        std::vector<const std::pair<const Cells, int>*> by_map;
        std::vector<std::pair<size_t, size_t>> order;  // (size, map position)
        by_map.reserve(rule_map.size());
        order.reserve(rule_map.size());
        for (const auto& kv : rule_map) {
            order.push_back({kv.first.size(), by_map.size()});
            by_map.push_back(&kv);
        }
        std::sort(order.begin(), order.end());
        std::vector<const std::pair<const Cells, int>*> keys;
        keys.reserve(order.size());
        for (const auto& o : order) keys.push_back(by_map[o.second]);
        std::map<Cells, int> inferred;
        for (size_t i = 0; i < keys.size(); ++i) {
            const Cells& small = keys[i]->first;
            for (size_t j = i + 1; j < keys.size(); ++j) {
                const Cells& big = keys[j]->first;
                if (small.size() >= big.size()) continue;
                if (!std::includes(big.begin(), big.end(), small.begin(), small.end())) continue;
                diff.clear();
                std::set_difference(big.begin(), big.end(), small.begin(), small.end(), std::back_inserter(diff));
                const int rhs = keys[j]->second - keys[i]->second;
                if (rhs < 0 || rhs > int(diff.size())) return false;
                auto existing = rule_map.find(diff);
                if (existing != rule_map.end()) {
                    if (existing->second != rhs) return false;
                    continue;
                }
                auto ins = inferred.emplace(diff, rhs);
                if (ins.second) changed = true;
                else if (ins.first->second != rhs) return false;
            }
        }
        for (const auto& kv : inferred) rule_map.insert(kv);
        if (!changed) return true;
    }
}

// ---------------------------------------------------------------------------
// Step 3: supercells.

struct Rule {
    int mines  = 0;
    int ncells = 0;  // base cells
    Cells cells;     // supercell ids
};

inline void condense(const std::vector<std::pair<Cells, int>>& specs, int nvars,
                     std::vector<Rule>& rules, std::vector<int>& super_size, std::vector<int>& var_super) {
    std::vector<std::vector<int>> var_rules(nvars);
    for (int r = 0; r < int(specs.size()); ++r)
        for (int v : specs[r].first) var_rules[v].push_back(r);

    std::map<std::vector<int>, int> by_signature;
    var_super.assign(nvars, -1);
    super_size.clear();
    for (int v = 0; v < nvars; ++v) {
        if (var_rules[v].empty()) continue;
        auto ins = by_signature.emplace(var_rules[v], int(super_size.size()));
        if (ins.second) super_size.push_back(0);
        var_super[v] = ins.first->second;
        super_size[ins.first->second] += 1;
    }

    rules.clear();
    for (const auto& s : specs) {
        Rule r;
        r.mines  = s.second;
        r.ncells = int(s.first.size());
        for (int v : s.first) r.cells.push_back(var_super[v]);
        std::sort(r.cells.begin(), r.cells.end());
        r.cells.erase(std::unique(r.cells.begin(), r.cells.end()), r.cells.end());
        rules.push_back(std::move(r));
    }
}

// ---------------------------------------------------------------------------
// Step 4: logical reduction. Rules are interned by value (mines, cells), as
// Rule_ equality was in Python, so a rule that comes back is the same rule.

class Reducer {
public:
    explicit Reducer(const std::vector<int>& super_size)
        : size_(super_size), cell_rules_(super_size.size()) {}

    bool add_rule(int mines, int ncells, Cells cells) {
        if (mines < 0 || mines > ncells) return false;
        if (mines == 0 || mines == ncells) {
            // completely empty or full: split into one rule per supercell
            for (int c : cells) {
                const int s = size_[c];
                add_base(intern(mines > 0 ? s : 0, s, Cells{c}));
            }
            return true;
        }
        add_base(intern(mines, ncells, std::move(cells)));
        return true;
    }

    bool reduce_all() {
        Cells diff;
        while (!heap_.empty()) {
            const Cand c = heap_.top();
            heap_.pop();
            if (!active_[c.sup] || !active_[c.sub]) continue;
            const Rule sup = rules_[c.sup];
            const Rule sub = rules_[c.sub];
            diff.clear();
            std::set_difference(sup.cells.begin(), sup.cells.end(), sub.cells.begin(), sub.cells.end(),
                                std::back_inserter(diff));
            remove(c.sup);
            if (!add_rule(sup.mines - sub.mines, sup.ncells - sub.ncells, diff)) return false;
        }
        return true;
    }

    std::vector<int> active_ids() const {
        std::vector<int> out;
        for (int i = 0; i < int(rules_.size()); ++i)
            if (active_[i]) out.push_back(i);
        return out;
    }

    const Rule& rule(int id) const { return rules_[id]; }

private:
    struct Cand {
        int a, b;     // superrule cells, subrule cells
        double c;     // distance of reduced mines from half the reduced cells
        int64_t seq;
        int sup, sub;
    };
    // priority_queue pops the largest: biggest rules first, then most extreme
    // mine ratio, then insertion order.
    struct CandLess {
        bool operator()(const Cand& x, const Cand& y) const {
            if (x.a != y.a) return x.a < y.a;
            if (x.b != y.b) return x.b < y.b;
            if (x.c != y.c) return x.c < y.c;
            return x.seq > y.seq;
        }
    };

    int intern(int mines, int ncells, Cells cells) {
        auto key = std::make_pair(mines, cells);
        auto it = index_.find(key);
        if (it != index_.end()) return it->second;
        const int id = int(rules_.size());
        index_.emplace(std::move(key), id);
        Rule r;
        r.mines  = mines;
        r.ncells = ncells;
        r.cells  = std::move(cells);
        rules_.push_back(std::move(r));
        active_.push_back(0);
        mark_.push_back(0);
        return id;
    }

    void add_base(int id) {
        if (active_[id]) return;
        active_[id] = 1;
        for (int c : rules_[id].cells) cell_rules_[c].push_back(id);
        update_reduceables(id);
    }

    void remove(int id) {
        active_[id] = 0;
        for (int c : rules_[id].cells) {
            auto& v = cell_rules_[c];
            v.erase(std::find(v.begin(), v.end(), id));
        }
    }

    void overlapping(int id, std::vector<int>& out) {
        out.clear();
        ++stamp_;
        mark_[id] = stamp_;
        for (int c : rules_[id].cells)
            for (int o : cell_rules_[c])
                if (mark_[o] != stamp_) { mark_[o] = stamp_; out.push_back(o); }
    }

    void update_reduceables(int id) {
        std::vector<int> ov;
        overlapping(id, ov);
        const Rule& r = rules_[id];
        for (int o_id : ov) {
            const Rule& o = rules_[o_id];
            if (o.ncells <= r.ncells) {
                if (o.mines <= r.mines && (o.ncells - o.mines) <= (r.ncells - r.mines) &&
                    std::includes(r.cells.begin(), r.cells.end(), o.cells.begin(), o.cells.end()))
                    push(id, o_id);
            } else {
                if (r.mines <= o.mines && (r.ncells - r.mines) <= (o.ncells - o.mines) &&
                    std::includes(o.cells.begin(), o.cells.end(), r.cells.begin(), r.cells.end()))
                    push(o_id, id);
            }
        }
    }

    void push(int sup, int sub) {
        const Rule& a = rules_[sup];
        const Rule& b = rules_[sub];
        const int cells = a.ncells - b.ncells;
        const int mines = a.mines - b.mines;
        heap_.push(Cand{a.ncells, b.ncells, std::fabs(mines - 0.5 * cells), seq_++, sup, sub});
    }

    const std::vector<int>& size_;
    std::vector<Rule> rules_;
    std::map<std::pair<int, Cells>, int> index_;
    std::vector<char> active_;
    std::vector<int> mark_;
    int stamp_ = 0;
    std::vector<std::vector<int>> cell_rules_;
    std::priority_queue<Cand, std::vector<Cand>, CandLess> heap_;
    int64_t seq_ = 0;
};

// ---------------------------------------------------------------------------
// Step 5: permutations.

inline void perms_rec(const std::vector<int>& sizes, const std::vector<int>& cap, int i, int left,
                      Perm& cur, std::vector<Perm>& out) {
    if (i == int(sizes.size())) {
        if (left == 0) out.push_back(cur);
        return;
    }
    const int hi = std::min(left, sizes[i]);
    const int lo = std::max(0, left - cap[i + 1]);
    for (int m = hi; m >= lo; --m) {
        cur[i] = uint8_t(m);
        perms_rec(sizes, cap, i + 1, left - m, cur, out);
    }
}

// Every way to spread k mines over cells of the given sizes.
inline std::vector<Perm> all_perms(const std::vector<int>& sizes, int k) {
    std::vector<int> cap(sizes.size() + 1, 0);
    for (int i = int(sizes.size()) - 1; i >= 0; --i) cap[i] = cap[i + 1] + sizes[i];
    std::vector<Perm> out;
    Perm cur(sizes.size(), 0);
    perms_rec(sizes, cap, 0, k, cur, out);
    return out;
}

struct PermSet {
    Cells cells;
    int k = 0;
    std::vector<Perm> perms;  // sorted, unique
};

// If the set is the cartesian product of its projections onto `in_pos` and
// the remaining positions, return both factors.
inline bool split(const PermSet& ps, const std::vector<int>& in_pos, PermSet& sub, PermSet& rem) {
    const int n = int(ps.cells.size());
    std::vector<char> is_in(n, 0);
    for (int p : in_pos) is_in[p] = 1;
    std::vector<int> rem_pos;
    for (int i = 0; i < n; ++i)
        if (!is_in[i]) rem_pos.push_back(i);

    std::map<Perm, std::vector<Perm>> groups;
    int k_sub = -1;
    Perm a, b;
    for (const Perm& p : ps.perms) {
        a.clear();
        b.clear();
        int ka = 0;
        for (int i : in_pos) { a.push_back(p[i]); ka += p[i]; }
        for (int i : rem_pos) b.push_back(p[i]);
        if (k_sub < 0) k_sub = ka;
        else if (ka != k_sub) return false;
        groups[a].push_back(b);
    }
    const std::vector<Perm>* first = nullptr;
    for (auto& g : groups) {
        std::sort(g.second.begin(), g.second.end());
        if (!first) first = &g.second;
        else if (g.second != *first) return false;
    }
    if (!first) return false;

    sub.cells.clear();
    for (int i : in_pos) sub.cells.push_back(ps.cells[i]);
    sub.k = k_sub;
    sub.perms.clear();
    for (const auto& g : groups) sub.perms.push_back(g.first);

    rem.cells.clear();
    for (int i : rem_pos) rem.cells.push_back(ps.cells[i]);
    rem.k = ps.k - k_sub;
    rem.perms = *first;
    return true;
}

// Factor a permutation set into the finest cartesian product it allows.
inline void decompose(const PermSet& ps, int k_floor, std::vector<PermSet>& out) {
    const int n = int(ps.cells.size());
    std::vector<int> combo;
    for (int kk = k_floor; kk <= n / 2; ++kk) {
        combo.resize(kk);
        for (int i = 0; i < kk; ++i) combo[i] = i;
        for (;;) {
            PermSet sub, rem;
            if (split(ps, combo, sub, rem)) {
                out.push_back(std::move(sub));
                decompose(rem, kk, out);
                return;
            }
            int i = kk - 1;
            while (i >= 0 && combo[i] == n - kk + i) --i;
            if (i < 0) break;
            ++combo[i];
            for (int j = i + 1; j < kk; ++j) combo[j] = combo[j - 1] + 1;
        }
    }
    out.push_back(ps);
}

class Ruleset {
public:
    struct Entry {
        int mines  = 0;
        int ncells = 0;
        Cells cells;
        std::vector<Perm> perms;
        bool constrained = false;
        bool alive       = true;
    };
    std::vector<Entry> entries;

    explicit Ruleset(const std::vector<int>& super_size)
        : size_(super_size), cell_rules_(super_size.size()) {}

    void add(int mines, Cells cells, std::vector<Perm> perms) {
        auto key = std::make_pair(mines, cells);
        auto it = index_.find(key);
        if (it != index_.end()) {
            entries[it->second].perms       = std::move(perms);
            entries[it->second].constrained = false;
            return;
        }
        const int id = int(entries.size());
        index_.emplace(std::move(key), id);
        for (int c : cells) cell_rules_[c].push_back(id);
        Entry e;
        e.mines = mines;
        for (int c : cells) e.ncells += size_[c];
        e.cells = std::move(cells);
        e.perms = std::move(perms);
        entries.push_back(std::move(e));
        mark_.push_back(0);
    }

    // Drop permutations that no permutation of an overlapping rule agrees
    // with, until nothing changes.
    bool cross_eliminate() {
        std::deque<std::pair<int, int>> work;
        std::set<std::pair<int, int>> queued;
        std::vector<int> ov, pa, pb;
        for (int r = 0; r < int(entries.size()); ++r) {
            if (!entries[r].alive) continue;
            overlapping(r, ov);
            for (int o : ov)
                if (queued.insert({r, o}).second) work.push_back({r, o});
        }
        std::vector<uint64_t> keys;
        while (!work.empty()) {
            const std::pair<int, int> edge = work.front();
            work.pop_front();
            queued.erase(edge);
            Entry& r = entries[edge.first];
            const Entry& o = entries[edge.second];
            shared_positions(r.cells, o.cells, pa, pb);
            keys.clear();
            for (const Perm& q : o.perms) keys.push_back(perm_key(q, pb));
            std::sort(keys.begin(), keys.end());
            size_t w = 0;
            for (size_t i = 0; i < r.perms.size(); ++i) {
                if (std::binary_search(keys.begin(), keys.end(), perm_key(r.perms[i], pa))) {
                    if (w != i) r.perms[w] = std::move(r.perms[i]);
                    ++w;
                }
            }
            const bool changed = w != r.perms.size();
            r.perms.resize(w);
            if (r.perms.empty()) return false;
            if (changed) {
                r.constrained = true;
                overlapping(edge.first, ov);
                for (int x : ov)
                    if (queued.insert({x, edge.first}).second) work.push_back({x, edge.first});
            }
        }
        return true;
    }

    // Replace constrained rules whose permutations factor by their factors;
    // this can split one front into several.
    void rereduce() {
        std::vector<int> superseded;
        std::map<Cells, PermSet> parts_by_cells;
        for (int r = 0; r < int(entries.size()); ++r) {
            Entry& e = entries[r];
            if (!e.alive || !e.constrained) continue;
            PermSet ps;
            ps.cells = e.cells;
            ps.k     = e.mines;
            ps.perms = e.perms;
            std::sort(ps.perms.begin(), ps.perms.end());
            std::vector<PermSet> parts;
            decompose(ps, 1, parts);
            if (parts.size() > 1) {
                superseded.push_back(r);
                for (auto& p : parts) parts_by_cells[p.cells] = std::move(p);
            }
        }
        for (int r : superseded) remove(r);
        for (auto& kv : parts_by_cells) add(kv.second.k, kv.second.cells, std::move(kv.second.perms));
    }

    // Connected groups of rules (sharing at least one cell).
    std::vector<std::vector<int>> fronts() const {
        const int n = int(entries.size());
        std::vector<int> parent(n);
        for (int i = 0; i < n; ++i) parent[i] = i;
        auto find = [&parent](int x) {
            while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
            return x;
        };
        for (const auto& rs : cell_rules_)
            for (size_t i = 1; i < rs.size(); ++i) parent[find(rs[i])] = find(rs[0]);
        std::map<int, std::vector<int>> groups;
        for (int i = 0; i < n; ++i)
            if (entries[i].alive) groups[find(i)].push_back(i);
        std::vector<std::vector<int>> out;
        for (auto& g : groups) out.push_back(std::move(g.second));
        return out;
    }

private:
    void remove(int id) {
        Entry& e = entries[id];
        e.alive = false;
        index_.erase(std::make_pair(e.mines, e.cells));
        for (int c : e.cells) {
            auto& v = cell_rules_[c];
            v.erase(std::find(v.begin(), v.end(), id));
        }
    }

    void overlapping(int id, std::vector<int>& out) {
        out.clear();
        ++stamp_;
        mark_[id] = stamp_;
        for (int c : entries[id].cells)
            for (int o : cell_rules_[c])
                if (mark_[o] != stamp_) { mark_[o] = stamp_; out.push_back(o); }
    }

    const std::vector<int>& size_;
    std::vector<std::vector<int>> cell_rules_;
    std::map<std::pair<int, Cells>, int> index_;
    std::vector<int> mark_;
    int stamp_ = 0;
};

// ---------------------------------------------------------------------------
// Step 6: enumerate a front.

// Per-front tally: for each possible mine count k in the front, the weighted
// number of configurations and the expected mines in each supercell.
struct Tally {
    Cells cells;
    std::vector<int> ks;                    // ascending
    std::vector<double> totals;
    std::vector<std::vector<double>> mean;  // [k index][cell index]
};

inline Tally trivial_tally(int cell, int ncells, int mines) {
    Tally t;
    t.cells  = {cell};
    t.ks     = {mines};
    t.totals = {choose(ncells, mines)};
    t.mean   = {{double(mines)}};
    return t;
}

class FrontEnumerator {
public:
    FrontEnumerator(const Ruleset& rs, const std::vector<int>& ids, const std::vector<int>& super_size) {
        R_ = int(ids.size());
        for (int id : ids) {
            const auto& e = rs.entries[id];
            perms_.push_back(&e.perms);
            if (e.perms.empty() || int(e.perms.size()) > kMaxPerms || int(e.cells.size()) > kMaxRuleCells)
                ok_ = false;
        }

        std::map<int, int> cell_index;
        for (int id : ids)
            for (int c : rs.entries[id].cells) cell_index.emplace(c, 0);
        for (auto& kv : cell_index) {
            kv.second = int(cells_.size());
            cells_.push_back(kv.first);
            cell_size_.push_back(super_size[kv.first]);
        }
        const int nc = int(cells_.size());

        std::vector<std::vector<int>> rules_of_cell(nc);
        cell_local_.resize(R_);
        for (int r = 0; r < R_; ++r)
            for (int c : rs.entries[ids[r]].cells) {
                const int lc = cell_index[c];
                cell_local_[r].push_back(lc);
                rules_of_cell[lc].push_back(r);
            }

        nbrs_.resize(R_);
        std::vector<int> mark(R_, -1);
        for (int r = 0; r < R_; ++r) {
            mark[r] = r;
            for (int lc : cell_local_[r])
                for (int o : rules_of_cell[lc])
                    if (mark[o] != r) { mark[o] = r; nbrs_[r].push_back(o); }
        }

        if (!ok_) return;
        compat_.resize(R_);
        std::vector<int> pa, pb;
        std::vector<std::pair<uint64_t, int>> keys;
        for (int r = 0; r < R_; ++r) {
            const auto& er  = rs.entries[ids[r]];
            const int deg   = int(nbrs_[r].size());
            compat_[r].assign(er.perms.size() * deg, Bits{});
            for (int j = 0; j < deg; ++j) {
                const auto& eo = rs.entries[ids[nbrs_[r][j]]];
                shared_positions(er.cells, eo.cells, pa, pb);
                keys.clear();
                for (int q = 0; q < int(eo.perms.size()); ++q) keys.push_back({perm_key(eo.perms[q], pb), q});
                std::sort(keys.begin(), keys.end());
                for (int p = 0; p < int(er.perms.size()); ++p) {
                    const uint64_t kp = perm_key(er.perms[p], pa);
                    auto it = std::lower_bound(keys.begin(), keys.end(), std::make_pair(kp, -1));
                    Bits& b = compat_[r][size_t(p) * deg + j];
                    for (; it != keys.end() && it->first == kp; ++it) b.set(it->second);
                }
            }
        }

        int max_size = 0, max_k = 0;
        for (int s : cell_size_) { max_size = std::max(max_size, s); max_k += s; }
        choose_.assign(size_t(max_size + 1) * (max_size + 1), 0.0);
        for (int n = 0; n <= max_size; ++n)
            for (int k = 0; k <= n; ++k) choose_[size_t(n) * (max_size + 1) + k] = choose(n, k);
        choose_stride_ = max_size + 1;
        totals_.assign(max_k + 1, 0.0);
        sums_.assign(max_k + 1, std::vector<double>());
        seen_.assign(max_k + 1, 0);
        cfg_.assign(nc, 0);
    }

    bool run(Tally& out) {
        if (!ok_) return false;
        free_.assign(R_ + 2, std::vector<Bits>(R_));
        fixed_.assign(R_ + 2, std::vector<int16_t>(R_, -1));
        for (int r = 0; r < R_; ++r)
            for (int p = 0; p < int(perms_[r]->size()); ++p) free_[0][r].set(p);
        dfs(0, R_);

        out.cells = cells_;
        out.ks.clear();
        out.totals.clear();
        out.mean.clear();
        for (int k = 0; k < int(totals_.size()); ++k) {
            if (!seen_[k]) continue;
            out.ks.push_back(k);
            out.totals.push_back(totals_[k]);
            std::vector<double> m(sums_[k]);
            for (double& x : m) x /= totals_[k];
            out.mean.push_back(std::move(m));
        }
        return !out.ks.empty();
    }

private:
    void dfs(int depth, int nfree) {
        const std::vector<Bits>& fr    = free_[depth];
        const std::vector<int16_t>& fx = fixed_[depth];
        if (nfree == 0) {
            emit(fx);
            return;
        }
        int best = -1, best_n = 1 << 30;
        for (int r = 0; r < R_; ++r) {
            if (fx[r] >= 0) continue;
            const int n = fr[r].count();
            if (n < best_n) { best_n = n; best = r; }
        }
        Bits todo = fr[best];
        while (todo.any()) {
            const int p = todo.first();
            todo.clear(p);
            free_[depth + 1]  = free_[depth];
            fixed_[depth + 1] = fixed_[depth];
            int nf = nfree;
            if (propagate(free_[depth + 1], fixed_[depth + 1], nf, best, p)) dfs(depth + 1, nf);
        }
    }

    // Fix permutation p for rule r and narrow every overlapping free rule;
    // a rule left with one option is fixed in turn.
    bool propagate(std::vector<Bits>& fr, std::vector<int16_t>& fx, int& nfree, int r0, int p0) {
        stack_.clear();
        stack_.push_back({r0, p0});
        while (!stack_.empty()) {
            const std::pair<int, int> top = stack_.back();
            stack_.pop_back();
            const int r = top.first, p = top.second;
            if (fx[r] >= 0) continue;
            fx[r] = int16_t(p);
            --nfree;
            const std::vector<int>& nb = nbrs_[r];
            const int deg   = int(nb.size());
            const Bits* row = compat_[r].data() + size_t(p) * deg;
            for (int j = 0; j < deg; ++j) {
                const int o = nb[j];
                if (fx[o] >= 0) continue;
                fr[o] &= row[j];
                const int c = fr[o].count();
                if (c == 0) return false;
                if (c == 1) stack_.push_back({o, fr[o].first()});
            }
        }
        return true;
    }

    void emit(const std::vector<int16_t>& fx) {
        for (int r = 0; r < R_; ++r) {
            const Perm& p = (*perms_[r])[fx[r]];
            const std::vector<int>& lc = cell_local_[r];
            for (size_t i = 0; i < lc.size(); ++i) cfg_[lc[i]] = p[i];
        }
        int k = 0;
        double mult = 1.0;
        for (size_t c = 0; c < cfg_.size(); ++c) {
            k += cfg_[c];
            mult *= choose_[size_t(cell_size_[c]) * choose_stride_ + cfg_[c]];
        }
        if (!seen_[k]) {
            seen_[k] = 1;
            sums_[k].assign(cfg_.size(), 0.0);
        }
        totals_[k] += mult;
        std::vector<double>& s = sums_[k];
        for (size_t c = 0; c < cfg_.size(); ++c) s[c] += cfg_[c] * mult;
    }

    bool ok_ = true;
    int R_   = 0;
    std::vector<const std::vector<Perm>*> perms_;
    Cells cells_;
    std::vector<int> cell_size_;
    std::vector<std::vector<int>> cell_local_;
    std::vector<std::vector<int>> nbrs_;
    std::vector<std::vector<Bits>> compat_;  // [rule][perm * degree + neighbour]
    std::vector<double> choose_;
    int choose_stride_ = 1;

    std::vector<std::vector<Bits>> free_;     // per depth
    std::vector<std::vector<int16_t>> fixed_; // per depth
    std::vector<std::pair<int, int>> stack_;
    std::vector<int> cfg_;

    std::vector<double> totals_;
    std::vector<std::vector<double>> sums_;
    std::vector<char> seen_;
};

// ---------------------------------------------------------------------------
// Step 7: weight fronts against each other under the fixed mine total.

struct Poly {
    int lo = 0;               // mine count of c[0]
    std::vector<double> c;
};

inline Poly convolve(const Poly& a, const Poly& b) {
    Poly out;
    out.lo = a.lo + b.lo;
    out.c.assign(a.c.size() + b.c.size() - 1, 0.0);
    for (size_t i = 0; i < a.c.size(); ++i) {
        if (a.c[i] == 0.0) continue;
        for (size_t j = 0; j < b.c.size(); ++j) out.c[i + j] += a.c[i] * b.c[j];
    }
    return out;
}

// (a * b)[t]
inline double convolve_at(const Poly& a, const Poly& b, int t) {
    double s = 0.0;
    for (size_t i = 0; i < a.c.size(); ++i) {
        const int j = t - a.lo - int(i) - b.lo;
        if (j < 0) break;
        if (j < int(b.c.size())) s += a.c[i] * b.c[j];
    }
    return s;
}

// Reweights every multi-k tally in place so its totals are relative
// likelihoods given the board total. Returns the probability for each
// "uncharted" cell (hidden, touched by no rule).
inline bool combine_fronts(std::vector<Tally>& tallies, int uncharted, int total_mines, double& p_other) {
    int min_all = 0, max_all = 0;
    for (const Tally& t : tallies) { min_all += t.ks.front(); max_all += t.ks.back(); }
    if (min_all > total_mines || total_mines > max_all + uncharted) return false;

    int at_large = total_mines;
    std::vector<Tally*> dyn;
    for (Tally& t : tallies) {
        if (t.ks.size() == 1) at_large -= t.ks[0];
        else dyn.push_back(&t);
    }

    int min_t = 0, max_t = 0;
    for (const Tally* t : dyn) { min_t += t->ks.front(); max_t += t->ks.back(); }
    const int min_other = std::max(at_large - max_t, 0);
    const int max_other = std::min(std::max(at_large - min_t, 0), uncharted);
    if (min_other > max_other) return false;

    // One polynomial per dynamic front, plus the uncharted region last. Each
    // is scaled to max 1: only ratios within an item matter.
    std::vector<Poly> items;
    for (const Tally* t : dyn) {
        Poly p;
        p.lo = t->ks.front();
        p.c.assign(t->ks.back() - p.lo + 1, 0.0);
        double mx = 0.0;
        for (size_t i = 0; i < t->ks.size(); ++i) {
            p.c[t->ks[i] - p.lo] = t->totals[i];
            mx = std::max(mx, t->totals[i]);
        }
        if (mx > 0.0)
            for (double& x : p.c) x /= mx;
        items.push_back(std::move(p));
    }
    {
        // C(U, n) / C(U, max_other), built downward from n = max_other
        Poly p;
        p.lo = min_other;
        p.c.assign(max_other - min_other + 1, 0.0);
        double r   = 1.0;
        p.c.back() = 1.0;
        for (int n = max_other; n > min_other; --n) {
            r *= double(n) / double(uncharted - n + 1);
            p.c[n - 1 - min_other] = r;
        }
        items.push_back(std::move(p));
    }

    const int m = int(items.size());
    std::vector<Poly> pre(m + 1), suf(m + 1);
    pre[0].c = {1.0};
    suf[m].c = {1.0};
    for (int i = 0; i < m; ++i) pre[i + 1] = convolve(pre[i], items[i]);
    for (int i = m - 1; i >= 0; --i) suf[i] = convolve(items[i], suf[i + 1]);

    for (int i = 0; i < m; ++i) {
        const Poly& it = items[i];
        std::vector<double> w(it.c.size(), 0.0);
        for (size_t x = 0; x < it.c.size(); ++x)
            if (it.c[x] != 0.0) w[x] = it.c[x] * convolve_at(pre[i], suf[i + 1], at_large - (it.lo + int(x)));
        if (i < int(dyn.size())) {
            Tally* t = dyn[i];
            for (size_t q = 0; q < t->ks.size(); ++q) t->totals[q] = w[t->ks[q] - it.lo];
        } else {
            double s = 0.0, e = 0.0;
            for (size_t x = 0; x < w.size(); ++x) { s += w[x]; e += w[x] * (it.lo + int(x)); }
            if (!(s > 0.0)) return false;
            p_other = uncharted > 0 ? e / s / uncharted : 0.0;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Steps 3-7 for one simplified rule set over variables 0..nvars-1.
// p[v] gets each variable's mine probability.

inline bool solve_rules(const std::vector<std::pair<Cells, int>>& specs, int nvars, int total_mines,
                        std::vector<double>& p) {
    std::vector<Rule> rules;
    std::vector<int> super_size, var_super;
    condense(specs, nvars, rules, super_size, var_super);

    Reducer red(super_size);
    for (const Rule& r : rules)
        if (!red.add_rule(r.mines, r.ncells, r.cells)) return false;
    if (!red.reduce_all()) return false;

    std::vector<Tally> tallies;
    Ruleset rs(super_size);
    std::vector<int> sizes;
    for (int id : red.active_ids()) {
        const Rule& r = red.rule(id);
        if (r.cells.size() == 1) {
            tallies.push_back(trivial_tally(r.cells[0], r.ncells, r.mines));
            continue;
        }
        if (int(r.cells.size()) > kMaxRuleCells) return false;
        sizes.clear();
        for (int c : r.cells) sizes.push_back(super_size[c]);
        rs.add(r.mines, r.cells, all_perms(sizes, r.mines));
    }
    if (!rs.cross_eliminate()) return false;
    rs.rereduce();

    for (const std::vector<int>& front : rs.fronts()) {
        const auto& e0 = rs.entries[front[0]];
        if (front.size() == 1 && e0.cells.size() == 1) {
            tallies.push_back(trivial_tally(e0.cells[0], e0.ncells, e0.mines));
            continue;
        }
        Tally t;
        FrontEnumerator fe(rs, front, super_size);
        if (!fe.run(t)) return false;
        tallies.push_back(std::move(t));
    }

    int covered = 0;
    for (int s : super_size) covered += s;
    double p_other = 0.0;
    if (!combine_fronts(tallies, nvars - covered, total_mines, p_other)) return false;

    std::vector<double> cell_p(super_size.size(), std::numeric_limits<double>::quiet_NaN());
    for (const Tally& t : tallies) {
        double s = 0.0;
        for (double x : t.totals) s += x;
        if (!(s > 0.0)) return false;
        for (size_t ci = 0; ci < t.cells.size(); ++ci) {
            double e = 0.0;
            for (size_t q = 0; q < t.ks.size(); ++q) e += t.totals[q] / s * t.mean[q][ci];
            cell_p[t.cells[ci]] = e / super_size[t.cells[ci]];
        }
    }

    p.assign(nvars, p_other);
    for (int v = 0; v < nvars; ++v) {
        const int c = var_super[v];
        if (c >= 0 && !std::isnan(cell_p[c])) p[v] = cell_p[c];
    }
    return true;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Public entry point.
//
// Mine probability for every cell of a rows x cols board, from the visible
// state only. out[flat] is NaN for revealed cells, [0, 1] for hidden ones.
//
//   revealed, counts : rows*cols bytes (counts = adjacent mines, read only
//                      where revealed)
//   nbr_idx, nbr_len : neighbour table, 8 slots per cell, -1 padded
//   known_mine_flat  : the exploded cell after a loss, else -1
inline void mine_probabilities(int rows, int cols, const uint8_t* revealed, const uint8_t* counts,
                               const int32_t* nbr_idx, const int8_t* nbr_len, int mine_count,
                               bool started, int known_mine_flat, double* out) {
    using namespace detail;
    const int n      = rows * cols;
    const double nan = std::numeric_limits<double>::quiet_NaN();

    std::vector<int> hidden;
    std::vector<int> tag(n, -1);
    for (int f = 0; f < n; ++f) {
        out[f] = nan;
        if (!revealed[f]) {
            tag[f] = int(hidden.size());
            hidden.push_back(f);
        }
    }
    if (hidden.empty()) return;

    auto fill_uniform = [out](const std::vector<int>& flats, int mines) {
        const double p = clamped_probability(mines, int(flats.size()));
        for (int f : flats) out[f] = p;
    };

    if (!started) {
        fill_uniform(hidden, mine_count);
        return;
    }

    int mines_remaining = std::max(mine_count - (known_mine_flat >= 0 ? 1 : 0), 0);

    // Step 1: one rule per revealed clue.
    std::map<Cells, int> rule_map;
    bool bad = false;
    Cells vars;
    for (int f = 0; f < n && !bad; ++f) {
        if (!revealed[f] || f == known_mine_flat) continue;
        int rhs = counts[f];
        vars.clear();
        for (int k = 0; k < nbr_len[f]; ++k) {
            const int g = nbr_idx[f * 8 + k];
            if (!revealed[g]) vars.push_back(tag[g]);
            else if (g == known_mine_flat) --rhs;
        }
        if (vars.empty()) {
            if (rhs != 0) bad = true;
            continue;
        }
        std::sort(vars.begin(), vars.end());
        rhs = std::min(std::max(rhs, 0), int(vars.size()));
        auto ins = rule_map.emplace(vars, rhs);
        if (!ins.second && ins.first->second != rhs) bad = true;
    }
    if (bad || rule_map.empty()) {
        fill_uniform(hidden, mines_remaining);
        return;
    }

    // Step 2.
    std::vector<int8_t> fixed(hidden.size(), -1);
    if (!simplify(rule_map, fixed)) {
        fill_uniform(hidden, mines_remaining);
        return;
    }

    std::vector<int> remap(hidden.size(), -1);
    std::vector<int> unresolved;
    int fixed_mines = 0;
    for (int t = 0; t < int(hidden.size()); ++t) {
        if (fixed[t] >= 0) {
            out[hidden[t]] = double(fixed[t]);
            fixed_mines += fixed[t];
        } else {
            remap[t] = int(unresolved.size());
            unresolved.push_back(hidden[t]);
        }
    }
    mines_remaining = std::max(mines_remaining - fixed_mines, 0);
    if (unresolved.empty()) return;
    if (rule_map.empty()) {
        fill_uniform(unresolved, mines_remaining);
        return;
    }

    std::vector<std::pair<Cells, int>> specs;
    specs.reserve(rule_map.size());
    for (const auto& kv : rule_map) {
        Cells c;
        c.reserve(kv.first.size());
        for (int t : kv.first) c.push_back(remap[t]);
        specs.push_back({std::move(c), kv.second});
    }

    // Steps 3-7.
    std::vector<double> p;
    if (!solve_rules(specs, int(unresolved.size()), mines_remaining, p)) {
        fill_uniform(unresolved, mines_remaining);
        return;
    }
    for (int i = 0; i < int(unresolved.size()); ++i) out[unresolved[i]] = p[i];
}

}  // namespace msw
