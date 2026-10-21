// Reveal-only Minesweeper engine (no flags, no chording).
//
// Flat row-major board, first click is never a mine, flood fill on zeros, and
// an incremental pool of still-hidden safe cells (used by the cheating
// baseline agents). Mine probabilities come from solver.hpp.
//
// Board arrays are plain std::vectors allocated once in the constructor; the
// *_addr() accessors hand their addresses to Python, which wraps them as
// zero-copy numpy views. Nothing here may reallocate them afterwards.
//
// No exceptions (see solver.hpp). Out-of-range input returns -1.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "solver.hpp"

namespace msw {

// xoshiro256** seeded through splitmix64. Hand-rolled rather than <random>
// so a given seed deals the same board on every compiler and platform.
struct Rng {
    uint64_t s[4] = {0, 0, 0, 0};

    void seed(uint64_t x) {
        for (uint64_t& w : s) {
            uint64_t z = (x += 0x9E3779B97F4A7C15ull);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            w = z ^ (z >> 31);
        }
    }

    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

    uint64_t next() {
        const uint64_t r = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return r;
    }

    // Uniform in [0, n), unbiased.
    uint64_t below(uint64_t n) {
        const uint64_t threshold = (0 - n) % n;
        for (;;) {
            const uint64_t r = next();
            if (r >= threshold) return r % n;
        }
    }
};

class Engine {
public:
    static constexpr int OVER    = 0;
    static constexpr int RUNNING = 1;

    int rows             = 0;
    int cols             = 0;
    int mine_count       = 0;
    int flat_size        = 0;
    int total_safe_cells = 0;

    int state               = RUNNING;
    int started             = 0;
    int hit_mine            = 0;
    int won                 = 0;
    int move_count          = 0;
    int covered_count       = 0;
    int revealed_safe_cells = 0;
    int safe_pool_size      = 0;
    int exploded_flat       = -1;
    int min_win_moves       = -1;  // fewest moves that win this board, set by the first click
    uint64_t version        = 0;   // bumped on every change; lets callers cache derived data

    std::vector<uint8_t> mines;
    std::vector<uint8_t> revealed;
    std::vector<uint8_t> neighbor_counts;
    std::vector<int32_t> safe_cells;    // first safe_pool_size entries: hidden safe cells
    std::vector<int32_t> safe_pos;      // flat -> index in safe_cells, -1 if not in pool
    std::vector<int32_t> neighbor_idx;  // flat * 8 + k, -1 padded
    std::vector<int8_t>  neighbor_len;

    Engine(int rows_, int cols_, int mine_count_, uint64_t seed)
        : rows(rows_), cols(cols_), mine_count(mine_count_), flat_size(rows_ * cols_),
          total_safe_cells(rows_ * cols_ - mine_count_),
          mines(flat_size, 0), revealed(flat_size, 0), neighbor_counts(flat_size, 0),
          safe_cells(flat_size, 0), safe_pos(flat_size, -1),
          neighbor_idx(size_t(flat_size) * 8, -1), neighbor_len(flat_size, 0),
          queue_(flat_size, 0), pool_(flat_size, 0) {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) {
                const int f = i * cols + j;
                int k = 0;
                for (int ni = std::max(i - 1, 0); ni <= std::min(i + 1, rows - 1); ++ni)
                    for (int nj = std::max(j - 1, 0); nj <= std::min(j + 1, cols - 1); ++nj)
                        if (ni != i || nj != j) neighbor_idx[size_t(f) * 8 + k++] = ni * cols + nj;
                neighbor_len[f] = int8_t(k);
            }
        reset(seed);
    }

    void reset(uint64_t seed) {
        rng_.seed(seed);
        std::fill(mines.begin(), mines.end(), 0);
        std::fill(revealed.begin(), revealed.end(), 0);
        std::fill(neighbor_counts.begin(), neighbor_counts.end(), 0);
        std::fill(safe_pos.begin(), safe_pos.end(), -1);
        covered_count       = flat_size;
        safe_pool_size      = 0;
        state               = RUNNING;
        started             = 0;
        revealed_safe_cells = 0;
        exploded_flat       = -1;
        hit_mine            = 0;
        won                 = 0;
        move_count          = 0;
        min_win_moves       = -1;
        ++version;
    }

    // Load a mid-game position: rows*cols bytes each, nonzero = true. Mines
    // must already match mine_count. State (running / won / lost) is
    // inferred: a revealed mine is a loss, all safe cells revealed is a win.
    void load_state(uintptr_t mines_addr, uintptr_t revealed_addr) {
        const uint8_t* m = reinterpret_cast<const uint8_t*>(mines_addr);
        const uint8_t* r = reinterpret_cast<const uint8_t*>(revealed_addr);
        for (int f = 0; f < flat_size; ++f) {
            mines[f]    = m[f] ? 1 : 0;
            revealed[f] = r[f] ? 1 : 0;
        }
        compute_counts();
        started       = 1;
        move_count    = 0;
        covered_count = 0;
        exploded_flat = -1;
        int revealed_safe = 0;
        for (int f = 0; f < flat_size; ++f) {
            if (!revealed[f]) ++covered_count;
            else if (mines[f]) { if (exploded_flat < 0) exploded_flat = f; }
            else ++revealed_safe;
        }
        revealed_safe_cells = revealed_safe;
        if (exploded_flat >= 0) {
            hit_mine = 1;
            won      = 0;
            state    = OVER;
        } else {
            hit_mine = 0;
            won      = revealed_safe_cells == total_safe_cells;
            state    = won ? OVER : RUNNING;
        }
        safe_pool_size = 0;
        for (int f = 0; f < flat_size; ++f) {
            if (!mines[f] && !revealed[f]) {
                safe_cells[safe_pool_size] = f;
                safe_pos[f] = safe_pool_size++;
            } else {
                safe_pos[f] = -1;
            }
        }
        min_win_moves = min_clicks_left();
        ++version;
    }

    // Reveal one cell. Returns safe cells uncovered (0 on a mine or a no-op),
    // -1 if (row, col) is off the board.
    int reveal_count(int row, int col) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) return -1;
        const int flat = row * cols + col;
        if (state == OVER || revealed[flat]) return 0;
        const bool first = !started;
        const int n = reveal_flat(flat);
        ++move_count;
        // Mines only exist from the first click on, so that is when the
        // board's best possible move count becomes known.
        if (first) min_win_moves = move_count + min_clicks_left();
        return n;
    }

    // Fewest clicks that clear every hidden safe cell, for a player who
    // knows where the mines are (Minesweeper players call this 3BV):
    //
    //   one click per opening -- a connected patch of hidden zeros; clicking
    //     any cell of it uncovers the whole patch plus its border numbers --
    //   plus one click per hidden number cell that touches no hidden zero,
    //     since nothing but a direct click can uncover it.
    //
    // No order of play can do better, and clicking openings first then the
    // leftover numbers always achieves it.
    int min_clicks_left() const {
        std::vector<uint8_t> covered(flat_size, 0);  // uncovered by some opening
        std::vector<int32_t> queue;
        queue.reserve(flat_size);
        int clicks = 0;
        for (int f = 0; f < flat_size; ++f) {
            if (revealed[f] || mines[f] || neighbor_counts[f] != 0 || covered[f]) continue;
            ++clicks;  // a new opening
            covered[f] = 1;
            queue.clear();
            queue.push_back(f);
            for (size_t head = 0; head < queue.size(); ++head) {
                const int cur     = queue[head];
                const int32_t* nb = &neighbor_idx[size_t(cur) * 8];
                for (int k = 0; k < neighbor_len[cur]; ++k) {
                    const int g = nb[k];
                    if (revealed[g] || covered[g]) continue;
                    covered[g] = 1;  // a zero's neighbours are never mines
                    if (neighbor_counts[g] == 0) queue.push_back(g);
                }
            }
        }
        for (int f = 0; f < flat_size; ++f)
            if (!revealed[f] && !mines[f] && !covered[f]) ++clicks;  // a lone number
        return clicks;
    }

    // rows*cols int8: -1 hidden, 0..8 clue, 9 mine (only after a loss, if asked).
    void public_view(uintptr_t out_addr, bool reveal_mines_on_loss) const {
        int8_t* v = reinterpret_cast<int8_t*>(out_addr);
        const bool show = state == OVER && hit_mine && reveal_mines_on_loss;
        for (int f = 0; f < flat_size; ++f) {
            v[f] = (revealed[f] && !mines[f]) ? int8_t(neighbor_counts[f]) : int8_t(-1);
            if (show && mines[f]) v[f] = 9;
        }
    }

    // rows*cols float64: NaN where revealed, exact mine probability elsewhere.
    void mine_probabilities(uintptr_t out_addr) const {
        msw::mine_probabilities(rows, cols, revealed.data(), neighbor_counts.data(), neighbor_idx.data(),
                                neighbor_len.data(), mine_count, started != 0, hit_mine ? exploded_flat : -1,
                                reinterpret_cast<double*>(out_addr));
    }

    uintptr_t mines_addr() { return reinterpret_cast<uintptr_t>(mines.data()); }
    uintptr_t revealed_addr() { return reinterpret_cast<uintptr_t>(revealed.data()); }
    uintptr_t neighbor_counts_addr() { return reinterpret_cast<uintptr_t>(neighbor_counts.data()); }
    uintptr_t safe_cells_addr() { return reinterpret_cast<uintptr_t>(safe_cells.data()); }
    uintptr_t neighbor_idx_addr() { return reinterpret_cast<uintptr_t>(neighbor_idx.data()); }
    uintptr_t neighbor_len_addr() { return reinterpret_cast<uintptr_t>(neighbor_len.data()); }

private:
    int reveal_flat(int flat) {
        if (!started) {
            place_mines(flat);
            started = 1;
        }
        ++version;
        if (mines[flat]) {
            revealed[flat] = 1;
            --covered_count;
            state         = OVER;
            exploded_flat = flat;
            hit_mine      = 1;
            won           = 0;
            return 0;
        }

        int head = 0, tail = 1, count = 0;
        queue_[0]      = flat;
        revealed[flat] = 1;
        while (head < tail) {
            const int cur = queue_[head++];
            ++count;
            --covered_count;
            remove_from_pool(cur);
            if (neighbor_counts[cur] != 0) continue;
            const int32_t* nb = &neighbor_idx[size_t(cur) * 8];
            for (int k = 0; k < neighbor_len[cur]; ++k) {
                const int g = nb[k];
                if (!revealed[g] && !mines[g]) {
                    revealed[g]    = 1;
                    queue_[tail++] = g;
                }
            }
        }
        revealed_safe_cells += count;
        if (revealed_safe_cells == total_safe_cells) {
            state = OVER;
            won   = 1;
        }
        return count;
    }

    // Choose mine_count distinct cells, never `first`.
    void place_mines(int first) {
        std::fill(mines.begin(), mines.end(), 0);
        const int pool = flat_size - 1;
        const int m    = std::min(mine_count, pool);
        for (int i = 0; i < pool; ++i) pool_[i] = i;
        for (int i = 0; i < m; ++i) {
            const int j = i + int(rng_.below(uint64_t(pool - i)));
            std::swap(pool_[i], pool_[j]);
            mines[pool_[i] + (pool_[i] >= first ? 1 : 0)] = 1;
        }
        compute_counts();
        safe_pool_size = 0;
        for (int f = 0; f < flat_size; ++f) {
            if (!mines[f]) {
                safe_cells[safe_pool_size] = f;
                safe_pos[f] = safe_pool_size++;
            } else {
                safe_pos[f] = -1;
            }
        }
    }

    void compute_counts() {
        for (int f = 0; f < flat_size; ++f) {
            int c = 0;
            const int32_t* nb = &neighbor_idx[size_t(f) * 8];
            for (int k = 0; k < neighbor_len[f]; ++k) c += mines[nb[k]];
            neighbor_counts[f] = uint8_t(c);
        }
    }

    void remove_from_pool(int flat) {
        const int pos = safe_pos[flat];
        if (pos < 0) return;
        const int last      = safe_pool_size - 1;
        const int last_flat = safe_cells[last];
        if (pos != last) {
            safe_cells[pos]     = last_flat;
            safe_pos[last_flat] = pos;
        }
        safe_pos[flat] = -1;
        --safe_pool_size;
    }

    Rng rng_;
    std::vector<int32_t> queue_;
    std::vector<int32_t> pool_;
};

}  // namespace msw
