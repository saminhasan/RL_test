# Minesweeper

A Minesweeper engine, an exact Bayesian mine solver, a benchmark of agents,
learning agents in PyTorch, and a pygame front-end.

The engine and solver are C++ ([cpp/msw/](cpp/msw/)), used from Python through
[cppyy](https://cppyy.readthedocs.io/). There is no build step: cppyy compiles
the headers when Python first loads them.

Reveal-only rules: no flags in the engine, no chording. Flags exist in the GUI
as a cosmetic marker only.

## Layout

| File | What it does |
| --- | --- |
| [cpp/msw/engine.hpp](cpp/msw/engine.hpp) | The game. Flat board, flood fill, first click is never a mine, pool of hidden safe cells. |
| [cpp/msw/solver.hpp](cpp/msw/solver.hpp) | Exact mine probabilities from the visible board. |
| [_native.py](_native.py) | Loads the C++ through cppyy (and sorts out the Windows headers, see below). |
| [engine.py](engine.py) | `MinesweeperEngine`, the Python API on top of the C++ engine. |
| [ai_agents.py](ai_agents.py) | Agents plus a parallel benchmark. |
| [ui.py](ui.py) | pygame GUI with two render styles and a probability overlay. |
| [rl/](rl/) | Learning agents (PyTorch): env, model, learners, evaluation. See [Learning agents](#learning-agents). |
| [models/](models/) | Saved checkpoints and sweep results. |
| [tests/test_engine.py](tests/test_engine.py) | Engine checks; solver and move count checked against brute force. |
| [tests/test_rl.py](tests/test_rl.py) | Learning-code checks, e.g. a "known safe" cell is never a mine. |

## Setup

```
python -m venv .venv
.venv\Scripts\Activate.ps1          # PowerShell
# .venv/Scripts/activate            # bash / git-bash
pip install -U pip
pip install numpy scipy pygame shapely joblib tqdm "cppyy==3.1.2"
pip install torch --index-url https://download.pytorch.org/whl/cu128   # for rl/; drop --index-url for CPU only
```

**Windows needs two things:**

- **cppyy 3.1.2, not the latest.** cppyy 3.5.0 fails to build on Windows (a link
  error in `CPyCppyy`).
- **The MSVC 14.29 C++ headers.** cppyy compiles with its own clang 13, which
  cannot read the standard library headers of newer Visual Studio versions.
  Install "MSVC v142 - VS 2019 C++ x64/x86 build tools" from the Visual Studio
  Installer (it installs next to newer toolsets). [_native.py](_native.py)
  finds it on its own and points cppyy at it. To pick a different folder, set
  `MSW_MSVC_INCLUDE` to `...\VC\Tools\MSVC\<version>\include`.

Linux needs only the pip install. It is untested here.

**cppyy and PyTorch in one process:** cppyy ships an old copy of the Microsoft
C++ runtime. If it loads first, PyTorch fails to start ("DLL initialization
routine failed"). [_native.py](_native.py) loads the system's newer copy first,
so either import order works.

**Startup cost:** the very first `import cppyy` builds a precompiled header
once (a few seconds here; cppyy warns it can take a minute). After that, every
Python process spends about 1 s on import and about 3 s compiling the solver
the first time it is used. This includes each joblib worker.

## Playing

```
python ui.py
```

| Key | |
| --- | --- |
| `1`–`4` | test / easy / medium / hard |
| `H` | toggle help + probability overlay |
| `C` | toggle coloured numbers |
| `T` | switch between the `simple` and `nice` render styles |
| `Z` | let the agent take one known-safe move |
| `R` | restart |
| `Esc` | quit |

Left click reveals, right click plants a flag. With the overlay on, hidden cells
are tinted green→red by mine probability.

## The engine

```python
from engine import MinesweeperEngine

eng = MinesweeperEngine(level=2, seed=7)   # 1=test 6x6, 2=easy 9x9, 3=medium 16x16, 4=hard 16x30
eng.reveal(4, 4)             # or reveal_count(row, col) -> safe cells uncovered
eng.get_public_view()        # int8 board: -1 hidden, 0-8 clue, 9 mine (after a loss)
eng.mine_probs               # float64 board: NaN where revealed, [0,1] where hidden
eng.state, eng.won, eng.move_count
eng.min_win_moves            # fewest moves that win this board (known from the first click on)
eng.min_clicks_left()        # fewest clicks that clear what is still hidden
```

Mines are placed on the first reveal, never on the clicked cell. The same seed
always gives the same board. `reset()` replays the same board and
`reset(seed)` deals a new one. `MinesweeperEngine.from_state(mines, revealed)`
builds an engine from any mid-game position and works out won/lost/running from
the arrays.

The board arrays (`revealed`, `_mines`, `_neighbor_counts`, and the flat
versions) are numpy views straight into the C++ engine's memory, so reading
them costs nothing. `mine_probs` is computed when first read after a move, then
cached until the next move.

## The solver

It is exact, not an estimate. Each revealed clue becomes a rule ("these hidden
cells hold N mines"). The rules are simplified and merged. They are then split
into independent groups, and each group is enumerated in full. Finally the
groups are weighted against each other and against the cells no clue touches,
using the fixed total mine count. If the board is ever inconsistent, the solver
falls back to a uniform probability instead of failing.

It follows mrgriscom's minesweepr algorithm.
[tests/test_engine.py](tests/test_engine.py) checks it against a brute-force
count of every possible mine layout on small boards.

Average time per solve over real game states:

| level | time |
| --- | --- |
| easy 9×9 | 0.03 ms |
| medium 16×16 | 0.08 ms |
| hard 16×30 | 0.14 ms |

## Agents

| Agent | Sees the mines? | Behaviour |
| --- | --- | --- |
| `random_any` | no | Random hidden cell. Dies constantly. The floor. |
| `random_safe` | yes | Random safe cell. Never loses; wastes moves. |
| `bayes` | no | Lowest mine probability, ties broken at random. The plain baseline. |
| `bayes_corner` | no | `bayes`, but the first click goes in a corner. |
| `bayes_open` | no | `bayes`, but among equally safe cells it picks the one likeliest to be a zero. The strongest honest agent. |
| `oracle` | yes | Wins every board in the fewest possible moves. The ceiling. |

`bayes_corner`: a corner has 3 neighbours instead of 8, so a first click there
is the likeliest to land on a zero and open a patch.

`bayes_open`: a zero opens a whole patch in one click, while a number at the
edge of an unopened patch is a wasted click. Its "likeliest to be a zero"
multiplies the neighbours' chances of being safe as if they were independent.
That is exact when the neighbours are already known, which is the usual case.
Before the first click every cell is equally safe, so it starts in a corner too.

All agents get the same seeds. Agents whose first click is the same random draw
(all except `bayes_corner` and `bayes_open`) play the same boards, game for game.

### The fewest possible moves

Mines are placed on the first click, so nobody (oracle included) knows the
board before it. From then on, if you know where the mines are, the fewest
clicks needed is:

- **one click per opening.** An opening is a connected patch of zeros. Clicking
  any cell in it uncovers the whole patch plus the numbers around its edge.
- **plus one click per lone number.** A lone number is a number cell that touches
  no zero, so no opening uncovers it and it has to be clicked directly.

Nothing can do better, and the order doesn't matter. Example (`X` = mine):

```
0 0 1 1 1 0 0
0 0 1 X 1 0 0
0 0 1 1 1 0 0
```

There are two openings (left and right), and they uncover everything except the
two 1s above and below the mine. Those two are lone numbers, so the fewest clicks
is 2 + 2 = 4. Clicking a 1 at the edge of an opening wastes a click, because the
opening would have uncovered it anyway.

`eng.min_win_moves` is that number for the whole game (first click included),
and `eng.min_clicks_left()` is the count for whatever is still hidden. The
`oracle` agent plays it exactly: it clicks openings first, then lone numbers.
Players call this count 3BV.

### Benchmark

```python
from ai_agents import benchmark_agents

benchmark_agents(levels=[1, 2, 3, 4], num_games=1000)
```

It runs in parallel through joblib with `backend="loky"` (worker processes),
which is what works on Windows. Under WSL pass `backend="threading"`. For each
agent it reports win rate, moves, and, for wins:

```
efficiency  = fewest possible moves / moves taken     (1.0 = perfect)
extra moves = moves taken - fewest possible moves
```

Results for 2000 games per level, seed 42:

| level | `bayes` | `bayes_corner` | `bayes_open` | `bayes` efficiency | `bayes_open` efficiency |
| --- | --- | --- | --- | --- | --- |
| test 6×6 | 77.8% | 81.5% | 82.6% | 0.86 | 0.94 |
| easy 9×9 | 84.6% | 89.1% | 91.4% | 0.82 | 0.93 |
| medium 16×16 | 70.9% | 75.2% | 76.6% | 0.81 | 0.91 |
| hard 16×30 | 33.8% | 36.3% | 37.3% | 0.84 | 0.92 |

The first three columns are win rates. Starting in a corner gives most of the
win-rate gain. Preferring zeros gives the efficiency gain plus a little more
win rate. `oracle` wins everything at 1.00, by definition.

Plain `bayes` is no more move-efficient than clicking random safe cells (0.84–0.87).

## Learning agents

The goal: a learning agent that comes close to `bayes_open` on both win rate and
efficiency. It can't beat `oracle`, which sees the mines. Code is in [rl/](rl/):

| File | What it does |
| --- | --- |
| [rl/env.py](rl/env.py) | Many games stepped together: input layers, allowed moves, free labels from the true board. |
| [rl/model.py](rl/model.py) | The network, built from a config dict. Checkpoints hold config + weights + stats, auto-named. |
| [rl/agent.py](rl/agent.py) | Turns a model (or the `bayes_open` rule) into moves; `make_agent` gives a one-game agent like the ones in `ai_agents.py`. |
| [rl/train_safe.py](rl/train_safe.py) | Learns which known-safe cell to click (moves). |
| [rl/train_guess.py](rl/train_guess.py) | Learns which cell to guess (wins), from game results. |
| [rl/train_ppo.py](rl/train_ppo.py) | PPO with a switch per suspect, to find out why plain RL stalls. |
| [rl/evaluate.py](rl/evaluate.py) | Compares models and scripted agents on the same seeds. |
| [rl/sweep.py](rl/sweep.py) | Trains the safe-click part at many sizes to find how small the net can be. |

### How a game splits up

The solver's numbers sort every position into one of three kinds:

- **FIRST**, before the first click. Every cell is safe.
- **SAFE**, some hidden cells are known safe. The agent only picks among those.
- **GUESS**, nothing is known safe. The agent picks among cells that aren't known mines.

Clicking known-safe cells in any order ends at the same board, so SAFE moves
never change whether a game is won. They only change how many moves it takes.
Only FIRST and GUESS moves decide wins, and there are few of them (measured with
`bayes_open`):

| level | guesses per game | games with no guess after the first click | share of moves |
| --- | --- | --- | --- |
| easy | 0.7 | 54% | 4.6% |
| medium | 1.4 | 34% | 2.3% |
| hard | 3.4 | 4% | 3.0% |

The env applies two rules from logic. They aren't cheating, just deduction from
the visible board:

1. never click a known mine
2. if a cell is known safe, click a known-safe cell

### Version 1: the network sees the solver

**Inputs, per cell:**
- hidden, clue 0, …, clue 8
- on board (zero padding marks the edge)
- mines left ÷ hidden cells
- the solver's mine chance
- known safe, known mine

`inputs="board"` drops the last three, for version 2.

**Network:** 3×3 conv layers only, so one net plays every board size.
Optionally each layer has a shortcut (`residual`), so deep nets train. The
default is 32 channels × 6 layers with shortcuts (about 51k weights).

**Outputs:**

| output | meaning | used for |
| --- | --- | --- |
| `zero` | chance the cell is a zero | SAFE moves |
| `wasted` | chance clicking it is a wasted move (a number next to a hidden zero) | SAFE moves |
| `guess` | chance of winning if the cell is clicked and turns out safe | FIRST / GUESS moves |
| `policy`, `value` | click scores and expected return | PPO only |

**How it picks:**
- **SAFE:** the lowest `wasted`, ties going to the highest `zero`.
- **FIRST / GUESS:** the highest P(safe) × `guess`. P(safe) is exact from the solver (1 on the first click).

**The two learners:**

- **Safe clicks** ([train_safe.py](rl/train_safe.py)) use plain supervised
  learning. Every known-safe cell gets free labels from the true board: is it a
  zero, would clicking it be wasted. Guesses stay with the `bayes_open` rule.
- **Guesses** ([train_guess.py](rl/train_guess.py)) learn from game results,
  but only at guess moves. Each round plays games (exploring among the safest
  few cells), then labels every safe guess with the game's result (1 won, 0
  lost) and trains the `guess` output on those labels. A round is kept only if
  it beats the best so far on validation boards.

### Results

All models below were trained on easy (9×9). They were scored on boards they
never saw and that were never used to pick checkpoints (`rl.evaluate` seeds).

**Easy, 5000 games** (win rates are about ±0.8):

| agent | wins | efficiency | extra moves per win |
| --- | --- | --- | --- |
| `bayes` | 84.9% | 0.824 | 3.66 |
| `bayes_open` | 91.3% | 0.925 | 1.36 |
| `oracle` | 100% | 1.000 | 0 |
| safe clicks, 16 ch × 3 layers (6,901 weights) | 90.9% | 0.936 | 1.15 |
| safe clicks, 8 ch × 2 layers (1,717 weights) | 91.5% | 0.936 | 1.14 |
| safe clicks + learned guesses | 91.4% | 0.934 | 1.19 |
| PPO, fixed setup | 91.4% | 0.926 | 1.32 |
| PPO, all suspects on | 89.8% | 0.607 | 11.48 |
| PPO, shaped reward only | 90.3% | 0.695 | 7.52 |

**The same easy-trained models on bigger boards** (medium: 3000 games, about ±1.5;
hard: 2000 games, about ±2.1):

| agent | medium wins | medium efficiency | hard wins | hard efficiency |
| --- | --- | --- | --- | --- |
| `bayes` | 69.8% | 0.812 | 33.8% | 0.842 |
| `bayes_open` | 76.7% | 0.909 | 37.0% | 0.918 |
| safe clicks, 16 × 3 | 77.2% | 0.920 | 38.8% | 0.921 |
| safe clicks, 8 × 2 | 77.4% | 0.921 | 37.4% | 0.925 |
| safe clicks + learned guesses | 71.7% | 0.917 | 34.3% | 0.918 |
| PPO, fixed setup | 76.2% | 0.910 | 26.6% | 0.917 |

What this says:

1. **Learning the safe clicks works, and it carries over to other board sizes.**
   - Efficiency beats `bayes_open` at every size: 0.936 vs 0.925 on easy, 0.92 vs 0.91 on bigger boards.
   - Win rate is unchanged, as it should be.
   - The ceiling is lower than 1.0: picking safe clicks with the true labels (cheating) reaches 0.954 on easy and 0.939 on medium. Sometimes every known-safe cell is a wasted click, and clicking one still beats guessing.
2. **Nothing beats `bayes_open`'s win rate by more than noise.**
   - Guess learning from game results reaches rule level on easy but gets worse on bigger boards.
   - PPO holds rule level on easy, then falls apart on hard (26.6% vs 37.0%).
3. **What the PPO switches showed:**
   - **A shaped reward (+0.3 per safe click) ruins efficiency.** With only that switch flipped, efficiency slides from 0.93 to 0.70 while the win rate stays flat. The policy learns to take many small clicks. With every suspect switched on, it drops to 0.61.
   - **The win signal is too weak to move.** There are 0.7–3.4 guesses per game, and a gain of 1–2 points needs tens of thousands of games just to see. Every PPO run's win rate wandered within ±1.5 points for all 400 updates.
   - **The other suspects showed no measurable effect at this length:** entropy bonus, untrained value estimate at the start, and no logic rules.
4. **Size:** for safe clicks, 8 channels × 2 layers (1,717 weights) was as good
   as nets 20× bigger, from all three starting seeds tried. Nets with 4
   channels work from some starting weights but not others, and 1 layer is not
   enough. Results are in `models/sweep-*.json`.

**Bigger and deeper:** 32 channels × 6 layers with shortcuts (50,757 weights).
Safe clicks were trained on easy + medium, guesses on medium. Scored on the same
fresh boards as the tables above:

| agent | easy wins | easy efficiency | medium wins | medium efficiency | hard wins | hard efficiency |
| --- | --- | --- | --- | --- | --- | --- |
| `bayes_open` | 91.3% | 0.925 | 76.7% | 0.909 | 37.0% | 0.918 |
| deep, safe clicks | 91.8% | 0.937 | 77.1% | 0.925 | 37.8% | 0.926 |
| deep, safe clicks + guesses | | | 77.8% | 0.925 | 33.1% | 0.923 |

Depth bought a little efficiency on medium (0.920 → 0.925; the ceiling there is
0.939) and nothing on win rate. Guess learning on medium followed the same path
as on easy: round 1 dropped to 72%, later rounds got back to rule level, and it
did worse on hard. With the solver as input, the size of the net is not what
holds things back; the guess signal is. Depth should matter more for version 2,
where the net has to work out the mine chances itself.

### Running it

```
python -m rl.train_safe  --levels 2 3 --games 10000 --keep 0.1      # safe clicks (default 32 ch x 6 layers)
python -m rl.train_guess --level 3 --model <safe checkpoint>         # guesses, from game results
python -m rl.train_ppo   --level 2 --updates 400                     # PPO, fixed setup
python -m rl.train_ppo   --level 2 --updates 400 --no-critic-warmup --entropy 0.01 --no-rules --reward shaped
python -m rl.sweep       --level 2 --channels 4 8 16 --layers 2 3 --repeats 3
python -m rl.evaluate    --level 3 --games 3000 --models <name> <name>
```

Checkpoints land in `models/` with names like `safe-solver-c32x6r-<date>-<time>.pt`
(`r` = shortcuts). Training seeds, validation seeds (for picking checkpoints) and
evaluation seeds (for reporting) never overlap. `make_agent(model)` from
[rl/agent.py](rl/agent.py) gives a one-game agent for the GUI or `ai_agents.py`.

### What next

- **Better guess signal.** The final result is a noisy label for one guess. A
  closer one is whether the guess leads to new known-safe cells before the next
  guess, which strong scripted solvers use. The solver can compute that directly.
- **Version 2** (`inputs="board"`): the net doesn't see the solver and has to
  learn mine chances itself, using the free `mine` labels. It needs honest
  probabilities rather than a loss that bends them, because choosing between
  two risky guesses depends on the exact numbers. That version also needs the
  deeper nets.
- **GUI:** start, pause and stop training, saving and loading models, a model
  builder and viewer, and model-vs-agent play on the same boards. The
  config-driven model and `make_agent` are the hooks for it.

## Changing the C++

Edits to the headers take effect the next time Python starts. There is nothing
to rebuild. Two cppyy-on-Windows rules apply:

- **No C++ exceptions.** Any `throw` aborts Python, even one that gets caught.
  The code reports bad input by returning `false` or `-1`.
- **If a function fails to link, the whole process is broken.** cling prints
  `Failed to materialize symbols`, and from then on every later C++ call in that
  process fails too. The known trigger is `std::stable_sort`, which needs
  `std::exception` RTTI that cling cannot find. Use `std::sort` with an index
  tiebreak instead.

Run the tests after any change:

```
python tests/test_engine.py
python tests/test_rl.py
```

## Notes

`benchmark_agents` under the `loky` backend starts worker processes, so on
Windows put your own calls under `if __name__ == "__main__":`. `ui.py` uses
`ctypes.windll` for screen metrics when it can, and falls back to pygame's
display info elsewhere.
