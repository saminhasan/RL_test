# Reward study: easy Minesweeper

Goal: a board-only learned policy with a repeatable win-rate advantage over the
current full Bayesian policy on easy 9x9 boards. Move efficiency is a second
metric. A move-efficiency gain alone does not satisfy the win-rate goal.

## Frozen starting point

Git baseline: `13e3083`, tag `reward-study-baseline-20261024`, branch
`reward-study-20261024`. The baseline includes the user's previously uncommitted
progress-scaled reveal reward and exponential learning-rate schedule.

`baseline/manifest.json` records checkpoint hashes and original arguments.
Frozen local models (ignored by git):

- `models/reward-study-20261024-baseline-best.pt`: 2,800,000 clicks.
- `models/reward-study-20261024-baseline-latest.pt`: 5,300,032 clicks.

`baseline/status.json` and `baseline/metrics.jsonl` are snapshots, not live files.
The original `easy-exp2` process finished naturally at 10M clicks. Its best
legacy evaluation was 85.8% at 5.6M; its final score was 81.2%. The
`original-final/` snapshot and frozen models preserve that final state.

## Protocol

1. Use the frozen best model for all initial ablations and warm starts.
2. Compare inference precision and symmetry averaging without retraining.
3. Screen reward ratios, progress/late/burst rewards, and eventual-win objectives
   with equal additional click budgets and the same training seed.
4. Choose settings using validation boards only. Then repeat the chosen training
   recipe with at least three training seeds. Report that these are warm-start
   repeats from one frozen network, rather than independent training from scratch.
5. Freeze the finalists and their inference settings before evaluating the test
   split. Report paired per-board win differences and confidence intervals for
   every repeat, not just the best observed checkpoint. Further tuning after
   seeing test results requires a newly reserved test block.

Training boards are drawn below `2**31`. The legacy evaluation block begins at
10 billion and has already been used to select the frozen starting checkpoint.
Study validation for level 2 begins at **1,108,101,562,368**. Study test begins at
**2,207,613,190,144**. The test block remains unopened during screening.
The trainer's matching validation argument is `--eval-seed0 1108081562368` because
it adds `level * 10,000,000` to its base.

`rl.experiment_eval` saves requested seeds, effective board seeds, per-board
records, source and checkpoint hashes, device/precision settings, paired
bootstrap intervals and exact McNemar tests. Statistics from validation are
exploratory and unadjusted for the number of candidates. Repeatability across
training seeds and a preselected test comparison are required for a success
claim. The full Bayes policy uses its existing zero-opening tie break; no weaker
reference is substituted. The learned policy receives no solver probabilities,
oracle actions, mine-layout labels, or extra first-click privilege.

## Findings before screening

- Current `gamma=0` trains expected immediate reward, not eventual win probability.
- Under BCE normalization, mine/reveal settings `10000/100` and `1/0.01` give
  identical targets. The ratio matters; multiplying both does not.
- `--reward-burst` was not wired into the environment; this is now fixed.
- Repeated flat safe-click payments can exceed the old BCE return bounds with
  lookahead. Those combinations now require Huber instead of silently clipping
  their targets. Potential shaping in any BCE stage is also rejected.
- bf16 caused tied best scores on 16.7% of 461 sampled positions. fp32 changed
  8.9% of choices, but this small diagnostic did not show safer choices. See
  `diagnose_precision.py` and `precision_baseline.json`.
- On 2,000 new validation boards, the frozen bf16 baseline won **85.10%** versus
  **88.60%** for Bayes; paired difference **-3.50 percentage points**, 95% interval
  **[-5.35, -1.65]**. This is the starting gap, not a successful improvement.

## Commands

```powershell
.venv\Scripts\python.exe -m rl.reward_sweep --phase inference
.venv\Scripts\python.exe -m rl.study_io reward --phase screen
.venv\Scripts\python.exe -m rl.study_io auxiliary --wait-for-screen
.venv\Scripts\python.exe -m rl.study_io confirm --wait-for-sweeps
.venv\Scripts\python.exe -m rl.experiment_eval --model models/<frozen-finalist>.pt --split test --games 10000 --output experiments/reward-study-20261024/final-test.json
```

Create `experiments/reward-study-20261024/STOP` to stop study work cleanly. This
control is separate from the original run. Completed experiment artifacts stay
on disk and rerunning the same command resumes or skips completed work.

The auxiliary batch waits for the first screen, then runs six equal-budget
experiments from the pure-risk checkpoint. It compares separate conditional
opening predictions and optional consistency losses from public clues and the
public mine count. These losses do not use the solver or hidden-layout labels.
Each case uses seed 201 and 500k additional clicks. Policy opening coefficients
are selected on validation only. See `separated_heads_design.md` for assumptions
and `RESULTS.md` for the evolving comparison table.

`rl.study_io` adds bounded retries for Windows sharing locks during atomic JSON
publication. It leaves the frozen algorithm source and existing experiment
seals unchanged; `io-supervision.jsonl` records wrapper hashes and invocations.
The original controller encountered this Windows error during `burst2`. Its
trained checkpoint was recovered and the remaining trials resumed without
changing their training configuration. The auxiliary cases completed in the
intervening period; their independent initialization and click budgets remain
unchanged.

On 2026-10-25 UTC, the remaining workers were found absent with no recorded
failure cause. The interrupted `future_win3` trial had reached 81,408 clicks,
before its first scheduled checkpoint. Its original run directory and manifest
were archived under `interruptions/20261025-before-first-checkpoint/`, then the
same seed and full 500k-click protocol were restarted from the frozen init.
The discarded partial trajectory was not evaluated. Completed trials were
preserved; confirmation still waits for the full screen before opening test.
