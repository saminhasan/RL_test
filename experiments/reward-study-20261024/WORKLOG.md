# Study work log

The requested work is ongoing: preserve the current repo/run and investigate
repeatable learned-agent gains over the full Bayesian policy. No superiority
claim has been established. The reserved test split remains unused.

## Checkpoints and active work

- Branch: `reward-study-20261024`.
- Original-code/run snapshot: commit `13e3083`, tag
  `reward-study-baseline-20261024`.
- Controlled experiment implementation: commit `1c44cb8`.
- Original `easy-exp2-c64x16-20261024-040042` continues independently. Its live
  files and process have not been targeted for stopping or replacement.
- Frozen best/latest models are `models/reward-study-20261024-baseline-*.pt`.
  Their hashes and arguments are in `baseline/manifest.json`.
- Main screen is running via `python -m rl.reward_sweep --phase screen --jobs 4`.
  It runs all 13 recipes sequentially, 500k additional clicks each, seed 101.
  The source/config hashes are locked; changing its core source mid-screen
  makes it stop before the next job. New independent modules do not change the
  frozen core source. Inspect `status.json` and each `screen/*/manifest.json`.
- Create this study directory's `STOP` file to cleanly stop study work. This is
  separate from the original run's stop controls.
- Refresh human/machine reports with `python experiments/reward-study-20261024/report.py`.

## Results established so far

All figures below use validation, not the final test split.

The full Bayesian policy wins 88.60% on the 2,000-board study validation block,
with move efficiency 0.9359. The frozen bf16 model wins 85.10%, efficiency
0.8602, paired gap -3.50 points (95% interval [-5.35, -1.65]).

Higher-precision inference did not improve that baseline. Eight-way symmetry
averaging scored 85.85%, efficiency 0.8735; it still trails Bayes.

Completed equal-budget screen cases:

| Recipe | Best checkpoint / latest wins | Efficiency of latest |
| --- | ---: | ---: |
| Unchanged objective, original precision | 84.85% / 85.80% | 0.862 |
| Unchanged objective, fp32 output head | 83.50% / 83.50% | 0.849 |
| Pure immediate risk | 84.75% / 84.55% | 0.783 |
| Mine/reveal ratio 10 | 84.05% / 84.05% | 0.896 |

These are single-training-seed screens and exploratory comparisons. The
`RESULTS.md`/`summary.json` reports and per-case result JSONs are authoritative
as further cases finish.

## Diagnostic evidence

- `--reward-burst` was ignored by the original CLI wiring. It is now passed to
  the environment and tested.
- BCE normalization makes joint reward rescaling redundant; changing a ratio
  changes the objective, multiplying all rewards does not.
- `tiny_objective.py` exactly reproduces 77.2487% for immediate/Bayes play
  versus 81.3492% optimal future-win play on 3x3/two-mine boards. This is a toy
  objective diagnostic, not evidence of a gain on 9x9.
- `baseline-policy-audit.json`: in 500 CPU/fp32 validation games, 18 choices
  ignored an available safe cell and 73 forced guesses were riskier than the
  safest available. Fatal-only breakdowns omit earlier errors that survive.
- Historical model validation: wall4 87.40% / efficiency 0.7875, stage3m40
  86.60% / efficiency 0.8688. Wall4 inherited exact-solver coin-flip filtering;
  it is a separate confounded reference, not a pure-reward result.
- Scripted solver-assisted risk/opening references gained at most one win in
  2,000 boards. That is not a reliable improvement and is not a learned-agent
  success.
- `risk_consistency.json`: on 1,000 Bayes-distribution states, most public clue
  constraints are nearly satisfied, with larger tail errors. Global mine-count
  MAE is 0.48 mines. Low predicted-risk bins have substantial overconfidence.
  The audit is measurement only; no solver targets entered training.

## Prepared next stage

New independent modules `rl/safety_opening.py` and `rl/public_constraints.py`
separate selected-click binary safety from conditional zero/reveal prediction.
Optional consistency losses use only visible clues and the public mine count.
The prototype passed focused CPU tests, checkpoint/resume and CLI evaluation
checks, plus a 1,024-click CUDA gradient/validation smoke. The smoke's 20-game
score is too small to interpret as a performance result.

An auxiliary sweep is being prepared in `rl/auxiliary_sweep.py`: safety control,
zero auxiliary alone, zero-driven policy, public consistency alone, and two
combined weights. It should wait for the main screen and preserve one study
training child at a time. Initialize from the immutable pure-risk model:

`screen/risk_only-s101-n500000-v2000/checkpoints/latest.pt`

SHA256: `8ec3c4a583fe1508f212273824174d908831289b7416f8d5bbbfbc76c542d4ad`.

Choose finalists using validation only, freeze their policy settings, then run
at least three training-seed repeats and a predeclared held-out comparison.
Report warm-start repeatability honestly; these starts share one frozen model.

One conservative guard in the main BCE trainer rejects all nonzero flat safe
rewards with lookahead. Some negative-cost cases are actually bounded when
`-safe <= mine * (1 - gamma)`. See historical metadata before a later guard
revision; do not change frozen screen source while it is running.
