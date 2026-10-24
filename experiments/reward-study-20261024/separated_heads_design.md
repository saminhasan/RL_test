# A board-only policy with separate safety and opening predictions

This is a proposed second-stage experiment, not a measured result. No training
source or frozen-screen configuration was changed to write this note.

The current screen's parent-reported validation results are 85.1% wins for the
frozen baseline, 84.9% with FP32 inference, and 85.85% with eight symmetries,
against 88.6% for Bayes. Precision changes alone have not closed the win gap.

## Why separate the predictions

With mine penalty 10,000 and maximum reveal reward 100, the current normalized
BCE target is 0 for a mine and approximately 0.9902--1 for a safe click. One
output must simultaneously encode safety and comparatively small opening gains.
Changing the ratio also changes that encoding and the preferred tradeoff.

A separate safety head can instead learn the actual binary observation:
`P(the selected cell is safe | visible board, selected cell)`. An auxiliary
opening head can improve the shared representation without changing the safety
target, and an independently controlled opening term can change action ranking.
This is a hypothesis to test, not a guarantee that this factorization is better.

## Labels available from the selected click

Use the transition's real post-click observation, including `final_obs` for a
completed episode, before the vector environment substitutes its reset board.
For the legal selected cell `a`, the current observation must be hidden (0).
The encoded post-click selected-cell code has this meaning:

| Post-click code at `a` | Selected-click label |
| --- | --- |
| 0 | Mine: it remains hidden in the public view when mine display is disabled |
| 1 | Safe and zero |
| 2--9 | Safe and clue 1--8 |
| 10 | Invalid off-board sample: reject it |

This inference relies on the actual current engine/public-view contract. Assert
it in a small integration test before any training, and retain an explicit
clicked-mine outcome field if the observation contract changes. A no-op on an
already revealed cell is not a mine label: require pre-click legality.

For one legal selected click, derive:

```
safe = 1 <= next_view[a] <= 9
zero = next_view[a] == 1                         # train only when safe
k = count((view == 0) & (1 <= next_view <= 9))   # actual opening from this click
S = number_of_on_board_cells - total_mine_count
reveal_fraction = k / S                        # train only when safe
won = count(1 <= next_view <= 9) == S
```

Only the selected action receives these targets. Do not invent counterfactual
opening sizes or mine labels for unclicked hidden cells. The total mine count,
visible clues, and changes to the visible board are public observations. Reward
sign need not be decoded, so these labels remain valid across reward recipes.

For future-win learning, retain explicit `terminated`, `truncated`, episode ID,
terminal won/lost outcome, behavior epsilon and policy version. In the current
gamma-zero replay, every stored bootstrap discount is zero; that field cannot
distinguish a continuing click from the end of an episode. Truncated episodes
are censored, not observed losses. An episode interrupted by a shutdown is also
censored unless a separate bootstrap rule is explicitly chosen.

## Heads and losses

Keep the existing 64x16 trunk and compute all output heads in FP32.

* `safe_logit`: binary BCE, safe=1 and mine=0, for every selected legal click.
* `zero_logit`: conditional BCE for zero versus nonzero on safe selected clicks.
* `reveal_logit`: sigmoid gives expected revealed fraction conditional on a
  safe click; train with MSE against `k/S` on safe selected clicks.
* Optional `win_logit`: eventual-win probability conditional on the selected
  click being safe and following the specified continuation policy.

Start with `L = L_safe + 0.05 L_zero + L_reveal`; these are proposed initial
weights, not tuned constants. Record gradient magnitudes and include a detached
auxiliary-trunk ablation if auxiliary gradients degrade safety. A Huber or
log-opening target is a possible separate experiment, but its optimum is not in
general the arithmetic expected opening size required by an expected-reward
formula. Initialize the reveal output near a small opening fraction, not 0.5.

Use ordinary unweighted safety BCE to retain its proper-probability objective.
If balancing rare mine examples or prioritizing samples, apply inverse sampling
weights; otherwise a class-weighted predictor is not a calibrated probability
without correcting its prior/odds. Calibrate and score NLL/Brier on fresh
selected-click outcomes, never solver mine-probability labels.

## Action selection using only learned scores

The only hard legality mask is the existing public hidden-cell mask. Predictions
do not become logically known-safe cells.

Begin with safety only: choose the largest `safe_logit`. This isolates the effect
of binary labels from the efficiency objective. Then compare these alternatives:

1. Auxiliary heads train, but actions still use safety only. This tests whether
   additional selected-click observations improve the shared representation.
2. `score = log_sigmoid(safe_logit) + lambda * P(zero | safe)`.
3. `score = log_sigmoid(safe_logit) + lambda * E(k/S | safe)`.

Try a small declared lambda grid such as 0, 0.001, 0.003, 0.01, exclusively on
validation. For small risks `log P(safe)` is approximately minus mine risk, so
the opening term has an explicit, bounded scale. It can still accept risk to get
an opening. A finite lambda is not a logical safety guarantee, and finite-data
probability calibration does not establish that a predicted risk is exactly 0.
Use `log_sigmoid` directly rather than `log(1 - predicted_mine)` to avoid numeric
cancellation at very small predicted mine probabilities.

If symmetry averaging is enabled, map and average each head's probabilities or
predicted reveal fractions first, then form the composite action score.
`SymmetryEnsemble(probabilities=True)` on the final composite score would apply
an inappropriate sigmoid and would not implement this policy.

## Future-win head: separate, later experiment

The Bayes baseline minimizes immediate mine probability with a zero-opening tie
break. Surpassing its win rate can require preferring a safe outcome that leaves
better future decisions. Pure safety/zero learning does not promise that gain.

For a conditional continuation head `w`, use
`P(win after this action) = P(safe) * w`. This makes the proposed score grounded
in eventual wins rather than merely opening reward. The conditional head has no
label on a mine action. It gets the episode's terminal win/loss on safe selected
actions, so all its labels remain observed outcomes.

Start this study with fresh, nearly greedy self-play on training seeds, a frozen
or slowly updated continuation policy, and recent-policy episode replay. Never
use validation/test boards to create training labels. Labeling highly
exploratory or old-policy outcomes as current-policy win probabilities creates
off-policy bias: those probabilities describe the behavior that actually ran.
Record policy version/epsilon, and do not claim that ordinary MC labels solve
that bias. Prefer recent greedy rollouts to high-variance whole-trajectory
importance ratios for the first ablation.

A later TD variant can bootstrap safe continuing clicks from the target
network's full `P(safe)*w` at the next online-selected action, with exact terminal
targets won=1/lost=0. Keep the safety target binary and separate. This introduces
bootstrapping/model error and should be compared against observed-outcome MC,
not silently substituted for it. Clip neither binary targets nor logits during
training. Episode labels are correlated; episode-aware sampling or weighting
can stop long games from dominating the continuation loss.

## Warm start and compatibility

Copy the existing body and output-head weights into the safety head. Its old
normalized-reward predictions are useful initial rankings, but not calibrated
safety probabilities; train against the new binary labels. Do not invert the
old reward-to-sigmoid transform and clamp it as a training head, which can create
zero-gradient saturation and retains unknown opening contamination.

Keep the initial acting policy frozen during a short head warmup, then release
it. One proposed schedule is 20k clicks with frozen trunk/new heads at 1e-4,
followed by trunk at 1e-5 and heads at 1e-4, measured at 100k/300k/1M clicks.
Record all these choices and compare equal interaction/update budgets.

Use a versioned architecture/config and explicit checkpoint loader dispatch.
Keep an immutable reference checkpoint. Existing single-head checkpoint loading
must continue to work, and adding auxiliary heads must not silently load them as
trained weights. A model's `forward` may return its configured scalar action
score for the current evaluator, while a separate method returns all heads for
training. Store loss definitions, selection formula, lambda, calibration, and
head version in the checkpoint; also record that inference may require
head-aware symmetry averaging.

## Small, ordered ablation rather than every combination

After the frozen reward screen finishes, use its best compatible warm start:

1. Existing reward head, unchanged, continued for the same budget.
2. Binary safety head only.
3. Binary safety plus zero auxiliary, safety-only actions.
4. Binary safety plus reveal auxiliary, safety-only actions.
5. Best auxiliary model plus the predeclared small opening-score grid.
6. Only then the conditional future-win experiment.

Use three independent training seeds for finalists. Choose settings/checkpoints
on the validation split, compare paired wins and common-board outcomes against
Bayes, then freeze the finalist before one untouched test evaluation. Report
each training seed and aggregate uncertainty across training seeds separately
from paired board confidence intervals. A favorable selected checkpoint on the
same boards is not by itself a repeatable win over Bayes.

## Audit every decision, including mistakes that survived

The existing loss breakdown looks only at the fatal click. An earlier needless
guess may survive, change the subsequent trajectory, and end in a death labeled
"unlucky". Those labels do not establish that all earlier choices were sound.

For measurement only, run the exact solver before every policy-selected click
and record chosen risk, minimum available risk, their difference, whether any
known-safe choice exists, whether the policy nevertheless guesses, and whether
a forced guess chooses above the minimum risk. Also retain forced-guess counts,
progress at guesses, and per-board trajectories. Report rates both per board
and per decision, since policies that clear more boards also make more moves.
`rl/policy_audit.py` provides this independent audit without training labels or
any policy access to solver values.

The sum of immediate excess mine probabilities over actually visited positions
is a diagnostic. It is not a proof of optimal eventual-win regret: another
action can change future information and future positions, and agents visit
different trajectories. A lower fatal-click error count alone can also obscure
earlier errors.

## Keep solver-filtered training separate in provenance

The original frozen easy-exp2 baseline trained without solver filtering. A run
using `--skip-coin-flips` calls the exact solver during training and excludes
selected transitions on solver-defined coin flips. Even though the network
still receives visible-board observations, that is solver-assisted sample
selection. A wall=4 recipe that also enables this switch changes both the
reward ratio and training data selection. It must not be reported as a clean
reward-only ablation or as a strictly unassisted successor to the baseline.

Record both the reward configuration and every solver/filter switch in results.
For the proposed selected-outcome multi-head study, leave this switch off in
the primary comparison. If tested separately, name it as solver-filtered
training and compare paired with/without variants under the same reward.

## Public-clue and mine-budget consistency auxiliary

A pure binary safety head also permits a dense, differentiable constraint loss
without solver probabilities or hidden labels. Let `p_i = sigmoid(-safe_logit_i)`
on a hidden cell and zero on a visible/off-board cell. A revealed clue c requires
the sum of neighbouring hidden-cell mine indicators to equal c, so their
conditional expectations must sum to c too. The sum of all hidden-cell mine
probabilities must equal the public mine total. The true posterior satisfies
both relations because expectation is linear; no independence assumption is
needed to justify these equalities.

`rl/public_constraints.py` implements:

```
clue_loss, budget_loss = public_constraint_loss(obs, safe_logits)
L = selected_click_BCE + lambda_clue * clue_loss + lambda_budget * budget_loss
```

For each revealed clue with at least one hidden neighbour, its local residual
is `(sum_neighbour_p - clue) / number_of_hidden_neighbours`. Square this, average
over active clues within each board, then average over boards. This makes a
residual comparable to a per-cell probability error, preserves strong
single-unknown deductions, and avoids assigning greater overall weight just
because a board has more clues. Boards without active clues have local loss 0.

The budget residual is `(sum_hidden_p - public_mines) / max(public_mines, 1)`.
Its square penalizes relative mine-count error and remains active without any
revealed clue. A proposed initial weighting is lambda_clue=1 and
lambda_budget=0.1. Compare 0/0, 1/0, 0/0.1 and 1/0.1 before adding a weight grid.
Track each loss and its gradient magnitude; these numbers are a starting
experiment, not evidence that a weight is optimal. Normalizing the global
residual by hidden-cell count is an alternative, but can make its gradient very
small when few of many hidden cells contain mines; it is a different ablation.

All prediction arithmetic uses FP32 and `sigmoid(-safe_logit)` rather than
`1 - sigmoid(safe_logit)`, which loses precision near certainty. Visible clue
cells and off-board padding are excluded from the mine budget. A convolution
with a 3x3 ones kernel and zero center sums the eight neighbours correctly at
edges. Use the actual pure-safety logits, not the old normalized-reward logits
or a composite action score. Apply constraints to policy inputs/replay states,
not to a hidden layout. The observations and current action legality remain
unchanged.

These soft linear equations underidentify the posterior. Many fractional maps
and sometimes arbitrarily chosen legal mine assignments achieve zero loss;
zero constraint residual does not mean calibrated individual risks, correct
correlations, or optimal action selection. Retain proper binary selected-click
BCE as the empirical anchor, ordinary unweighted sampling (or appropriate
inverse-sampling correction), and fresh-outcome calibration checks. Uniform
sampling of replay states is not the same as uniform sampling of cells, and
class balancing without correction changes the target probabilities. Overlapping
clues produce correlated constraints; do not describe their number as an
independent labeled sample count. Monitor chosen-cell NLL/Brier and win rate,
as well as all-board constraint residuals. Strong auxiliary weights can harm
calibration even while satisfying public counts more accurately.

Finally, compare this auxiliary under the same interaction/update budget and
without solver-filtered replay. Keep zero/reveal auxiliaries and action-score
changes separate until the public-consistency effect is established.
