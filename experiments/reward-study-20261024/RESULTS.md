# Reward study results

Updated 2026-10-25T01:16:02+00:00.

Baseline code checkpoint: `13e3083`, tag `reward-study-baseline-20261024`.

**No repeatable win-rate advantage over Bayes has been established.**

Screening numbers below are validation results. The final test split is still unused.

Current: **future_win3**, training, 92,352 / 500,000 clicks.

## Frozen model: inference ablations

Same 2,000 validation boards; Bayes wins 88.60% with efficiency 0.936.

| Inference | Wins | Efficiency | Paired win gap vs Bayes | 95% interval |
| --- | ---: | ---: | ---: | ---: |
| [symmetry-bf16](baseline-validation-symmetry-bf16.json) | 85.85% | 0.874 | -2.75 points | [-4.55, -0.95] |
| [symmetry-fp32](baseline-validation-symmetry-fp32.json) | 85.60% | 0.874 | -3.00 points | [-4.80, -1.20] |
| [bf16](baseline-validation-bf16.json) | 85.10% | 0.860 | -3.50 points | [-5.35, -1.65] |
| [fp32head](baseline-validation-fp32head.json) | 84.95% | 0.857 | -3.65 points | [-5.55, -1.75] |
| [fp32](baseline-validation-fp32.json) | 84.90% | 0.859 | -3.70 points | [-5.55, -1.85] |

## Equal-budget training screen

13 recipes, each warm-started from the same frozen model and trained for 500,000 additional clicks with seed 101. Best checkpoints are selected on 500 validation boards, then best and latest are measured on the same 2,000-board validation set. These comparisons are exploratory.

| Recipe | State | Checkpoint | Wins | Efficiency | Gap vs Bayes |
| --- | --- | --- | ---: | ---: | ---: |
| [burst2](screen\burst2-s101-n500000-v2000\result.json) | complete | best at 300,032 | 84.75% | 0.833 | -3.85 points |
| [burst2](screen\burst2-s101-n500000-v2000\result.json) | complete | latest at 500,032 | 84.15% | 0.837 | -4.45 points |
| [burst5](screen\burst5-s101-n500000-v2000\result.json) | complete | best at 200,000 | 84.60% | 0.820 | -4.00 points |
| [burst5](screen\burst5-s101-n500000-v2000\result.json) | complete | latest at 500,032 | 81.55% | 0.816 | -7.05 points |
| [future_win1](screen\future_win1-s101-n500000-v2000\result.json) | complete | best at 500,032 | 86.65% | 0.723 | -1.95 points |
| [future_win1](screen\future_win1-s101-n500000-v2000\result.json) | complete | latest at 500,032 | 86.65% | 0.723 | -1.95 points |
| future_win3 | training | pending | pending | pending | pending |
| [late10](screen\late10-s101-n500000-v2000\result.json) | complete | best at 400,000 | 85.35% | 0.862 | -3.25 points |
| [late10](screen\late10-s101-n500000-v2000\result.json) | complete | latest at 500,032 | 85.40% | 0.862 | -3.20 points |
| [late2](screen\late2-s101-n500000-v2000\result.json) | complete | best at 400,000 | 84.40% | 0.862 | -4.20 points |
| [late2](screen\late2-s101-n500000-v2000\result.json) | complete | latest at 500,032 | 84.45% | 0.866 | -4.15 points |
| [old_precision_control](screen\old_precision_control-s101-n500000-v2000\result.json) | complete | best at 300,032 | 84.85% | 0.863 | -3.75 points |
| [old_precision_control](screen\old_precision_control-s101-n500000-v2000\result.json) | complete | latest at 500,032 | 85.80% | 0.862 | -2.80 points |
| [precise_ratio100](screen\precise_ratio100-s101-n500000-v2000\result.json) | complete | best at 300,032 | 83.50% | 0.849 | -5.10 points |
| [precise_ratio100](screen\precise_ratio100-s101-n500000-v2000\result.json) | complete | latest at 500,032 | 83.50% | 0.849 | -5.10 points |
| [ratio10](screen\ratio10-s101-n500000-v2000\result.json) | complete | best at 500,032 | 84.05% | 0.896 | -4.55 points |
| [ratio10](screen\ratio10-s101-n500000-v2000\result.json) | complete | latest at 500,032 | 84.05% | 0.896 | -4.55 points |
| [ratio1000](screen\ratio1000-s101-n500000-v2000\result.json) | complete | best at 100,032 | 83.95% | 0.807 | -4.65 points |
| [ratio1000](screen\ratio1000-s101-n500000-v2000\result.json) | complete | latest at 500,032 | 82.65% | 0.818 | -5.95 points |
| [reveal_progress](screen\reveal_progress-s101-n500000-v2000\result.json) | complete | best at 300,032 | 85.30% | 0.862 | -3.30 points |
| [reveal_progress](screen\reveal_progress-s101-n500000-v2000\result.json) | complete | latest at 500,032 | 85.10% | 0.862 | -3.50 points |
| [risk_only](screen\risk_only-s101-n500000-v2000\result.json) | complete | best at 400,000 | 84.75% | 0.785 | -3.85 points |
| [risk_only](screen\risk_only-s101-n500000-v2000\result.json) | complete | latest at 500,032 | 84.55% | 0.783 | -4.05 points |

## Separate safety/opening heads and public consistency

Each checkpoint's displayed lambda is selected from its validation grid. This is additional validation selection, not a held-out success claim. Complete grids remain in each linked result.

| Recipe | State | Checkpoint | Selected lambda | Wins | Efficiency | Gap vs Bayes |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| [public_rules](auxiliary\public_rules-s201-n500000-v2000\best-validation.json) | complete | best | 0 | 84.45% | 0.780 | -4.15 points |
| [public_rules](auxiliary\public_rules-s201-n500000-v2000\latest-validation.json) | complete | latest | 0 | 82.05% | 0.777 | -6.55 points |
| [safety_control](auxiliary\safety_control-s201-n500000-v2000\best-validation.json) | complete | best | 0 | 85.60% | 0.781 | -3.00 points |
| [safety_control](auxiliary\safety_control-s201-n500000-v2000\latest-validation.json) | complete | latest | 0 | 84.20% | 0.783 | -4.40 points |
| [zero_policy](auxiliary\zero_policy-s201-n500000-v2000\best-validation.json) | complete | best | 0.01 | 84.80% | 0.894 | -3.80 points |
| [zero_policy](auxiliary\zero_policy-s201-n500000-v2000\latest-validation.json) | complete | latest | 0 | 84.30% | 0.737 | -4.30 points |
| [zero_public](auxiliary\zero_public-s201-n500000-v2000\best-validation.json) | complete | best | 0.03 | 83.70% | 0.908 | -4.90 points |
| [zero_public](auxiliary\zero_public-s201-n500000-v2000\latest-validation.json) | complete | latest | 0.03 | 83.70% | 0.908 | -4.90 points |
| [zero_public_strong](auxiliary\zero_public_strong-s201-n500000-v2000\best-validation.json) | complete | best | 0.001 | 84.00% | 0.863 | -4.60 points |
| [zero_public_strong](auxiliary\zero_public_strong-s201-n500000-v2000\latest-validation.json) | complete | latest | 0 | 83.65% | 0.735 | -4.95 points |
| [zero_representation](auxiliary\zero_representation-s201-n500000-v2000\best-validation.json) | complete | best | 0.001 | 85.60% | 0.847 | -3.00 points |
| [zero_representation](auxiliary\zero_representation-s201-n500000-v2000\latest-validation.json) | complete | latest | 0.001 | 85.10% | 0.871 | -3.50 points |

## Interpretation

Changing the immediate reward ratio alone cannot train future-win predictions when gamma is zero. The screen also includes terminal win/loss objectives with gamma near one. Higher precision alone did not close the measured gap.

This study tests 500,000-click warm starts from fixed pretrained weights. It does not establish that the same settings are best from scratch or after a longer training budget, nor that reward design cannot close the gap.

A successful recipe must be frozen, repeated with fresh training seeds, and confirmed on the reserved test boards. Validation p-values are not evidence of repeatable success after searching many variants.

See [protocol and commands](README.md). Create `STOP` in this directory to stop the study cleanly; this does not target the original easy-exp2 run.
