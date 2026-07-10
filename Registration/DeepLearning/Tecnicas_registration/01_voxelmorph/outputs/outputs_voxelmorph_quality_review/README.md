# VoxelMorph-style quality review

This review closes the first instance-specific VoxelMorph-style experiment by considering both mask overlap and deformation plausibility.

## Summary

- Mean classical baseline Dice: 0.913
- Mean VoxelMorph-style Dice: 0.984
- Mean Dice gain over baseline: +0.072
- Mean negative-Jacobian fraction: 0.099

## Case-level decision

| Subject | Baseline Dice | VoxelMorph Dice | Gain | P95 flow (px) | Negative Jacobian | Status |
|---|---:|---:|---:|---:|---:|---|
| SB012 | 0.970 | 0.996 | +0.026 | 50.0 | 0.049 | acceptable_exploratory |
| SB013 | 0.984 | 0.996 | +0.012 | 23.9 | 0.019 | acceptable_exploratory |
| SB017 | 0.609 | 0.946 | +0.337 | 84.1 | 0.227 | reject_as_final |
| SB018 | 0.977 | 0.976 | -0.000 | 82.4 | 0.117 | reject_as_final |
| SB019 | 0.971 | 0.995 | +0.024 | 84.8 | 0.103 | reject_as_final |
| SB020 | 0.965 | 0.995 | +0.030 | 63.4 | 0.079 | needs_regularized_rerun |

## Interpretation

- The experiment is successful as a proof of concept: the neural instance-specific refinement can increase mask overlap after the classical initialization.
- It is not yet acceptable as a final registration method: several cases have high displacement and a large fraction of negative Jacobians, indicating folding or non-plausible local deformation.
- The next technical step should be a regularized VoxelMorph rerun, increasing smoothness/topology constraints and comparing the trade-off between Dice and deformation plausibility.

## Recommended use in the TFM

Present the current VoxelMorph-style experiment as an exploratory deep-learning registration result, not as the definitive HSI-H&E registration pipeline.