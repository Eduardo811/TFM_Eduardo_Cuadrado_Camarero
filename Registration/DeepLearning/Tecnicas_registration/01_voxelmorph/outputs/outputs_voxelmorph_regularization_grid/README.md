# VoxelMorph regularization grid

This experiment reruns the instance-specific VoxelMorph-style method with stronger smoothness and lower maximum displacement. The goal is not to maximize Dice alone, but to reduce deformation folding and obtain a more plausible registration.

## Config-level summary

| Config | Mean Dice | Mean gain | Mean p95 flow | Mean negative Jacobian | Acceptable | Review | Reject | Not better |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current_s008_d022 | 0.984 | +0.072 | 64.8 | 0.099 | 2 | 1 | 3 | 0 |
| reg_s015_d018 | 0.977 | +0.064 | 58.5 | 0.086 | 2 | 2 | 2 | 0 |
| reg_s030_d015 | 0.968 | +0.055 | 48.9 | 0.077 | 2 | 2 | 2 | 0 |
| reg_s050_d012 | 0.956 | +0.043 | 42.3 | 0.066 | 3 | 2 | 1 | 0 |
| reg_s080_d010 | 0.946 | +0.033 | 35.9 | 0.058 | 3 | 2 | 1 | 0 |

## Case-level results

| Config | Subject | Baseline Dice | VM Dice | Gain | p95 flow | Negative Jacobian | Status |
|---|---|---:|---:|---:|---:|---:|---|
| current_s008_d022 | SB012 | 0.970 | 0.996 | +0.026 | 50.0 | 0.049 | acceptable_exploratory |
| current_s008_d022 | SB013 | 0.984 | 0.996 | +0.012 | 23.9 | 0.019 | acceptable_exploratory |
| current_s008_d022 | SB017 | 0.609 | 0.946 | +0.337 | 84.1 | 0.227 | reject_as_final |
| current_s008_d022 | SB018 | 0.977 | 0.976 | -0.000 | 82.4 | 0.117 | reject_as_final |
| current_s008_d022 | SB019 | 0.971 | 0.995 | +0.024 | 84.8 | 0.103 | reject_as_final |
| current_s008_d022 | SB020 | 0.965 | 0.995 | +0.030 | 63.4 | 0.079 | needs_review |
| reg_s015_d018 | SB012 | 0.970 | 0.996 | +0.026 | 56.7 | 0.037 | acceptable_exploratory |
| reg_s015_d018 | SB013 | 0.984 | 0.997 | +0.013 | 29.6 | 0.020 | acceptable_exploratory |
| reg_s015_d018 | SB017 | 0.609 | 0.919 | +0.310 | 69.0 | 0.201 | reject_as_final |
| reg_s015_d018 | SB018 | 0.977 | 0.970 | -0.007 | 67.7 | 0.088 | needs_review |
| reg_s015_d018 | SB019 | 0.971 | 0.988 | +0.017 | 70.1 | 0.105 | reject_as_final |
| reg_s015_d018 | SB020 | 0.965 | 0.992 | +0.027 | 57.7 | 0.063 | needs_review |
| reg_s030_d015 | SB012 | 0.970 | 0.996 | +0.026 | 43.7 | 0.030 | acceptable_exploratory |
| reg_s030_d015 | SB013 | 0.984 | 0.997 | +0.013 | 23.4 | 0.014 | acceptable_exploratory |
| reg_s030_d015 | SB017 | 0.609 | 0.887 | +0.279 | 57.5 | 0.181 | reject_as_final |
| reg_s030_d015 | SB018 | 0.977 | 0.962 | -0.015 | 56.4 | 0.075 | needs_review |
| reg_s030_d015 | SB019 | 0.971 | 0.975 | +0.004 | 58.2 | 0.109 | reject_as_final |
| reg_s030_d015 | SB020 | 0.965 | 0.989 | +0.024 | 54.3 | 0.051 | needs_review |
| reg_s050_d012 | SB012 | 0.970 | 0.992 | +0.022 | 48.7 | 0.026 | acceptable_exploratory |
| reg_s050_d012 | SB013 | 0.984 | 0.996 | +0.012 | 23.0 | 0.011 | acceptable_exploratory |
| reg_s050_d012 | SB017 | 0.609 | 0.847 | +0.238 | 46.0 | 0.155 | reject_as_final |
| reg_s050_d012 | SB018 | 0.977 | 0.950 | -0.027 | 45.1 | 0.063 | needs_review |
| reg_s050_d012 | SB019 | 0.971 | 0.967 | -0.004 | 46.8 | 0.093 | needs_review |
| reg_s050_d012 | SB020 | 0.965 | 0.981 | +0.016 | 44.0 | 0.048 | acceptable_exploratory |
| reg_s080_d010 | SB012 | 0.970 | 0.989 | +0.018 | 41.2 | 0.031 | acceptable_exploratory |
| reg_s080_d010 | SB013 | 0.984 | 0.997 | +0.013 | 22.5 | 0.010 | acceptable_exploratory |
| reg_s080_d010 | SB017 | 0.609 | 0.817 | +0.208 | 38.3 | 0.124 | reject_as_final |
| reg_s080_d010 | SB018 | 0.977 | 0.940 | -0.037 | 37.6 | 0.066 | needs_review |
| reg_s080_d010 | SB019 | 0.971 | 0.956 | -0.015 | 39.0 | 0.083 | needs_review |
| reg_s080_d010 | SB020 | 0.965 | 0.975 | +0.010 | 36.9 | 0.034 | acceptable_exploratory |

## Interpretation

- Best current trade-off by the automatic criteria: `reg_s050_d012`.
- The original configuration (`current_s008_d022`) gives the highest Dice, but it also has the highest deformation risk: 3/6 cases are rejected as final.
- Increasing regularization progressively reduces mean p95 displacement and mean negative-Jacobian fraction. This confirms that the excessive deformation problem is controllable, at least partially.
- `reg_s050_d012` is the best practical compromise: it keeps a positive mean gain over the classical baseline (+0.043), reduces mean p95 flow from 64.8 px to 42.3 px, and reduces rejected cases from 3 to 1.
- `reg_s080_d010` is more conservative, but the extra safety is not enough to rescue SB017 and it loses more Dice in SB018/SB019. Therefore it is less attractive as the default configuration.
- SB017 remains the difficult case. Even under strong regularization it improves mask Dice, but still shows excessive folding risk. It should not be presented as a reliable final registration.
- This should still be checked visually, because mask Dice and Jacobian statistics do not guarantee correct histological correspondence.
- If all regularized configurations lose too much Dice, the defensible conclusion is that VoxelMorph is useful as a proof of concept but not yet reliable as the final pipeline with the current data.
