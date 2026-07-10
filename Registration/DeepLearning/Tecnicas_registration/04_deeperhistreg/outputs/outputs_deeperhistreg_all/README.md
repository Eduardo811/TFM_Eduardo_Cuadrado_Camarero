# DeeperHistReg exploratory evaluation

This experiment runs DeeperHistReg on the six prepared HSI-H&E pairs. It should be interpreted as an external feature/non-rigid registration baseline, not as a method designed specifically for HSI-H&E.

Important caveat: the DeeperHistReg Dice is approximate because the warped H&E mask is re-estimated from the black background of the warped source image.

## Summary

- Mean classical baseline Dice: 0.913
- Mean DeeperHistReg Dice: 0.738
- Mean DeeperHistReg gain vs baseline: -0.175

## Case-level comparison

| Subject | Classical baseline | VoxelMorph reg_s050_d012 | DeeperHistReg | DeeperHistReg - baseline |
|---|---:|---:|---:|---:|
| SB012 | 0.970 | 0.992 | 0.872 | -0.098 |
| SB013 | 0.984 | 0.996 | 0.857 | -0.127 |
| SB017 | 0.609 | 0.847 | 0.231 | -0.377 |
| SB018 | 0.977 | 0.950 | 0.770 | -0.207 |
| SB019 | 0.971 | 0.967 | 0.834 | -0.137 |
| SB020 | 0.965 | 0.981 | 0.865 | -0.100 |

## Interpretation

- DeeperHistReg provides a useful external comparison, but it is not directly competitive with the current classical baseline on these HSI-H&E pairs.
- The method is designed for histology-to-histology registration; HSI pseudo-RGB and H&E have a much larger modality gap.
- The result supports keeping DeeperHistReg as an exploratory baseline/future-work direction rather than as the main registration pipeline.