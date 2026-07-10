# VoxelMorph final deep-learning block

This folder closes the VoxelMorph-style experiments as the main deep-learning registration block for the TFM.

## What was tested

A 2D VoxelMorph-style CNN was used as an instance-specific deformable registration method. The model was not trained as a universal model across subjects because only six specimen pairs are available. Instead, one small network is optimized for each H&E-HSI pair.

The network receives signed distance maps from the H&E and HSI masks and predicts a dense deformation field. The loss combines mask-distance similarity, mask overlap and deformation smoothness.

## Selected configuration

- Selected config: `reg_s050_d012`
- Smooth weight: 0.50
- Max displacement norm: 0.12
- Mean classical baseline Dice: 0.913
- Mean selected VoxelMorph Dice: 0.956
- Mean gain vs classical baseline: +0.043
- Mean P95 displacement: 42.3 px
- Mean negative Jacobian fraction: 0.066
- Quality status count: 3 acceptable, 2 needs review, 1 reject as final

This configuration was selected because it provides a better compromise than the first aggressive VoxelMorph run: it preserves a positive mean Dice gain while reducing displacement magnitude and deformation-risk indicators.

## Case-level summary

| Subject | Baseline Dice | VoxelMorph Dice | Gain | P95 flow px | Neg. Jacobian frac. | Status |
|---|---:|---:|---:|---:|---:|---|
| SB012 | 0.970 | 0.992 | +0.022 | 48.7 | 0.026 | acceptable exploratory |
| SB013 | 0.984 | 0.996 | +0.012 | 23.0 | 0.011 | acceptable exploratory |
| SB017 | 0.609 | 0.847 | +0.238 | 46.0 | 0.155 | reject as final |
| SB018 | 0.977 | 0.950 | -0.027 | 45.1 | 0.063 | needs visual review |
| SB019 | 0.971 | 0.967 | -0.004 | 46.8 | 0.093 | needs visual review |
| SB020 | 0.965 | 0.981 | +0.016 | 44.0 | 0.048 | acceptable exploratory |

## Interpretation

The VoxelMorph-style approach is the most relevant deep-learning experiment developed in this project. It shows that a neural deformation model can improve overlap in several cases, especially when used as an instance-specific refinement after the classical alignment.

However, the result should not be presented as a fully validated final pipeline. Some cases still require visual review, and SB017 remains problematic because the gain in Dice is associated with a deformation that is too aggressive to accept as final.

The defensible conclusion is: VoxelMorph-style registration is promising for HSI-H&E refinement, but with the current dataset size it should be treated as a proof of concept rather than a generalizable trained model.

## Files in this folder

- `voxelmorph_final_selected_summary.csv`: clean selected case-level table.
- `voxelmorph_final_selected_summary.json`: same information in JSON format.
- `voxelmorph_selected_dice_comparison.png`: Dice comparison against the affine input and classical baseline.
- `voxelmorph_selected_quality_scatter.png`: Dice gain vs deformation magnitude.
- `voxelmorph_selected_summary_montage.png`: visual summary of the selected configuration.
- `regularization_tradeoff.png`: trade-off across all tested regularization settings.
- `regularization_overlay_grid.png`: visual grid from the regularization experiment.
- `selected_subject_outputs/`: copied per-subject overlays, warped H&E images and masks for the selected configuration.