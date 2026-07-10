# TransMorph-style final block

This folder summarizes the exploratory TransMorph-style registration experiment for the H&E -> HSI task.

## Why this is TransMorph-style, not the official TransMorph model

The original TransMorph paper proposes a hybrid Transformer-ConvNet architecture for unsupervised volumetric medical image registration. Our dataset is much smaller and 2D: six paired H&E-HSI specimens. For this reason, the experiment here adapts the idea to a lightweight 2D instance-specific setting rather than training the original 3D model.

Reference: Chen et al., `TransMorph: Transformer for unsupervised medical image registration`, Medical Image Analysis, 82, 102615, 2022. DOI: 10.1016/j.media.2022.102615.

## What was tested

A small 2D hybrid CNN-Transformer network receives signed distance maps from the H&E and HSI masks. A convolutional encoder extracts local structure, a Transformer bottleneck models longer-range spatial relationships, and a convolutional decoder predicts a dense deformation field.

As with the VoxelMorph-style experiment, the model is optimized separately for each subject. It should therefore be interpreted as an instance-specific deep registration proof of concept, not as a universal trained model.

## Selected configuration

- Selected config: `tm_s100_d008`
- Mean TransMorph-style Dice: 0.932
- Mean gain vs classical baseline: +0.020
- Mean gain vs selected VoxelMorph: -0.023
- Mean p95 displacement: 29.4 px
- Mean negative Jacobian fraction: 0.040
- Quality status count: 3 acceptable, 2 review, 0 reject, 1 not better

## Case-level selected results

| Subject | Baseline Dice | VoxelMorph selected | TransMorph-style | Gain vs baseline | Gain vs VoxelMorph | P95 flow | Neg. Jacobian | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| SB012 | 0.970 | 0.992 | 0.981 | +0.011 | -0.011 | 33.2 | 0.022 | acceptable exploratory |
| SB013 | 0.984 | 0.996 | 0.996 | +0.012 | -0.001 | 21.8 | 0.006 | acceptable exploratory |
| SB017 | 0.609 | 0.847 | 0.780 | +0.171 | -0.067 | 30.6 | 0.093 | needs visual review |
| SB018 | 0.977 | 0.950 | 0.927 | -0.050 | -0.023 | 30.0 | 0.040 | not better than baseline |
| SB019 | 0.971 | 0.967 | 0.944 | -0.027 | -0.023 | 31.1 | 0.057 | needs visual review |
| SB020 | 0.965 | 0.981 | 0.967 | +0.002 | -0.014 | 29.4 | 0.020 | acceptable exploratory |

## Interpretation

This experiment gives a second deep-learning family after VoxelMorph: a transformer-based registration variant. It is useful to show that the project explored both CNN-style and transformer-style deformable registration.

The main limitation is the same as before: there are only six image pairs, so the model cannot be presented as a trained general solution. The result is best framed as an exploratory instance-specific experiment.

If the selected TransMorph-style configuration does not outperform VoxelMorph, the correct conclusion is not that the experiment failed, but that VoxelMorph remains the stronger deep-learning block for this dataset.

## Files in this folder

- `transmorph_final_selected_summary.csv`: clean case-level selected table.
- `transmorph_final_selected_summary.json`: same information in JSON format.
- `transmorph_selected_method_comparison.png`: baseline vs VoxelMorph vs TransMorph-style.
- `transmorph_selected_quality_scatter.png`: Dice gain vs deformation magnitude.
- `transmorph_selected_summary_montage.png`: visual summary of selected TransMorph-style outputs.
- `selected_subject_outputs/`: copied outputs for the selected configuration.