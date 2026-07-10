# TransMorph-style grid

This folder contains the regularization grid for the 2D TransMorph-style exploratory experiment.

| Config | Mean Dice | Gain vs baseline | Gain vs VoxelMorph | Mean p95 flow | Mean neg. Jacobian | Accept | Review | Reject | Not better |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tm_s040_d012 | 0.955 | +0.042 | -0.001 | 42.3 | 0.059 | 3 | 1 | 1 | 1 |
| tm_s060_d010 | 0.945 | +0.032 | -0.011 | 36.2 | 0.054 | 3 | 2 | 1 | 0 |
| tm_s100_d008 | 0.932 | +0.020 | -0.023 | 29.4 | 0.040 | 3 | 2 | 0 | 1 |

Selected automatic compromise: `tm_s100_d008`.