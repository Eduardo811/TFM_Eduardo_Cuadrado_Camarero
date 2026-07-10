# VoxelMorph-style instance-specific vs baseline clasico

Este experimento entrena una red 2D tipo VoxelMorph por sujeto. No es un modelo generalizable; es un ajuste instance-specific para comprobar si una CNN puede estimar una deformacion densa util para cada pareja H&E-HSI.

| Sujeto | Dice inicial | Dice baseline | Dice VoxelMorph | VM - baseline | p95 flow px | max flow px | Jacobian negativo |
|---|---:|---:|---:|---:|---:|---:|---:|
| SB012 | 0.926 | 0.970 | 0.996 | +0.026 | 50.03 | 70.05 | 0.0485 |
| SB013 | 0.941 | 0.984 | 0.996 | +0.012 | 23.86 | 51.88 | 0.0185 |
| SB017 | 0.609 | 0.609 | 0.946 | +0.337 | 84.14 | 84.32 | 0.2265 |
| SB018 | 0.839 | 0.977 | 0.976 | -0.000 | 82.36 | 82.76 | 0.1170 |
| SB019 | 0.852 | 0.971 | 0.995 | +0.024 | 84.84 | 85.87 | 0.1028 |
| SB020 | 0.890 | 0.965 | 0.995 | +0.030 | 63.41 | 79.63 | 0.0791 |

## Interpretacion

- `Dice inicial` es el solape antes de la CNN, normalmente tras el registro afin/rigido del baseline.
- `Dice baseline` es el resultado final aceptado del pipeline clasico.
- `Dice VoxelMorph` es el resultado de la red instance-specific.
- Las metricas de flujo sirven como control: si Dice sube pero el campo es enorme o plegado, el resultado no debe aceptarse automaticamente.