# Tecnicas de registration

Esta carpeta esta organizada por familias de metodos para registrar H&E -> HSI.

## Estructura

- `00_baseline_clasico/`: baseline clasico ya desarrollado y cerrado.
- `01_voxelmorph/`: bloque deep learning principal, con experimentos instance-specific y grid de regularizacion.
- `02_transmorph/`: bloque deep learning secundario, con adaptacion TransMorph-style 2D y comparacion completa.
- `03_elastix_bsplines_demons/`: espacio reservado para metodos clasicos deformables adicionales.
- `04_deeperhistreg/`: prueba exploratoria con DeeperHistReg como metodo externo histologico.
- `05_feature_based_matching/`: pruebas de matching por caracteristicas, comenzando con LoFTR sobre SB013 segmentado.
- `notebooks/22_comparacion_deep_learning.ipynb`: sintesis global de baseline, VoxelMorph, TransMorph y DeeperHistReg.
- `outputs/outputs_comparacion_deep_learning/`: tablas y figuras generadas por la comparacion global.

## Baseline actual

El baseline clasico usa:

1. registro rigido por mascaras,
2. registro afin por mascaras,
3. registro no rigido TV-L1 sobre mapas de distancia,
4. comparacion final y export del mejor resultado por sujeto.

El resultado principal del baseline esta en:

`00_baseline_clasico/outputs/outputs_baseline_clasico_final/`

## Comparacion deep learning

La comparacion global ejecutada esta en:

`notebooks/22_comparacion_deep_learning.ipynb`

Resultados medios seleccionados:

| Metodo | Mean Dice | Gain vs baseline | Mejor metodo por sujeto |
|---|---:|---:|---:|
| Classical baseline | 0.913 | +0.000 | 2/6 |
| VoxelMorph-style | 0.956 | +0.043 | 4/6 |
| TransMorph-style | 0.932 | +0.020 | 0/6 |
| DeeperHistReg approximate | 0.738 | -0.175 | 0/6 |

Conclusion: VoxelMorph-style es el resultado deep learning principal. TransMorph-style aporta una segunda familia transformer-based, pero no supera globalmente a VoxelMorph. DeeperHistReg se mantiene como comparacion externa exploratoria.
