# 02 TransMorph

Experimento exploratorio con una arquitectura **TransMorph-style 2D** para registration H&E -> HSI.

## Idea

TransMorph es un metodo deep learning basado en una arquitectura hibrida Transformer-ConvNet para registro medico no supervisado. El articulo original trabaja con registro volumetrico 3D, por lo que no se ha usado directamente el modelo oficial.

En este proyecto se ha implementado una version ligera y 2D inspirada en esa idea:

- entrada: mapas de distancia firmados de las mascaras H&E y HSI,
- encoder convolucional para estructura local,
- bottleneck Transformer para relaciones espaciales mas globales,
- decoder convolucional para predecir un campo de deformacion denso,
- entrenamiento instance-specific: una optimizacion independiente por sujeto.

Esto debe presentarse como **TransMorph-style**, no como el TransMorph oficial completo.

Referencia principal:

Chen, J., Frey, E. C., He, Y., Segars, W. P., Li, Y., & Du, Y. (2022). TransMorph: Transformer for unsupervised medical image registration. *Medical Image Analysis*, 82, 102615. DOI: 10.1016/j.media.2022.102615.

## Estructura

- `notebooks/20_transmorph_exploratorio_SB013.ipynb`: explicacion interactiva de la arquitectura y revision de una ejecucion individual sobre SB013.
- `notebooks/21_transmorph_resultados_todos.ipynb`: comparacion de configuraciones y resultados sobre los seis sujetos.
- `scripts/run_transmorph_instance.py`: entrena y evalua un sujeto concreto.
- `scripts/run_transmorph_grid.py`: ejecuta la grid completa, genera metricas, figuras y reportes.
- `scripts/create_transmorph_notebooks.py`: genera de forma reproducible los dos notebooks anteriores.
- `outputs/outputs_transmorph_grid/`: resultados de todas las configuraciones probadas.
- `outputs/outputs_transmorph_final_block/`: cierre limpio del experimento TransMorph-style.

Los notebooks sirven para recorrer, visualizar y explicar el experimento. La implementacion completa se mantiene en los scripts para evitar duplicar el modelo y para que las ejecuciones sean reproducibles desde terminal o desde Jupyter.

Comparar siempre contra:

`../00_baseline_clasico/outputs/outputs_baseline_clasico_final/`

y, para el bloque deep, contra:

`../01_voxelmorph/outputs/outputs_voxelmorph_final_block/`

## Configuraciones probadas

Se probaron tres configuraciones con distinta regularizacion:

| Config | Smooth weight | Max disp norm | Mean Dice | Gain vs baseline | Gain vs VoxelMorph | Mean p95 flow | Mean neg. Jacobian | Status general |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `tm_s040_d012` | 0.40 | 0.12 | 0.955 | +0.042 | -0.001 | 42.3 px | 0.059 | mejor Dice, pero 1 caso rechazado |
| `tm_s060_d010` | 0.60 | 0.10 | 0.945 | +0.032 | -0.011 | 36.2 px | 0.054 | compromiso intermedio |
| `tm_s100_d008` | 1.00 | 0.08 | 0.932 | +0.020 | -0.023 | 29.4 px | 0.040 | mas conservador, 0 casos rechazados |

La configuracion seleccionada automaticamente para el bloque final es:

`tm_s100_d008`

Se selecciona porque elimina los casos rechazados por calidad de deformacion, aunque sacrifica parte del Dice.

## Resultado seleccionado

| Subject | Baseline Dice | VoxelMorph selected | TransMorph-style | Gain vs baseline | Gain vs VoxelMorph | Status |
|---|---:|---:|---:|---:|---:|---|
| SB012 | 0.970 | 0.992 | 0.981 | +0.011 | -0.011 | acceptable exploratory |
| SB013 | 0.984 | 0.996 | 0.996 | +0.012 | -0.001 | acceptable exploratory |
| SB017 | 0.609 | 0.847 | 0.780 | +0.171 | -0.067 | needs visual review |
| SB018 | 0.977 | 0.950 | 0.927 | -0.050 | -0.023 | not better than baseline |
| SB019 | 0.971 | 0.967 | 0.944 | -0.027 | -0.023 | needs visual review |
| SB020 | 0.965 | 0.981 | 0.967 | +0.002 | -0.014 | acceptable exploratory |

## Interpretacion

Este experimento es util porque aporta una segunda familia deep learning: no solo CNN-style como VoxelMorph, sino tambien una variante con Transformer.

La lectura principal es:

- TransMorph-style mejora el baseline clasico medio si se mira el Dice global.
- La configuracion flexible `tm_s040_d012` casi iguala a VoxelMorph en Dice medio, pero introduce mas riesgo de deformacion.
- La configuracion conservadora `tm_s100_d008` es mas segura, pero queda por debajo de VoxelMorph.
- VoxelMorph sigue siendo el bloque deep learning principal para este dataset.
- TransMorph-style queda como experimento exploratorio defendible, no como pipeline final.

Conclusion breve para explicar al tutor:

> Se exploro una adaptacion 2D instance-specific inspirada en TransMorph. El metodo confirma que las arquitecturas transformer-based pueden aplicarse al problema HSI-H&E, pero en este conjunto reducido no supera de forma robusta al bloque VoxelMorph-style, que sigue siendo la opcion deep learning mas prometedora.

## Outputs principales

- `outputs/outputs_transmorph_final_block/README.md`
- `outputs/outputs_transmorph_final_block/transmorph_final_selected_summary.csv`
- `outputs/outputs_transmorph_final_block/transmorph_selected_method_comparison.png`
- `outputs/outputs_transmorph_final_block/transmorph_selected_quality_scatter.png`
- `outputs/outputs_transmorph_final_block/transmorph_selected_summary_montage.png`
- `outputs/outputs_transmorph_grid/transmorph_grid_config_summary.csv`
- `outputs/outputs_transmorph_grid/transmorph_regularization_overlay_grid.png`
- `outputs/outputs_transmorph_grid/transmorph_regularization_tradeoff.png`
