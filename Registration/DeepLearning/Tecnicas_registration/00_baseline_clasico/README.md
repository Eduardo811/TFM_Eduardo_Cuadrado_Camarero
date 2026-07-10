# 00 Baseline clasico

Baseline clasico de registration H&E -> HSI.

## Contenido

- `notebooks/`: notebooks del pipeline clasico.
- `outputs/`: salidas generadas por cada etapa.
- `debug/`: salidas diagnosticas antiguas de pruebas intermedias.

## Notebooks

- `11_registro_rigido_mascaras_SB013.ipynb`: prueba inicial sobre SB013.
- `12_registro_rigido_mascaras_todos.ipynb`: registro rigido por mascaras.
- `13_registro_afin_mascaras_todos.ipynb`: refinamiento afin.
- `14_registro_no_rigido_mascaras_todos.ipynb`: registro deformable TV-L1.
- `15_comparacion_final_registration.ipynb`: comparacion rigido/afin/no rigido.
- `16_exportar_baseline_clasico_final.ipynb`: export del baseline final.

## Output principal

`outputs/outputs_baseline_clasico_final/`

Este directorio es el que debe usarse como referencia para comparar contra futuros metodos deep learning.

