# Baseline clasico final de registration H&E -> HSI

Esta carpeta contiene el mejor resultado clasico aceptado por sujeto.

Direccion del registro:

- Imagen fija: HSI
- Imagen movil: H&E

Metodos considerados:

1. Rigido por mascaras: rotacion + traslacion.
2. Afin por mascaras: OpenCV ECC sobre mapas de distancia firmados.
3. No rigido: TV-L1 optical flow sobre mapas de distancia firmados.

Regla de decision:

1. Si el no rigido fue aceptado, se usa no rigido.
2. Si no, pero el afin fue aceptado, se usa afin.
3. Si no, se usa rigido.

Archivos principales:

- `baseline_clasico_summary.csv`: resumen por sujeto.
- `baseline_clasico_manifest.json`: manifest completo.
- `baseline_clasico_overlay_grid.png`: montaje de overlays finales.
- `baseline_clasico_contour_grid.png`: montaje de contornos finales.

Flags de revision:

- `ok`: resultado aceptado sin avisos fuertes.
- `review_moderate_nonrigid_displacement`: revisar visualmente por desplazamiento no rigido moderado.
- `review_high_nonrigid_displacement`: revisar con cuidado por desplazamiento no rigido alto.
- `review_rejected_nonrigid`: el no rigido se rechazo; el baseline vuelve a rigido/afin.

Nota importante:

Este baseline registra principalmente la geometria del especimen mediante mascaras/contornos. No garantiza por si solo alineamiento perfecto de estructuras internas H&E-HSI. Debe usarse como referencia clasica frente a futuros metodos deep learning como VoxelMorph o TransMorph.
