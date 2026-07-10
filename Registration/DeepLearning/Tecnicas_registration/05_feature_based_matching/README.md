# 05 Feature-based matching

Experimentos de registration basados en correspondencias entre caracteristicas visuales.

## LoFTR

LoFTR es un matcher detector-free basado en Transformers. A diferencia de VoxelMorph y TransMorph-style, no predice directamente un campo de deformacion. Primero encuentra correspondencias entre dos imagenes y despues se usa RANSAC para estimar una transformacion geometrica.

La primera prueba se realizo sobre SB013 para comprobar si LoFTR mejora al eliminar la caja, el fondo y las regiones externas al especimen.

## Estructura

- `notebooks/23_loftr_SB013_segmentado.ipynb`: explicacion interactiva y resultados ejecutados.
- `scripts/run_loftr_sb013_segmented.py`: reproduce LoFTR, affine, homografia y metricas.
- `scripts/create_loftr_segmented_notebook.py`: genera el notebook a partir de los resultados.
- `outputs/outputs_loftr_SB013_segmented/`: imagenes, matches, overlays y resumen CSV/JSON.

## Casos comparados

| Caso | Matches | Affine inliers | Homography inliers | Affine Dice | Homography Dice |
|---|---:|---:|---:|---:|---:|
| Original full images | 55 | 6 | 9 | N/A | N/A |
| Scale-matched unmasked | 114 | 16 | 23 | 0.785 | 0.838 |
| Segmented and cropped | 142 | 12 | 19 | 0.764 | 0.903 |

## Interpretacion

La segmentacion aumenta claramente el numero de correspondencias y mejora el Dice de la homografia de 0.838 a 0.903. Sin embargo, el ratio de inliers disminuye y la transformacion affine no mejora. Por tanto, eliminar el fondo ayuda, pero no convierte automaticamente todos los matches en correspondencias fiables.

LoFTR segmentado es util como inicializacion feature-based y como mejora de la primera prueba exploratoria. No sustituye al baseline clasico de SB013, que alcanza aproximadamente 0.984 de Dice.

## Direccion

Para replicar el notebook inicial se mantiene:

- moving: HSI pseudo-RGB,
- fixed: H&E,
- transformacion estimada: HSI -> H&E.
