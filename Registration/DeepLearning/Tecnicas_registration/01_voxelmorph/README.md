# 01 VoxelMorph

Experimentos con una red 2D estilo VoxelMorph para H&E -> HSI.

## Enfoque actual

Con solo 6 especimenes no se plantea entrenar un modelo universal. El primer experimento es **instance-specific**:

1. se toma la H&E ya inicializada por el baseline afin,
2. se toma la HSI como imagen fija,
3. la red recibe dos canales: mapa de distancia firmado de la mascara H&E y mapa de distancia firmado de la mascara HSI,
4. una CNN tipo U-Net predice un campo de deformacion denso,
5. la perdida combina similitud de mapas de distancia, solape de mascaras y suavidad del campo.

Esto es deep learning, pero se debe presentar como ajuste por muestra, no como modelo generalizable.

## Estructura

- `notebooks/`: notebooks de preparacion, entrenamiento/inferencia y evaluacion.
- `outputs/`: resultados, modelos, overlays y metricas.
- `scripts/run_voxelmorph_instance.py`: prueba instance-specific reproducible.

Comparar siempre contra:

`../00_baseline_clasico/outputs/outputs_baseline_clasico_final/`

## Resultados actuales

Comparacion frente al baseline clasico final:

| Sujeto | Dice inicial | Dice baseline | Dice VoxelMorph-style | VM - baseline |
|---|---:|---:|---:|---:|
| SB012 | 0.926 | 0.970 | 0.996 | +0.026 |
| SB013 | 0.941 | 0.984 | 0.996 | +0.012 |
| SB017 | 0.609 | 0.609 | 0.946 | +0.337 |
| SB018 | 0.839 | 0.977 | 0.976 | -0.000 |
| SB019 | 0.852 | 0.971 | 0.995 | +0.024 |
| SB020 | 0.890 | 0.965 | 0.995 | +0.030 |

Estos valores necesitan revision visual porque una red puede mejorar Dice deformando de forma agresiva. En esta primera version hay casos con desplazamientos grandes y fraccion de Jacobiano negativo, especialmente SB017/SB018/SB019/SB020. Por tanto, el resultado es prometedor como experimento deep learning, pero no debe aceptarse automaticamente sin una version mas regularizada.

## Revision de calidad de la deformacion

Se ha cerrado una primera revision de calidad en:

`outputs/outputs_voxelmorph_quality_review/`

Esta revision no mira solo Dice, sino tambien:

- desplazamiento medio, p95 y maximo del campo,
- suavidad del campo,
- minimo y percentil 1 del determinante Jacobiano,
- fraccion de Jacobiano negativo.

Conclusion actual:

| Sujeto | Estado | Motivo principal |
|---|---|---|
| SB012 | acceptable_exploratory | Buen Dice y deformacion relativamente controlada. |
| SB013 | acceptable_exploratory | Mejor caso: buen Dice y bajo riesgo de deformacion. |
| SB017 | reject_as_final | Mejora mucho el Dice, pero con desplazamiento y plegamiento excesivos. |
| SB018 | reject_as_final | No mejora al baseline y presenta deformacion agresiva. |
| SB019 | reject_as_final | Mejora Dice, pero con fraccion de Jacobiano negativo demasiado alta. |
| SB020 | needs_regularized_rerun | Resultado prometedor, pero necesita mas regularizacion. |

Por tanto, esta version debe presentarse como **proof of concept** de registration deep instance-specific, no como metodo final. El siguiente paso recomendado es repetir VoxelMorph con mayor regularizacion y comparar explicitamente el trade-off entre Dice y plausibilidad de la deformacion.

## Grid de regularizacion

Se ejecuto una grid adicional en:

`outputs/outputs_voxelmorph_regularization_grid/`

Configuraciones probadas:

| Config | Smooth weight | Max disp norm |
|---|---:|---:|
| current_s008_d022 | 0.08 | 0.22 |
| reg_s015_d018 | 0.15 | 0.18 |
| reg_s030_d015 | 0.30 | 0.15 |
| reg_s050_d012 | 0.50 | 0.12 |
| reg_s080_d010 | 0.80 | 0.10 |

Resultado principal: al aumentar la regularizacion baja el Dice medio, pero tambien bajan el desplazamiento p95 y la fraccion de Jacobiano negativo. El mejor compromiso automatico es `reg_s050_d012`: mantiene mejora media frente al baseline clasico (+0.043), reduce el p95 medio del campo de 64.8 px a 42.3 px, y reduce los casos rechazados de 3 a 1.

La conclusion metodologica es que VoxelMorph-style puede ser util como refinamiento instance-specific, pero todavia no es suficientemente robusto para aceptarlo como pipeline final en todos los casos. SB017 sigue siendo el caso mas problematico incluso con regularizacion fuerte, por lo que debe tratarse como prueba de concepto o caso a revisar manualmente.

## Bloque final deep-learning

El cierre limpio del experimento VoxelMorph esta en:

`outputs/outputs_voxelmorph_final_block/`

Esta carpeta debe usarse como referencia principal para explicar el bloque deep-learning. Incluye:

- tabla limpia con la configuracion seleccionada `reg_s050_d012`,
- comparacion Dice frente al baseline clasico,
- figura de trade-off entre mejora y deformacion,
- montage visual de los seis sujetos,
- outputs seleccionados por sujeto,
- interpretacion metodologica lista para defender.

Conclusion breve: VoxelMorph-style es el experimento deep-learning principal del proyecto. Es prometedor como refinamiento instance-specific, pero no debe presentarse como modelo universal entrenado ni como pipeline final totalmente validado, porque solo hay seis pares de imagenes y algunos casos todavia requieren revision visual.
