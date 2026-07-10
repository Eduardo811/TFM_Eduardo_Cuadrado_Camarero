# 04_deeperhistreg

Prueba exploratoria con DeeperHistReg como metodo externo histologico para registrar H&E -> HSI.

## Idea

DeeperHistReg esta pensado para registro histologico 2D/WSI. En esta prueba no se usa el cubo HSI completo, sino las imagenes ya preparadas en `Imagenes_a_escala`:

- source/moving: H&E enmascarada,
- target/fixed: HSI pseudo-RGB enmascarada,
- registro inicial por caracteristicas SIFT,
- optimizacion no rigida basada en PyTorch.

## Estado actual

Se ha extendido la prueba a los seis sujetos preparados:

- script general: `scripts/run_deeperhistreg_all.py`
- script inicial de SB013: `scripts/run_deeperhistreg_sb013.py`
- notebook inicial: `notebooks/17_deeperhistreg_exploratorio_SB013.ipynb`
- outputs generales: `outputs/outputs_deeperhistreg_all/`
- outputs iniciales de SB013: `outputs/outputs_deeperhistreg_SB013/`

La metrica Dice guardada aqui es orientativa: se calcula reestimando la mascara de la H&E registrada a partir del fondo negro, no aplicando directamente la deformacion a la mascara original.

## Resultados generales

Resumen del experimento con los seis casos:

- Mean classical baseline Dice: 0.913
- Mean DeeperHistReg Dice: 0.738
- Mean DeeperHistReg gain vs baseline: -0.175

Comparacion por sujeto:

| Subject | Classical baseline | VoxelMorph reg_s050_d012 | DeeperHistReg | DeeperHistReg - baseline |
|---|---:|---:|---:|---:|
| SB012 | 0.970 | 0.992 | 0.872 | -0.098 |
| SB013 | 0.984 | 0.996 | 0.857 | -0.127 |
| SB017 | 0.609 | 0.847 | 0.231 | -0.377 |
| SB018 | 0.977 | 0.950 | 0.770 | -0.207 |
| SB019 | 0.971 | 0.967 | 0.834 | -0.137 |
| SB020 | 0.965 | 0.981 | 0.865 | -0.100 |

## Interpretacion

DeeperHistReg aporta una comparacion externa util, pero no es directamente competitivo con el baseline clasico en estos pares HSI-H&E. La razon mas probable es que el metodo esta disenado principalmente para histologia-histologia, mientras que aqui se registra H&E contra una representacion pseudo-RGB de HSI, con una diferencia de modalidad mayor.

Por tanto, se puede mencionar como prueba exploratoria/future work, pero no conviene presentarlo como pipeline principal.
