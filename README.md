# Registro multimodal HSI-H&E en SPECTRA-BREAST

Repositorio asociado al Trabajo Fin de Master **"Registro multimodal entre imagen hiperespectral e histologia H&E en el proyecto SPECTRA-BREAST"**.

El objetivo del trabajo es estudiar y evaluar un pipeline de registro entre imagen hiperespectral (HSI) e imagen histologica H&E de especimenes de cancer de mama. El repositorio conserva los notebooks, scripts, resultados derivados y figuras principales usados para documentar el desarrollo.

## Contenido principal

- `Registration/DeepLearning/`: notebooks de preprocesado, segmentacion HSI/H&E, orientacion, escalado fisico y preparacion de pares H&E-HSI.
- `Registration/DeepLearning/Imagenes_a_escala/`: datos derivados ligeros usados por los metodos de registro: imagenes segmentadas, mascaras, contornos, mapas de distancia firmada y metadatos.
- `Registration/DeepLearning/Tecnicas_registration/`: nucleo experimental del registro, organizado por metodo.
- `Datos/`: notebooks exploratorios iniciales del caso SB013. No incluye datos brutos.
- `Fotos_Memoria/`: figuras utilizadas o preparadas para la memoria del TFM.
- `docs/`: guia de datos, inventario explicado y pasos recomendados para publicar en GitHub.

## Metodos evaluados

| Metodo | Carpeta | Papel en el TFM |
|---|---|---|
| Baseline clasico | `00_baseline_clasico/` | Registro rigido, afin y deformable clasico basado en mascaras/contornos. |
| VoxelMorph-style | `01_voxelmorph/` | Experimento principal de aprendizaje profundo, optimizado por especimen. |
| TransMorph-style | `02_transmorph/` | Variante exploratoria 2D inspirada en TransMorph. |
| DeeperHistReg | `04_deeperhistreg/` | Comparacion externa exploratoria con una herramienta de registro histologico. |
| LoFTR | `05_feature_based_matching/` | Pruebas de correspondencias visuales y efecto del preprocesado. |

Resumen cuantitativo global de los resultados finales:

| Metodo | Dice medio | Ganancia frente al baseline |
|---|---:|---:|
| Baseline clasico | 0.913 | 0.000 |
| VoxelMorph-style | 0.956 | +0.043 |
| TransMorph-style | 0.932 | +0.020 |
| DeeperHistReg aproximado | 0.738 | -0.175 |

Estos resultados deben interpretarse como pruebas experimentales sobre seis pares H&E-HSI. Los metodos VoxelMorph-style y TransMorph-style se usan como optimizacion por especimen, no como modelos clinicos generalizables.

## Datos incluidos y no incluidos

Este repositorio no incluye los datos brutos originales: cubos HSI, archivos `.hdr/.hrd/raw`, preparaciones completas `.mrxs/.dat`, instaladores, papers descargados ni documentos clinicos o privados. En su lugar se incluyen derivados ligeros suficientes para entender el pipeline y revisar los resultados.

La politica completa de datos esta en `docs/DATOS.md`. Antes de hacer publico el repositorio, conviene confirmar que las figuras y derivados de imagen incluidos pueden publicarse segun las condiciones del proyecto SPECTRA-BREAST.

## Uso rapido

Desde la raiz del repositorio:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

En Windows, la activacion del entorno seria:

```powershell
.venv\Scripts\Activate.ps1
```

Los notebooks se han dejado sin outputs embebidos para que GitHub los cargue bien y para evitar rutas locales. Las figuras, CSV, JSON y salidas principales ya estan guardadas en las carpetas `outputs/`.

## Guia de lectura

Para entender el trabajo sin ejecutar nada, empezar por:

1. `docs/GUIA_ARCHIVOS.md`
2. `Registration/DeepLearning/Tecnicas_registration/README.md`
3. `Registration/DeepLearning/Tecnicas_registration/outputs/outputs_comparacion_deep_learning/README.md`
4. `Fotos_Memoria/`

Para reproducir o inspeccionar el pipeline, abrir los notebooks en orden aproximado:

1. `Registration/DeepLearning/9_funcion_preprocesado.ipynb`
2. `Registration/DeepLearning/10_guardar_imagenes_a_escala.ipynb`
3. `Registration/DeepLearning/Tecnicas_registration/00_baseline_clasico/notebooks/`
4. `Registration/DeepLearning/Tecnicas_registration/01_voxelmorph/notebooks/`
5. `Registration/DeepLearning/Tecnicas_registration/notebooks/22_comparacion_deep_learning.ipynb`

## Licencia y permisos

La licencia debe confirmarse antes de publicar. Recomendacion: separar licencia de codigo y licencia de material grafico/datos derivados, ya que el repositorio contiene imagenes procedentes de un proyecto biomédico colaborativo.
