from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


ROOT = Path.cwd()
METHOD_DIR = ROOT / "Registration" / "DeepLearning" / "Tecnicas_registration" / "05_feature_based_matching"
OUTPUT_DIR = METHOD_DIR / "outputs" / "outputs_loftr_SB013_segmented"
NOTEBOOK_DIR = METHOD_DIR / "notebooks"
NOTEBOOK_PATH = NOTEBOOK_DIR / "23_loftr_SB013_segmentado.ipynb"


def load_summaries() -> dict[str, dict]:
    with open(OUTPUT_DIR / "loftr_SB013_comparison_summary.json", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {summary["case"]: summary for summary in payload["summaries"]}


def main() -> None:
    summaries = load_summaries()
    original = summaries["original_full"]
    scaled = summaries["scaled_unmasked"]
    segmented = summaries["segmented_cropped"]
    homography_gain = segmented["homography_mask_dice"] - scaled["homography_mask_dice"]

    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3.9", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9"},
        }
    )
    nb.cells = [
        nbf.v4.new_markdown_cell(
            """# 23 - LoFTR sobre SB013 segmentado

Este notebook repite la primera prueba deep de matching realizada con LoFTR, pero incorpora las segmentaciones H&E y HSI desarrolladas posteriormente.

La pregunta es sencilla: **si eliminamos la caja, el fondo y las regiones que no pertenecen al especimen, mejora el matching de LoFTR?**"""
        ),
        nbf.v4.new_markdown_cell(
            """## Donde encaja este experimento

LoFTR no es una red de registration deformable como VoxelMorph o TransMorph. Es un matcher de caracteristicas detector-free basado en Transformers y preentrenado sobre imagenes naturales.

En esta prueba:

1. LoFTR encuentra correspondencias entre la HSI pseudo-RGB y la H&E.
2. RANSAC elimina correspondencias inconsistentes.
3. Se estima una transformacion affine y una homografia.
4. La HSI se transforma al sistema de referencia de la H&E, igual que en `1_Prueba_Deep.ipynb`.

No se entrena LoFTR con nuestros seis sujetos."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display

ROOT = Path.cwd()
METHOD_DIR = ROOT / 'Registration' / 'DeepLearning' / 'Tecnicas_registration' / '05_feature_based_matching'
SCRIPT = METHOD_DIR / 'scripts' / 'run_loftr_sb013_segmented.py'
OUTPUT_DIR = METHOD_DIR / 'outputs' / 'outputs_loftr_SB013_segmented'
CSV_PATH = OUTPUT_DIR / 'loftr_SB013_comparison_summary.csv'
JSON_PATH = OUTPUT_DIR / 'loftr_SB013_comparison_summary.json'

print('Script:', SCRIPT)
print('Outputs:', OUTPUT_DIR)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Casos comparados

- `original_full`: HSI pseudo-RGB completa y macrofotografia H&E completa, reproduciendo el planteamiento inicial.
- `scaled_unmasked`: imagenes preparadas a escala fisica comparable, antes del recorte exacto por mascara.
- `segmented_cropped`: solo el especimen segmentado, con fondo negro y recorte al bounding box de cada mascara.

Se mantiene `KF.LoFTR(pretrained="outdoor")` para no cambiar el matcher respecto a la prueba original."""
        ),
        nbf.v4.new_code_cell(
            """# Cambia a True si quieres volver a ejecutar LoFTR y regenerar todos los outputs.
RUN_EXPERIMENT = False

if RUN_EXPERIMENT:
    result = subprocess.run([sys.executable, str(SCRIPT)], cwd=str(ROOT), text=True, capture_output=True)
    print(result.stdout[-8000:])
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError('Fallo el experimento LoFTR segmentado')"""
        ),
        nbf.v4.new_code_cell(
            """metrics = pd.read_csv(CSV_PATH)
display(metrics.round(4))"""
        ),
        nbf.v4.new_code_cell(
            """fig, axes = plt.subplots(2, 2, figsize=(16, 13))
axes[0, 0].imshow(Image.open(OUTPUT_DIR / 'original_full_inputs.png'))
axes[0, 0].set_title('Entradas originales completas')
axes[0, 1].imshow(Image.open(OUTPUT_DIR / 'segmented_cropped_inputs.png'))
axes[0, 1].set_title('Entradas segmentadas y recortadas')
axes[1, 0].imshow(Image.open(OUTPUT_DIR / 'original_full_loftr_matches.png'))
axes[1, 0].set_title('Matches LoFTR originales')
axes[1, 1].imshow(Image.open(OUTPUT_DIR / 'segmented_cropped_loftr_matches.png'))
axes[1, 1].set_title('Matches LoFTR con segmentacion')
for ax in axes.ravel():
    ax.axis('off')
plt.tight_layout()"""
        ),
        nbf.v4.new_code_cell(
            """plt.figure(figsize=(16, 5))
plt.imshow(Image.open(OUTPUT_DIR / 'loftr_SB013_comparison_summary.png'))
plt.axis('off')
plt.title('Comparacion cuantitativa de las tres variantes')
plt.tight_layout()"""
        ),
        nbf.v4.new_markdown_cell(
            f"""## Resultado principal

La segmentacion produce mas correspondencias:

- Prueba original: **{original['num_matches']} matches**.
- Imagenes preparadas sin recorte explicito: **{scaled['num_matches']} matches**.
- Especimen segmentado y recortado: **{segmented['num_matches']} matches**.

Para la homografia, el Dice de mascara pasa de **{scaled['homography_mask_dice']:.3f}** a **{segmented['homography_mask_dice']:.3f}**, una mejora de **{homography_gain:+.3f}**.

Sin embargo, el ratio de inliers de la homografia baja de **{scaled['homography_inlier_ratio']:.3f}** a **{segmented['homography_inlier_ratio']:.3f}**. Esto significa que aparecen mas matches, pero una proporcion mayor es inconsistente y debe ser rechazada por RANSAC."""
        ),
        nbf.v4.new_code_cell(
            """fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(Image.open(OUTPUT_DIR / 'scaled_unmasked_homography_overlay.png'))
axes[0].set_title('Sin recorte explicito | Homography')
axes[1].imshow(Image.open(OUTPUT_DIR / 'segmented_cropped_affine_overlay.png'))
axes[1].set_title('Segmentado | Affine')
axes[2].imshow(Image.open(OUTPUT_DIR / 'segmented_cropped_homography_overlay.png'))
axes[2].set_title('Segmentado | Homography')
for ax in axes:
    ax.axis('off')
plt.tight_layout()"""
        ),
        nbf.v4.new_markdown_cell(
            f"""## Lectura critica

La segmentacion **si ayuda a LoFTR**, especialmente cuando las correspondencias se utilizan para estimar una homografia:

- El numero de matches aumenta de {original['num_matches']} a {segmented['num_matches']}.
- El Dice segmentado con homografia alcanza {segmented['homography_mask_dice']:.3f}.
- La transformacion affine segmentada queda en {segmented['affine_mask_dice']:.3f}, por lo que el beneficio no aparece en todos los modelos geometricos.
- Los matches siguen siendo ruidosos y se concentran principalmente en bordes y estructuras de contraste fuerte.
- El baseline clasico de SB013 alcanza aproximadamente 0.984 de Dice, por encima de LoFTR + homografia.

**Conclusion:** eliminar caja y fondo mejora la utilidad de LoFTR como inicializacion feature-based, pero LoFTR por si solo no sustituye al pipeline de registration. La version segmentada puede presentarse como una mejora clara respecto a la primera prueba exploratoria."""
        ),
    ]
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Created: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
