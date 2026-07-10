from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path.cwd()
METHOD_DIR = ROOT / "Registration" / "DeepLearning" / "Tecnicas_registration" / "02_transmorph"
NOTEBOOK_DIR = METHOD_DIR / "notebooks"


def metadata() -> dict:
    return {
        "kernelspec": {
            "display_name": "Python 3.9",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.9",
        },
    }


def create_exploratory_notebook() -> None:
    nb = nbf.v4.new_notebook(metadata=metadata())
    nb.cells = [
        nbf.v4.new_markdown_cell(
            """# 20 - TransMorph-style exploratorio sobre SB013

Este notebook explica y reproduce una prueba de registration deep learning inspirada en TransMorph.

No se utiliza directamente el TransMorph 3D oficial. Se usa una adaptacion ligera 2D porque las bandas de la HSI representan informacion espectral, no profundidad anatomica. La deformacion predicha es espacial 2D y se optimiza por separado para cada pareja H&E-HSI."""
        ),
        nbf.v4.new_markdown_cell(
            """## Metodo

- Moving: H&E inicializada mediante el registro afin clasico.
- Fixed: HSI pseudo-RGB y su mascara.
- Entrada: mapas de distancia firmados de las mascaras H&E y HSI.
- Encoder CNN: extrae estructura local.
- Bottleneck Transformer: modela relaciones espaciales globales.
- Decoder CNN: predice un campo denso de deformacion 2D.
- Perdida: similitud de mapas de distancia + Dice de mascaras + suavidad.

El modelo es **instance-specific**: no aprende un modelo universal con seis sujetos, sino que optimiza una red para la pareja seleccionada."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = Path.cwd()
METHOD_DIR = ROOT / 'Registration' / 'DeepLearning' / 'Tecnicas_registration' / '02_transmorph'
SCRIPT_DIR = METHOD_DIR / 'scripts'
SCRIPT = SCRIPT_DIR / 'run_transmorph_instance.py'
OUTPUT_TAG = 'tm_s100_d008'
SUBJECT = 'SB013'
OUTPUT_DIR = METHOD_DIR / 'outputs' / f'outputs_transmorph_instance_{OUTPUT_TAG}_{SUBJECT}'

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

print('Script:', SCRIPT)
print('Resultado existente:', OUTPUT_DIR)"""
        ),
        nbf.v4.new_code_cell(
            """# Arquitectura utilizada en el experimento.
from run_transmorph_instance import TransMorphLite2D

model = TransMorphLite2D(
    input_size=192,
    max_disp_norm=0.08,
    transformer_depth=2,
    transformer_heads=4,
)
num_params = sum(parameter.numel() for parameter in model.parameters())
print(f'Parametros entrenables: {num_params:,}')
print(model)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Ejecucion opcional

El resultado seleccionado ya existe. Deja `RUN_TRAINING = False` para revisar los resultados sin volver a entrenar. Cambialo a `True` solo si quieres repetir SB013 desde cero; se guardara con la etiqueta `notebook_demo` para no sobrescribir la grid final."""
        ),
        nbf.v4.new_code_cell(
            """RUN_TRAINING = False

if RUN_TRAINING:
    command = [
        sys.executable,
        str(SCRIPT),
        '--subject', SUBJECT,
        '--iterations', '140',
        '--size', '192',
        '--lr', '0.0015',
        '--smooth-weight', '1.00',
        '--dice-weight', '0.45',
        '--max-disp-norm', '0.08',
        '--transformer-depth', '2',
        '--transformer-heads', '4',
        '--output-tag', 'notebook_demo',
    ]
    result = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True)
    print(result.stdout[-5000:])
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError('Fallo el entrenamiento TransMorph-style de demostracion')"""
        ),
        nbf.v4.new_code_cell(
            """summary_path = OUTPUT_DIR / f'{SUBJECT}_transmorph_instance_summary.json'
with open(summary_path, encoding='utf-8') as handle:
    summary = json.load(handle)

keys = [
    'subject', 'method', 'device', 'input_size', 'iterations',
    'initial_dice', 'final_dice', 'flow_p95_px',
    'jacobian_det_negative_fraction',
]
display(pd.DataFrame([{key: summary[key] for key in keys}]).T.rename(columns={0: 'value'}))"""
        ),
        nbf.v4.new_code_cell(
            """summary_image = OUTPUT_DIR / f'{SUBJECT}_transmorph_instance_summary.png'
plt.figure(figsize=(16, 6))
plt.imshow(Image.open(summary_image))
plt.axis('off')
plt.title('SB013: entrada afin, resultado TransMorph-style y curva de optimizacion')
plt.tight_layout()"""
        ),
        nbf.v4.new_markdown_cell(
            """## Interpretacion

SB013 es uno de los casos mas estables. La configuracion conservadora mejora el Dice del baseline clasico de aproximadamente 0.984 a 0.996 y mantiene indicadores de deformacion relativamente controlados.

Este buen resultado no demuestra generalizacion. Demuestra que una arquitectura hibrida CNN-Transformer puede optimizar una deformacion util para una pareja concreta H&E-HSI."""
        ),
    ]
    nbf.write(nb, NOTEBOOK_DIR / "20_transmorph_exploratorio_SB013.ipynb")


def create_results_notebook() -> None:
    nb = nbf.v4.new_notebook(metadata=metadata())
    nb.cells = [
        nbf.v4.new_markdown_cell(
            """# 21 - TransMorph-style en todos los sujetos

Este notebook resume las tres configuraciones TransMorph-style probadas sobre los seis sujetos y compara la configuracion final con el baseline clasico y VoxelMorph-style."""
        ),
        nbf.v4.new_markdown_cell(
            """## Preguntas que responde

1. Cuanto mejora cada configuracion respecto al baseline clasico.
2. Si TransMorph-style supera al VoxelMorph-style seleccionado.
3. Cuanto desplazamiento y plegamiento introduce la deformacion.
4. Que configuracion ofrece el compromiso mas seguro."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import subprocess
import sys

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display

ROOT = Path.cwd()
METHOD_DIR = ROOT / 'Registration' / 'DeepLearning' / 'Tecnicas_registration' / '02_transmorph'
GRID_DIR = METHOD_DIR / 'outputs' / 'outputs_transmorph_grid'
FINAL_DIR = METHOD_DIR / 'outputs' / 'outputs_transmorph_final_block'
GRID_SCRIPT = METHOD_DIR / 'scripts' / 'run_transmorph_grid.py'

CONFIG_CSV = GRID_DIR / 'transmorph_grid_config_summary.csv'
GRID_CSV = GRID_DIR / 'transmorph_grid_summary.csv'
FINAL_CSV = FINAL_DIR / 'transmorph_final_selected_summary.csv'

print('Grid:', GRID_DIR)
print('Final:', FINAL_DIR)"""
        ),
        nbf.v4.new_code_cell(
            """# Cambia a True solo si quieres repetir las 18 optimizaciones.
# El script reutiliza resultados existentes cuando encuentra sus JSON.
RUN_FULL_GRID = False

if RUN_FULL_GRID:
    result = subprocess.run([sys.executable, str(GRID_SCRIPT)], cwd=str(ROOT), text=True, capture_output=True)
    print(result.stdout[-8000:])
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError('Fallo la grid completa TransMorph-style')"""
        ),
        nbf.v4.new_code_cell(
            """config_df = pd.read_csv(CONFIG_CSV)
display(config_df.round(4))"""
        ),
        nbf.v4.new_code_cell(
            """selected_df = pd.read_csv(FINAL_CSV)
columns = [
    'subject', 'classical_baseline_dice', 'voxelmorph_selected_dice',
    'transmorph_dice', 'transmorph_minus_baseline',
    'transmorph_minus_voxelmorph', 'flow_p95_px',
    'negative_jacobian_fraction', 'quality_status',
]
display(selected_df[columns].round(4))

means = pd.Series({
    'Classical baseline': selected_df['classical_baseline_dice'].mean(),
    'VoxelMorph selected': selected_df['voxelmorph_selected_dice'].mean(),
    'TransMorph-style selected': selected_df['transmorph_dice'].mean(),
}, name='mean_dice')
display(means.to_frame().round(4))"""
        ),
        nbf.v4.new_code_cell(
            """fig, axes = plt.subplots(2, 1, figsize=(16, 13))
axes[0].imshow(Image.open(GRID_DIR / 'transmorph_regularization_tradeoff.png'))
axes[0].axis('off')
axes[0].set_title('Trade-off de regularizacion')

axes[1].imshow(Image.open(FINAL_DIR / 'transmorph_selected_method_comparison.png'))
axes[1].axis('off')
axes[1].set_title('Baseline clasico vs VoxelMorph vs TransMorph-style')
plt.tight_layout()"""
        ),
        nbf.v4.new_code_cell(
            """plt.figure(figsize=(18, 14))
plt.imshow(Image.open(FINAL_DIR / 'transmorph_selected_summary_montage.png'))
plt.axis('off')
plt.title('Resumen visual de la configuracion TransMorph-style seleccionada')
plt.tight_layout()"""
        ),
        nbf.v4.new_markdown_cell(
            """## Lectura critica

- La configuracion flexible `tm_s040_d012` alcanza un Dice medio cercano al VoxelMorph seleccionado, pero conserva un caso rechazable por deformacion.
- La configuracion conservadora `tm_s100_d008` reduce desplazamientos y evita casos clasificados como rechazo final, aunque pierde Dice.
- SB012 y SB013 son los casos mas favorables.
- SB017 mejora mucho respecto al baseline, pero sigue necesitando revision por deformacion.
- SB018 y SB019 no mejoran al baseline con la configuracion conservadora.

**Conclusion:** TransMorph-style constituye una segunda familia deep learning valida como exploracion, pero VoxelMorph-style sigue siendo el resultado deep principal para este dataset."""
        ),
    ]
    nbf.write(nb, NOTEBOOK_DIR / "21_transmorph_resultados_todos.ipynb")


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    create_exploratory_notebook()
    create_results_notebook()
    print(f"Created notebooks in: {NOTEBOOK_DIR}")


if __name__ == "__main__":
    main()
