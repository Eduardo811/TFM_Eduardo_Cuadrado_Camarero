from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path.cwd()
TECHNIQUES_DIR = ROOT / "Registration" / "DeepLearning" / "Tecnicas_registration"
NOTEBOOK_DIR = TECHNIQUES_DIR / "notebooks"
NOTEBOOK_PATH = NOTEBOOK_DIR / "22_comparacion_deep_learning.ipynb"


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3.9",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.9"},
        }
    )

    nb.cells = [
        nbf.v4.new_markdown_cell(
            """# 22 - Comparacion global de registration deep learning

Este notebook reune los resultados finales del baseline clasico y de las tres lineas deep/exploratorias estudiadas:

- VoxelMorph-style instance-specific.
- TransMorph-style 2D instance-specific.
- DeeperHistReg como herramienta externa histologica.

El objetivo es obtener una lectura conjunta y decidir que metodo constituye el resultado deep learning principal del proyecto."""
        ),
        nbf.v4.new_markdown_cell(
            """## Alcance y cautelas

Esta comparacion no representa una evaluacion de modelos generalizables:

1. Solo hay seis pares H&E-HSI.
2. VoxelMorph-style y TransMorph-style se optimizan por separado para cada sujeto.
3. Ambos parten de una inicializacion clasica afin/rigida; son refinamientos y no pipelines independientes end-to-end.
4. El Dice de DeeperHistReg es aproximado porque su mascara registrada se reestima desde el fondo negro.
5. Un Dice alto no garantiza una deformacion anatomica plausible. Tambien se revisan desplazamientos y Jacobianos negativos."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display

ROOT = Path.cwd()
TECHNIQUES_DIR = ROOT / 'Registration' / 'DeepLearning' / 'Tecnicas_registration'
OUTPUT_DIR = TECHNIQUES_DIR / 'outputs' / 'outputs_comparacion_deep_learning'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_CSV = TECHNIQUES_DIR / '00_baseline_clasico' / 'outputs' / 'outputs_baseline_clasico_final' / 'baseline_clasico_summary.csv'
VOXELMORPH_CSV = TECHNIQUES_DIR / '01_voxelmorph' / 'outputs' / 'outputs_voxelmorph_final_block' / 'voxelmorph_final_selected_summary.csv'
TRANSMORPH_CSV = TECHNIQUES_DIR / '02_transmorph' / 'outputs' / 'outputs_transmorph_final_block' / 'transmorph_final_selected_summary.csv'
DEEPERHISTREG_CSV = TECHNIQUES_DIR / '04_deeperhistreg' / 'outputs' / 'outputs_deeperhistreg_all' / 'deeperhistreg_all_summary.csv'

print('Outputs globales:', OUTPUT_DIR)"""
        ),
        nbf.v4.new_code_cell(
            """baseline = pd.read_csv(BASELINE_CSV).rename(columns={
    'subject_id': 'subject',
    'dice_final': 'classical_baseline_dice',
})
voxelmorph = pd.read_csv(VOXELMORPH_CSV)
transmorph = pd.read_csv(TRANSMORPH_CSV)
deeperhistreg = pd.read_csv(DEEPERHISTREG_CSV).rename(columns={
    'deeperhistreg_dice_mask_from_black_background': 'deeperhistreg_dice_approx',
})

comparison = (
    baseline[['subject', 'classical_baseline_dice', 'visual_review_flag']]
    .merge(voxelmorph[['subject', 'voxelmorph_dice', 'quality_status']], on='subject', how='inner')
    .rename(columns={'quality_status': 'voxelmorph_quality_status'})
    .merge(transmorph[['subject', 'transmorph_dice', 'quality_status']], on='subject', how='inner')
    .rename(columns={'quality_status': 'transmorph_quality_status'})
    .merge(deeperhistreg[['subject', 'deeperhistreg_dice_approx']], on='subject', how='inner')
)

comparison['voxelmorph_minus_baseline'] = comparison['voxelmorph_dice'] - comparison['classical_baseline_dice']
comparison['transmorph_minus_baseline'] = comparison['transmorph_dice'] - comparison['classical_baseline_dice']
comparison['deeperhistreg_minus_baseline_approx'] = comparison['deeperhistreg_dice_approx'] - comparison['classical_baseline_dice']

comparison.to_csv(OUTPUT_DIR / 'deep_learning_comparison_by_subject.csv', index=False)
display(comparison.round(4))"""
        ),
        nbf.v4.new_markdown_cell(
            """## Comparacion cuantitativa

Para VoxelMorph y TransMorph se utilizan las configuraciones conservadoras seleccionadas en sus respectivos bloques finales. DeeperHistReg se incluye como referencia externa, pero su Dice se marca como aproximado."""
        ),
        nbf.v4.new_code_cell(
            """method_columns = {
    'Classical baseline': 'classical_baseline_dice',
    'VoxelMorph-style': 'voxelmorph_dice',
    'TransMorph-style': 'transmorph_dice',
    'DeeperHistReg (approx.)': 'deeperhistreg_dice_approx',
}

method_summary = []
for method, column in method_columns.items():
    values = comparison[column]
    gain = values.mean() - comparison['classical_baseline_dice'].mean()
    wins = sum(values >= comparison[list(method_columns.values())].max(axis=1) - 1e-12)
    method_summary.append({
        'method': method,
        'mean_dice': values.mean(),
        'std_dice': values.std(ddof=1),
        'mean_gain_vs_baseline': gain,
        'best_subject_count': int(wins),
    })

method_summary_df = pd.DataFrame(method_summary)
method_summary_df.to_csv(OUTPUT_DIR / 'deep_learning_method_summary.csv', index=False)
display(method_summary_df.round(4))"""
        ),
        nbf.v4.new_code_cell(
            """subjects = comparison['subject'].tolist()
x = np.arange(len(subjects))
width = 0.20
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#777777']

fig, ax = plt.subplots(figsize=(14, 6))
for index, ((method, column), color) in enumerate(zip(method_columns.items(), colors)):
    offset = (index - 1.5) * width
    ax.bar(x + offset, comparison[column], width, label=method, color=color)

ax.set_ylim(0, 1.06)
ax.set_ylabel('Mask Dice')
ax.set_xticks(x)
ax.set_xticklabels(subjects)
ax.set_title('Registration comparison by subject')
ax.grid(axis='y', alpha=0.25)
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'deep_learning_dice_by_subject.png', dpi=180)
plt.show()"""
        ),
        nbf.v4.new_code_cell(
            """fig, axes = plt.subplots(1, 2, figsize=(14, 5))

summary_plot = method_summary_df.set_index('method')
axes[0].bar(summary_plot.index, summary_plot['mean_dice'], color=colors)
axes[0].set_ylim(0, 1.02)
axes[0].set_ylabel('Mean mask Dice')
axes[0].set_title('Mean Dice across six subjects')
axes[0].tick_params(axis='x', rotation=20)
axes[0].grid(axis='y', alpha=0.25)

gain_methods = ['VoxelMorph-style', 'TransMorph-style', 'DeeperHistReg (approx.)']
gain_values = summary_plot.loc[gain_methods, 'mean_gain_vs_baseline']
axes[1].bar(gain_methods, gain_values, color=['#ff7f00', '#4daf4a', '#777777'])
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_ylabel('Mean Dice gain vs baseline')
axes[1].set_title('Mean improvement over classical baseline')
axes[1].tick_params(axis='x', rotation=20)
axes[1].grid(axis='y', alpha=0.25)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'deep_learning_mean_and_gain.png', dpi=180)
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## Calidad de la deformacion

La siguiente tabla compara los indicadores de deformacion de los dos modelos desarrollados especificamente en el proyecto. Una fraccion alta de Jacobiano negativo indica plegamientos o deformaciones localmente no invertibles."""
        ),
        nbf.v4.new_code_cell(
            """vm_quality = voxelmorph[[
    'subject', 'flow_p95_px', 'negative_jacobian_fraction', 'quality_status'
]].rename(columns={
    'flow_p95_px': 'voxelmorph_flow_p95_px',
    'negative_jacobian_fraction': 'voxelmorph_negative_jacobian_fraction',
    'quality_status': 'voxelmorph_quality_status',
})

tm_quality = transmorph[[
    'subject', 'flow_p95_px', 'negative_jacobian_fraction', 'quality_status'
]].rename(columns={
    'flow_p95_px': 'transmorph_flow_p95_px',
    'negative_jacobian_fraction': 'transmorph_negative_jacobian_fraction',
    'quality_status': 'transmorph_quality_status',
})

quality = vm_quality.merge(tm_quality, on='subject')
quality.to_csv(OUTPUT_DIR / 'deep_learning_deformation_quality.csv', index=False)
display(quality.round(4))"""
        ),
        nbf.v4.new_code_cell(
            """fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(
    quality['voxelmorph_flow_p95_px'],
    quality['voxelmorph_negative_jacobian_fraction'],
    s=100,
    label='VoxelMorph-style',
    color='#ff7f00',
    edgecolor='black',
)
ax.scatter(
    quality['transmorph_flow_p95_px'],
    quality['transmorph_negative_jacobian_fraction'],
    s=100,
    label='TransMorph-style',
    color='#4daf4a',
    edgecolor='black',
)
for _, row in quality.iterrows():
    ax.annotate(row['subject'], (row['voxelmorph_flow_p95_px'], row['voxelmorph_negative_jacobian_fraction']), xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.annotate(row['subject'], (row['transmorph_flow_p95_px'], row['transmorph_negative_jacobian_fraction']), xytext=(5, -10), textcoords='offset points', fontsize=8)

ax.axhline(0.10, color='#cc4444', linestyle='--', linewidth=1, label='0.10 reference')
ax.set_xlabel('P95 displacement (px)')
ax.set_ylabel('Negative Jacobian fraction')
ax.set_title('Deformation magnitude and folding risk')
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'deep_learning_deformation_quality.png', dpi=180)
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## Resumen metodologico

| Metodo | Tipo | Uso de deep learning | Entrenamiento | Lectura principal |
|---|---|---|---|---|
| Baseline clasico | Rigido + afin + TV-L1 | No | No | Referencia fuerte y estable en 5/6 casos |
| VoxelMorph-style | CNN deformable 2D | Si | Instance-specific | Mejor resultado deep medio, pero con riesgo en SB017 |
| TransMorph-style | CNN + Transformer 2D | Si | Instance-specific | Segundo bloque deep, mas conservador pero inferior a VoxelMorph |
| DeeperHistReg | Framework histologico externo | Parcial/hibrido | Sin entrenamiento propio | No se adapta bien al salto H&E-HSI |

VoxelMorph-style y TransMorph-style no utilizan todavia el cubo espectral HSI completo: trabajan principalmente sobre mascaras y mapas de distancia. Esto deja abierta una futura linea spectral-aware con deformacion espacial 2D."""
        ),
        nbf.v4.new_code_cell(
            """baseline_mean = comparison['classical_baseline_dice'].mean()
vm_mean = comparison['voxelmorph_dice'].mean()
tm_mean = comparison['transmorph_dice'].mean()
dhr_mean = comparison['deeperhistreg_dice_approx'].mean()

report = f'''# Deep-learning registration comparison

## Mean Dice

- Classical baseline: {baseline_mean:.3f}
- VoxelMorph-style selected: {vm_mean:.3f} ({vm_mean - baseline_mean:+.3f} vs baseline)
- TransMorph-style selected: {tm_mean:.3f} ({tm_mean - baseline_mean:+.3f} vs baseline)
- DeeperHistReg approximate: {dhr_mean:.3f} ({dhr_mean - baseline_mean:+.3f} vs baseline)

## Conclusion

VoxelMorph-style is the main deep-learning result for the current dataset. TransMorph-style provides a valid transformer-based exploratory comparison but does not outperform VoxelMorph globally. DeeperHistReg is not competitive for the H&E-HSI modality gap. All neural results remain instance-specific proof-of-concept experiments because only six paired specimens are available.

Important: DeeperHistReg Dice is approximate and should not be interpreted as strictly equivalent to the other mask Dice values.
'''
(OUTPUT_DIR / 'README.md').write_text(report, encoding='utf-8')
print(report)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Conclusion final

1. **VoxelMorph-style es el bloque deep learning principal**: consigue el mejor Dice medio y gana en cuatro de los seis sujetos.
2. **TransMorph-style aporta valor metodologico** al explorar una arquitectura con Transformer, pero queda por debajo de VoxelMorph con la configuracion conservadora.
3. **El baseline clasico sigue siendo preferible en SB018 y SB019**, lo que demuestra que no conviene seleccionar un metodo solo por su media global.
4. **SB017 sigue siendo el caso critico**: las redes mejoran mucho el solape, pero la deformacion necesita revision.
5. El siguiente experimento deep con mayor valor seria incorporar informacion espectral HSI como canales o componentes PCA, manteniendo una deformacion espacial 2D."""
        ),
    ]

    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Created: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
