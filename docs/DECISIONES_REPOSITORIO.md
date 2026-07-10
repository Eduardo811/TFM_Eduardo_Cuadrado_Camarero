# Decisiones de inclusion y exclusion

Este documento resume como se ha transformado la carpeta local de trabajo en una version preparada para GitHub.

## Incluido

| Origen local | Destino en repo | Motivo |
|---|---|---|
| `Registration/DeepLearning/*.ipynb` | mismo path | Notebooks principales de preprocesado, segmentacion, orientacion y preparacion de pares. |
| `Registration/DeepLearning/4_Orientacion_HSI/` | mismo path | Documenta la linea de trabajo basada en tintas, incluida su limitacion. |
| `Registration/DeepLearning/Imagenes_a_escala/` | mismo path | Derivados ligeros usados para registro: imagenes segmentadas, mascaras, contornos, mapas de distancia y metadatos. |
| `Registration/DeepLearning/Tecnicas_registration/` | mismo path | Nucleo reproducible: baseline, VoxelMorph-style, TransMorph-style, DeeperHistReg, LoFTR y comparacion global. |
| `Datos/Primeras_pruebass.ipynb` y `Datos/SB013/*.ipynb` | mismo path parcial | Historial exploratorio inicial del caso SB013, sin copiar datos brutos. |
| `Fotos_Memoria/` | mismo path | Figuras de la memoria y figuras de apoyo con nombres conservados. |
| `Memoria_registration_SPECTRA_BREAST.md` | raiz | Resumen temprano del trabajo y estado del arte, util como contexto. |
| `Registration/Marc_metodo/Pipeline.txt` | mismo path | Nota metodologica propia sobre pipeline razonable de registro. |
| `Datos/SpectraBreast_samples.xlsx` | `docs/metadata/sample_availability_public.csv` | Se sustituye el Excel completo por un CSV tecnico minimo sin fecha de cirugia ni histotipo. |

## Excluido

| Origen local | Motivo |
|---|---|
| `Datos/SB012`, `Datos/SB013`, `Datos/SB017_uomo`, `Datos/SB018`, `Datos/SB019`, `Datos/SB020` brutos | Datos HSI/H&E enormes y potencialmente sensibles. Solo se copian notebooks exploratorios concretos. |
| Archivos `.hdr`, `.hrd`, `raw`, `nrm`, `nrm_edu`, `.mrxs`, `.dat`, `.ini` | Datos originales o auxiliares de imagen completa, no adecuados para GitHub publico. |
| `Documentos_enrique/` | Datos externos muy pesados y fuera del nucleo del TFM final. |
| `Papers/` y `Registration_estado_del_arte/**/*.pdf` | PDFs de terceros, no publicables por copyright. |
| `Spectra_Breast/` | Documentacion de proyecto/consorcio, no necesaria para reproducir el trabajo publico. |
| `Presentacion_*.pptx`, `Entrega_25_Ingles*.docx`, `MemoriaTFM*.docx` | Versiones de entrega/presentacion no necesarias en el repositorio de codigo. |
| `MemoriaTFM_2.0.pdf` | Se recomienda incluir solo el PDF final ya corregido y con enlace al repo, si se desea una copia autocontenida. |
| `revision_memoria_2_0_*` | Auditoria interna de la memoria, util para trabajar pero no para el repositorio publico. |
| `Info.txt`, `q_hacer.txt`, `Registration/DeepLearning/Deep.txt` | Contienen enlaces privados, correos, notas sueltas o referencias de chat. |
| `Summary.zip`, `Registration/DeepLearning.zip` | Duplicados comprimidos y pesados. |
| `Datos/.venv/`, `.git/`, temporales `~WR*`, `~$*`, `.DS_Store`, `._*` | Ruido tecnico/local. |

## Opcional antes de publicar

- Incluir `MemoriaTFM_2.0.pdf` solo cuando sea la version final definitiva.
- Crear una licencia formal cuando esten claros los permisos de codigo, figuras y derivados de imagen.
- Si el repositorio se quiere aligerar, mover `*.npz` y `*.mha` a Git LFS o publicar solo outputs finales.
