# Guia de archivos

## Lectura general del repositorio

`Registration/DeepLearning/` contiene la evolucion tecnica del trabajo: generacion de pseudo-RGB HSI, segmentacion, orientacion, homogeneizacion de escala y preparacion de pares comparables. Los notebooks `1_` a `10_` documentan esta fase.

`Registration/DeepLearning/Imagenes_a_escala/` contiene las entradas derivadas usadas por los registros finales. Cada caso (`SB012`, `SB013`, `SB017`, `SB018`, `SB019`, `SB020`) tiene imagen H&E, imagen HSI, mascaras, contornos, mapas de distancia firmada, una vista previa y metadatos.

`Registration/DeepLearning/Tecnicas_registration/` es el nucleo del TFM. Esta dividido por familias de metodos: baseline clasico, VoxelMorph-style, TransMorph-style, DeeperHistReg y LoFTR.

`Fotos_Memoria/` contiene las figuras preparadas para la memoria, con nombres conservados para que sea facil relacionarlas con el documento.

`Datos/SB013/` contiene notebooks exploratorios iniciales. Son importantes como historial de trabajo, pero no son el pipeline final.

## Notebooks principales

| Archivo | Descripcion |
|---|---|
| `Datos/Primeras_pruebass.ipynb` | Primeras pruebas de lectura de datos HSI/H&E y visualizacion. |
| `Datos/SB013/1_SB013.ipynb` | Exploracion inicial del caso SB013 y formatos disponibles. |
| `Datos/SB013/2_Rois_Segmentacion.ipynb` | Primeras pruebas con ROIs, XML/anotaciones y segmentacion. |
| `Datos/SB013/3_Segmentacion_especimen.ipynb` | Pruebas de segmentacion del especimen en H&E. |
| `Datos/SB013/4_Level4.ipynb` | Exploracion del nivel 4 de la WSI y procesamiento por tiles. |
| `Datos/SB013/5_Level4.ipynb` | Desarrollo intensivo de tiles/contornos en nivel 4; se mantiene como historial exploratorio. |
| `Datos/SB013/6_ROI.ipynb` | Pruebas de ROI y depuracion de region de interes. |
| `Datos/SB013/7_Final.ipynb` | Cierre de la fase inicial de segmentacion/contorno en SB013. |
| `Datos/SB013/8_Depuraciones.ipynb` | Depuraciones finales de esa linea exploratoria. |
| `Registration/DeepLearning/1_Prueba_Deep.ipynb` | Primeras pruebas de pseudo-RGB HSI, LoFTR y preparacion para registro. |
| `Registration/DeepLearning/2_Segmentacion_HSI.ipynb` | Segmentacion de imagen hiperespectral. |
| `Registration/DeepLearning/2.5_Comprobaciones_Segmentacion_HSI.ipynb` | Comprobaciones visuales de segmentacion HSI. |
| `Registration/DeepLearning/3_Segmentacion_H&E.ipynb` | Segmentacion de imagen histologica H&E. |
| `Registration/DeepLearning/4_Orientacion_HSI/4_Orientacion.ipynb` | Experimento de orientacion usando tintas y contornos. |
| `Registration/DeepLearning/4_Orientacion_HSI/4_2_OrientacionFinal.ipynb` | Version consolidada de la orientacion por tintas; documenta por que no se adopta como solucion general. |
| `Registration/DeepLearning/5_Pruebas_Deep.ipynb` | Preparacion de pares de imagenes para pruebas de registro/aprendizaje profundo. |
| `Registration/DeepLearning/6_Todas_las_imagenes.ipynb` | Revision de todos los casos disponibles. |
| `Registration/DeepLearning/7_Deteccion_caja_azul_HSI.ipynb` | Deteccion del biocassette/caja azul en HSI y delimitacion de ROI. |
| `Registration/DeepLearning/8_igualamos_tamaño.ipynb` | Reescalado y compatibilizacion de tamanos entre H&E e HSI. |
| `Registration/DeepLearning/9_funcion_preprocesado.ipynb` | Funciones reutilizables de preprocesado H&E/HSI. |
| `Registration/DeepLearning/10_guardar_imagenes_a_escala.ipynb` | Exportacion de imagenes, mascaras, contornos y mapas de distancia a escala fisica comun. |
| `Registration/DeepLearning/Comprobacion_H&E.ipynb` | Comprobacion final de salidas H&E. |
| `Registration/DeepLearning/Comprobacion_HSI.ipynb` | Comprobacion final de salidas HSI. |
| `Registration/DeepLearning/Tecnicas_registration/00_baseline_clasico/notebooks/11_registro_rigido_mascaras_SB013.ipynb` | Registro rigido inicial sobre SB013. |
| `Registration/DeepLearning/Tecnicas_registration/00_baseline_clasico/notebooks/12_registro_rigido_mascaras_todos.ipynb` | Registro rigido para todos los sujetos. |
| `Registration/DeepLearning/Tecnicas_registration/00_baseline_clasico/notebooks/13_registro_afin_mascaras_todos.ipynb` | Refinamiento afin con mascaras/mapas de distancia. |
| `Registration/DeepLearning/Tecnicas_registration/00_baseline_clasico/notebooks/14_registro_no_rigido_mascaras_todos.ipynb` | Registro deformable clasico TV-L1. |
| `Registration/DeepLearning/Tecnicas_registration/00_baseline_clasico/notebooks/15_comparacion_final_registration.ipynb` | Comparacion de etapas rigida, afin y no rigida. |
| `Registration/DeepLearning/Tecnicas_registration/00_baseline_clasico/notebooks/16_exportar_baseline_clasico_final.ipynb` | Exportacion del baseline final seleccionado por sujeto. |
| `Registration/DeepLearning/Tecnicas_registration/04_deeperhistreg/notebooks/17_deeperhistreg_exploratorio_SB013.ipynb` | Prueba inicial con DeeperHistReg sobre SB013. |
| `Registration/DeepLearning/Tecnicas_registration/01_voxelmorph/notebooks/18_voxelmorph_instance_specific_SB013_SB017.ipynb` | Primeros experimentos VoxelMorph-style por especimen. |
| `Registration/DeepLearning/Tecnicas_registration/01_voxelmorph/notebooks/19_voxelmorph_instance_todos_vs_baseline.ipynb` | Comparacion VoxelMorph-style frente al baseline en los seis sujetos. |
| `Registration/DeepLearning/Tecnicas_registration/02_transmorph/notebooks/20_transmorph_exploratorio_SB013.ipynb` | Adaptacion exploratoria TransMorph-style 2D sobre SB013. |
| `Registration/DeepLearning/Tecnicas_registration/02_transmorph/notebooks/21_transmorph_resultados_todos.ipynb` | Resultados TransMorph-style sobre todos los sujetos. |
| `Registration/DeepLearning/Tecnicas_registration/notebooks/22_comparacion_deep_learning.ipynb` | Comparacion global baseline, VoxelMorph, TransMorph y DeeperHistReg. |
| `Registration/DeepLearning/Tecnicas_registration/05_feature_based_matching/notebooks/23_loftr_SB013_segmentado.ipynb` | LoFTR sobre SB013 segmentado. |
| `Registration/DeepLearning/Tecnicas_registration/05_feature_based_matching/notebooks/24_comparacion_loftr_preprocesado_SB013.ipynb` | Comparacion controlada del efecto del preprocesado en LoFTR. |

## Resultados clave

- `00_baseline_clasico/outputs/outputs_baseline_clasico_final/`: resultado clasico final por sujeto.
- `01_voxelmorph/outputs/outputs_voxelmorph_final_block/`: bloque final de VoxelMorph-style con configuracion seleccionada.
- `02_transmorph/outputs/outputs_transmorph_final_block/`: bloque final de TransMorph-style.
- `04_deeperhistreg/outputs/outputs_deeperhistreg_all/`: resultados exploratorios con DeeperHistReg.
- `05_feature_based_matching/outputs/`: resultados LoFTR y comparacion de preprocesado.
- `outputs/outputs_comparacion_deep_learning/`: tablas y figuras de comparacion global.

## Archivos excluidos de forma deliberada

Los datos brutos, papers, instaladores, notas privadas, temporales de Word y versiones antiguas de entregables no forman parte del repositorio publico. La razon es una combinacion de peso, copyright, privacidad y claridad.

