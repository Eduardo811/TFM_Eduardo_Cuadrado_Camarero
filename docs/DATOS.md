# Guia de datos

## Que se incluye

Este repositorio incluye datos derivados ligeros y resultados ya procesados:

- Imagenes H&E/HSI segmentadas y reescaladas en `Registration/DeepLearning/Imagenes_a_escala/`.
- Mascaras binarias, contornos y mapas de distancia firmada usados por los metodos de registro.
- Metadatos tecnicos de escala, forma de imagen y area de mascara.
- Resultados de registro en `Registration/DeepLearning/Tecnicas_registration/**/outputs/`.
- Figuras finales y de apoyo en `Fotos_Memoria/`.
- Un CSV publico de disponibilidad tecnica de los seis casos usados: `docs/metadata/sample_availability_public.csv`.

## Que no se incluye

No se han incluido en el repositorio publico:

- Cubos hiperespectrales brutos o preparados (`.hdr`, `.hrd`, archivos `raw`, `nrm`, `nrm_edu` sin extension).
- Imagenes histologicas completas `.mrxs` y sus archivos auxiliares `.dat`, `Index.dat` y `Slidedat.ini`.
- Fotos de adquisicion completas que no sean necesarias como figura/documentacion.
- Papers descargados y documentos de terceros.
- Instaladores de SlideViewer, SlideMaster, QuPath u otras herramientas.
- Entornos virtuales, caches, temporales de Word y archivos personales con enlaces privados.
- El Excel original `SpectraBreast_samples.xlsx`, porque contiene campos clinicos adicionales como fecha de cirugia e histotipo.

## CSV publico de disponibilidad

El archivo `docs/metadata/sample_availability_public.csv` resume solo la disponibilidad tecnica de los seis especimenes utilizados en la fase final del TFM. No incluye fechas, diagnosticos, histotipo ni campos clinicos descriptivos.

## Nota de confidencialidad

Aunque los archivos incluidos son derivados y usan codigos de muestra, siguen procediendo de imagenes biomédicas. Antes de publicar el repositorio en abierto, hay que confirmar que el consorcio/proyecto permite publicar estas figuras y derivados. Si la respuesta no esta clara, las opciones prudentes son:

- publicar el repositorio sin `Registration/DeepLearning/Imagenes_a_escala/`;
- publicar solo scripts, notebooks y figuras no sensibles;
- mantener el repositorio privado y conceder acceso bajo solicitud.

