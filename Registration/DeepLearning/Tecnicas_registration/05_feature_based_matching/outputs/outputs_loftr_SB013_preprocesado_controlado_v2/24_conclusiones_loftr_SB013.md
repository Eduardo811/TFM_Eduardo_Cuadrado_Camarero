### Lectura de los resultados

- Las escenas completas producen **30 correspondencias**. Al aplicar las máscaras sin recortar se obtienen **70** (+40).
- La eliminación del fondo eleva el Dice afín de **0.401** a **0.768** (+0.367) y el Dice de homografía de **0.573** a **0.829** (+0.256).
- El recorte independiente aumenta las correspondencias de **70** a **82** (+12), pero reduce el Dice afín en -0.320 y el Dice de homografía en -0.409.
- Por tanto, la evidencia favorable se concentra en la aplicación de las máscaras: eliminar el fondo facilita LoFTR, mientras que un recorte independiente no garantiza una transformación geométricamente mejor.
- Un aumento del número de correspondencias no implica necesariamente una mejora geométrica: deben considerarse conjuntamente los inliers de RANSAC y el solapamiento de las máscaras.
- El experimento utiliza únicamente SB013 y un LoFTR preentrenado sobre imágenes naturales. Sus resultados sirven como comprobación preliminar del preprocesado, pero no permiten generalizar al conjunto completo.
