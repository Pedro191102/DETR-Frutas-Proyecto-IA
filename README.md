# DETR

Extrae la carpeta completa. Abre **DETR_WINDOWS.bat** para iniciar un nuevo ajuste parcial desde los pesos base. Al finalizar todas las épocas, se abre la app con DETR 2 seleccionado. El lanzador utiliza CPython 3.15 para Windows x64. Lo busca primero con py -3.15 y después con python en PATH. Crea un entorno propio para 3.15. La primera instalación requiere internet. El entorno NVIDIA instala PyTorch 2.15.0 y torchvision 0.29.2 para CUDA 13.4. INICIAR_WINDOWS_CPU.bat usa los paquetes de CPU en su propio entorno. FruitVision.bat abre la misma aplicación. Las siguientes ejecuciones reutilizan el entorno.

## Cámara

Selecciona DETR en la barra lateral. Sus pesos y procesador se leen de models/base. El modelo utiliza la resolución de su procesador original; la velocidad depende del equipo. La cámara aplica la confianza seleccionada y DETR conserva una clase por propuesta y suprime cajas con mucho solapamiento para contar piezas. El panel amarillo agrupa las detecciones del fotograma por alimento. S captura la vista con el panel, la imagen original y los datos en JSON. Q o Esc finalizan la sesión. YOLOE sigue disponible como detector alternativo con sus propios pesos.

Las kcal se calculan con una unidad comestible de referencia por caja visible. Una rodaja puede recibir la misma etiqueta que una fruta entera. Para porciones cortadas usa la revisión manual de una foto. Si un alimento carece de referencia, aparece s/d y se indica el subtotal conocido. Puedes editar pesos y fuentes en config/nutrition_reference.csv.

## Entrenamiento local

Desde PowerShell, dentro de esta carpeta:

- Entrenar y abrir la app: `.\DETR_WINDOWS.bat`
- Abrir directamente la cámara o fotos: `.\DETR_WINDOWS.bat app`
- Ajuste parcial rápido: `.\DETR_WINDOWS.bat entrenar`
- Ajustar todos los parámetros desde los pesos incluidos: `.\DETR_WINDOWS.bat completo`
- Elegir tamaño y duración: `.\DETR_WINDOWS.bat --modo completo --train 1024 --val 128 --epochs 5 --batch 1`

En un entorno Python ya preparado: `python entrenar_detr.py --modo parcial` o `python entrenar_detr.py --modo completo`. Ambos descargan las anotaciones oficiales LVIS v1 y las fotos necesarias, con caché en data/lvis. Por defecto usan hasta 512 fotos de entrenamiento y 128 de validación durante cinco épocas, entre las 58 clases definidas en fruitvision/lvis_clases.py. El modo completo necesita más memoria y tiempo. Los resultados reales de cada corrida (configuración, clases, IDs de fotos, pérdidas y pesos de la mejor época) quedan en runs/detr. El doble clic abre la app al finalizar. Si entrenas por separado, abre o reinicia la app y selecciona DETR 2. La selección se actualiza después de validar la última época y guardar los mejores pesos. Una interrupción conserva la selección anterior. Para llevar un ajuste existente a esta versión, copia su carpeta runs y models/detr_actual.txt conservando las rutas relativas. La procedencia y la licencia de los pesos base están en models/base/ORIGEN.txt.
