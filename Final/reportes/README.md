# Reportes de la versión 4

Estos son resultados medidos localmente con 5 000 imágenes COCO y CLIP en CPU. Cada JSON incluye fecha UTC, configuración y huellas de las entradas para verificar cuándo debe repetirse la medición.

| Archivo | Qué mide | Comando para reproducir |
| --- | --- | --- |
| `resultados_benchmark.json` | Latencia de FAISS Flat y HNSW con 10 000, 100 000 y 500 000 vectores sintéticos; 200 consultas por tamaño. | `python benchmark.py --force` |
| `resultados_evaluacion.json` | Recall@1/5/10 y MRR@10 en una muestra fija de 1 000 captions COCO. | `python evaluar.py --muestra 1000 --force` |
| `resultados_carga.json` | Rendimiento HTTP texto a imagen con 24 consultas únicas y concurrencia 1, 4 y 8. | `python carga.py --force` |

El benchmark FAISS no incluye inferencia ni red. La prueba HTTP es breve y depende de la máquina y la carga del momento. Estas cifras no garantizan el mismo rendimiento en producción.
