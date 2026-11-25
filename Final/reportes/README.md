# Reportes de PinSearch: OpenCLIP ViT-B/16 LAION-2B

## Comparación nueva de tres modelos

`comparacion_v4.json`, `comparacion_vit_b16.json` y `comparacion_rn50.json` son las mediciones del mismo evaluador `comparar_modelos.py`, con 20 000 imágenes y muestras idénticas. `comparativa_modelos.md` resume sus resultados y limitaciones. `resultados_indexacion_modelo.json` documenta el último tramo de indexación de esta rama; si se reanudó un proceso anterior, su throughput **no** representa el tiempo de las 20 000 imágenes completas.

Las mediciones se hicieron con la misma GPU. Las métricas de objetos usan categorías COCO como proxy y el coseno no es una probabilidad calibrada. Los archivos de la sección siguiente se heredaron de v4 y **no miden OpenCLIP ViT-B/16 LAION-2B**.

## Reportes heredados de v4: PostgreSQL + pgvector, 20 000 imágenes COCO

| Archivo | Evidencia | Repetición |
| --- | --- | --- |
| `resultados_escalabilidad_pgvector.json` y `escalabilidad_pgvector.png` | 1 000, 5 000, 10 000 y 20 000 embeddings COCO reales; HNSW frente a búsqueda exacta, p50/p95, Recall@10 y tiempo de construcción | `python benchmark.py --force` |
| `resultados_evaluacion_pgvector.json` | Recuperación texto→imagen con 1 000 captions COCO: Recall@1/5/10 y MRR@10 | `python evaluar.py --muestra 1000 --force` |
| `resultados_evaluacion_imagenes.json` | Recuperación imagen→imagen con 500 consultas: acierto de categoría COCO en los primeros 1/5/10 | `python evaluar_imagenes.py --muestra 500 --force` |
| `resultados_carga.json` | API HTTP con 100 consultas distintas por nivel de concurrencia 1, 4, 8 y 16 | `python carga.py --requests 100 --concurrency 1 4 8 16 --force` |

El benchmark usa tablas temporales y verifica con `EXPLAIN` que HNSW consulte el índice y el modo exacto haga un barrido. La métrica imagen→imagen usa objetos COCO como referencia semántica parcial, no equivalencias visuales evaluadas por humanos. Las pruebas de carga son locales en CPU y no garantizan el mismo rendimiento en otro equipo. Los JSON guardan configuración y estado de la base; si no cambian, se reutilizan sin recalcular.

## Referencia histórica medida en v4

| Prueba | Resultado |
| --- | --- |
| Búsqueda SQL, 10 000 → 20 000 imágenes | HNSW p50: 5,865 → 6,261 ms; exacta p50: 26,284 → 47,647 ms; Recall@10 de HNSW frente a la búsqueda exacta: 100 % en las 100 consultas sintéticas de cada tamaño |
| Texto → imagen, 20 000 candidatas | Recall@10: 45,1 % con 1 000 captions COCO |
| Imagen → imagen, 20 000 candidatas | ObjectHit@10: 90,4 % con 500 imágenes; coincidencia de categoría COCO como proxy |
| API HTTP, 20 000 imágenes | 400/400 solicitudes correctas; a 16 solicitudes simultáneas, p95: 580,28 ms y 30,06 solicitudes/s |

La comparación SQL excluye la inferencia del modelo y la API HTTP. La prueba HTTP sí las incluye. El rendimiento HTTP alcanzó su máximo medido a concurrencia 4 (38,92 solicitudes/s) y descendió a 8 y 16; esta configuración de CPU necesita más capacidad de servicio para mantener el rendimiento al aumentar las solicitudes simultáneas. Los resultados de recuperación usan referencias parciales de COCO y no implican satisfacción humana del 100 %.

## Referencias históricas

`resultados_evaluacion_imagenes_10k.json` y `resultados_benchmark_pgvector.json` documentan la etapa PostgreSQL de 10 000 imágenes. `resultados_benchmark.json`, `resultados_evaluacion.json` y `benchmark_latencia.png` pertenecen al motor FAISS/SQLite anterior de 5 000 imágenes. No deben mezclarse con la medición de 20 000.
