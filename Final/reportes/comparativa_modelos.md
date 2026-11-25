# Comparativa de modelos CLIP en PinSearch

Catálogo: **20 000 imágenes COCO**. Dispositivo: **cuda** (NVIDIA GeForce GTX 1650). Captions: **500**. Consultas imagen→imagen: **100**. Categorías COCO: **79**.

Las tres mediciones usaron los mismos identificadores de consulta y el mismo código de evaluación. Los hashes de muestra se comprobaron antes de crear esta tabla.

## Calidad de recuperación

| Métrica | openai/clip-vit-base-patch32 | OpenCLIP ViT-B/16 (LAION-2B) | CLIP RN50 (OpenAI) |
|---|---:|---:|---:|
| Caption Recall@1 | 18.8% | 25.6% | 17.4% |
| Caption Recall@5 | 35.0% | 44.8% | 30.0% |
| Caption Recall@10 | 41.8% | 55.0% | 39.6% |
| Caption MRR@10 | 0.2589 | 0.3377 | 0.2346 |
| Objetos mAP@10 | 0.8203 | 0.8877 | 0.7985 |
| Objetos nDCG@10 | 0.8749 | 0.9187 | 0.8491 |
| Objetos Hit@10 | 100.0% | 100.0% | 100.0% |
| Objetos Precision@10 | 86.6% | 91.0% | 84.4% |
| Imagen→imagen ObjectHit@1 | 74.0% | 74.0% | 71.0% |
| Imagen→imagen ObjectHit@5 | 88.0% | 93.0% | 90.0% |
| Imagen→imagen ObjectHit@10 | 91.0% | 97.0% | 93.0% |
| Imagen→imagen ObjectPrecision@10 | 69.2% | 72.7% | 68.3% |
| HNSW vs. exacta: solapamiento@10 | 99.3% | 99.6% | 99.2% |
| Coseno medio top 1, captions | 0.3354 | 0.3159 | 0.2832 |
| Coseno medio top 1, objetos | 0.3248 | 0.3077 | 0.2713 |

## Velocidad y recursos

| Métrica | openai/clip-vit-base-patch32 | OpenCLIP ViT-B/16 (LAION-2B) | CLIP RN50 (OpenAI) |
|---|---:|---:|---:|
| Inferencia texto p50 (ms) | 12.54 | 6.38 | 7.68 |
| Inferencia texto p95 (ms) | 14.25 | 6.48 | 9.41 |
| Inferencia imagen p50 (ms) | 14.9 | 32.01 | 13.52 |
| Inferencia imagen p95 (ms) | 15.68 | 32.92 | 14.15 |
| Embeddings de imágenes/s, lote 16 | 61.74 | 32.56 | 70.49 |
| Búsqueda HNSW p50 (ms) | 6.11 | 6.09 | 9.61 |
| Búsqueda HNSW p95 (ms) | 8.43 | 8.6 | 16.22 |
| Búsqueda exacta p50 (ms) | 80.6 | 80.83 | 99.65 |
| Búsqueda exacta p95 (ms) | 89.03 | 89.42 | 112.43 |
| Dimensión | 512 | 512 | 1024 |
| Parámetros | 151277313 | 149620737 | 102007137 |
| RAM adicional al cargar (MiB) | 447.2 | 245.1 | 900.3 |
| VRAM pico (MiB) | 633.4 | 702.4 | 651.2 |
| Índice HNSW (MiB) | 52.09 | 52.09 | 156.26 |
| Tabla + índices (MiB) | 114.81 | 114.34 | 271.8 |

## Interpretación

- Recall@K de captions usa como objetivo la propia foto; captions parecidas o duplicadas pueden penalizar injustamente a un modelo.
- Las métricas de objetos usan categorías COCO como relevancia proxy; no sustituyen evaluación humana de similitud semántica.
- Los tiempos se midieron en esta máquina y dependen de CPU/GPU, caché y carga del sistema.
- La RAM diferencial incluye librerías cargadas por Python; no es una medición aislada de los pesos del modelo.
- El coseno medio del primer resultado no se presenta como confianza ni se usa para elegir ganador: las escalas entre modelos no están calibradas.
- El throughput incluye lectura/preprocesamiento e inferencia, pero excluye INSERT y construcción del índice HNSW.

## Lectura de estos resultados

- ViT-B/16 alcanzó 55.0% de Recall@10 en captions, 13.2 puntos por encima del modelo original en esta muestra. También lideró mAP@10 de objetos.
- Su inferencia de imagen tuvo p50 de 32.01 ms, frente a 14.9 ms del original; la ganancia de calidad tiene costo de latencia visual.
- RN50 logró 70.49 imágenes/s en lotes de 16, pero su Recall@10 textual fue inferior al de ViT-B/16 y su índice HNSW ocupó aproximadamente tres veces más espacio.

Archivos JSON de origen: `comparacion_v4.json`, `comparacion_vit_b16.json`, `comparacion_rn50.json`.
