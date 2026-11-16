# Reportes de PinSearch

## Versión actual: PostgreSQL + pgvector, 10 000 imágenes COCO

| Archivo | Medición | Repetición |
| --- | --- | --- |
| `resultados_benchmark_pgvector.json` | 50 consultas reales, HNSW frente a búsqueda exacta, latencia SQL y Recall@10 de HNSW | `python benchmark.py --queries 50 --force` |
| `resultados_evaluacion_pgvector.json` | Recuperación texto a imagen con 1 000 captions COCO: Recall@1/5/10 y MRR@10 | `python evaluar.py --muestra 1000 --force` |
| `resultados_carga.json` | API HTTP con 24 consultas distintas y concurrencia 1, 4 y 8 | `python carga.py --force` |

Los JSON se reutilizan mientras la configuración y el estado de la base no cambien. Las mediciones son locales en CPU y no establecen una garantía de producción.

## Históricos de FAISS / SQLite (5 000 imágenes)

`resultados_benchmark.json`, `resultados_evaluacion.json` y `benchmark_latencia.png` corresponden al motor anterior y no describen PostgreSQL. Se conservan solo para comparación histórica.
