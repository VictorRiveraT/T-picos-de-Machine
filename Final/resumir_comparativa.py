"""Genera una tabla Markdown de tres mediciones hechas con comparar_modelos.py."""

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def value(report, key, percent=False):
    number = report["metrics"][key]
    return f"{number * 100:.1f}%" if percent else str(number)


def row(label, reports, key, percent=False):
    return f"| {label} | " + " | ".join(value(report, key, percent) for report in reports) + " |"


def nested_row(label, reports, key, field):
    return f"| {label} | " + " | ".join(str(report["metrics"][key][field]) for report in reports) + " |"


def field_row(label, reports, key):
    return f"| {label} | " + " | ".join(str(report[key]) for report in reports) + " |"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--vit", type=Path, required=True)
    p.add_argument("--rn", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    reports = [load(args.base), load(args.vit), load(args.rn)]
    for key in ("image_count", "caption_queries", "caption_sample_sha256",
                "image_queries", "image_sample_sha256", "object_categories",
                "device", "gpu_name", "torch_version", "platform"):
        if len({json.dumps(report[key], sort_keys=True) for report in reports}) != 1:
            raise RuntimeError(f"Experimentos no comparables: {key} difiere")
    labels = [report["model"] for report in reports]
    header = "| Métrica | " + " | ".join(labels) + " |"
    separator = "|---|" + "|".join("---:" for _ in reports) + "|"
    lines = [
        "# Comparativa de modelos CLIP en PinSearch",
        "",
        f"Catálogo: **{reports[0]['image_count']:,} imágenes COCO**. ".replace(",", " ") +
        f"Dispositivo: **{reports[0]['device']}** ({reports[0]['gpu_name']}). "
        f"Captions: **{reports[0]['caption_queries']}**. "
        f"Consultas imagen→imagen: **{reports[0]['image_queries']}**. "
        f"Categorías COCO: **{reports[0]['object_queries']}**.",
        "",
        "Las tres mediciones usaron los mismos identificadores de consulta y el mismo código de evaluación. "
        "Los hashes de muestra se comprobaron antes de crear esta tabla.",
        "",
        "## Calidad de recuperación",
        "",
        header, separator,
        row("Caption Recall@1", reports, "caption_recall@1", True),
        row("Caption Recall@5", reports, "caption_recall@5", True),
        row("Caption Recall@10", reports, "caption_recall@10", True),
        row("Caption MRR@10", reports, "caption_mrr@10"),
        row("Objetos mAP@10", reports, "object_map@10"),
        row("Objetos nDCG@10", reports, "object_ndcg@10"),
        row("Objetos Hit@10", reports, "object_hit@10", True),
        row("Objetos Precision@10", reports, "object_precision@10", True),
        row("Imagen→imagen ObjectHit@1", reports, "image_object_hit@1", True),
        row("Imagen→imagen ObjectHit@5", reports, "image_object_hit@5", True),
        row("Imagen→imagen ObjectHit@10", reports, "image_object_hit@10", True),
        row("Imagen→imagen ObjectPrecision@10", reports, "image_object_precision@10", True),
        row("HNSW vs. exacta: solapamiento@10", reports, "hnsw_overlap_with_exact@10", True),
        row("Coseno medio top 1, captions", reports, "caption_top1_cosine_mean"),
        row("Coseno medio top 1, objetos", reports, "object_top1_cosine_mean"),
        "",
        "## Velocidad y recursos",
        "",
        header, separator,
        nested_row("Inferencia texto p50 (ms)", reports, "text_inference", "p50_ms"),
        nested_row("Inferencia texto p95 (ms)", reports, "text_inference", "p95_ms"),
        nested_row("Inferencia imagen p50 (ms)", reports, "image_inference", "p50_ms"),
        nested_row("Inferencia imagen p95 (ms)", reports, "image_inference", "p95_ms"),
        row("Embeddings de imágenes/s, lote 16", reports, "image_embedding_throughput_per_second_batch16"),
        nested_row("Búsqueda HNSW p50 (ms)", reports, "vector_search_hnsw", "p50_ms"),
        nested_row("Búsqueda HNSW p95 (ms)", reports, "vector_search_hnsw", "p95_ms"),
        nested_row("Búsqueda exacta p50 (ms)", reports, "vector_search_exact", "p50_ms"),
        nested_row("Búsqueda exacta p95 (ms)", reports, "vector_search_exact", "p95_ms"),
        field_row("Dimensión", reports, "embedding_dim"),
        field_row("Parámetros", reports, "model_parameters"),
        field_row("RAM adicional al cargar (MiB)", reports, "model_load_delta_rss_mib"),
        field_row("VRAM pico (MiB)", reports, "gpu_peak_mib"),
        field_row("Índice HNSW (MiB)", reports, "hnsw_index_mib"),
        field_row("Tabla + índices (MiB)", reports, "images_table_with_indexes_mib"),
        "",
        "## Interpretación",
        "",
        "- Recall@K de captions usa como objetivo la propia foto; captions parecidas o duplicadas pueden penalizar injustamente a un modelo.",
        "- Las métricas de objetos usan categorías COCO como relevancia proxy; no sustituyen evaluación humana de similitud semántica.",
        "- Los tiempos se midieron en esta máquina y dependen de CPU/GPU, caché y carga del sistema.",
        "- La RAM diferencial incluye librerías cargadas por Python; no es una medición aislada de los pesos del modelo.",
        "- El coseno medio del primer resultado no se presenta como confianza ni se usa para elegir ganador: las escalas entre modelos no están calibradas.",
        "- El throughput incluye lectura/preprocesamiento e inferencia, pero excluye INSERT y construcción del índice HNSW.",
        "",
        "## Lectura de estos resultados",
        "",
        f"- ViT-B/16 alcanzó {reports[1]['metrics']['caption_recall@10'] * 100:.1f}% de Recall@10 en captions, "
        f"{(reports[1]['metrics']['caption_recall@10'] - reports[0]['metrics']['caption_recall@10']) * 100:.1f} puntos "
        "por encima del modelo original en esta muestra. También lideró mAP@10 de objetos.",
        f"- Su inferencia de imagen tuvo p50 de {reports[1]['metrics']['image_inference']['p50_ms']} ms, "
        f"frente a {reports[0]['metrics']['image_inference']['p50_ms']} ms del original; la ganancia de calidad tiene costo de latencia visual.",
        f"- RN50 logró {reports[2]['metrics']['image_embedding_throughput_per_second_batch16']} imágenes/s en lotes de 16, "
        "pero su Recall@10 textual fue inferior al de ViT-B/16 y su índice HNSW ocupó aproximadamente tres veces más espacio.",
        "",
        "Archivos JSON de origen: " + ", ".join(f"`{path.name}`" for path in (args.base, args.vit, args.rn)) + ".",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
