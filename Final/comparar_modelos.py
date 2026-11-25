"""Compara un checkpoint CLIP sobre el mismo catálogo y consultas COCO.

Ejecutar una vez por rama/base. No mezcla embeddings de modelos distintos.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image
import psutil


def quantile_ms(values, q):
    return round(float(np.percentile(values, q)), 2)


def timed(call, repeats=16, warmup=3):
    for _ in range(warmup):
        call()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        times.append((time.perf_counter() - started) * 1000)
    return {"p50_ms": quantile_ms(times, 50), "p95_ms": quantile_ms(times, 95)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--database-url", required=True, help="URL de la base exclusiva del modelo")
    parser.add_argument("--caption-queries", type=int, default=500)
    parser.add_argument("--expected-count", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.caption_queries < 1:
        parser.error("--caption-queries debe ser positivo")
    model_dir = args.model_dir.resolve()
    sys.path.insert(0, str(model_dir))
    os.environ["DATABASE_URL"] = args.database_url

    rss_before_model = psutil.Process().memory_info().rss
    from catalogo import Catalog, connect
    from embeddings import DEVICE, MODEL_NAME, embed_imagenes, embed_texto, model
    import torch
    if (model_dir / "model_config.py").is_file():
        from model_config import MODEL_ID
    else:
        MODEL_ID = MODEL_NAME
    rss_after_model = psutil.Process().memory_info().rss

    metadata_path = model_dir / "data/coco_metadata.json"
    image_dir = model_dir / "imagenes"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["records"]
    with connect() as conn:
        db_rows = conn.execute("SELECT filename, attributes->'objetos' FROM images ORDER BY filename").fetchall()
        index_bytes, table_bytes = conn.execute(
            "SELECT pg_relation_size('images_embedding_hnsw'), pg_total_relation_size('images')"
        ).fetchone()
    if len(db_rows) != args.expected_count:
        raise RuntimeError(f"Índice incompleto: {len(db_rows)} filas; esperadas {args.expected_count}")
    catalog = Catalog()
    names = [name for name, _ in db_rows]
    labels = {name: set(objects or []) for name, objects in db_rows}
    caption_rows = [(name, metadata[name]["caption"]) for name in names if metadata[name]["caption"]]
    rng = np.random.default_rng(42)
    selected = [caption_rows[int(i)] for i in rng.choice(
        len(caption_rows), size=min(args.caption_queries, len(caption_rows)), replace=False
    )]
    sample_hash = hashlib.sha256("\n".join(name for name, _ in selected).encode()).hexdigest()

    caption_hits = {1: 0, 5: 0, 10: 0}
    reciprocal = 0.0
    scores = []
    hnsw_ms, exact_ms, hnsw_overlap = [], [], []
    for number, (name, caption) in enumerate(selected, 1):
        vector = embed_texto([caption])[0]
        start = time.perf_counter()
        found = catalog.search(vector, 10, mode="hnsw")
        hnsw_ms.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        exact = catalog.search(vector, 10, mode="flat")
        exact_ms.append((time.perf_counter() - start) * 1000)
        found_names = [row["filename"] for row in found]
        exact_names = {row["filename"] for row in exact}
        hnsw_overlap.append(len(set(found_names) & exact_names) / 10)
        for k in caption_hits:
            caption_hits[k] += name in found_names[:k]
        if name in found_names:
            reciprocal += 1 / (found_names.index(name) + 1)
        if found:
            scores.append(found[0]["score"])
        if number % 25 == 0:
            print(f"Captions: {number}/{len(selected)}", flush=True)

    frequencies = Counter(label for objects in labels.values() for label in objects)
    categories = sorted(label for label, n in frequencies.items() if 25 <= n <= 4000)
    if not categories:
        raise RuntimeError("No hay categorías COCO elegibles")
    ap, ndcg, hit, precision, object_scores = [], [], [], [], []
    for category in categories:
        vector = embed_texto([f"a photo of a {category}"])[0]
        found = catalog.search(vector, 10, mode="hnsw")
        flags = [category in labels[row["filename"]] for row in found]
        relevant = min(10, frequencies[category])
        cumulative = 0
        average_precision = 0.0
        dcg = 0.0
        for rank, flag in enumerate(flags, 1):
            cumulative += flag
            if flag:
                average_precision += cumulative / rank
                dcg += 1 / math.log2(rank + 1)
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, relevant + 1))
        ap.append(average_precision / relevant)
        ndcg.append(dcg / ideal)
        hit.append(any(flags))
        precision.append(sum(flags) / 10)
        if found:
            object_scores.append(found[0]["score"])

    image_candidates = []
    for name in names:
        eligible = [label for label in labels[name] if 25 <= frequencies[label] <= 4000]
        if eligible:
            category = min(eligible, key=lambda label: (frequencies[label], label))
            image_candidates.append((name, category))
    image_rng = np.random.default_rng(43)
    image_selected = [image_candidates[int(i)] for i in image_rng.choice(
        len(image_candidates), size=min(100, len(image_candidates)), replace=False
    )]
    image_sample_hash = hashlib.sha256("\n".join(name for name, _ in image_selected).encode()).hexdigest()
    image_hits = {1: 0, 5: 0, 10: 0}
    image_precision = []
    for name, category in image_selected:
        vector = catalog.get_by_filename(name)["embedding"]
        found = catalog.search(vector, 10, mode="hnsw", exclude=name)
        flags = [category in labels[row["filename"]] for row in found]
        for k in image_hits:
            image_hits[k] += any(flags[:k])
        image_precision.append(sum(flags) / 10)

    probe_name = selected[0][0]
    with Image.open(image_dir / probe_name) as source:
        probe = source.convert("RGB")
    text_timing = timed(lambda: embed_texto([selected[0][1]]))
    image_timing = timed(lambda: embed_imagenes([probe]))
    batch_names = names[:64]
    started = time.perf_counter()
    for start in range(0, len(batch_names), 16):
        images = []
        for name in batch_names[start:start + 16]:
            with Image.open(image_dir / name) as source:
                images.append(source.convert("RGB"))
        embed_imagenes(images)
    batch_seconds = time.perf_counter() - started

    dimension = int(embed_texto(["a photo"])[0].size)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "model_id": MODEL_ID,
        "embedding_dim": dimension,
        "image_count": len(names),
        "caption_queries": len(selected),
        "caption_sample_sha256": sample_hash,
        "image_queries": len(image_selected),
        "image_sample_sha256": image_sample_hash,
        "object_queries": len(categories),
        "object_categories": categories,
        "device": DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "model_load_delta_rss_mib": round((rss_after_model - rss_before_model) / 2**20, 1),
        "loaded_process_rss_mib": round(rss_after_model / 2**20, 1),
        "process_rss_mib": round(psutil.Process().memory_info().rss / 2**20, 1),
        "gpu_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1) if DEVICE == "cuda" else None,
        "hnsw_index_mib": round(index_bytes / 2**20, 2),
        "images_table_with_indexes_mib": round(table_bytes / 2**20, 2),
        "metrics": {
            **{f"caption_recall@{k}": round(caption_hits[k] / len(selected), 4) for k in caption_hits},
            "caption_mrr@10": round(reciprocal / len(selected), 4),
            "object_map@10": round(float(np.mean(ap)), 4),
            "object_ndcg@10": round(float(np.mean(ndcg)), 4),
            "object_hit@10": round(float(np.mean(hit)), 4),
            "object_precision@10": round(float(np.mean(precision)), 4),
            **{f"image_object_hit@{k}": round(image_hits[k] / len(image_selected), 4) for k in image_hits},
            "image_object_precision@10": round(float(np.mean(image_precision)), 4),
            "caption_top1_cosine_mean": round(float(np.mean(scores)), 4),
            "object_top1_cosine_mean": round(float(np.mean(object_scores)), 4),
            "hnsw_overlap_with_exact@10": round(float(np.mean(hnsw_overlap)), 4),
            "text_inference": text_timing,
            "image_inference": image_timing,
            "image_embedding_throughput_per_second_batch16": round(len(batch_names) / batch_seconds, 2),
            "vector_search_hnsw": {"p50_ms": quantile_ms(hnsw_ms, 50), "p95_ms": quantile_ms(hnsw_ms, 95)},
            "vector_search_exact": {"p50_ms": quantile_ms(exact_ms, 50), "p95_ms": quantile_ms(exact_ms, 95)},
        },
        "methodology": {
            "caption": "Una caption COCO identifica su propia imagen; Recall@K y MRR@10 sobre muestra fija semilla 42.",
            "object": "Prompts ingleses por categoría oficial COCO con 25-4000 apariciones; relevancia=presencia de categoría; promedios macro.",
            "image": "100 consultas imagen-imagen sin la propia imagen; categoría COCO elegible menos frecuente como relevancia proxy; semilla 43.",
            "cosine": "Coseno medio del primer resultado, no probabilidad ni confianza calibrada; comparar con cautela entre modelos.",
            "latency": "Tiempos de pared en este equipo; inferencia incluye preprocesamiento; búsqueda incluye SQL y red local.",
            "throughput": "64 imágenes en lotes de 16, incluyendo lectura de archivos e inferencia, sin INSERT ni construcción HNSW.",
            "memory": "RSS del proceso y diferencia antes/después de importar el modelo; incluye dependencias Python, no equivale a memoria exclusiva de pesos.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model": MODEL_NAME, "metrics": report["metrics"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
