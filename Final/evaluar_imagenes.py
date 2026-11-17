"""Evalúa búsqueda imagen-imagen con categorías oficiales COCO como proxy semántico."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from catalogo import Catalog, connect


def execute(args):
    if args.muestra < 1:
        raise ValueError("--muestra debe ser positivo")
    with connect() as conn:
        rows = conn.execute("SELECT filename, attributes->'objetos' FROM images ORDER BY filename").fetchall()
        count, max_id, latest = conn.execute("SELECT count(*), max(id), max(created_at) FROM images").fetchone()
    key = {"schema": 1, "count": count, "max_id": max_id,
           "latest": latest.isoformat(), "sample": args.muestra, "seed": 42,
           "target_frequency_range": [25, 4000],
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.output.exists() and not args.force:
        old = json.loads(args.output.read_text(encoding="utf-8"))
        if old.get("cache_key") == key:
            print("Evaluación imagen-imagen reutilizada sin recalcular")
            return old
    objects = {name: set(categories or []) for name, categories in rows}
    frequencies = Counter(obj for categories in objects.values() for obj in categories)
    candidates = []
    for name, categories in objects.items():
        eligible = [obj for obj in categories if 25 <= frequencies[obj] <= 4000]
        if eligible:
            target = min(eligible, key=lambda obj: (frequencies[obj], obj))
            candidates.append((name, target))
    if len(candidates) < args.muestra:
        raise RuntimeError(f"Solo hay {len(candidates)} imágenes evaluables con categorías COCO")
    rng = np.random.default_rng(42)
    chosen = [candidates[int(i)] for i in rng.choice(len(candidates), args.muestra, replace=False)]
    catalog = Catalog()
    hits = {1: 0, 5: 0, 10: 0}
    precision_sum = 0.0
    random_baseline = 0.0
    examples = []
    for index, (name, target) in enumerate(chosen, 1):
        vector = catalog.get_by_filename(name)["embedding"]
        found = catalog.search(vector, k=10, exclude=name)
        flags = [target in objects[row["filename"]] for row in found]
        for k in hits:
            hits[k] += any(flags[:k])
        precision_sum += sum(flags) / len(flags)
        random_baseline += (frequencies[target] - 1) / (count - 1)
        if len(examples) < 8:
            examples.append({"query": name, "target_object": target,
                             "top_result": found[0]["filename"], "top_result_has_object": flags[0]})
        if index % 100 == 0:
            print(f"Imágenes evaluadas: {index}/{len(chosen)}", flush=True)
    metrics = {f"ObjectHit@{k}": round(hits[k] / len(chosen) * 100, 2) for k in hits}
    metrics["ObjectPrecision@10"] = round(precision_sum / len(chosen) * 100, 2)
    metrics["random_object_prevalence_percent"] = round(random_baseline / len(chosen) * 100, 2)
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "engine": "PostgreSQL pgvector HNSW", "image_count": count,
              "query_count": len(chosen), "cache_key": key,
              "methodology": "Consultas imagen-imagen sin la propia imagen. Relevancia proxy: compartir la categoría COCO menos frecuente de la consulta (25-4000 apariciones). No es una evaluación de similitud visual juzgada por humanos.",
              "metrics": metrics, "examples": examples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp.json")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, args.output)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--muestra", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("reportes/resultados_evaluacion_imagenes.json"))
    parser.add_argument("--force", action="store_true")
    execute(parser.parse_args())
