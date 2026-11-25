"""Evalúa recuperación texto-imagen contra PostgreSQL usando captions COCO."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from catalogo import BASE_DIR, Catalog, connect
from model_config import MODEL_ID


def execute(args):
    metadata = json.loads((BASE_DIR / "data/coco_metadata.json").read_text(encoding="utf-8"))["records"]
    with connect() as conn:
        count, max_id = conn.execute("SELECT count(*), max(id) FROM images").fetchone()
    config = {"sample": args.muestra, "seed": 42, "count": count, "max_id": max_id, "schema": 2,
              "model_id": MODEL_ID}
    if args.output.exists() and not args.force:
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        if prior.get("config") == config:
            print("Evaluación reutilizada sin recalcular")
            return prior
    rows = [(name, item["caption"]) for name, item in metadata.items() if item["caption"]]
    rng = np.random.default_rng(42)
    selected = rng.choice(len(rows), size=min(args.muestra, len(rows)), replace=False)
    sample = [rows[i] for i in selected]
    from embeddings import embed_texto
    catalog = Catalog()
    hits = {1: 0, 5: 0, 10: 0}
    reciprocal = 0.0
    for start in range(0, len(sample), 64):
        batch = sample[start:start+64]
        vectors = embed_texto([caption for _, caption in batch])
        for (name, _), vector in zip(batch, vectors):
            found = [r["filename"] for r in catalog.search(vector, 10)]
            for k in hits:
                hits[k] += name in found[:k]
            if name in found:
                reciprocal += 1 / (found.index(name) + 1)
        print(f"Evaluadas: {min(start+64,len(sample))}/{len(sample)}", flush=True)
    metrics = {f"Recall@{k}": round(v / len(sample) * 100, 2) for k, v in hits.items()}
    metrics["MRR@10"] = round(reciprocal / len(sample), 4)
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "engine": "PostgreSQL pgvector HNSW", "image_count": count,
              "caption_count": len(sample), "config": config, "metrics": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--muestra", type=int, default=1000)
    p.add_argument("--output", type=Path, default=Path("reportes/resultados_evaluacion_pgvector.json"))
    p.add_argument("--force", action="store_true")
    execute(p.parse_args())
