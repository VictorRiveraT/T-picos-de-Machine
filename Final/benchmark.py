"""Benchmark reproducible de pgvector HNSW frente a búsqueda exacta."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

from catalogo import Catalog, connect


def snapshot(conn):
    return conn.execute("SELECT count(*), max(id), max(created_at) FROM images").fetchone()


def execute(args):
    with connect() as conn:
        count, max_id, latest = snapshot(conn)
        if count == 0:
            raise RuntimeError("PostgreSQL no contiene imágenes")
        names = [r[0] for r in conn.execute("SELECT filename FROM images ORDER BY filename LIMIT %s", (args.queries,))]
    key = {"count": count, "max_id": max_id, "latest": latest.isoformat(), "queries": len(names), "k": args.k, "schema": 1}
    if args.output.exists() and not args.force:
        old = json.loads(args.output.read_text(encoding="utf-8"))
        if old.get("cache_key") == key:
            print("Benchmark reutilizado sin recalcular")
            return old
    catalog = Catalog()
    times = {"flat": [], "hnsw": []}
    recalls = []
    for name in names:
        vec = catalog.get_by_filename(name)["embedding"]
        found = {}
        for mode in ("flat", "hnsw"):
            t0 = time.perf_counter()
            found[mode] = catalog.search(vec, args.k, mode)
            times[mode].append((time.perf_counter() - t0) * 1000)
        truth = {r["filename"] for r in found["flat"]}
        recalls.append(len(truth & {r["filename"] for r in found["hnsw"]}) / len(truth))
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "engine": "PostgreSQL 17 + pgvector; cosine distance; HNSW vs exact sequential scan",
              "methodology": "Consultas con embeddings reales almacenados; latencia SQL y red local, sin CLIP ni HTTP",
              "cache_key": key, "image_count": count, "query_count": len(names), "k": args.k,
              "flat_p50_ms": round(statistics.median(times["flat"]), 3),
              "hnsw_p50_ms": round(statistics.median(times["hnsw"]), 3),
              "hnsw_recall_at_k": round(statistics.mean(recalls), 4)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", type=int, default=50)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--output", type=Path, default=Path("reportes/resultados_benchmark_pgvector.json"))
    p.add_argument("--force", action="store_true")
    execute(p.parse_args())
