"""Mide crecimiento real de pgvector con subconjuntos de embeddings COCO."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from catalogo import connect
from model_config import MODEL_ID


SCHEMA_VERSION = 4
INDEX_SETTINGS = {"m": 32, "ef_construction": 100, "ef_search": 100}


def percentile(values, q):
    return round(float(np.percentile(values, q)), 3)


def plan(conn, vector, mode, k):
    with conn.transaction():
        conn.execute("SET LOCAL enable_seqscan = off" if mode == "hnsw" else "SET LOCAL enable_indexscan = off")
        explanation = "\n".join(row[0] for row in conn.execute(
            "EXPLAIN SELECT id FROM bench_images ORDER BY embedding <=> %s LIMIT %s", (vector, k)))
    required = "Index Scan using bench_images_hnsw" if mode == "hnsw" else "Seq Scan on bench_images"
    if required not in explanation:
        raise RuntimeError(f"Plan SQL inesperado para {mode}: {explanation[:250]}")
    return required


def search(conn, vector, mode, k):
    started = time.perf_counter()
    with conn.transaction():
        if mode == "hnsw":
            conn.execute("SET LOCAL enable_seqscan = off")
            conn.execute("SET LOCAL hnsw.ef_search = 100")
        else:
            conn.execute("SET LOCAL enable_indexscan = off")
        ids = [row[0] for row in conn.execute(
            "SELECT id FROM bench_images ORDER BY embedding <=> %s LIMIT %s", (vector, k))]
    return ids, (time.perf_counter() - started) * 1000


def graph(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    sizes = [r["images"] for r in rows]
    for mode, label, color in (("hnsw", "HNSW", "#d81b60"), ("exact", "Exacta", "#2b6cb0")):
        ax.plot(sizes, [r[f"{mode}_p50_ms"] for r in rows], "o-", color=color, label=f"{label} p50")
        ax.plot(sizes, [r[f"{mode}_p95_ms"] for r in rows], "o--", color=color, alpha=0.65, label=f"{label} p95")
    ax.set_xlabel("Imágenes COCO indexadas")
    ax.set_ylabel("Latencia SQL local (ms)")
    ax.set_title("Escalabilidad PostgreSQL + pgvector")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.png")
    fig.savefig(temp, dpi=180)
    plt.close(fig)
    os.replace(temp, path)


def execute(args):
    if (args.queries < 1 or args.k < 1 or args.noise < 0 or not args.sizes
            or sorted(set(args.sizes)) != args.sizes or args.queries > min(args.sizes)):
        raise ValueError("Usa tamaños ordenados y únicos; k y consultas positivos, consultas <= tamaño menor y ruido >= 0")
    with connect() as conn:
        count, max_id, latest = conn.execute("SELECT count(*), max(id), max(created_at) FROM images").fetchone()
    if max(args.sizes) > count or min(args.sizes) < args.k:
        raise ValueError(f"Los tamaños deben estar entre k={args.k} y las {count} imágenes presentes")
    key = {"schema": SCHEMA_VERSION, "model_id": MODEL_ID, "count": count, "max_id": max_id,
           "latest": latest.isoformat(), "sizes": args.sizes, "queries": args.queries,
           "k": args.k, "noise": args.noise, "seed": 42,
           "index_settings": INDEX_SETTINGS,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.output.exists() and not args.force:
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        if prior.get("cache_key") == key:
            if not args.plot.exists():
                graph(prior["results"], args.plot)
            print("Benchmark de escalabilidad reutilizado sin recalcular")
            return prior

    rng = np.random.default_rng(42)
    results = []
    with connect() as conn:
        conn.autocommit = True
        for size in args.sizes:
            conn.execute("DROP TABLE IF EXISTS bench_images")
            started = time.perf_counter()
            conn.execute("CREATE TEMP TABLE bench_images AS SELECT id, embedding FROM images ORDER BY id LIMIT %s", (size,))
            conn.execute("CREATE INDEX bench_images_hnsw ON bench_images USING hnsw (embedding vector_cosine_ops) WITH (m=32, ef_construction=100)")
            conn.execute("ANALYZE bench_images")
            build_s = round(time.perf_counter() - started, 3)
            table_bytes, index_bytes = conn.execute(
                "SELECT pg_table_size('bench_images'), pg_relation_size('bench_images_hnsw')").fetchone()
            vectors = [row[0].to_numpy() for row in conn.execute(
                "SELECT embedding FROM bench_images ORDER BY md5(id::text) LIMIT %s", (args.queries,))]
            queries = []
            for vector in vectors:
                noisy = vector + rng.normal(0, args.noise, vector.shape).astype("float32")
                queries.append(noisy / np.linalg.norm(noisy))
            plans = {mode: plan(conn, queries[0], mode, args.k) for mode in ("hnsw", "exact")}
            for mode in ("hnsw", "exact"):
                for vector in queries[:min(5, len(queries))]:
                    search(conn, vector, mode, args.k)
            times = {"hnsw": [], "exact": []}
            overlaps = []
            for vector in queries:
                exact, exact_ms = search(conn, vector, "exact", args.k)
                approx, hnsw_ms = search(conn, vector, "hnsw", args.k)
                times["exact"].append(exact_ms)
                times["hnsw"].append(hnsw_ms)
                overlaps.append(len(set(exact) & set(approx)) / len(exact))
            row = {"images": size, "queries": len(queries), "build_s": build_s,
                   "table_mb": round(table_bytes / 1_000_000, 2),
                   "hnsw_index_mb": round(index_bytes / 1_000_000, 2),
                   "exact_p50_ms": percentile(times["exact"], 50),
                   "exact_p95_ms": percentile(times["exact"], 95),
                   "hnsw_p50_ms": percentile(times["hnsw"], 50),
                   "hnsw_p95_ms": percentile(times["hnsw"], 95),
                   "hnsw_recall_at_k": round(float(np.mean(overlaps)), 4),
                   "plans": plans}
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        conn.execute("DROP TABLE IF EXISTS bench_images")

    report = {"schema_version": SCHEMA_VERSION,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "engine": "PostgreSQL + pgvector HNSW vs exact sequential scan",
              "methodology": "Subconjuntos temporales de embeddings COCO reales, consultas con ruido gaussiano normalizado; tiempos incluyen SQL y red local, excluyen CLIP y HTTP. Calentamiento: 5 consultas por modo.",
              "cache_key": key, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp.json")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, args.output)
    graph(results, args.plot)
    print(f"Guardados {args.output} y {args.plot}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1_000, 5_000, 10_000, 20_000])
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=Path("reportes/resultados_escalabilidad_pgvector.json"))
    parser.add_argument("--plot", type=Path, default=Path("reportes/escalabilidad_pgvector.png"))
    parser.add_argument("--force", action="store_true")
    execute(parser.parse_args())
