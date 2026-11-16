"""Compara Flat y HNSW y reutiliza resultados si el índice y la configuración no cambian."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import faiss
import numpy as np

from catalogo import INDEX_PATH, ensure_data_files

SCHEMA_VERSION = 2


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sintetico(rng, reales, n, ruido):
    sample = rng.integers(0, len(reales), n)
    vectors = reales[sample] + rng.normal(0, ruido, (n, reales.shape[1])).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors.astype("float32")


def latencia_ms(index, consultas, k):
    inicio = time.perf_counter()
    for query in consultas:
        index.search(query[None, :], k)
    return (time.perf_counter() - inicio) / len(consultas) * 1000


def graficar(resultados, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib no está instalado; se conservan los resultados JSON.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    ns = [row["n"] for row in resultados]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ns, [row["flat_latencia_ms"] for row in resultados], "o-", label="Exacto (Flat)")
    ax.plot(ns, [row["hnsw_latencia_ms"] for row in resultados], "s-", label="Aproximado (HNSW)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Número de imágenes indexadas")
    ax.set_ylabel("Latencia por consulta (ms)")
    ax.set_title("Escalabilidad de la búsqueda: Flat vs HNSW")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(temp, dpi=200, format="png")
    plt.close(fig)
    os.replace(temp, path)


def ejecutar(args):
    ensure_data_files()
    if not INDEX_PATH.exists():
        raise FileNotFoundError("Falta index.faiss. Ejecuta python indexar.py primero.")
    if args.queries < 1 or args.k < 1 or not args.sizes or any(n < args.k for n in args.sizes):
        raise ValueError("--queries y --k deben ser positivos; cada tamaño debe ser al menos k.")
    config = {
        "sizes": args.sizes,
        "queries": args.queries,
        "k": args.k,
        "noise": 0.02,
        "m": 32,
        "ef_construction": 100,
        "ef_search": 64,
        "seed": 42,
    }
    fingerprint = sha256_file(INDEX_PATH)
    if args.output.exists() and not args.force:
        try:
            with args.output.open(encoding="utf-8") as source:
                cached = json.load(source)
            if (
                isinstance(cached, dict)
                and cached.get("schema_version") == SCHEMA_VERSION
                and cached.get("source_index_sha256") == fingerprint
                and cached.get("config") == config
                and cached.get("results")
            ):
                if not args.plot.exists():
                    graficar(cached["results"], args.plot)
                print(f"Benchmark reutilizado sin recalcular: {args.output}")
                return cached
        except (OSError, ValueError, TypeError):
            pass

    rng = np.random.default_rng(config["seed"])
    base = faiss.read_index(str(INDEX_PATH))
    if not base.ntotal:
        raise ValueError("El índice base está vacío.")
    reales = base.reconstruct_n(0, base.ntotal)
    dimension = reales.shape[1]
    resultados = []
    for n in args.sizes:
        print(f"\n=== {n:,} vectores ===", flush=True)
        datos = sintetico(rng, reales, n, config["noise"])
        consultas = sintetico(rng, reales, args.queries, config["noise"])

        flat = faiss.IndexFlatIP(dimension)
        inicio = time.perf_counter()
        flat.add(datos)
        build_flat = time.perf_counter() - inicio
        lat_flat = latencia_ms(flat, consultas, args.k)
        _, verdad = flat.search(consultas, args.k)

        hnsw = faiss.IndexHNSWFlat(dimension, config["m"], faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = config["ef_construction"]
        inicio = time.perf_counter()
        hnsw.add(datos)
        build_hnsw = time.perf_counter() - inicio
        hnsw.hnsw.efSearch = config["ef_search"]
        lat_hnsw = latencia_ms(hnsw, consultas, args.k)
        _, aproximado = hnsw.search(consultas, args.k)
        recall = float(np.mean([
            len(set(verdad[i]) & set(aproximado[i])) / args.k
            for i in range(len(consultas))
        ]))
        row = {
            "n": n,
            "flat_build_s": round(build_flat, 2),
            "flat_latencia_ms": round(lat_flat, 2),
            "hnsw_build_s": round(build_hnsw, 2),
            "hnsw_latencia_ms": round(lat_hnsw, 2),
            "hnsw_recall_vs_exacto": round(recall * 100, 2),
            "memoria_vectores_mb": round(n * dimension * 4 / 1e6, 1),
        }
        resultados.append(row)
        print(json.dumps(row, indent=2), flush=True)
        del flat, hnsw, datos

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_index_sha256": fingerprint,
        "methodology": "Vectores sintéticos derivados de embeddings reales; latencia del índice, sin inferencia ni HTTP.",
        "config": config,
        "results": resultados,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as target:
            json.dump(report, target, ensure_ascii=False, indent=2)
        os.replace(temp, args.output)
    finally:
        temp.unlink(missing_ok=True)
    graficar(resultados, args.plot)
    print(f"Benchmark guardado en {args.output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[10_000, 100_000, 500_000])
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("reportes/resultados_benchmark.json"))
    parser.add_argument("--plot", type=Path, default=Path("reportes/benchmark_latencia.png"))
    parser.add_argument("--force", action="store_true", help="Recalcular aunque haya un resultado válido")
    ejecutar(parser.parse_args())


if __name__ == "__main__":
    main()
