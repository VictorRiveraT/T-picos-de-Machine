"""Mide carga HTTP del servicio y guarda un reporte reutilizable."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import urlopen

from catalogo import INDEX_PATH, ensure_data_files


SCHEMA_VERSION = 1
PHRASES = (
    "a cat sleeping on a sofa", "a red sports car on a road",
    "a dog playing on a beach", "an airplane in a blue sky",
    "a bicycle parked by a wall", "a pizza on a wooden table",
    "a person skiing on snow", "a cup of coffee by a window",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(sorted_values, fraction):
    return sorted_values[min(len(sorted_values) - 1, max(0, math.ceil(len(sorted_values) * fraction) - 1))]


def ejecutar(args):
    ensure_data_files()
    if args.requests < 1 or not args.concurrency or any(n < 1 for n in args.concurrency):
        raise ValueError("Solicitudes y niveles de concurrencia deben ser positivos.")
    base_url = args.url.rstrip("/")
    config = {
        "url": base_url,
        "requests_per_level": args.requests,
        "concurrency": args.concurrency,
        "mode": "hnsw",
        "translation": False,
        "query_pattern": "frases en inglés con sufijo único por ejecución para evitar la caché de CLIP",
    }
    source_hashes = {
        "app_sha256": sha256_file(Path("app.py")),
        "index_sha256": sha256_file(INDEX_PATH),
        "load_script_sha256": sha256_file(Path(__file__)),
    }
    if args.output.exists() and not args.force:
        try:
            with args.output.open(encoding="utf-8") as source:
                cached = json.load(source)
            if (
                isinstance(cached, dict)
                and cached.get("schema_version") == SCHEMA_VERSION
                and cached.get("source_hashes") == source_hashes
                and cached.get("config") == config
                and cached.get("results")
            ):
                print(f"Prueba de carga reutilizada sin generar tráfico: {args.output}")
                return cached
        except (OSError, ValueError, TypeError):
            pass

    run_nonce = time.time_ns()

    def request_one(serial):
        phrase = PHRASES[serial % len(PHRASES)] + f" variant {run_nonce}-{serial}"
        query = urlencode({"q": phrase, "modo": "hnsw", "traducir": "false", "k": 6})
        started = time.perf_counter()
        with urlopen(f"{base_url}/api/buscar/texto?{query}", timeout=30) as response:
            payload = json.load(response)
            if response.status != 200 or "resultados" not in payload:
                raise RuntimeError(f"Respuesta inesperada: HTTP {response.status}")
        return (time.perf_counter() - started) * 1000

    results = []
    for level in args.concurrency:
        latencies = []
        errors = []
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=level) as executor:
            futures = [executor.submit(request_one, level * 100000 + i) for i in range(args.requests)]
            for future in as_completed(futures):
                try:
                    latencies.append(future.result())
                except Exception as error:
                    errors.append(str(error))
        elapsed = time.perf_counter() - started
        latencies.sort()
        row = {
            "concurrency": level,
            "requests": args.requests,
            "successful": len(latencies),
            "failed": len(errors),
            "wall_seconds": round(elapsed, 3),
            "throughput_rps": round(len(latencies) / elapsed, 2),
            "p50_ms": round(percentile(latencies, 0.50), 2) if latencies else None,
            "p95_ms": round(percentile(latencies, 0.95), 2) if latencies else None,
            "errors": errors[:5],
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_nonce": run_nonce,
        "methodology": "HTTP texto-a-imagen con consultas únicas; incluye inferencia CLIP, SQLite y FAISS HNSW. Es una prueba local, no una garantía de producción.",
        "source_hashes": source_hashes,
        "config": config,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as target:
            json.dump(report, target, ensure_ascii=False, indent=2)
        os.replace(temp, args.output)
    finally:
        temp.unlink(missing_ok=True)
    print(f"Prueba de carga guardada en {args.output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--output", type=Path, default=Path("reportes/resultados_carga.json"))
    parser.add_argument("--force", action="store_true", help="Volver a medir aunque haya un reporte válido")
    ejecutar(parser.parse_args())


if __name__ == "__main__":
    main()
