import json
import time

import faiss
import numpy as np

TAMANOS = [10_000, 100_000, 500_000]  # agrega 1_000_000 si tienes 8 GB+ de RAM libres
N_CONSULTAS = 200
K = 10
RUIDO = 0.02
M = 32
EF_CONSTRUCCION = 100
EF_BUSQUEDA = 64

rng = np.random.default_rng(42)
base = faiss.read_index("index.faiss")
reales = base.reconstruct_n(0, base.ntotal)
D = reales.shape[1]


def sintetico(n):
    idx = rng.integers(0, len(reales), n)
    v = reales[idx] + rng.normal(0, RUIDO, (n, D)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v.astype("float32")


def latencia_ms(index, consultas):
    inicio = time.perf_counter()
    for q in consultas:
        index.search(q[None, :], K)
    return (time.perf_counter() - inicio) / len(consultas) * 1000


resultados = []
for n in TAMANOS:
    print(f"\n=== {n:,} vectores ===")
    datos = sintetico(n)
    consultas = sintetico(N_CONSULTAS)

    # Índice exacto
    flat = faiss.IndexFlatIP(D)
    t0 = time.perf_counter()
    flat.add(datos)
    build_flat = time.perf_counter() - t0
    lat_flat = latencia_ms(flat, consultas)
    _, verdad = flat.search(consultas, K)

    # Índice aproximado HNSW
    hnsw = faiss.IndexHNSWFlat(D, M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = EF_CONSTRUCCION
    t0 = time.perf_counter()
    hnsw.add(datos)
    build_hnsw = time.perf_counter() - t0
    hnsw.hnsw.efSearch = EF_BUSQUEDA
    lat_hnsw = latencia_ms(hnsw, consultas)
    _, aprox = hnsw.search(consultas, K)

    recall = float(np.mean([len(set(verdad[i]) & set(aprox[i])) / K for i in range(len(consultas))]))

    fila = {
        "n": n,
        "flat_build_s": round(build_flat, 2),
        "flat_latencia_ms": round(lat_flat, 2),
        "hnsw_build_s": round(build_hnsw, 2),
        "hnsw_latencia_ms": round(lat_hnsw, 2),
        "hnsw_recall_vs_exacto": round(recall * 100, 2),
        "memoria_vectores_mb": round(n * D * 4 / 1e6, 1),
    }
    resultados.append(fila)
    print(json.dumps(fila, indent=2))

    del flat, hnsw, datos

with open("resultados_benchmark.json", "w") as f:
    json.dump(resultados, f, indent=2)
print("\nGuardado en resultados_benchmark.json")

# Gráfica para el informe
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["n"] for r in resultados]
    plt.figure(figsize=(7, 4.5))
    plt.plot(ns, [r["flat_latencia_ms"] for r in resultados], "o-", label="Exacto (Flat)")
    plt.plot(ns, [r["hnsw_latencia_ms"] for r in resultados], "s-", label="Aproximado (HNSW)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Número de imágenes indexadas")
    plt.ylabel("Latencia por consulta (ms)")
    plt.title("Escalabilidad de la búsqueda: Flat vs HNSW")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("benchmark_latencia.png", dpi=200)
    print("Gráfica guardada en benchmark_latencia.png")
except ImportError:
    print("Instala matplotlib para generar la gráfica: pip install matplotlib")