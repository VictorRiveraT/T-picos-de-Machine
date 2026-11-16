"""Evalúa recuperación texto-imagen y reutiliza métricas de entradas sin cambios."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import faiss
import numpy as np

from catalogo import CAPTIONS_PATH, INDEX_PATH, MAPPING_PATH, ensure_catalog, ensure_data_files


SCHEMA_VERSION = 2
KS = (1, 5, 10)
BATCH = 256


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ejecutar(args):
    ensure_data_files()
    index_path = Path(args.indice)
    for path in (index_path, MAPPING_PATH, CAPTIONS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Falta {path}. Descarga el dataset o ejecuta python indexar.py.")
    if args.muestra is not None and args.muestra < 1:
        raise ValueError("--muestra debe ser un número positivo.")

    config = {
        "sample_size": args.muestra,
        "seed": 42,
        "k_values": list(KS),
        "batch": BATCH,
        "model": "openai/clip-vit-base-patch32",
    }
    source_hashes = {
        "index_sha256": sha256_file(index_path),
        "mapping_sha256": sha256_file(MAPPING_PATH),
        "captions_sha256": sha256_file(CAPTIONS_PATH),
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
                and cached.get("metrics")
            ):
                print(f"Evaluación reutilizada sin recalcular CLIP: {args.output}")
                return cached
        except (OSError, ValueError, TypeError):
            pass

    # La carga del modelo se evita por completo cuando el resultado guardado es válido.
    from embeddings import embed_texto

    index = faiss.read_index(str(index_path))
    catalogo = ensure_catalog(index.ntotal)
    positions = catalogo.positions()
    with CAPTIONS_PATH.open(encoding="utf-8") as source:
        annotations = json.load(source)["annotations"]

    captions = []
    targets = []
    for annotation in annotations:
        name = f"{annotation['image_id']:012d}.jpg"
        if name in positions:
            captions.append(annotation["caption"].strip())
            targets.append(positions[name])
    if not captions:
        raise RuntimeError("No hay captions que correspondan a las imágenes indexadas.")
    if args.muestra and args.muestra < len(captions):
        rng = np.random.default_rng(config["seed"])
        selected = rng.choice(len(captions), size=args.muestra, replace=False)
        captions = [captions[i] for i in selected]
        targets = [targets[i] for i in selected]
    print(f"Evaluando {len(captions)} captions sobre {catalogo.count()} imágenes", flush=True)

    vectors = []
    for start in range(0, len(captions), BATCH):
        vectors.append(embed_texto(captions[start:start + BATCH]))
        print(f"Captions procesados: {min(start + BATCH, len(captions))}/{len(captions)}", flush=True)
    _, ids = index.search(np.vstack(vectors), max(KS))
    objective = np.asarray(targets)
    metrics = {}
    for k in KS:
        recall = float((ids[:, :k] == objective[:, None]).any(axis=1).mean())
        metrics[f"Recall@{k}"] = round(recall * 100, 2)
        print(f"Recall@{k}: {metrics[f'Recall@{k}']:.2f}%", flush=True)
    hits = ids == objective[:, None]
    ranks = np.argmax(hits, axis=1) + 1
    reciprocal = np.where(hits.any(axis=1), 1.0 / ranks, 0.0)
    metrics["MRR@10"] = round(float(np.mean(reciprocal)), 4)
    print(f"MRR@10: {metrics['MRR@10']:.4f}", flush=True)

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": source_hashes,
        "config": config,
        "image_count": catalogo.count(),
        "caption_count": len(captions),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as target:
            json.dump(report, target, ensure_ascii=False, indent=2)
        os.replace(temp, args.output)
    finally:
        temp.unlink(missing_ok=True)
    print(f"Evaluación guardada en {args.output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--muestra", type=int, default=None)
    parser.add_argument("--indice", type=Path, default=INDEX_PATH)
    parser.add_argument("--output", type=Path, default=Path("reportes/resultados_evaluacion.json"))
    parser.add_argument("--force", action="store_true", help="Recalcular aunque haya un resultado válido")
    ejecutar(parser.parse_args())


if __name__ == "__main__":
    main()
