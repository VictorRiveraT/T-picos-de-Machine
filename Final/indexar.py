"""Calcula todos los embeddings de esta rama sin reutilizar vectores de otro modelo."""

import argparse
from datetime import datetime, timezone
import json
import time

from PIL import Image
from psycopg.types.json import Jsonb

from catalogo import BASE_DIR, connect, create_vector_index, init_schema
from model_config import EMBEDDING_DIM, MODEL_ID


def insert_batch(conn, entries):
    with conn.cursor() as cur:
        cur.executemany("""INSERT INTO images (filename, split, caption, attributes, embedding)
            VALUES (%s, %s, %s, %s, %s) ON CONFLICT (filename) DO NOTHING""", entries)
    conn.commit()


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch debe ser positivo")
    data = json.loads((BASE_DIR / "data/coco_metadata.json").read_text(encoding="utf-8"))["records"]
    image_dir = BASE_DIR / "imagenes"
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            data.setdefault(path.name, {"split": "custom", "caption": "", "attributes": {
                "objetos": [], "familias": [], "colores": [], "acciones": [], "escenas": []}})
    missing = [name for name in data if not (image_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Faltan {len(missing)} imágenes; primera: {missing[0]}")
    init_schema()
    with connect() as conn:
        existing = {r[0] for r in conn.execute("SELECT filename FROM images")}
        pending = [name for name in data if name not in existing]
        if pending:
            from embeddings import embed_imagenes
            for start in range(0, len(pending), args.batch):
                names = pending[start:start + args.batch]
                images = []
                for name in names:
                    with Image.open(image_dir / name) as source:
                        images.append(source.convert("RGB"))
                vectors = embed_imagenes(images)
                entries = [(name, data[name]["split"], data[name]["caption"],
                            Jsonb(data[name]["attributes"]), vector)
                           for name, vector in zip(names, vectors)]
                insert_batch(conn, entries)
                print(f"Nuevos embeddings: {min(start+args.batch,len(pending))}/{len(pending)}", flush=True)
        count = conn.execute("SELECT count(*) FROM images").fetchone()[0]
    create_vector_index()
    elapsed = time.perf_counter() - started
    report = {
        "model_id": MODEL_ID,
        "embedding_dim": EMBEDDING_DIM,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "images_total": count,
        "images_added_this_run": len(pending),
        "elapsed_seconds_including_hnsw": round(elapsed, 2),
        "images_per_second_this_run": round(len(pending) / elapsed, 2),
    }
    output = BASE_DIR / "reportes/resultados_indexacion_modelo.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PostgreSQL listo: {count} imágenes con embeddings y atributos; {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
