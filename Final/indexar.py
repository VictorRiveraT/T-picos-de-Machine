"""Carga incremental de embeddings CLIP, captions y atributos en PostgreSQL."""

import argparse
import json

import faiss
from PIL import Image
from psycopg.types.json import Jsonb

from catalogo import BASE_DIR, connect, create_vector_index, init_schema


def insert_batch(conn, entries):
    with conn.cursor() as cur:
        cur.executemany("""INSERT INTO images (filename, split, caption, attributes, embedding)
            VALUES (%s, %s, %s, %s, %s) ON CONFLICT (filename) DO NOTHING""", entries)
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=32)
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
        seed_names = json.loads((BASE_DIR / "archivos.json").read_text(encoding="utf-8"))
        seed = faiss.read_index(str(BASE_DIR / "index.faiss"))
        if len(seed_names) != seed.ntotal or seed.d != 512:
            raise RuntimeError("La semilla FAISS no coincide con archivos.json")
        for start in range(0, len(seed_names), args.batch):
            entries = []
            for i in range(start, min(start + args.batch, len(seed_names))):
                name = seed_names[i]
                if name in existing:
                    continue
                meta = data[name]
                entries.append((name, meta["split"], meta["caption"], Jsonb(meta["attributes"]), seed.reconstruct(i)))
            if entries:
                insert_batch(conn, entries)
            if start % 512 == 0:
                print(f"Semilla: {min(start+args.batch,len(seed_names))}/{len(seed_names)}", flush=True)
        existing.update(seed_names)
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
    print(f"PostgreSQL listo: {count} imágenes con embeddings y atributos.")


if __name__ == "__main__":
    main()
