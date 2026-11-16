"""Añade imágenes nuevas a FAISS y SQLite sin recalcular las ya indexadas."""

import argparse
import json
import os
from pathlib import Path

import faiss
from PIL import Image

from catalogo import (
    Catalog, DATA_DIR, DB_PATH, HNSW_PATH, INDEX_PATH, MAPPING_PATH,
    captions_from_coco, ensure_catalog, ensure_data_files,
)


EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp"}
FLAT_PATH = INDEX_PATH
CARPETA = Path("imagenes")


def construir_hnsw(flat):
    hnsw = faiss.IndexHNSWFlat(flat.d, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 100
    if flat.ntotal:
        hnsw.add(flat.reconstruct_n(0, flat.ntotal))
    hnsw.hnsw.efSearch = 64
    return hnsw


def cargar_existentes(rebuild):
    if rebuild:
        return None, None, [], None, False
    if not FLAT_PATH.exists():
        if DB_PATH.exists():
            raise RuntimeError("Falta el índice, pero existe el catálogo. Usa --rebuild.")
        return None, None, [], None, False
    if not MAPPING_PATH.exists():
        raise RuntimeError("Falta archivos.json; usa --rebuild para recuperar el índice.")
    flat = faiss.read_index(str(FLAT_PATH))
    with MAPPING_PATH.open(encoding="utf-8") as source:
        filenames = json.load(source)
    if len(filenames) != flat.ntotal:
        raise RuntimeError("archivos.json y FAISS no coinciden; usa --rebuild.")
    catalogo = ensure_catalog(flat.ntotal)
    hnsw_dirty = False
    if HNSW_PATH.exists():
        hnsw = faiss.read_index(str(HNSW_PATH))
        if hnsw.ntotal != flat.ntotal:
            hnsw = construir_hnsw(flat)
            hnsw_dirty = True
    elif Path("index_hnsw.faiss").exists():
        hnsw = faiss.read_index("index_hnsw.faiss")
        if hnsw.ntotal != flat.ntotal:
            hnsw = construir_hnsw(flat)
        hnsw_dirty = True
    else:
        hnsw = construir_hnsw(flat)
        hnsw_dirty = True
    return flat, hnsw, filenames, catalogo, hnsw_dirty


def guardar(flat, hnsw, filenames, rows, rebuild):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    flat_tmp = DATA_DIR / "index.faiss.pending"
    hnsw_tmp = DATA_DIR / "index_hnsw.faiss.pending"
    mapping_tmp = DATA_DIR / "archivos.json.pending"
    try:
        faiss.write_index(flat, str(flat_tmp))
        faiss.write_index(hnsw, str(hnsw_tmp))
        with mapping_tmp.open("w", encoding="utf-8") as target:
            json.dump(filenames, target)
        # La base de datos se actualiza al final; el arranque valida los conteos.
        os.replace(flat_tmp, FLAT_PATH)
        os.replace(hnsw_tmp, HNSW_PATH)
        os.replace(mapping_tmp, MAPPING_PATH)
        if rebuild or not DB_PATH.exists():
            Catalog.replace_all(rows)
        else:
            Catalog().append(rows)
    finally:
        for temp in (flat_tmp, hnsw_tmp, mapping_tmp):
            temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--rebuild", action="store_true", help="Recalcular todo desde cero")
    parser.add_argument("--verify-files", action="store_true", help="Comprobar todos los archivos ya indexados")
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch debe ser positivo")
    if not CARPETA.is_dir():
        parser.error(f"No existe la carpeta {CARPETA}")

    ensure_data_files()
    flat, hnsw, filenames, catalogo, hnsw_dirty = cargar_existentes(args.rebuild)
    if args.verify_files and filenames:
        missing = [name for name in filenames if not (CARPETA / name).is_file()]
        if missing:
            raise RuntimeError(
                f"Faltan {len(missing)} imágenes ya indexadas (por ejemplo {missing[0]}). "
                "Restaura los archivos o usa --rebuild para cambiar de dataset."
            )
    conocidos = set(filenames)
    candidatos = sorted(
        (entry for entry in CARPETA.iterdir()
         if entry.is_file() and entry.suffix.lower() in EXTENSIONES and entry.name not in conocidos),
        key=lambda entry: entry.name,
    )
    if not candidatos:
        if flat is None:
            raise RuntimeError("No hay imágenes para indexar.")
        if hnsw_dirty:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            faiss.write_index(hnsw, str(HNSW_PATH))
        print(f"Sin imágenes nuevas. Se reutilizan {flat.ntotal} vectores e índices existentes.")
        return

    # CLIP se carga sólo cuando hay imágenes nuevas, no en cada ejecución.
    from embeddings import embed_imagenes

    captions = captions_from_coco()
    nuevos = []
    rows = []
    for start in range(0, len(candidatos), args.batch):
        lote = []
        nombres = []
        for path in candidatos[start:start + args.batch]:
            try:
                with Image.open(path) as source:
                    lote.append(source.convert("RGB"))
                nombres.append(path.name)
            except Exception as error:
                print(f"Saltando {path.name}: {error}")
        if not lote:
            continue
        vectores = embed_imagenes(lote)
        if flat is None:
            flat = faiss.IndexFlatIP(vectores.shape[1])
            hnsw = construir_hnsw(flat)
        if vectores.shape[1] != flat.d:
            raise RuntimeError("La dimensión del modelo no coincide con el índice actual.")
        base_id = flat.ntotal
        flat.add(vectores)
        hnsw.add(vectores)
        nuevos.extend(nombres)
        rows.extend(
            (base_id + offset, name, captions.get(name, ""))
            for offset, name in enumerate(nombres)
        )
        print(f"Nuevas imágenes procesadas: {len(nuevos)}/{len(candidatos)}")

    if not nuevos:
        print("No se encontraron imágenes válidas nuevas.")
        return
    if not args.rebuild and catalogo and catalogo.count() != len(filenames):
        raise RuntimeError("El catálogo cambió durante la indexación. Reintenta con el servidor detenido.")
    filenames.extend(nuevos)
    guardar(flat, hnsw, filenames, rows, args.rebuild)
    print(f"Índices y SQLite guardados: {len(filenames)} imágenes; {len(nuevos)} añadidas.")
    print("Reinicia el servidor para cargar el nuevo índice.")


if __name__ == "__main__":
    main()
