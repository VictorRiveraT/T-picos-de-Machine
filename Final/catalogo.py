"""Catálogo persistente: FAISS guarda vectores y SQLite guarda sus metadatos."""

from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3


DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "catalogo.sqlite3"
INDEX_PATH = DATA_DIR / "index.faiss"
HNSW_PATH = DATA_DIR / "index_hnsw.faiss"
MAPPING_PATH = DATA_DIR / "archivos.json"
CAPTIONS_PATH = Path("anotaciones/captions_val2017.json")


def ensure_data_files():
    """Copia el índice de ejemplo del repositorio sólo en la primera ejecución."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for source, destination in (
        (Path("index.faiss"), INDEX_PATH),
        (Path("archivos.json"), MAPPING_PATH),
    ):
        if not destination.exists() and source.exists():
            shutil.copy2(source, destination)


def _connect(path=DB_PATH):
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def _schema(connection):
    connection.execute(
        """CREATE TABLE IF NOT EXISTS images (
            faiss_id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL UNIQUE,
            caption TEXT NOT NULL DEFAULT ''
        )"""
    )


def captions_from_coco(path=CAPTIONS_PATH):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as source:
        annotations = json.load(source).get("annotations", [])
    captions = {}
    for item in annotations:
        filename = f"{item['image_id']:012d}.jpg"
        captions.setdefault(filename, item.get("caption", "").strip())
    return captions


def ensure_catalog(expected_count, path=DB_PATH, mapping_path=MAPPING_PATH):
    """Migra una sola vez el JSON legado y comprueba la alineación con FAISS."""
    ensure_data_files()
    path = Path(path)
    catalog = Catalog(path)
    current_count = catalog.count() if path.exists() else None
    if current_count != expected_count:
        if not Path(mapping_path).exists():
            raise RuntimeError("Falta el catálogo y archivos.json. Ejecuta python indexar.py.")
        with Path(mapping_path).open(encoding="utf-8") as source:
            filenames = json.load(source)
        if len(filenames) != expected_count or len(set(filenames)) != len(filenames):
            raise RuntimeError("archivos.json no coincide con el índice FAISS. Reindexa los datos.")
        captions = captions_from_coco()
        Catalog.replace_all(
            ((i, name, captions.get(name, "")) for i, name in enumerate(filenames)), path
        )
    if catalog.count() != expected_count:
        raise RuntimeError(
            "El catálogo SQLite y el índice FAISS tienen distinto número de imágenes. "
            "Detén el servidor y ejecuta python indexar.py --rebuild."
        )
    return catalog


class Catalog:
    def __init__(self, path=DB_PATH):
        self.path = Path(path)

    def count(self):
        with closing(_connect(self.path)) as connection:
            return connection.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    def get_by_ids(self, ids):
        ids = [int(i) for i in ids if int(i) >= 0]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with closing(_connect(self.path)) as connection:
            rows = connection.execute(
                f"SELECT faiss_id, filename, caption FROM images WHERE faiss_id IN ({placeholders})",
                ids,
            ).fetchall()
        return {row["faiss_id"]: dict(row) for row in rows}

    def get_by_filename(self, filename):
        with closing(_connect(self.path)) as connection:
            row = connection.execute(
                "SELECT faiss_id, filename, caption FROM images WHERE filename = ?", (filename,)
            ).fetchone()
        return dict(row) if row else None

    def filenames(self):
        with closing(_connect(self.path)) as connection:
            return {
                row[0] for row in connection.execute("SELECT filename FROM images")
            }

    def positions(self):
        with closing(_connect(self.path)) as connection:
            return {
                row[0]: row[1]
                for row in connection.execute("SELECT filename, faiss_id FROM images")
            }

    def append(self, rows):
        with closing(_connect(self.path)) as connection:
            with connection:
                connection.executemany(
                    "INSERT INTO images(faiss_id, filename, caption) VALUES (?, ?, ?)", rows
                )

    @staticmethod
    def replace_all(rows, path=DB_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(_connect(path)) as connection:
            with connection:
                _schema(connection)
                connection.execute("DELETE FROM images")
                connection.executemany(
                    "INSERT INTO images(faiss_id, filename, caption) VALUES (?, ?, ?)", rows
                )
