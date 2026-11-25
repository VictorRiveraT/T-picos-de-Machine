"""Catálogo persistente y búsqueda vectorial en PostgreSQL + pgvector."""

import os
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from model_config import DATABASE_NAME, DB_PORT, EMBEDDING_DIM, MODEL_ID

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATABASE_URL = os.getenv("DATABASE_URL", f"postgresql://pinsearch:pinsearch_dev@127.0.0.1:{DB_PORT}/{DATABASE_NAME}")


def connect():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def init_schema():
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if conn.execute("SELECT to_regclass('images') IS NOT NULL AND to_regclass('model_info') IS NULL").fetchone()[0]:
            raise RuntimeError("Esta base contiene embeddings sin identidad de modelo. Usa una base vacía exclusiva para esta rama.")
        conn.execute("""CREATE TABLE IF NOT EXISTS model_info (
            singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
            model_id TEXT NOT NULL,
            dimension INTEGER NOT NULL
        )""")
        row = conn.execute("SELECT model_id, dimension FROM model_info WHERE singleton").fetchone()
        if row and row != (MODEL_ID, EMBEDDING_DIM):
            raise RuntimeError(f"Base de otro modelo: {row}; se esperaba {(MODEL_ID, EMBEDDING_DIM)}")
        if not row:
            conn.execute("INSERT INTO model_info (singleton, model_id, dimension) VALUES (TRUE, %s, %s)",
                         (MODEL_ID, EMBEDDING_DIM))
        conn.execute(f"""CREATE TABLE IF NOT EXISTS images (
            id BIGSERIAL PRIMARY KEY,
            filename TEXT NOT NULL UNIQUE,
            split TEXT NOT NULL,
            caption TEXT NOT NULL DEFAULT '',
            attributes JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS images_attributes_gin ON images USING gin (attributes)")


def create_vector_index():
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute("CREATE INDEX IF NOT EXISTS images_embedding_hnsw ON images USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 100)")
        conn.execute("ANALYZE images")


class Catalog:
    def __init__(self):
        self.pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=10,
                                   configure=register_vector, open=True)
        self.verify_model()

    def verify_model(self):
        with self.pool.connection() as conn:
            if conn.execute("SELECT to_regclass('model_info')").fetchone()[0] is None:
                if conn.execute("SELECT to_regclass('images')").fetchone()[0] is not None:
                    raise RuntimeError("Esta base tiene embeddings sin identidad de modelo; usa la base exclusiva de esta rama")
                raise RuntimeError("Base no inicializada para este modelo; ejecuta python indexar.py")
            row = conn.execute("SELECT model_id, dimension FROM model_info WHERE singleton").fetchone()
            if row != (MODEL_ID, EMBEDDING_DIM):
                raise RuntimeError(f"Base de otro modelo: {row}; se esperaba {(MODEL_ID, EMBEDDING_DIM)}")

    def count(self):
        with self.pool.connection() as conn:
            return conn.execute("SELECT count(*) FROM images").fetchone()[0]

    def get_by_filename(self, filename):
        with self.pool.connection() as conn:
            row = conn.execute("SELECT filename, caption, attributes, embedding FROM images WHERE filename = %s", (filename,)).fetchone()
        return dict(filename=row[0], caption=row[1], attributes=row[2],
                    embedding=row[3].to_numpy()) if row else None

    def search(self, vector, k=6, mode="hnsw", exclude=None, threshold=0.0, family=None):
        if mode not in ("hnsw", "flat"):
            raise ValueError("Modo de búsqueda inválido")
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        if query.shape != (EMBEDDING_DIM,):
            raise ValueError(f"El embedding debe tener {EMBEDDING_DIM} dimensiones")
        filters = []
        params = []
        if exclude:
            filters.append("filename <> %s")
            params.append(exclude)
        if family:
            filters.append("attributes @> %s")
            params.append(Jsonb({"familias": [family]}))
        where = "WHERE " + " AND ".join(filters) if filters else ""
        sql = f"SELECT filename, caption, attributes, 1 - (embedding <=> %s) AS score FROM images {where} ORDER BY embedding <=> %s LIMIT %s"
        with self.pool.connection() as conn:
            with conn.transaction():
                if mode == "flat":
                    conn.execute("SET LOCAL enable_indexscan = off")
                else:
                    conn.execute("SET LOCAL enable_seqscan = off")
                    conn.execute("SET LOCAL hnsw.ef_search = 100")
                    if family:
                        conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")
                rows = conn.execute(sql, (query, *params, query, max(k * 4, 100) if threshold > 0 else k)).fetchall()
        return [dict(filename=r[0], caption=r[1], attributes=r[2], score=float(r[3])) for r in rows if r[3] >= threshold][:k]


def ensure_catalog():
    catalog = Catalog()
    if catalog.count() == 0:
        raise RuntimeError("PostgreSQL está vacío. Ejecuta python indexar.py antes de iniciar el servidor.")
    return catalog
