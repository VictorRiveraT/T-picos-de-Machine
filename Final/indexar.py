import json
import os

import faiss
import numpy as np
from PIL import Image

from embeddings import embed_imagenes

CARPETA = "imagenes"
BATCH = 64
EXTENSIONES = (".jpg", ".jpeg", ".png", ".webp")

archivos = sorted(f for f in os.listdir(CARPETA) if f.lower().endswith(EXTENSIONES))
print(f"{len(archivos)} imágenes encontradas en {CARPETA}/")

vectores, validos = [], []
for i in range(0, len(archivos), BATCH):
    lote, nombres = [], []
    for nombre in archivos[i:i + BATCH]:
        try:
            lote.append(Image.open(os.path.join(CARPETA, nombre)).convert("RGB"))
            nombres.append(nombre)
        except Exception as e:
            print(f"Saltando {nombre}: {e}")
    if lote:
        vectores.append(embed_imagenes(lote))
        validos.extend(nombres)
    print(f"Procesadas {min(i + BATCH, len(archivos))}/{len(archivos)}")

matriz = np.vstack(vectores)
dimension = matriz.shape[1]

# 1) Indice Exacto (Flat)
print("Construyendo índice exacto IndexFlatIP...")
index_flat = faiss.IndexFlatIP(dimension)
index_flat.add(matriz)
faiss.write_index(index_flat, "index.faiss")

# 2) Indice Aproximado de Alta Escalabilidad (HNSW)
print("Construyendo índice aproximado IndexHNSWFlat...")
index_hnsw = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
index_hnsw.hnsw.efConstruction = 100
index_hnsw.add(matriz)
index_hnsw.hnsw.efSearch = 64
faiss.write_index(index_hnsw, "index_hnsw.faiss")

with open("archivos.json", "w", encoding="utf-8") as f:
    json.dump(validos, f, indent=2)

print("Índices guardados con éxito:")
print("  - index.faiss (Búsqueda exacta Flat)")
print("  - index_hnsw.faiss (Búsqueda escalable HNSW)")
print("  - archivos.json (Mapeo de nombres)")
