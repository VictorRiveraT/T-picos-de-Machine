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
print(f"{len(archivos)} imágenes encontradas")

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
index = faiss.IndexFlatIP(matriz.shape[1])  # producto interno = coseno (vectores normalizados)
index.add(matriz)

faiss.write_index(index, "index.faiss")
with open("archivos.json", "w") as f:
    json.dump(validos, f)
print("Índice guardado: index.faiss + archivos.json")