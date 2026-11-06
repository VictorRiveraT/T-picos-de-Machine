import json

import faiss
import numpy as np

from embeddings import embed_texto

BATCH = 256
KS = (1, 5, 10)

index = faiss.read_index("index.faiss")
with open("archivos.json") as f:
    archivos = json.load(f)
pos = {nombre: i for i, nombre in enumerate(archivos)}

with open("anotaciones/captions_val2017.json") as f:
    datos = json.load(f)

captions, objetivo = [], []
for a in datos["annotations"]:
    nombre = f"{a['image_id']:012d}.jpg"
    if nombre in pos:
        captions.append(a["caption"].strip())
        objetivo.append(pos[nombre])
objetivo = np.array(objetivo)
print(f"{len(captions)} captions sobre {len(archivos)} imágenes")

vectores = []
for i in range(0, len(captions), BATCH):
    vectores.append(embed_texto(captions[i:i + BATCH]))
    print(f"Captions procesados: {min(i + BATCH, len(captions))}/{len(captions)}")
vectores = np.vstack(vectores)

_, ids = index.search(vectores, max(KS))

resultados = {}
for k in KS:
    recall = float((ids[:, :k] == objetivo[:, None]).any(axis=1).mean())
    resultados[f"Recall@{k}"] = round(recall * 100, 2)
    print(f"Recall@{k}: {recall * 100:.2f}%")

with open("resultados_evaluacion.json", "w") as f:
    json.dump(resultados, f, indent=2)
print("Guardado en resultados_evaluacion.json")