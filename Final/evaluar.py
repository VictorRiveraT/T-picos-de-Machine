import argparse
import json
import os

import faiss
import numpy as np

from embeddings import embed_texto

parser = argparse.ArgumentParser(description="Evaluación de recuperación Information Retrieval (Text-to-Image)")
parser.add_argument("--muestra", type=int, default=None, help="Evaluar sobre una muestra de N captions para rapidez")
parser.add_argument("--indice", type=str, default="index.faiss", help="Archivo del índice a evaluar")
args = parser.parse_args()

BATCH = 256
KS = (1, 5, 10)

if not os.path.exists(args.indice):
    raise FileNotFoundError(f"No se encontró el índice {args.indice}. Ejecuta python indexar.py primero.")

index = faiss.read_index(args.indice)
with open("archivos.json", "r", encoding="utf-8") as f:
    archivos = json.load(f)
pos = {nombre: i for i, nombre in enumerate(archivos)}

captions_file = os.path.join("anotaciones", "captions_val2017.json")
if not os.path.exists(captions_file):
    raise FileNotFoundError(f"No se encontró {captions_file}. Ejecuta python descargar_dataset.py primero.")

with open(captions_file, "r", encoding="utf-8") as f:
    datos = json.load(f)

captions, objetivo = [], []
for a in datos["annotations"]:
    nombre = f"{a['image_id']:012d}.jpg"
    if nombre in pos:
        captions.append(a["caption"].strip())
        objetivo.append(pos[nombre])

if args.muestra and args.muestra < len(captions):
    rng = np.random.default_rng(42)
    indices = rng.choice(len(captions), size=args.muestra, replace=False)
    captions = [captions[i] for i in indices]
    objetivo = [objetivo[i] for i in indices]
    print(f"Evaluando sobre subconjunto de prueba: {len(captions)} captions seleccionados")
else:
    print(f"Evaluando sobre el conjunto completo: {len(captions)} captions sobre {len(archivos)} imágenes")

objetivo = np.array(objetivo)

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

# MRR (Mean Reciprocal Rank) sobre el top 10
aciertos = (ids == objetivo[:, None])
ranks = np.argmax(aciertos, axis=1) + 1
rr = np.where(aciertos.any(axis=1), 1.0 / ranks, 0.0)
mrr = float(np.mean(rr))
resultados["MRR@10"] = round(mrr, 4)
print(f"MRR@10: {mrr:.4f}")

with open("resultados_evaluacion.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2)
print("Resultados guardados en resultados_evaluacion.json")
