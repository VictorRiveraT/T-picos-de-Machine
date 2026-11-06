import io
import json

import faiss
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from embeddings import embed_imagenes, embed_texto

app = FastAPI(title="Buscador multimodal de imágenes")

index = faiss.read_index("index.faiss")
with open("archivos.json") as f:
    archivos = json.load(f)


def buscar(vector, k):
    scores, ids = index.search(vector, k)
    return [
        {"url": f"/imagenes/{archivos[i]}", "score": round(float(s), 4)}
        for s, i in zip(scores[0], ids[0])
        if i != -1
    ]


@app.get("/api/buscar/texto")
def buscar_texto(q: str, k: int = 6):
    return buscar(embed_texto([q]), k)


@app.post("/api/buscar/imagen")
async def buscar_imagen(archivo: UploadFile = File(...), k: int = 6):
    img = Image.open(io.BytesIO(await archivo.read())).convert("RGB")
    return buscar(embed_imagenes([img]), k)


@app.get("/")
def home():
    return FileResponse("static/index.html")


app.mount("/imagenes", StaticFiles(directory="imagenes"), name="imagenes")