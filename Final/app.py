import io
import json
import os
import time

import faiss
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from embeddings import DEVICE, MODEL_NAME, embed_imagenes, embed_texto, traducir_a_ingles

app = FastAPI(
    title="PinSearch - Buscador Multimodal de Imágenes",
    description="Motor de búsqueda multimodal de alta escalabilidad con CLIP y FAISS (Flat & HNSW)",
    version="2.0.0",
)

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Cargar metadatos de archivos
if not os.path.exists("archivos.json"):
    raise RuntimeError("No se encontró archivos.json. Ejecuta 'python indexar.py' primero.")

with open("archivos.json", "r", encoding="utf-8") as f:
    archivos = json.load(f)

archivos_pos = {nombre: i for i, nombre in enumerate(archivos)}

# 2. Cargar índice exacto (Flat)
if not os.path.exists("index.faiss"):
    raise RuntimeError("No se encontró index.faiss. Ejecuta 'python indexar.py' primero.")

index_flat = faiss.read_index("index.faiss")

# 3. Cargar o construir índice escalable (HNSW)
if os.path.exists("index_hnsw.faiss"):
    index_hnsw = faiss.read_index("index_hnsw.faiss")
    index_hnsw.hnsw.efSearch = 64
else:
    print("Construyendo índice HNSW en memoria desde Flat...")
    vectores = index_flat.reconstruct_n(0, index_flat.ntotal)
    index_hnsw = faiss.IndexHNSWFlat(vectores.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index_hnsw.hnsw.efConstruction = 100
    index_hnsw.add(vectores)
    index_hnsw.hnsw.efSearch = 64
    faiss.write_index(index_hnsw, "index_hnsw.faiss")

# 4. Cargar descripciones/captions si existen
captions_map = {}
captions_path = os.path.join("anotaciones", "captions_val2017.json")
if os.path.exists(captions_path):
    try:
        with open(captions_path, "r", encoding="utf-8") as f:
            datos_coco = json.load(f)
        for a in datos_coco.get("annotations", []):
            nombre = f"{a['image_id']:012d}.jpg"
            if nombre not in captions_map:
                captions_map[nombre] = a.get("caption", "").strip()
    except Exception as e:
        print(f"Aviso al cargar captions: {e}")


def _buscar(vector, k: int = 6, modo: str = "hnsw", excluir_nombre: str = None):
    """Búsqueda vectorial en el índice seleccionado midiendo la latencia exacta."""
    indice = index_hnsw if modo.lower() == "hnsw" else index_flat

    # Si se pide excluir un nombre, pedimos k + 1
    k_solicitado = k + 1 if excluir_nombre else k

    t0 = time.perf_counter()
    scores, ids = indice.search(vector, k_solicitado)
    latencia_ms = (time.perf_counter() - t0) * 1000

    resultados = []
    for s, i in zip(scores[0], ids[0]):
        if i == -1:
            continue
        nombre = archivos[i]
        if excluir_nombre and nombre == excluir_nombre:
            continue
        score_val = round(float(s), 4)
        pct = round(max(0.0, min(100.0, score_val * 100)), 1)
        caption = captions_map.get(nombre, f"Fotografía COCO ({nombre})")
        resultados.append({
            "id": nombre,
            "url": f"/imagenes/{nombre}",
            "score": score_val,
            "porcentaje": pct,
            "caption": caption,
        })
        if len(resultados) >= k:
            break

    return resultados, round(latencia_ms, 2)


@app.get("/api/info")
def info_sistema():
    """Información del sistema, modelo e índices disponibles."""
    return {
        "nombre": "PinSearch - Buscador Multimodal CLIP",
        "total_imagenes": len(archivos),
        "dimension_embedding": index_flat.d,
        "dispositivo": DEVICE,
        "modelo": MODEL_NAME,
        "indices": ["hnsw", "flat"],
        "indice_por_defecto": "hnsw",
    }


@app.get("/api/buscar/texto")
def buscar_texto(
    q: str = Query(..., min_length=1, description="Texto de búsqueda en español o inglés"),
    k: int = Query(6, ge=1, le=48),
    modo: str = Query("hnsw", pattern="^(hnsw|flat)$"),
    traducir: bool = Query(True, description="Traducir automáticamente si la consulta está en español"),
):
    """Búsqueda multimodal texto a imagen (Text-to-Image)."""
    q_limpio = q.strip()
    if not q_limpio:
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacía")

    t_inicio_total = time.perf_counter()

    # Traducción automática al inglés si se solicita
    q_procesado = traducir_a_ingles(q_limpio) if traducir else q_limpio

    # Vectorización con CLIP
    vector = embed_texto([q_procesado])

    # Búsqueda en FAISS
    resultados, latencia_indice_ms = _buscar(vector, k=k, modo=modo)
    tiempo_total_ms = (time.perf_counter() - t_inicio_total) * 1000

    return {
        "query_original": q_limpio,
        "query_procesada": q_procesado,
        "traducido": q_limpio.lower() != q_procesado.lower(),
        "modo": modo,
        "k": k,
        "tiempo_indice_ms": latencia_indice_ms,
        "tiempo_total_ms": round(tiempo_total_ms, 2),
        "total_encontrados": len(resultados),
        "resultados": resultados,
    }


@app.post("/api/buscar/imagen")
def buscar_imagen(
    archivo: UploadFile = File(...),
    k: int = Query(6, ge=1, le=48),
    modo: str = Query("hnsw", pattern="^(hnsw|flat)$"),
):
    """Búsqueda por similitud visual subiendo una imagen desde la computadora (Image-to-Image)."""
    if not archivo.filename:
        raise HTTPException(status_code=400, detail="No se seleccionó ningún archivo")

    t_inicio_total = time.perf_counter()
    try:
        contenido = archivo.file.read()
        img = Image.open(io.BytesIO(contenido)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Archivo de imagen inválido: {e}")

    # Vectorización con CLIP
    vector = embed_imagenes([img])

    # Búsqueda en FAISS
    resultados, latencia_indice_ms = _buscar(vector, k=k, modo=modo)
    tiempo_total_ms = (time.perf_counter() - t_inicio_total) * 1000

    return {
        "archivo": archivo.filename,
        "modo": modo,
        "k": k,
        "tiempo_indice_ms": latencia_indice_ms,
        "tiempo_total_ms": round(tiempo_total_ms, 2),
        "total_encontrados": len(resultados),
        "resultados": resultados,
    }


@app.get("/api/buscar/similar")
def buscar_similar(
    id: str = Query(..., description="Nombre del archivo de imagen indexada, ej: 000000001268.jpg"),
    k: int = Query(6, ge=1, le=48),
    modo: str = Query("hnsw", pattern="^(hnsw|flat)$"),
):
    """Búsqueda visual instantánea a partir de una imagen ya indexada (Visual Search estilo Pinterest)."""
    if id not in archivos_pos:
        raise HTTPException(status_code=404, detail=f"La imagen {id} no se encuentra en el índice")

    t_inicio_total = time.perf_counter()
    pos = archivos_pos[id]

    # Reconstrucción instantánea del embedding precalculado
    vector = index_flat.reconstruct(pos).reshape(1, -1)

    # Búsqueda en FAISS excluyendo la propia imagen
    resultados, latencia_indice_ms = _buscar(vector, k=k, modo=modo, excluir_nombre=id)
    tiempo_total_ms = (time.perf_counter() - t_inicio_total) * 1000

    return {
        "imagen_origen": id,
        "caption_origen": captions_map.get(id, id),
        "modo": modo,
        "k": k,
        "tiempo_indice_ms": latencia_indice_ms,
        "tiempo_total_ms": round(tiempo_total_ms, 2),
        "total_encontrados": len(resultados),
        "resultados": resultados,
    }


@app.get("/")
def home():
    """Ruta principal que sirve la interfaz web."""
    return FileResponse("static/index.html")


# Servir imágenes locales y estáticos
if not os.path.exists("imagenes"):
    os.makedirs("imagenes", exist_ok=True)

if not os.path.exists("static"):
    os.makedirs("static", exist_ok=True)

app.mount("/imagenes", StaticFiles(directory="imagenes"), name="imagenes")
app.mount("/static", StaticFiles(directory="static"), name="static")
