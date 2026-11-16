import io
import os
import time
from functools import lru_cache
from typing import Optional

import faiss
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Asegurar que el directorio de trabajo sea la carpeta Final para evitar problemas
# con caracteres especiales (como acentos en 'Tópicos') en las librerías C++ de FAISS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

from embeddings import DEVICE, MODEL_NAME, embed_imagenes, embed_texto, traducir_a_ingles
from catalogo import DATA_DIR, HNSW_PATH, INDEX_PATH, ensure_catalog, ensure_data_files

app = FastAPI(
    title="PinSearch - Buscador Multimodal de Imágenes",
    description="Motor de búsqueda multimodal de alta escalabilidad con CLIP, FAISS y Composed Image Retrieval (CIR)",
    version="3.0.0",
)

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Cargar índice exacto (Flat)
ensure_data_files()
if not INDEX_PATH.exists():
    raise RuntimeError("No se encontró index.faiss. Ejecuta 'python indexar.py' primero.")

index_flat = faiss.read_index(str(INDEX_PATH))
catalogo = ensure_catalog(index_flat.ntotal)

# 2. Cargar el índice escalable persistente; sólo se construye la primera vez.
DATA_DIR.mkdir(parents=True, exist_ok=True)
if HNSW_PATH.exists():
    index_hnsw = faiss.read_index(str(HNSW_PATH))
elif os.path.exists("index_hnsw.faiss"):
    index_hnsw = faiss.read_index("index_hnsw.faiss")
    faiss.write_index(index_hnsw, str(HNSW_PATH))
else:
    index_hnsw = None

if index_hnsw is None or index_hnsw.ntotal != index_flat.ntotal:
    print("Construyendo índice HNSW persistente desde Flat...")
    vectores = index_flat.reconstruct_n(0, index_flat.ntotal)
    index_hnsw = faiss.IndexHNSWFlat(vectores.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index_hnsw.hnsw.efConstruction = 100
    index_hnsw.add(vectores)
    faiss.write_index(index_hnsw, str(HNSW_PATH))
index_hnsw.hnsw.efSearch = 64


@lru_cache(maxsize=2048)
def _vector_texto(texto):
    """Evita repetir la inferencia CLIP para consultas textuales frecuentes."""
    return embed_texto([texto])


def _buscar(vector, k: int = 6, modo: str = "hnsw", excluir_nombre: str = None, umbral_min: float = 0.0):
    """Búsqueda vectorial en el índice seleccionado midiendo la latencia exacta y filtrando por umbral."""
    indice = index_hnsw if modo.lower() == "hnsw" else index_flat

    # Solicitamos un margen mayor si hay filtrado por umbral o exclusión
    k_solicitado = min(indice.ntotal, max(k * 3 if umbral_min > 0 else k + 1, k))

    t0 = time.perf_counter()
    scores, ids = indice.search(vector, k_solicitado)
    latencia_ms = (time.perf_counter() - t0) * 1000
    metadatos = catalogo.get_by_ids(ids[0])

    resultados = []
    for s, i in zip(scores[0], ids[0]):
        if i == -1:
            continue
        item = metadatos.get(int(i))
        if item is None:
            raise RuntimeError("El índice y el catálogo no coinciden. Ejecuta python indexar.py --rebuild.")
        nombre = item["filename"]
        if excluir_nombre and nombre == excluir_nombre:
            continue
        score_val = round(float(s), 4)

        # Filtro de umbral mínimo de similitud de coseno
        if umbral_min > 0 and score_val < umbral_min:
            continue

        pct = round(max(0.0, min(100.0, score_val * 100)), 1)
        caption = item["caption"] or f"Fotografía COCO ({nombre})"
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
        "nombre": "PinSearch - Buscador Multimodal CLIP con CIR y Docker",
        "total_imagenes": catalogo.count(),
        "dimension_embedding": index_flat.d,
        "dispositivo": DEVICE,
        "modelo": MODEL_NAME,
        "indices": ["hnsw", "flat"],
        "indice_por_defecto": "hnsw",
        "cir_soportado": True,
    }


@app.get("/api/buscar/texto")
def buscar_texto(
    q: str = Query(..., min_length=1, description="Texto de búsqueda en español o inglés"),
    k: int = Query(6, ge=1, le=48),
    modo: str = Query("hnsw", pattern="^(hnsw|flat)$"),
    traducir: bool = Query(True, description="Traducir automáticamente si la consulta está en español"),
    umbral: float = Query(0.0, ge=0.0, le=1.0, description="Umbral mínimo de similitud (0.0 a 1.0)"),
):
    """Búsqueda multimodal texto a imagen (Text-to-Image)."""
    q_limpio = q.strip()
    if not q_limpio:
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacía")

    t_inicio_total = time.perf_counter()

    # Traducción automática al inglés si se solicita
    q_procesado = traducir_a_ingles(q_limpio) if traducir else q_limpio

    # Vectorización con CLIP
    vector = _vector_texto(q_procesado)

    # Búsqueda en FAISS
    resultados, latencia_indice_ms = _buscar(vector, k=k, modo=modo, umbral_min=umbral)
    tiempo_total_ms = (time.perf_counter() - t_inicio_total) * 1000

    return {
        "query_original": q_limpio,
        "query_procesada": q_procesado,
        "traducido": q_limpio.lower() != q_procesado.lower(),
        "modo": modo,
        "k": k,
        "umbral": umbral,
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
    umbral: float = Query(0.0, ge=0.0, le=1.0),
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
    resultados, latencia_indice_ms = _buscar(vector, k=k, modo=modo, umbral_min=umbral)
    tiempo_total_ms = (time.perf_counter() - t_inicio_total) * 1000

    return {
        "archivo": archivo.filename,
        "modo": modo,
        "k": k,
        "umbral": umbral,
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
    umbral: float = Query(0.0, ge=0.0, le=1.0),
):
    """Búsqueda visual instantánea a partir de una imagen ya indexada (Visual Search estilo Pinterest)."""
    origen = catalogo.get_by_filename(id)
    if origen is None:
        raise HTTPException(status_code=404, detail=f"La imagen {id} no se encuentra en el índice")

    t_inicio_total = time.perf_counter()
    pos = origen["faiss_id"]

    # Reconstrucción instantánea del embedding precalculado
    vector = index_flat.reconstruct(pos).reshape(1, -1)

    # Búsqueda en FAISS excluyendo la propia imagen
    resultados, latencia_indice_ms = _buscar(vector, k=k, modo=modo, excluir_nombre=id, umbral_min=umbral)
    tiempo_total_ms = (time.perf_counter() - t_inicio_total) * 1000

    return {
        "imagen_origen": id,
        "caption_origen": origen["caption"] or id,
        "modo": modo,
        "k": k,
        "umbral": umbral,
        "tiempo_indice_ms": latencia_indice_ms,
        "tiempo_total_ms": round(tiempo_total_ms, 2),
        "total_encontrados": len(resultados),
        "resultados": resultados,
    }


@app.post("/api/buscar/compuesta")
def buscar_compuesta(
    archivo: Optional[UploadFile] = File(None),
    id_imagen: Optional[str] = Form(None),
    texto_modificador: str = Form(..., min_length=1, description="Texto modificador que complementa la imagen"),
    peso_imagen: float = Form(0.6, ge=0.05, le=0.95, description="Peso de la imagen (alpha) vs texto (1 - alpha)"),
    k: int = Form(6, ge=1, le=48),
    modo: str = Form("hnsw"),
    umbral: float = Form(0.0, ge=0.0, le=1.0),
):
    """Búsqueda Compuesta Multimodal (Composed Image Retrieval - CIR).
    Combina el vector visual de una imagen con el vector semántico de un texto modificador
    en el espacio latente de CLIP: v_compuesto = normalize(alpha * v_img + (1 - alpha) * v_text)."""
    t_inicio_total = time.perf_counter()

    # 1. Obtener vector de imagen (desde id existente o archivo subido)
    imagen_catalogo = catalogo.get_by_filename(id_imagen) if id_imagen else None
    if imagen_catalogo is not None:
        pos = imagen_catalogo["faiss_id"]
        v_img = index_flat.reconstruct(pos).reshape(1, -1)
        origen_desc = f"Pin {id_imagen}"
        excluir = id_imagen
    elif archivo and archivo.filename:
        try:
            contenido = archivo.file.read()
            img = Image.open(io.BytesIO(contenido)).convert("RGB")
            v_img = embed_imagenes([img])
            origen_desc = archivo.filename
            excluir = None
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Imagen inválida: {e}")
    else:
        raise HTTPException(
            status_code=400,
            detail="Debes proporcionar una imagen indexada ('id_imagen') o subir un archivo ('archivo')."
        )

    # 2. Traducir texto modificador si está en español
    mod_limpio = texto_modificador.strip()
    mod_procesado = traducir_a_ingles(mod_limpio)

    # 3. Vector de texto modificador con CLIP
    v_text = _vector_texto(mod_procesado)

    # 4. Composición de vectores en el espacio latente de CLIP
    alpha = float(peso_imagen)
    v_comp = alpha * v_img + (1.0 - alpha) * v_text
    norma = np.linalg.norm(v_comp, axis=1, keepdims=True)
    v_comp = (v_comp / np.maximum(norma, 1e-12)).astype("float32")

    # 5. Búsqueda en FAISS
    resultados, latencia_indice_ms = _buscar(v_comp, k=k, modo=modo, excluir_nombre=excluir, umbral_min=umbral)
    tiempo_total_ms = (time.perf_counter() - t_inicio_total) * 1000

    return {
        "tipo": "composed_image_retrieval",
        "imagen_origen": origen_desc,
        "modificador_original": mod_limpio,
        "modificador_procesado": mod_procesado,
        "traducido": mod_limpio.lower() != mod_procesado.lower(),
        "peso_imagen": alpha,
        "peso_texto": round(1.0 - alpha, 2),
        "modo": modo,
        "k": k,
        "umbral": umbral,
        "tiempo_indice_ms": latencia_indice_ms,
        "tiempo_total_ms": round(tiempo_total_ms, 2),
        "total_encontrados": len(resultados),
        "resultados": resultados,
    }


@app.get("/")
def home():
    """Ruta principal que sirve la interfaz web."""
    return FileResponse("static/index.html")


@app.get("/v1")
def home_v1():
    """Ruta alternativa para comparar con la primera versión (Mockup original del PDF)."""
    return FileResponse("static/v1.html")


# Servir imágenes locales y estáticos
if not os.path.exists("imagenes"):
    os.makedirs("imagenes", exist_ok=True)

if not os.path.exists("static"):
    os.makedirs("static", exist_ok=True)

app.mount("/imagenes", StaticFiles(directory="imagenes"), name="imagenes")
app.mount("/static", StaticFiles(directory="static"), name="static")
