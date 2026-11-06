# Buscador Multimodal de Imagenes (Texto e Imagen) - Version 2.0 (PinSearch)

Sistema de recuperacion y busqueda de imagenes basado en consultas textuales (Text-to-Image) y por similaridad de imagen (Image-to-Image), con interfaz estilo Pinterest y soporte multilingue, desarrollado para el curso **Topicos Avanzados de Machine Learning (2026-II)**.

---

## 1. Arquitectura y Nuevas Caracteristicas (v2.0)

* **Modelo Multimodal:** OpenAI CLIP (`openai/clip-vit-base-patch32`) mediante la libreria `transformers` de Hugging Face.
* **Motor de Busqueda Vectorial:** FAISS con conmutacion dinamica entre:
  * `IndexHNSWFlat` (busqueda aproximada ultrarrapida, latencia < 1 ms, alta escalabilidad).
  * `IndexFlatIP` (busqueda exacta por producto interno / similitud de coseno).
* **Traduccion Automatica al Vuelo:** Deteccion y traduccion de consultas en espanol hacia ingles antes de la inferencia con CLIP para maximizar la precision semantica.
* **Busqueda Visual Instantanea (Visual Search):** Extraccion directa de vectores precalculados al presionar "Similares" en cualquier imagen, con tiempos de respuesta de 0.1 ms a 0.5 ms.
* **Frontend Estilo Pinterest:**
  * Diseno Masonry responsivo en columnas.
  * Boton de busqueda visual integrada en la barra principal.
  * Chips de categorias y ejemplos listos para probar con un clic.
  * Vista previa inmediata de imagenes seleccionadas/arrastradas (drag & drop).
  * Modal detallado (Lightbox) con informacion del embedding, porcentaje de similitud y ranking.
  * Efectos de hover, skeleton loading animado y visualizacion de latencias.
* **Backend Asincrono Optimizado:** Endpoints desacoplados en FastAPI para evitar bloqueo de hilos de ejecucion (concurrencia segura).

---

## 2. Requisitos Previos e Instalacion

Se recomienda utilizar Python 3.10 o superior (compatible con 3.11 y 3.12).

1. **Crear y activar el entorno virtual:**
   * En Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   * En Linux o macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

2. **Instalar dependencias del proyecto:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. Flujo de Ejecucion Paso a Paso

### Paso 1: Descargar el Dataset de Imagenes
Ejecutar el script automatizado para descargar y descomprimir las 5,000 imagenes de COCO val2017 y el archivo de anotaciones `captions_val2017.json`:
```bash
python descargar_dataset.py
```
* Las imagenes se guardan en la carpeta `imagenes/`.
* Las anotaciones se guardan en `anotaciones/captions_val2017.json`.

### Paso 2: Generar los Indices FAISS (Flat y HNSW)
Extraer los vectores de caracteristicas de las imagenes usando CLIP y construir ambos indices vectoriales:
```bash
python indexar.py
```
* Genera `index.faiss` (indice exacto Flat).
* Genera `index_hnsw.faiss` (indice escalable HNSW).
* Genera `archivos.json` (mapeo ordenado de nombres de archivo).

### Paso 3: Iniciar el Servidor Web (PinSearch)
Levantar la aplicacion FastAPI:
```bash
uvicorn app:app --reload --port 8000
```
Abrir en el navegador: [http://localhost:8000](http://localhost:8000)

* **Busqueda por Texto:** Escribe en espanol o ingles (ej: `un gato durmiendo`, `sports car on road`, `pizza artesanal`).
* **Busqueda por Imagen:** Sube o arrastra una foto para ver la vista previa y buscar imagenes similares.
* **Visual Search en Resultados:** En cada tarjeta o modal, pulsa "Similares" para encontrar imagenes con caracteristicas visuales afines de manera instantanea.
* **Selector de Indice:** Permite alternar entre `HNSW` y `Flat` para comprobar la diferencia en tiempo de respuesta.

---

## 4. Evaluacion y Benchmarking

### Evaluacion de Informacion Recuperada (Recall@K y MRR)
Evalua la precision de recuperacion cruzando las 25,000 frases de COCO contra la imagen objetivo:
```bash
python evaluar.py
```
Para una evaluacion rapida sobre una muestra (por ejemplo, 1,000 captions):
```bash
python evaluar.py --muestra 1000
```
* Metricas: Recall@1, Recall@5, Recall@10 y MRR@10 (Mean Reciprocal Rank).
* Los resultados se guardan localmente en `resultados_evaluacion.json`.

### Benchmark de Escalabilidad
Compara la busqueda exacta frente a la aproximada sobre colecciones sinteticas escaladas hasta 500,000 vectores:
```bash
python benchmark.py
```
* Mide tiempo de construccion de indices, latencia por consulta (ms), consumo de memoria RAM y retencion de recall de HNSW frente a Flat.
* Genera `resultados_benchmark.json` y la grafica `benchmark_latencia.png`.
