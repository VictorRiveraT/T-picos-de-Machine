# Proyecto Final: Buscador Multimodal de Imágenes (Texto e Imagen)

Sistema de recuperación y búsqueda de imágenes basado en consultas textuales (Text-to-Image) y por similaridad de imagen (Image-to-Image), desarrollado para el curso **Tópicos Avanzados de Machine Learning (2026-II)**.

---

## 🚀 Arquitectura y Tecnologías

* **Modelo Multimodal:** OpenAI CLIP (`openai/clip-vit-base-patch32`) vía Hugging Face `transformers`.
* **Motor de Búsqueda Vectorial:** FAISS (`IndexFlatIP` para búsqueda exacta de similitud de coseno; comparado contra `IndexHNSWFlat` para escalabilidad).
* **Backend:** FastAPI + Uvicorn.
* **Frontend:** HTML5 / CSS3 / JavaScript interactivo con pestañas de búsqueda textual y por imagen según especificaciones de la rúbrica.
* **Dataset:** MS-COCO `val2017` (5,000 imágenes) con anotaciones oficiales de captions.

---

## 📦 Instalación y Configuración

1. **Crear y activar entorno virtual:**
   ```bash
   python -m venv .venv
   # En Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # En Linux / macOS:
   source .venv/bin/activate
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Descargar imágenes del dataset:**
   El repositorio ya incluye el índice FAISS (`index.faiss`), la lista de archivos (`archivos.json`) y las anotaciones (`anotaciones/captions_val2017.json`). Para descargar las 5,000 imágenes locales para visualización:
   ```bash
   python descargar_dataset.py
   ```

4. **(Opcional) Re-indexar imágenes:**
   Si se agregan nuevas imágenes o se desea reconstruir el índice:
   ```bash
   python indexar.py
   ```

---

## 🖥️ Ejecución de la Aplicación Web

Para iniciar el servidor de búsqueda:
```bash
uvicorn app:app --reload --port 8000
```
Luego abrir en el navegador: [http://localhost:8000](http://localhost:8000)

---

## 📊 Evaluación y Benchmarking

* **Evaluación de Recuperación (Information Retrieval):**
  Calcula $Recall@1$, $Recall@5$ y $Recall@10$ sobre los captions reales de COCO val2017.
  ```bash
  python evaluar.py
  ```
  *Resultados obtenidos:*
  * **Recall@1:** 30.45%
  * **Recall@5:** 54.79%
  * **Recall@10:** 66.24%

* **Benchmark de Escalabilidad:**
  Evalúa latencia por consulta, tiempo de indexación y retención de Recall comparando búsqueda exacta (`Flat`) vs aproximada (`HNSW`) sobre hasta 500,000 vectores sintéticos.
  ```bash
  python benchmark.py
  ```
  Genera el gráfico comparativo en `benchmark_latencia.png` y las métricas en `resultados_benchmark.json`.
