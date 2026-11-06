# Buscador Multimodal de Imagenes (Texto e Imagen)

Sistema de recuperacion y busqueda de imagenes basado en consultas textuales (Text-to-Image) y por similaridad de imagen (Image-to-Image), desarrollado para el curso **Topicos Avanzados de Machine Learning (2026-II)**.

---

## 1. Arquitectura y Tecnologias

* **Modelo Multimodal:** OpenAI CLIP (`openai/clip-vit-base-patch32`) mediante la libreria `transformers` de Hugging Face.
* **Motor de Busqueda Vectorial:** FAISS (`IndexFlatIP` para busqueda exacta por similitud de coseno, y comparativa con `IndexHNSWFlat` para escalabilidad).
* **Backend:** FastAPI con Uvicorn para servir los endpoints de busqueda multimodal.
* **Frontend:** HTML5, CSS y JavaScript vanilla con interfaz de dos paneles (busqueda textual y busqueda por imagen) segun los requerimientos del proyecto.
* **Dataset:** MS-COCO val2017 (5,000 imagenes) con anotaciones oficiales de captions.

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

Para correr el proyecto completo desde cero, seguir este orden:

### Paso 1: Descargar el Dataset de Imagenes
Ejecutar el script automatizado para descargar y descomprimir las 5,000 imagenes de COCO val2017 y el archivo de anotaciones `captions_val2017.json`:
```bash
python descargar_dataset.py
```
* Las imagenes se guardaran en la carpeta `imagenes/`.
* Las anotaciones se guardaran en `anotaciones/captions_val2017.json`.

### Paso 2: Generar los Embeddings e Indexar con FAISS
Extraer los vectores de caracteristicas de las imagenes usando CLIP y construir el indice FAISS:
```bash
python indexar.py
```
* Genera el indice vectorial `index.faiss`.
* Genera el mapeo de nombres de imagenes en `archivos.json`.

### Paso 3: Iniciar el Servidor Web
Levantar la aplicacion FastAPI para interactuar con la interfaz grafica:
```bash
uvicorn app:app --reload --port 8000
```
Abrir en el navegador: [http://localhost:8000](http://localhost:8000)

* **Panel 1 (Buscar por texto):** Permite ingresar una descripcion textual en ingles (por ejemplo: `a sleeping cat`, `an airplane flying in the sky`, `a person riding a bicycle`) y retorna las imagenes mas similares.
* **Panel 2 (Buscar por imagen):** Permite subir una imagen desde la computadora y buscar las imagenes visualmente mas cercanas en el dataset.

---

## 4. Evaluacion y Benchmarking

### Evaluacion de Informacion Recuperada (Recall@K)
Evalua la precision del sistema cruzando los captions reales de COCO contra la imagen correcta asociada:
```bash
python evaluar.py
```
* Metricas calculadas: Recall@1, Recall@5 y Recall@10.
* Guarda los resultados en `resultados_evaluacion.json`.

### Benchmark de Escalabilidad
Compara la busqueda exacta (`IndexFlatIP`) contra la busqueda aproximada (`IndexHNSWFlat`) sobre colecciones sinteticas escaladas hasta 500,000 vectores:
```bash
python benchmark.py
```
* Mide tiempo de construccion de indices, latencia por consulta (ms), consumo de memoria RAM y retencion de recall de HNSW frente a Flat.
* Genera el archivo `resultados_benchmark.json` y la grafica `benchmark_latencia.png`.
