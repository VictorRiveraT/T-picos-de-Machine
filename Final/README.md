# Buscador Multimodal de Imagenes (Texto e Imagen) - Version 3.0 (PinSearch)

Sistema de recuperacion y busqueda multimodal de imagenes (Text-to-Image, Image-to-Image y Composed Image Retrieval), con interfaz estilo Pinterest, tablero de guardados y despliegue universal en contenedores Docker, desarrollado para el curso **Topicos Avanzados de Machine Learning (2026-II)**.

---

## 1. Arquitectura y Capacidades del Sistema (v3.0)

* **Modelo Multimodal:** OpenAI CLIP (`openai/clip-vit-base-patch32`) mediante la libreria `transformers` de Hugging Face.
* **Motor de Busqueda Vectorial:** FAISS con conmutacion dinamica entre:
  * `IndexHNSWFlat` (busqueda aproximada; la latencia debe medirse en cada entorno).
  * `IndexFlatIP` (busqueda exacta por producto interno / similitud de coseno).
* **Catalogo SQLite:** `data/catalogo.sqlite3` guarda el ID de FAISS, nombre y caption de cada imagen. El servidor consulta SQLite por lote y no vuelve a leer las anotaciones COCO en cada arranque.
* **Indexacion incremental:** `indexar.py` procesa solo los archivos nuevos de `imagenes/` y los agrega a los dos indices. Los indices y el catalogo se conservan en `data/`; la copia incluida en el repositorio se usa solo para inicializar el primer arranque.
* **Busqueda Compuesta Multimodal (Composed Image Retrieval - CIR):**
  * Permite combinar una imagen base con un texto modificador en el espacio latente de CLIP.
  * Formula de composicion ponderada:
    $$v_{comp} = \text{norm}(\alpha \cdot v_{img} + (1 - \alpha) \cdot v_{text})$$
  * El usuario puede controlar el balance visual vs semantico mediante el factor alpha.
* **Filtro de Umbral de Confianza:**
  * Permite descartar imagenes cuya similitud este por debajo del umbral establecido (0% a 35%), evitando falsos positivos irrelevantes.
* **Traduccion Automatica al Vuelo:** Deteccion y traduccion de consultas en espanol hacia ingles antes de la inferencia con CLIP para maximizar la precision semantica.
* **Busqueda Visual Instantanea (Visual Search):** Extraccion directa de vectores precalculados al presionar "Similares" en cualquier pin, con tiempos de respuesta inferiores a 0.5 ms.
* **Tablero de Pines Guardados y Búsquedas Recientes:**
  * Boton de guardado interactivo por tarjeta.
  * Tablero privado persistente en el navegador mediante `localStorage`.
  * Menu desplegable con historial de busquedas recientes.
* **Contenedorizacion Docker:**
  * `Dockerfile` optimizado sobre `python:3.11-slim` con soporte para OpenMP y FAISS en Linux.
  * `docker-compose.yml` para despliegue reproducible en un solo comando.

---

## 2. Opciones de Ejecucion

### Opcion A: Ejecucion con Docker (Recomendada para despliegue rapido)

Requisitos: Tener instalado Docker y Docker Compose.

1. **Construir e iniciar el contenedor:**
   ```bash
   docker compose up --build
   ```

2. **Abrir la aplicacion:**
   Navegar en el navegador a: [http://localhost:8000](http://localhost:8000)

3. **Detener el servicio:**
   ```bash
   docker compose down
   ```

---

### Opcion B: Ejecucion Local con Python

Requisitos: Python 3.10 o superior (compatible con 3.11 y 3.12).

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

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Descargar imagenes del dataset (si no estan descargadas):**
   ```bash
   python descargar_dataset.py
   ```

4. **Agregar nuevas imagenes a los indices Flat, HNSW y SQLite:**
   ```bash
   python indexar.py
   ```
   Si no hay imagenes nuevas, no carga CLIP ni recalcula embeddings. Para reconstruir todo deliberadamente: `python indexar.py --rebuild`. Reinicia el servidor despues de indexar para que cargue los nuevos indices.

5. **Iniciar el servidor web FastAPI:**
   ```bash
   uvicorn app:app --reload --port 8000
   ```
   Abrir en el navegador: [http://localhost:8000](http://localhost:8000)

---

## 3. Modalidades de Busqueda Disponibles

1. **Busqueda por Texto (Text-to-Image):**
   Ingresa descripciones en espanol o ingles. Ejemplo: `un gato durmiendo`, `sports car on road`, `pizza artesanal`.

2. **Busqueda por Imagen (Image-to-Image):**
   Arrastra o selecciona una foto desde tu computadora para ver su vista previa y encontrar imagenes con atributos visuales afines.

3. **Busqueda Compuesta (Composed Image Retrieval - CIR):**
   Combina una foto con una instruccion textual (por ejemplo, subir un gato y escribir `en la nieve` o `de noche`). El sistema fusiona ambos vectores en el espacio de CLIP respetando el balance de peso seleccionado.

4. **Visual Search Directo:**
   Pulsa "Similares" en cualquier pin para navegar por imagenes analogas con tiempo de respuesta en submilisegundos.

5. **Pines Guardados:**
   Haz clic en "Guardar" para coleccionar tus imagenes favoritas en tu tablero personal.

---

## 4. Evaluacion y Benchmarking

* **Evaluacion de Recuperacion (Information Retrieval):**
  ```bash
  python evaluar.py --muestra 1000
  ```
  Calcula Recall@1, Recall@5, Recall@10 y MRR@10 sobre los captions oficiales de COCO. `reportes/resultados_evaluacion.json` guarda la muestra, fecha y SHA-256 del indice, mapeo y captions. Si no cambiaron, la siguiente ejecucion reutiliza las metricas sin cargar CLIP. Usa `--force` para recalcular.

* **Benchmark de Escalabilidad:**
  ```bash
  python benchmark.py
  ```
  Genera la comparativa de latencias entre Flat y HNSW hasta 500,000 vectores sinteticos. Guarda `reportes/resultados_benchmark.json` con fecha UTC, configuracion y SHA-256 del indice de origen, ademas de `reportes/benchmark_latencia.png`. Si la configuracion y el indice no cambiaron, reutiliza el JSON sin recalcular. Usa `python benchmark.py --force` para una medicion nueva. Para una prueba rapida: `python benchmark.py --sizes 10000 --queries 20 --output data/resultados_rapido.json --plot data/benchmark_rapido.png`.

* **Prueba de carga del servicio HTTP:** con el servidor activo, ejecuta `python carga.py`. Guarda `reportes/resultados_carga.json` con throughput, p50 y p95 para 1, 4 y 8 solicitudes simultaneas. Reutiliza el reporte si el codigo, indice y parametros no cambiaron; usa `python carga.py --force` para volver a medir en otro momento.

## 5. Persistencia y crecimiento

* Los archivos `data/index.faiss`, `data/index_hnsw.faiss`, `data/archivos.json` y `data/catalogo.sqlite3` forman el estado local de busqueda. La carpeta `data/` esta excluida de Git y se monta como volumen persistente en Docker.
* Para ampliar el conjunto, copia las imagenes nuevas a `imagenes/`, ejecuta `python indexar.py` con el servidor detenido y vuelve a iniciarlo. Los vectores anteriores se reutilizan. No sustituyas ni reordenes manualmente `data/archivos.json`: sus posiciones deben coincidir con FAISS y SQLite.
* En Docker, usa `docker compose stop pinsearch`, `docker compose run --rm pinsearch python indexar.py` y `docker compose up -d pinsearch`. La cache de CLIP tambien persiste dentro de `data/` para no descargar el modelo en cada contenedor nuevo.
* `resultados_benchmark.json` mide la busqueda del indice con vectores sinteticos derivados de COCO. No incluye inferencia CLIP, HTTP ni concurrencia; esos tiempos deben medirse por separado al evaluar el servicio completo.
