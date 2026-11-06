# Buscador Multimodal de Imagenes (Texto e Imagen) - Version 3.0 (PinSearch)

Sistema de recuperacion y busqueda multimodal de imagenes (Text-to-Image, Image-to-Image y Composed Image Retrieval), con interfaz estilo Pinterest, tablero de guardados y despliegue universal en contenedores Docker, desarrollado para el curso **Topicos Avanzados de Machine Learning (2026-II)**.

---

## 1. Arquitectura y Capacidades del Sistema (v3.0)

* **Modelo Multimodal:** OpenAI CLIP (`openai/clip-vit-base-patch32`) mediante la libreria `transformers` de Hugging Face.
* **Motor de Busqueda Vectorial:** FAISS con conmutacion dinamica entre:
  * `IndexHNSWFlat` (busqueda aproximada ultrarrapida, latencia < 1 ms, alta escalabilidad).
  * `IndexFlatIP` (busqueda exacta por producto interno / similitud de coseno).
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

4. **Indexar con FAISS (genera Flat y HNSW):**
   ```bash
   python indexar.py
   ```

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
  Calcula Recall@1, Recall@5, Recall@10 y MRR@10 sobre los captions oficiales de COCO.

* **Benchmark de Escalabilidad:**
  ```bash
  python benchmark.py
  ```
  Genera la comparativa de latencias entre Flat y HNSW hasta 500,000 vectores sinteticos.
