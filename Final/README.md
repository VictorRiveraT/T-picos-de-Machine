# PinSearch v4: búsqueda multimodal con PostgreSQL

PinSearch permite buscar imágenes COCO 2017 por texto, imagen o una composición de imagen y texto (CIR). CLIP genera embeddings de 512 dimensiones. PostgreSQL 17 con pgvector guarda los embeddings, captions y atributos agrupados, y calcula la similitud de coseno. El índice HNSW acelera las consultas; el modo `flat` ejecuta una búsqueda exacta en la misma base.

## Dataset y persistencia

El conjunto local consta de **10 000 imágenes reales**: 5 000 de `val2017` y 5 000 de `train2017`. Las categorías de objetos provienen de anotaciones oficiales COCO. Los grupos `familias`, `colores`, `acciones` y `escenas` se derivan con reglas de categorías y captions; no son etiquetas humanas verificadas para cada imagen. Se guardan en `images.attributes` (JSONB) y se devuelven en el API. `/api/buscar/texto` permite filtrar con `familia=animales`, `vehiculos`, `personas`, `comida`, `muebles` o `electronica`.

La tabla `images` conserva `filename`, `split`, `caption`, `attributes` y `embedding vector(512)`. La búsqueda usa `embedding <=> query` en PostgreSQL y transforma distancia coseno a similitud `1 - distancia`. FAISS se utiliza solamente como fuente de migración para los primeros 5 000 vectores; las consultas del servidor no usan FAISS ni SQLite. Las siguientes imágenes se procesan por lotes y se insertan de forma incremental. Volver a correr `indexar.py` no recalcula imágenes ya guardadas.

Para ampliar COCO a otro tamaño (por ejemplo 20 000), ejecuta `python ampliar_dataset.py --target 20000`, `python preparar_metadata.py` y `python indexar.py`. Los embeddings anteriores permanecen en PostgreSQL y el índice HNSW se mantiene con las inserciones. Vuelve a ejecutar los benchmarks para medir el nuevo tamaño; los resultados de 10 000 no predicen automáticamente la latencia a cualquier escala.

El traductor español-inglés usa MarianMT en el mismo proceso, con modelo `Helsinki-NLP/opus-mt-es-en` almacenado en caché. La detección combina un vocabulario breve para consultas cortas con `langdetect`. Una consulta inglesa pasa directo a CLIP.

## Preparación local

Desde esta carpeta `Final`:

```powershell
python -m venv ..\.venv
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
docker compose up -d db
..\.venv\Scripts\python.exe descargar_dataset.py
..\.venv\Scripts\python.exe ampliar_dataset.py --target 10000
..\.venv\Scripts\python.exe preparar_metadata.py
..\.venv\Scripts\python.exe indexar.py
..\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Abre [http://localhost:8000](http://localhost:8000). La base está publicada solo en `127.0.0.1:5433`; para cambiar la contraseña, establece `POSTGRES_PASSWORD` en Compose y `DATABASE_URL` en el proceso Python. El volumen `postgres_data` conserva la base entre reinicios. Para usar la aplicación en Docker, después de completar la indexación ejecuta `docker compose up -d --build pinsearch`. El contenedor necesita acceso a los modelos CLIP y MarianMT la primera vez.

## API y medidas

- `GET /api/info`: cantidad de imágenes, modelo y motor de búsqueda.
- `GET /api/buscar/texto?q=un gato durmiendo&modo=hnsw&familia=animales`: consulta textual y filtro opcional.
- `POST /api/buscar/imagen`: foto cargada por el usuario.
- `GET /api/buscar/similar?id=...`: usa el embedding persistido de una foto.
- `POST /api/buscar/compuesta`: combinación ponderada de foto y texto.
- `modo=flat` sirve como referencia exacta; `modo=hnsw` consulta el índice aproximado.

```powershell
..\.venv\Scripts\python.exe benchmark.py --queries 50
..\.venv\Scripts\python.exe evaluar.py --muestra 1000
..\.venv\Scripts\python.exe carga.py --requests 24 --concurrency 1 4 8
```

Los JSON de `reportes/*_pgvector.json` y `resultados_carga.json` incluyen la configuración y se reutilizan mientras el estado de la base y los parámetros no cambien. Usa `--force` para repetir. Los reportes FAISS anteriores documentan v3 y **no miden el motor PostgreSQL actual**.
