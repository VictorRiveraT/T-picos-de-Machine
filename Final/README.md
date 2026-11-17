# PinSearch v4: búsqueda multimodal con PostgreSQL

PinSearch busca imágenes COCO 2017 por texto, por una imagen cargada o por una composición de imagen y texto (CIR). CLIP genera embeddings de 512 dimensiones. PostgreSQL con pgvector guarda los embeddings, captions y atributos agrupados, y calcula similitud de coseno. HNSW acelera consultas; `modo=flat` ejecuta una búsqueda exacta en la misma base.

## Datos y arquitectura

El conjunto local contiene **20 000 imágenes reales**: 5 000 de `val2017` y 15 000 de `train2017`. Las categorías de objetos provienen de anotaciones COCO. Los grupos `familias`, `colores`, `acciones` y `escenas` se derivan por reglas de categorías y captions; no son etiquetas humanas verificadas para cada foto. Se guardan en `images.attributes` (JSONB), se devuelven en el API y se pueden filtrar en `/api/buscar/texto` con `familia=animales`, `vehiculos`, `personas`, `comida`, `muebles` o `electronica`.

La tabla `images` guarda `filename`, `split`, `caption`, `attributes` y `embedding vector(512)`. La búsqueda SQL ordena por `embedding <=> consulta` y convierte distancia coseno en puntuación `1 - distancia`. FAISS se usa únicamente como fuente para migrar los primeros 5 000 vectores ya calculados; el servidor no consulta FAISS ni SQLite. `indexar.py` procesa solo imágenes aún ausentes de PostgreSQL y mantiene el índice HNSW al crecer el conjunto.

El traductor español-inglés usa MarianMT local (`Helsinki-NLP/opus-mt-es-en`) con caché. Para consultas cortas, la detección combina un vocabulario español con `langdetect`; una consulta inglesa pasa directamente a CLIP.

## Instalación desde un clon

Desde la carpeta `Final`, con Python 3.11 o 3.12 y Docker:

```powershell
python -m venv ..\.venv
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
docker compose up -d db
..\.venv\Scripts\python.exe descargar_dataset.py
..\.venv\Scripts\python.exe ampliar_dataset.py --target 20000
..\.venv\Scripts\python.exe preparar_metadata.py
..\.venv\Scripts\python.exe indexar.py
docker compose up -d --build pinsearch
..\.venv\Scripts\python.exe verificar_despliegue.py --expected-count 20000
```

Abre [http://localhost:8000](http://localhost:8000). `descargar_dataset.py` aporta las primeras 5 000 imágenes y las anotaciones; `ampliar_dataset.py` descarga 15 000 de `train2017`. Las descargas y la indexación se pueden reanudar sin recalcular embeddings ya guardados. Los archivos de imágenes y la base son grandes y se conservan localmente fuera de Git. El volumen Docker `postgres_data` mantiene PostgreSQL entre reinicios; `./imagenes` y `./data` se montan en el contenedor. La primera ejecución descarga los modelos CLIP y MarianMT a la caché de `data/`.

Para ejecutar la API desde el entorno Python en vez del contenedor, inicia solo `docker compose up -d db` y luego `..\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000`. La base está publicada en `127.0.0.1:5433`. Para cambiar la contraseña de desarrollo, establece `POSTGRES_PASSWORD` en Compose y `DATABASE_URL` en Python.

## API

- `GET /api/info`: cantidad de imágenes, modelo y motor de búsqueda.
- `GET /api/buscar/texto?q=un gato durmiendo&modo=hnsw&familia=animales`: búsqueda textual y filtro opcional.
- `POST /api/buscar/imagen`: búsqueda por foto cargada.
- `GET /api/buscar/similar?id=...`: foto ya indexada, usando su vector persistido.
- `POST /api/buscar/compuesta`: combinación ponderada de foto y texto.
- `modo=flat`: referencia exacta; `modo=hnsw`: índice aproximado.

## Evaluación y escalabilidad

```powershell
..\.venv\Scripts\python.exe benchmark.py --sizes 1000 5000 10000 20000 --queries 100
..\.venv\Scripts\python.exe evaluar.py --muestra 1000
..\.venv\Scripts\python.exe evaluar_imagenes.py --muestra 500
..\.venv\Scripts\python.exe carga.py --requests 100 --concurrency 1 4 8 16
```

El benchmark crea tablas temporales con subconjuntos de embeddings COCO reales; compara HNSW y búsqueda exacta, verifica ambos planes con `EXPLAIN`, y guarda p50, p95, Recall@10 y tiempo de construcción. No altera el catálogo de producción. La evaluación imagen-imagen utiliza coincidencia de objetos COCO como proxy semántico y **no sustituye** un juicio humano de similitud visual. La prueba HTTP incluye inferencia CLIP y red local. Los reportes JSON se reutilizan si la configuración y el estado de la base no cambiaron; `--force` repite la medición. Los reportes FAISS anteriores son históricos de v3 y no miden PostgreSQL.

Para ampliar el conjunto (por ejemplo a 30 000), ejecuta `python ampliar_dataset.py --target 30000`, `python preparar_metadata.py` y `python indexar.py`; luego repite las evaluaciones. Los resultados medidos hasta 20 000 no predicen automáticamente la latencia a cualquier escala.
