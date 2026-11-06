# Traductor Inglés → Español con Transformer entrenado desde cero

Proyecto desarrollado para el curso **Tópicos Avanzados de Machine Learning** (UPCH, 2026-II).

Este repositorio está diseñado para albergar el progreso completo del curso (**Entrega Parcial** y **Entrega Final**). Actualmente contiene la solución completa, notebooks, reportes, datos y presentaciones correspondientes al **Examen Parcial**.

> **Integrantes:** Victor Rivera, Vanesa Rioja y Jesús Morales

---

## Resumen del Proyecto (Entrega Parcial)

Se construyó e implementó un modelo de traducción neuronal **Encoder-Decoder (Transformer)** entrenado **estrictamente desde cero** (pesos e inicialización aleatoria Xavier, sin embeddings preentrenados y con un tokenizador BPE propio), utilizando un corpus paralelo inglés–español de **2.000.000 de pares** curado desde cuatro fuentes públicas.

| Elemento | Detalle |
|---|---|
| **Tarea** | Traducción automática Inglés → Español (`en` → `es`) |
| **Modelo** | Transformer Encoder-Decoder (PyTorch `nn.Transformer`), **22,7 M de parámetros** |
| **Inicialización** | Aleatoria (Xavier). **Sin pesos, embeddings ni tokenizadores preentrenados** |
| **Tokenizador** | BPE a nivel de bytes (*ByteLevel*), vocabulario de **16.000 tokens**, entrenado solo con datos de train |
| **Corpus** | 2.000.000 de pares limpios (Kaggle, TED2020, NewsCommentary, Europarl) |
| **Descarga Dataset** | [Google Drive - Dataset (Transformer)](https://drive.google.com/drive/folders/1_63ZbrdyPn6uv4MdBPMSdKjnT8RVbI0L?usp=sharing) |
| **Particionado** | 1.995.994 Train / 2.000 Validación / 2.000 Test (semilla fija `2026`) |
| **Resultados Test** | **BLEU:** `42.98` \| **chrF++:** `64.42` \| **COMET:** `0.8653` |
| **Entorno** | Google Colab / GPU local (RTX PRO 6000 / RTX 5050) |

---

## Estructura del Repositorio

```text
.
├── Parcial/                                 # Entregables y artefactos del Examen Parcial
│   ├── data/
│   │   ├── eda_avanzado.png                 # Visualizaciones del EDA del corpus
│   │   └── manifest.json                    # Manifiesto con conteos por fuente, rechazos y SHA-256
│   ├── docs/
│   │   ├── Informe (Traductor).pdf          # Informe académico completo del parcial
│   │   └── Presentación (Traductor).pdf     # Diapositivas de la exposición (~12 min)
│   ├── final report/
│   │   ├── report.json                      # Manifiesto de división del dataset y tokenizador
│   │   ├── test_final_metrics.json          # Métricas cuantitativas en test (BLEU, chrF++, BP)
│   │   ├── test_final_comet.json            # Métrica semántica basada en redes (COMET score)
│   │   └── test_final_translations.jsonl    # 2.000 traducciones generadas en test
│   └── notebooks/
│       ├── Dataset_(Traductor).ipynb        # Extracción, normalización, filtrado, deduplicación y EDA
│       ├── Transformer_(Traductor)_300k.ipynb # Entrenamiento de la versión Piloto (300k pares, 8 épocas)
│       └── Transformer_(Traductor)_2M.ipynb   # Entrenamiento de la versión Completa (2M pares, 15 épocas)
├── Final/                                   # Buscador multimodal de imágenes con CLIP y FAISS (Proyecto Final)
├── PC/                                      # Prácticas calificadas del curso (PC)
├── Documentos/                              # Rúbricas y lineamientos del curso
├── .gitignore                               # Exclusión de archivos pesados y entornos virtuales
└── README.md                                # Documentación principal del repositorio
```

> **Nota sobre el Dataset (`pares_entrenamiento_2M.xlsx`):** Debido al límite de 100 MB impuesto por GitHub para archivos individuales, el archivo Excel comprimido/completo de 240 MB está alojado en Google Drive y se puede descargar desde el enlace: [Descargar pares_entrenamiento_2M.xlsx de Google Drive](https://drive.google.com/drive/folders/1_63ZbrdyPn6uv4MdBPMSdKjnT8RVbI0L?usp=sharing).

---

## 1. Construcción y Composición del Corpus

El dataset fue procesado en flujo mediante el notebook `Dataset_(Traductor).ipynb` usando SQLite y deduplicación basada en hashes SHA-256.

- **Acceso directo al archivo comprimido:** [Carpeta Google Drive con `pares_entrenamiento_2M.xlsx`](https://drive.google.com/drive/folders/1_63ZbrdyPn6uv4MdBPMSdKjnT8RVbI0L?usp=sharing)

### Fuentes de Datos y Filtrado (Extraído de `Parcial/data/manifest.json`)

| Fuente | Leídos | Rechazo Básico | Rechazo Idioma (fastText) | Aceptados Únicos |
|---|---:|---:|---:|---:|
| **Kaggle** (*Lonnie Qin*) | 118.964 | 686 | 4.935 | **95.532** |
| **TED2020 v1** (*OPUS*) | 416.846 | 15.092 | 10.103 | **386.501** |
| **News Commentary** (*Helsinki-NLP*) | 238.872 | 6.333 | 5.671 | **226.149** |
| **Europarl v7** (*StatMT*) | 1.965.734 | 101.807 | 16.183 | **1.805.049** |
| **Total Pool Elegible** | **2.739.416** | **123.918** | **36.892** | **2.513.231** |

- **Filtros aplicados:** Normalización Unicode (NFC), 2 a 60 palabras por oración, ratio de longitud $\le 3,0$, eliminación de HTML/URLs, filtro de idioma fastText `lid.176` (confianza $\ge 0,60$) y deduplicación estricta SHA-256 en ambos lados (ninguna frase se repite ni en inglés ni en español).
- **Selección final:** Se extrajo una muestra estratificada pseudoaleatoria de **2.000.000 de pares** (Semilla `2026`).
- **División:** 1.995.994 train (6 pares excluidos por superar 128 tokens BPE), 2.000 validación y 2.000 test.

---

## 2. Arquitectura del Modelo y Tokenización

- **Tokenizador:** BPE ByteLevel propio de **16.000 tokens** entrenado exclusivamente sobre los datos de entrenamiento (sin ver frases de validación ni test). Tokens especiales: `<pad>`, `<bos>`, `<eos>`, `<unk>`.
- **Arquitectura Transformer:**

| Hiperparámetro | Valor |
|---|---|
| Dimensiones de modelo (`d_model`) | 384 |
| Cabezas de atención (`nhead`) | 6 |
| Capas Encoder / Decoder | 4 / 4 |
| Dimensión Feedforward (`d_ff`) | 1536 |
| Dropout | 0,1 |
| Normalización | Pre-norm (`norm_first=True`) |
| Codificación de Posición | Sinusoidal fija |
| Compartición de Embeddings | Entre entrada del decoder y capa lineal de salida |
| **Parámetros Totales** | **22.709.760 (~22,7 M)** |

---

## 3. Entrenamiento

- **Optimizador:** AdamW ($\beta_1=0,9$, $\beta_2=0,98$, $\epsilon=10^{-9}$, weight decay 0,01).
- **Scheduler:** Noam LR Scheduler con 2.000 pasos de warmup.
- **Regularización:** *Label smoothing* de 0,1 en pérdida de train.
- **Técnicas:** Precisión mixta (FP16/BF16), *gradient clipping* a 1,0 y acumulación de gradientes (batch efectivo de 256).

### Comparativa entre Versiones

| Métrica / Parámetro | Versión Piloto (300k) | Versión Completa (2M) |
|---|---:|---:|
| **Pares de Train** | 300.000 | 1.995.994 |
| **Épocas Entrenadas** | 8 | 15 |
| **Pasos Totales** | 9.376 | 116.955 |
| **Pérdida Validación** | 1,428 | **1,213** |
| **Perplejidad Validación** | 4,170 | **3,364** |
| **BLEU (Test)** | — | **42,98** |
| **chrF++ (Test)** | — | **64,42** |
| **COMET Score (Test)** | — | **0,8653** |

---

## 4. Resultados Evaluados en Test (2.000 Pares)

Evaluación realizada sobre el conjunto de test independiente utilizando **Beam Search** ($beam=3, \alpha=0,6$).

### Métricas Globales (Extraídas de `Parcial/final report/test_final_metrics.json` y `test_final_comet.json`)

- **BLEU (SacreBLEU):** `42.98`
- **chrF++:** `64.42`
- **COMET Score:** `0.8653`
- **Brevity Penalty (BP):** `0.983` (Ratio de longitud de hipótesis vs referencia: `0.983`)

### Rendimiento según Longitud de Oración

| Palabras en Oración (EN) | Cantidad ($n$) | BLEU | chrF++ Medio |
|---|---:|---:|---:|
| 1 – 10 palabras | 407 | **44,30** | 63,58 |
| 11 – 20 palabras | 616 | **42,42** | 63,26 |
| 21 – 30 palabras | 533 | **42,92** | 64,25 |
| 31+ palabras | 444 | **41,85** | 64,12 |

---

## 5. Ejemplos de Traducción Real del Modelo

Extraídos directamente de `Parcial/final report/test_final_translations.jsonl`:

| Entrada Inglés (`en`) | Referencia Esperada (`es`) | Traducción del Modelo (`hypothesis`) | Tipo de Evaluación |
|---|---|---|---|
| *But when we arrived, it was clear that there was only one story in town.* | Pero cuando llegamos, estaba claro que solo había una historia en la ciudad. | **Pero cuando llegamos, estaba claro que solo había una historia en la ciudad.** | **Coincidencia Exacta** (100% idéntico) |
| *We also need an increase in the present limits on our coasts.* | También necesitamos un incremento de los actuales límites sobre nuestras costas. | **También necesitamos un aumento de los límites actuales en nuestras costas.** | **Traducción Fluida / Sinónimos** |
| *I believe what he says.* | Creo lo que él dice. | **Creo lo que él dice.** | **Coincidencia Exacta** |
| *Stay positive, work hard, make it happen* | *(Texto interactivo libre)* | **Quédate positivo, trabaje duro, haga que suceda** | **Prueba Interactiva** |

---

## 6. Cómo Ejecutar y Reproducir

1. **Generación / Obtención del Dataset:**
   - Descargar el archivo `pares_entrenamiento_2M.xlsx` desde [Google Drive](https://drive.google.com/drive/folders/1_63ZbrdyPn6uv4MdBPMSdKjnT8RVbI0L?usp=sharing) o volver a procesarlo con [`Parcial/notebooks/Dataset_(Traductor).ipynb`](Parcial/notebooks/Dataset_(Traductor).ipynb).

2. **Entrenamiento y Evaluación del Transformer:**
   - Cargar `pares_entrenamiento_2M.xlsx` y `manifest.json` en la carpeta de trabajo.
   - Ejecutar [`Parcial/notebooks/Transformer_(Traductor)_2M.ipynb`](Parcial/notebooks/Transformer_(Traductor)_2M.ipynb) en un entorno con GPU PyTorch.
   - El notebook entrena el BPE, construye el Transformer, efectúa la validación por época y ejecuta la evaluación automática de test (exportando `test_final_metrics.json`, `test_final_comet.json` y `test_final_translations.jsonl`).

---

## Entregables del Examen Parcial

- [x] **Notebooks de Código:** [`Parcial/notebooks/Dataset_(Traductor).ipynb`](Parcial/notebooks/Dataset_(Traductor).ipynb) y [`Parcial/notebooks/Transformer_(Traductor)_2M.ipynb`](Parcial/notebooks/Transformer_(Traductor)_2M.ipynb)
- [x] **Manifiesto y Datos Limpios:** [`Parcial/data/manifest.json`](Parcial/data/manifest.json) y Enlace de descarga en Google Drive
- [x] **Informe Escrito:** [`Parcial/docs/Informe (Traductor).pdf`](Parcial/docs/Informe (Traductor).pdf)
- [x] **Presentación Ejecutiva:** [`Parcial/docs/Presentación (Traductor).pdf`](Parcial/docs/Presentación (Traductor).pdf)
- [x] **Reporte y Métricas de Test:** [`Parcial/final report/test_final_metrics.json`](Parcial/final report/test_final_metrics.json) y [`Parcial/final report/test_final_translations.jsonl`](Parcial/final report/test_final_translations.jsonl)

---

## Referencias

- Vaswani et al. (2017). *Attention Is All You Need*.
- Koehn, P. (2005). *Europarl: A Parallel Corpus for Statistical Machine Translation*.
- OPUS TED2020 · Helsinki-NLP News Commentary · Kaggle *English-Spanish Translation Dataset*.
- Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores* (SacreBLEU).
- Popović, M. (2015, 2017). *chrF & chrF++: character n-gram F-score for automatic MT evaluation*.
- Rei et al. (2020). *COMET: A Neural Framework for MT Evaluation*.
