from functools import lru_cache
import re

import numpy as np
import open_clip
import torch
from langdetect import DetectorFactory, detect
from transformers import MarianMTModel, MarianTokenizer

from model_config import EMBEDDING_DIM, MODEL_ARCH, MODEL_NAME, PRETRAINED

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRANSLATOR_NAME = "Helsinki-NLP/opus-mt-es-en"
DetectorFactory.seed = 0
SPANISH_WORDS = {"un", "una", "unos", "unas", "el", "la", "los", "las", "de", "del", "con", "en", "sobre", "para", "gato", "gatos", "perro", "perros", "durmiendo", "bicicleta", "roja", "rojo", "nieve", "avion", "aviones", "coche", "coches", "auto", "autos", "playa", "taza", "cafe", "persona", "personas", "niño", "niña", "comiendo", "jugando", "volando", "corriendo", "buscando", "imagen", "imagenes", "similar", "similares"}

model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ARCH, pretrained=PRETRAINED)
model = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer(MODEL_ARCH)


def _normalizar(feats):
    vectors = torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy().astype(np.float32)
    if vectors.shape[1] != EMBEDDING_DIM:
        raise RuntimeError(f"{MODEL_NAME} produjo {vectors.shape[1]} dimensiones; se esperaban {EMBEDDING_DIM}")
    return vectors


@torch.inference_mode()
def embed_imagenes(imagenes):
    if not imagenes:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    batch = torch.stack([preprocess(image.convert("RGB")) for image in imagenes]).to(DEVICE)
    return _normalizar(model.encode_image(batch))


@torch.inference_mode()
def embed_texto(textos):
    if not textos:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    tokens = tokenizer(textos).to(DEVICE)
    return _normalizar(model.encode_text(tokens))


@lru_cache(maxsize=4096)
def traducir_a_ingles(texto: str) -> str:
    """Traducción local español-inglés; inglés pasa directamente a CLIP."""
    texto = texto.strip()
    if not texto:
        return texto
    words = set(re.findall(r"[a-záéíóúñ]+", texto.lower()))
    try:
        spanish = bool(words & SPANISH_WORDS) or detect(texto) == "es"
    except Exception:
        spanish = False
    if not spanish:
        return texto
    tokenizer, translator = _translator()
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        output = translator.generate(**inputs, max_new_tokens=128)
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()


@lru_cache(maxsize=1)
def _translator():
    tokenizer = MarianTokenizer.from_pretrained(TRANSLATOR_NAME)
    translator = MarianMTModel.from_pretrained(TRANSLATOR_NAME).eval()
    return tokenizer, translator
