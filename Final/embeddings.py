from functools import lru_cache
import re

import torch
from langdetect import DetectorFactory, detect
from transformers import CLIPModel, CLIPProcessor, MarianMTModel, MarianTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"
TRANSLATOR_NAME = "Helsinki-NLP/opus-mt-es-en"
DetectorFactory.seed = 0
SPANISH_WORDS = {"un", "una", "unos", "unas", "el", "la", "los", "las", "de", "del", "con", "en", "sobre", "para", "gato", "gatos", "perro", "perros", "durmiendo", "bicicleta", "roja", "rojo", "nieve", "avion", "aviones", "coche", "coches", "auto", "autos", "playa", "taza", "cafe", "persona", "personas", "niño", "niña", "comiendo", "jugando", "volando", "corriendo", "buscando", "imagen", "imagenes", "similar", "similares"}

model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(MODEL_NAME)


def _normalizar(feats):
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")


@torch.no_grad()
def embed_imagenes(imagenes):
    """imagenes: lista de PIL.Image -> array (n, 512) normalizado."""
    inputs = processor(images=imagenes, return_tensors="pt").to(DEVICE)
    return _normalizar(model.get_image_features(**inputs))


@torch.no_grad()
def embed_texto(textos):
    """textos: lista de str -> array (n, 512) normalizado."""
    inputs = processor(text=textos, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    return _normalizar(model.get_text_features(**inputs))


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
