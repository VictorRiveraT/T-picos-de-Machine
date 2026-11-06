import json
import urllib.parse
import urllib.request

import torch
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"

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


def traducir_a_ingles(texto: str) -> str:
    """Traduce la consulta a inglés si el usuario escribe en español u otro idioma.
    Si falla la conexión o ya está en inglés, retorna el texto original de forma segura."""
    texto = texto.strip()
    if not texto:
        return texto
    try:
        url = (
            "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q="
            + urllib.parse.quote(texto)
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            traducido = "".join([part[0] for part in data[0] if part[0]]).strip()
            return traducido if traducido else texto
    except Exception:
        return texto
