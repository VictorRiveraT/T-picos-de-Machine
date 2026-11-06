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