"""Identidad del checkpoint y del espacio vectorial de esta rama."""

MODEL_ARCH = "ViT-B-16"
PRETRAINED = "laion2b_s34b_b88k"
MODEL_ID = f"open_clip:{MODEL_ARCH}:{PRETRAINED}"
MODEL_NAME = "OpenCLIP ViT-B/16 (LAION-2B)"
EMBEDDING_DIM = 512
DATABASE_NAME = "pinsearch_vitb16"
DB_PORT = 5434
