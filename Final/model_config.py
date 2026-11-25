"""Identidad del checkpoint y del espacio vectorial de esta rama."""

MODEL_ARCH = "RN50-quickgelu"
PRETRAINED = "openai"
MODEL_ID = f"open_clip:{MODEL_ARCH}:{PRETRAINED}"
MODEL_NAME = "CLIP RN50 (OpenAI)"
EMBEDDING_DIM = 1024
DATABASE_NAME = "pinsearch_rn50"
DB_PORT = 5435
