"""Agrupa atributos derivados de categorías COCO y de sus captions oficiales."""

import re


FAMILIAS = {
    "person": "personas",
    "bicycle": "vehiculos", "car": "vehiculos", "motorcycle": "vehiculos",
    "airplane": "vehiculos", "bus": "vehiculos", "train": "vehiculos",
    "truck": "vehiculos", "boat": "vehiculos",
    "bird": "animales", "cat": "animales", "dog": "animales", "horse": "animales",
    "sheep": "animales", "cow": "animales", "elephant": "animales",
    "bear": "animales", "zebra": "animales", "giraffe": "animales",
    "banana": "comida", "apple": "comida", "sandwich": "comida",
    "orange": "comida", "broccoli": "comida", "carrot": "comida",
    "hot dog": "comida", "pizza": "comida", "donut": "comida", "cake": "comida",
    "chair": "muebles", "couch": "muebles", "potted plant": "muebles",
    "bed": "muebles", "dining table": "muebles", "toilet": "muebles",
    "tv": "electronica", "laptop": "electronica", "mouse": "electronica",
    "remote": "electronica", "keyboard": "electronica", "cell phone": "electronica",
}
COLORES = {"red", "blue", "green", "yellow", "orange", "black", "white", "brown", "pink", "purple", "gray", "grey"}
ACCIONES = {"sleeping", "running", "jumping", "riding", "eating", "cooking", "sitting", "standing", "walking", "flying", "playing", "swimming", "driving"}
ESCENAS = {"beach", "street", "road", "kitchen", "bedroom", "snow", "forest", "park", "field", "room", "water", "mountain", "city"}


def agrupar(captions, objetos):
    texto = " ".join(captions).lower()
    palabras = set(re.findall(r"[a-z]+", texto))
    objetos = sorted(set(objetos))
    return {
        "objetos": objetos,
        "familias": sorted({FAMILIAS[objeto] for objeto in objetos if objeto in FAMILIAS}),
        "colores": sorted(palabras & COLORES),
        "acciones": sorted(palabras & ACCIONES),
        "escenas": sorted(palabras & ESCENAS),
    }
