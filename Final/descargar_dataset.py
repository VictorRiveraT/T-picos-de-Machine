import os
import shutil
import urllib.request
import zipfile

URL_IMAGENES = "http://images.cocodataset.org/zips/val2017.zip"
URL_ANOTACIONES = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

os.makedirs("descargas", exist_ok=True)
os.makedirs("imagenes", exist_ok=True)
os.makedirs("anotaciones", exist_ok=True)


def descargar(url, destino):
    if os.path.exists(destino):
        print(f"Ya existe {destino}, se omite la descarga")
        return

    def progreso(bloques, tam_bloque, total):
        if total > 0:
            pct = min(100, bloques * tam_bloque * 100 / total)
            print(f"\r{destino}: {pct:5.1f}%", end="")

    urllib.request.urlretrieve(url, destino, progreso)
    print()


# 1) Imágenes de val2017 (5000 imágenes, ~780 MB)
descargar(URL_IMAGENES, "descargas/val2017.zip")
with zipfile.ZipFile("descargas/val2017.zip") as z:
    z.extractall("descargas")
for nombre in os.listdir("descargas/val2017"):
    shutil.move(os.path.join("descargas/val2017", nombre), os.path.join("imagenes", nombre))
print(f"{len(os.listdir('imagenes'))} imágenes listas en imagenes/")

# 2) Captions para la evaluación (solo se extrae captions_val2017.json)
descargar(URL_ANOTACIONES, "descargas/annotations.zip")
with zipfile.ZipFile("descargas/annotations.zip") as z:
    z.extract("annotations/captions_val2017.json", ".")
shutil.move("annotations/captions_val2017.json", "anotaciones/captions_val2017.json")
print("Captions guardados en anotaciones/captions_val2017.json")