"""Amplía COCO val2017 con un subconjunto reproducible de train2017."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen
import zipfile

import ijson
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
SEED_MAPPING = BASE_DIR / "archivos.json"
ANNOTATIONS_ZIP = BASE_DIR / "descargas/annotations.zip"
IMAGES_DIR = BASE_DIR / "imagenes"
MANIFEST = BASE_DIR / "data/train_manifest.json"
SOURCE = "http://images.cocodataset.org/train2017/"


def selected_train_ids(target_total):
    with SEED_MAPPING.open(encoding="utf-8") as source:
        seed_names = set(json.load(source))
    needed = target_total - len(seed_names)
    if needed <= 0:
        raise ValueError("El objetivo debe superar el tamaño del conjunto base.")
    with zipfile.ZipFile(ANNOTATIONS_ZIP) as archive:
        with archive.open("annotations/captions_train2017.json") as source:
            ids = sorted(
                int(item["id"])
                for item in ijson.items(source, "images.item")
                if f"{int(item['id']):012d}.jpg" not in seed_names
            )
    if len(ids) < needed:
        raise RuntimeError("COCO train2017 no tiene suficientes imágenes distintas.")
    return ids[:needed], len(seed_names)


def valid_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def download_one(image_id):
    name = f"{image_id:012d}.jpg"
    target = IMAGES_DIR / name
    if target.exists() and valid_image(target):
        return "cached"
    temp = IMAGES_DIR / f"{name}.part"
    for attempt in range(3):
        try:
            request = Request(SOURCE + name, headers={"User-Agent": "PinSearch-COCO-Dataset/1.0"})
            with urlopen(request, timeout=60) as response, temp.open("wb") as output:
                shutil.copyfileobj(response, output)
            if not valid_image(temp):
                raise ValueError("El servidor no devolvió una imagen válida")
            os.replace(temp, target)
            return "downloaded"
        except Exception:
            temp.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers debe ser positivo")
    if not ANNOTATIONS_ZIP.exists() or not SEED_MAPPING.exists():
        parser.error("Ejecuta primero python descargar_dataset.py para obtener COCO val2017 y las anotaciones")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    ids, seed_count = selected_train_ids(args.target)
    manifest = {
        "source": "COCO train2017",
        "selection": "primeros IDs ordenados que no están en val2017",
        "seed_count": seed_count,
        "target_total": args.target,
        "train_ids": ids,
    }
    temp = MANIFEST.with_suffix(".json.tmp")
    with temp.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
    os.replace(temp, MANIFEST)
    completed = 0
    downloaded = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, image_id): image_id for image_id in ids}
        for future in as_completed(futures):
            completed += 1
            try:
                downloaded += future.result() == "downloaded"
            except Exception as error:
                failures.append((futures[future], str(error)))
            if completed % 250 == 0 or completed == len(ids):
                print(f"{completed}/{len(ids)} revisadas; {downloaded} descargadas; {len(failures)} fallidas", flush=True)
    if failures:
        raise RuntimeError(f"Faltaron {len(failures)} imágenes; reejecuta el comando. Ejemplo: {failures[0]}")
    print(f"COCO listo: {seed_count} val + {len(ids)} train = {args.target} imágenes.")


if __name__ == "__main__":
    main()
