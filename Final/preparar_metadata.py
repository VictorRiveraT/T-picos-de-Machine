"""Extrae captions y atributos agrupados de las anotaciones oficiales COCO."""

import json
import os
from pathlib import Path
import zipfile

import ijson

from atributos import agrupar


BASE_DIR = Path(__file__).resolve().parent
MAPPING = BASE_DIR / "archivos.json"
MANIFEST = BASE_DIR / "data/train_manifest.json"
ARCHIVE = BASE_DIR / "descargas/annotations.zip"
OUTPUT = BASE_DIR / "data/coco_metadata.json"


def iter_items(archive, member, prefix):
    with archive.open(member) as source:
        yield from ijson.items(source, prefix)


def main():
    if not MANIFEST.exists() or not ARCHIVE.exists():
        raise RuntimeError("Ejecuta primero python ampliar_dataset.py y conserva descargas/annotations.zip")
    with MAPPING.open(encoding="utf-8") as source:
        val_names = json.load(source)
    with MANIFEST.open(encoding="utf-8") as source:
        manifest = json.load(source)
        train_ids = manifest["train_ids"]
    val_ids = {int(Path(name).stem) for name in val_names}
    train_ids = set(train_ids)
    records = {
        f"{image_id:012d}.jpg": {"split": split, "captions": [], "objects": set()}
        for split, ids in (("val2017", val_ids), ("train2017", train_ids))
        for image_id in ids
    }
    with zipfile.ZipFile(ARCHIVE) as archive:
        for split, ids in (("val2017", val_ids), ("train2017", train_ids)):
            member = f"annotations/captions_{split}.json"
            for item in iter_items(archive, member, "annotations.item"):
                if item["image_id"] in ids:
                    records[f"{item['image_id']:012d}.jpg"]["captions"].append(item["caption"].strip())
            print(f"Captions {split} extraídos", flush=True)

        categories = {
            item["id"]: item["name"]
            for item in iter_items(archive, "annotations/instances_val2017.json", "categories.item")
        }
        for split, ids in (("val2017", val_ids), ("train2017", train_ids)):
            member = f"annotations/instances_{split}.json"
            for item in iter_items(archive, member, "annotations.item"):
                if item["image_id"] in ids:
                    name = categories.get(item["category_id"])
                    if name:
                        records[f"{item['image_id']:012d}.jpg"]["objects"].add(name)
            print(f"Objetos {split} extraídos", flush=True)

    output = {}
    for filename, record in records.items():
        captions = record["captions"]
        output[filename] = {
            "split": record["split"],
            "caption": captions[0] if captions else "",
            "captions": captions,
            "attributes": agrupar(captions, record["objects"]),
        }
    if len(output) != manifest["target_total"]:
        raise RuntimeError(f"Se esperaban {manifest['target_total']} registros; se obtuvieron {len(output)}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".json.tmp")
    with temp.open("w", encoding="utf-8") as target:
        json.dump({"source": "COCO 2017 official captions and instances", "records": output}, target, ensure_ascii=False)
    os.replace(temp, OUTPUT)
    print(f"Metadatos agrupados guardados para {len(output)} imágenes en {OUTPUT}")


if __name__ == "__main__":
    main()
