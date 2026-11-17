"""Comprueba los flujos públicos de PinSearch tras iniciar el servidor."""

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import uuid

from catalogo import BASE_DIR


def request_json(url, body=None, content_type=None):
    headers = {"Content-Type": content_type} if content_type else {}
    request = Request(url, data=body, headers=headers)
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def multipart(fields, file_field=None):
    boundary = "pinsearch-" + uuid.uuid4().hex
    parts = []
    for key, value in fields.items():
        parts.extend((f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n".encode(), str(value).encode(), b"\r\n"))
    if file_field:
        key, path = file_field
        parts.extend((f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"; filename=\"{path.name}\"\r\nContent-Type: image/jpeg\r\n\r\n".encode(), path.read_bytes(), b"\r\n"))
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--expected-count", type=int, default=20_000)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    info = request_json(f"{base}/api/info")
    assert info["total_imagenes"] == args.expected_count, info
    assert info["base_datos"] == "PostgreSQL + pgvector"
    text = request_json(f"{base}/api/buscar/texto?" + urlencode({"q": "un gato durmiendo", "k": 3, "familia": "animales"}))
    assert text["traducido"] and len(text["resultados"]) == 3, text
    assert all("animales" in row["atributos"]["familias"] for row in text["resultados"])
    name = text["resultados"][0]["id"]
    similar = request_json(f"{base}/api/buscar/similar?" + urlencode({"id": name, "k": 3}))
    assert len(similar["resultados"]) == 3 and all(row["id"] != name for row in similar["resultados"])
    path = BASE_DIR / "imagenes" / name
    body, content_type = multipart({}, ("archivo", path))
    upload = request_json(f"{base}/api/buscar/imagen?k=3", body, content_type)
    assert len(upload["resultados"]) == 3
    body, content_type = multipart({"id_imagen": name, "texto_modificador": "en la nieve", "k": 3})
    composed = request_json(f"{base}/api/buscar/compuesta", body, content_type)
    assert composed["traducido"] and len(composed["resultados"]) == 3
    print(json.dumps({"status": "ok", "images": info["total_imagenes"],
                      "engine": info["base_datos"],
                      "flows": ["text", "attribute_filter", "similar", "upload", "composed", "translation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
