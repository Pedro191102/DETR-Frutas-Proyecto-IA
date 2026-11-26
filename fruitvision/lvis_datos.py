"""Selección reproducible desde las particiones originales de LVIS v1."""
import json
import random
import shutil
import tempfile
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlparse

import ijson
from PIL import Image
from tqdm import tqdm

from .lvis_clases import CLASES


def descargar(urls, destino, validar):
    if destino.is_file():
        try:
            validar(destino)
            return
        except (OSError, ValueError, zipfile.BadZipFile):
            pass
    destino.parent.mkdir(parents=True, exist_ok=True)
    for intento in range(3):
        for url in urls:
            with tempfile.NamedTemporaryFile(dir=destino.parent, delete=False) as tmp:
                ruta = Path(tmp.name)
            try:
                with urlopen(url, timeout=90) as origen, ruta.open("wb") as salida:
                    shutil.copyfileobj(origen, salida, length=1024 * 1024)
                validar(ruta)
                ruta.replace(destino)
                return
            except (OSError, ValueError, zipfile.BadZipFile) as error:
                ultimo = error
            finally:
                ruta.unlink(missing_ok=True)
        time.sleep(2 ** intento)
    raise RuntimeError(f"Descarga fallida: {destino.name}. Repite el comando para reanudar.") from ultimo


def elegir(cajas, limite, semilla):
    rng = random.Random(semilla)
    por_clase = defaultdict(list)
    for imagen, objetos in cajas.items():
        for clase in sorted({a["category_id"] for a in objetos}):
            por_clase[clase].append(imagen)
    for grupo in por_clase.values():
        rng.shuffle(grupo)
    elegidos = set()
    while len(elegidos) < min(limite, len(cajas)):
        for clase in sorted(por_clase, key=lambda k: len(por_clase[k])):
            grupo = por_clase[clase]
            while grupo and grupo[-1] in elegidos:
                grupo.pop()
            if grupo:
                elegidos.add(grupo.pop())
            if len(elegidos) >= limite:
                break
    return elegidos


def preparar(raiz, split, limite, semilla):
    nombre = f"lvis_v1_{split}.json"
    archivo = raiz / (nombre + ".zip")
    def validar_zip(p):
        with zipfile.ZipFile(p) as z:
            z.getinfo(nombre)
    print(f"Preparando anotaciones originales de {split}...", flush=True)
    descargar([f"https://dl.fbaipublicfiles.com/LVIS/{archivo.name}"], archivo, validar_zip)
    manifiesto = raiz / f"fv58-v1-{split}-{limite}-{semilla}.json"
    if manifiesto.is_file():
        registros = json.loads(manifiesto.read_text(encoding="utf8"))
    else:
        print(f"Seleccionando hasta {limite} imágenes de {split}...", flush=True)
        registros = seleccionar_registros(archivo, nombre, limite, semilla)
    if not registros:
        raise ValueError(f"Partición vacía: {split}")
    def obtener(r):
        destino = raiz / "images" / r["ruta"]
        def validar_img(p):
            with Image.open(p) as im:
                im.verify()
        descargar(["https://s3.amazonaws.com/images.cocodataset.org/" + r["ruta"],
                   "https://images.cocodataset.org/" + r["ruta"]], destino, validar_img)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(tqdm(pool.map(obtener, registros), total=len(registros), desc=split))
    if not manifiesto.exists():
        with tempfile.NamedTemporaryFile(dir=raiz, mode="w", encoding="utf8", delete=False) as tmp:
            json.dump(registros, tmp)
        Path(tmp.name).replace(manifiesto)
    return registros


def seleccionar_registros(archivo, nombre, limite, semilla):
    with zipfile.ZipFile(archivo) as z:
        def filas(clave):
            with z.open(nombre) as f:
                yield from ijson.items(f, clave + ".item", use_float=True)
        categorias = {c["name"]: c["id"] for c in filas("categories")}
        ids = {categorias[n]: i for i, n in enumerate(CLASES)}
        cajas = defaultdict(list)
        for a in filas("annotations"):
            if a["category_id"] in ids and min(a["bbox"][2:]) > 0:
                cajas[a["image_id"]].append(dict(category_id=ids[a["category_id"]],
                    bbox=a["bbox"], area=a["bbox"][2]*a["bbox"][3], iscrowd=0))
        seleccion = elegir(cajas, limite, semilla)
        registros = []
        for img in filas("images"):
            if img["id"] in seleccion:
                ruta = urlparse(img["coco_url"]).path.strip("/")
                registros.append(dict(id=img["id"], ruta=ruta, objetos=cajas[img["id"]]))
    return registros
