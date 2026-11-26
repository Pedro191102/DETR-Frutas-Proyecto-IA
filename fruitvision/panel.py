"""Resumen del fotograma. Los pesos corresponden a unidades de referencia comestibles."""
import csv
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .utils import slugify


@lru_cache(maxsize=1)
def referencias():
    ruta = Path(__file__).resolve().parents[1] / "config/nutrition_reference.csv"
    tabla = {}
    with ruta.open(encoding="utf8", newline="") as f:
        for fila in csv.DictReader(f):
            kcal = None
            if fila["weight_avg_g"] and fila["kcal_100g"]:
                kcal = float(fila["weight_avg_g"]) * float(fila["kcal_100g"]) / 100
            for alias in [fila["product"], *fila["aliases"].split(";")]:
                if alias:
                    tabla[slugify(alias)] = (fila["display_es"], kcal)
    return tabla


def resumen(detecciones):
    tabla = referencias()
    grupos = Counter(tabla.get(slugify(d["label"]), (d["label"], None)) for d in detecciones)
    filas = [(nombre, n, None if kcal is None else round(n*kcal))
             for (nombre, kcal), n in sorted(grupos.items())]
    return tuple(filas)


@lru_cache(maxsize=8)
def dibujar_panel(filas, alto):
    paso = 26
    capacidad = max(1, (alto-146)//paso)
    columnas = max(1, (len(filas)+capacidad-1)//capacidad)
    ancho = 276*columnas
    panel = Image.new("RGB", (ancho, alto), (24,24,24))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default(size=17)
    amarillo = (255,222,48)
    def texto(x,y,s):
        draw.text((x,y),s,font=font,fill=amarillo)
    texto(12,10,"Calorías estimadas")
    texto(12,37,"Por unidad de referencia")
    if not filas:
        texto(12,75,"Sin alimentos detectados")
    for j,(nombre,n,kcal) in enumerate(filas):
        x=12+276*(j//capacidad)
        y=73+paso*(j%capacidad)
        texto(x,y,f"x{n} {nombre}")
        draw.text((x+252,y), "s/d" if kcal is None else f"{kcal} kcal",
                  anchor="ra",font=font,fill=amarillo)
    parcial = any(kcal is None for _,_,kcal in filas)
    total = sum(kcal for _,_,kcal in filas if kcal is not None)
    draw.line((12,alto-68,ancho-12,alto-68),fill=amarillo)
    texto(12,alto-58,f"{'Subtotal conocido' if parcial else 'TOTAL'}: {total} kcal")
    texto(12,alto-31,"s/d: falta referencia" if parcial else "Estimación por fotograma")
    return np.asarray(panel)[:,:,::-1]
