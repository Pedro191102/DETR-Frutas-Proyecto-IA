"""Ajuste parcial de Deformable DETR sobre imágenes originales de LVIS."""
import argparse
import json
import math
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from fruitvision.lvis_clases import BASE, CLASES
from fruitvision.lvis_datos import preparar

ROOT = Path(__file__).resolve().parent
SIZE = {"shortest_edge": 480, "longest_edge": 640}


class Fotos(Dataset):
    def __init__(self, registros, raiz):
        self.registros = registros
        self.raiz = raiz

    def __len__(self):
        return len(self.registros)

    def __getitem__(self, idx):
        r = self.registros[idx]
        with Image.open(self.raiz / "images" / r["ruta"]) as im:
            return im.convert("RGB"), {"image_id": r["id"], "annotations": r["objetos"]}


class Lotes:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, fotos):
        # Las cajas de Deformable DETR se normalizan sobre el área válida de cada foto.
        partes = [self.processor(images=im, annotations=ann, return_tensors="pt",
                                 do_pad=False) for im, ann in fotos]
        pixels = [p.pixel_values[0] for p in partes]
        alto = max(p.shape[-2] for p in pixels)
        ancho = max(p.shape[-1] for p in pixels)
        pares = [self.processor.pad(p, (alto, ancho)) for p in pixels]
        return dict(pixel_values=torch.stack([p[0] for p in pares]),
                    pixel_mask=torch.stack([p[1] for p in pares]),
                    labels=[{k:p.labels[0][k] for k in ("boxes","class_labels")} for p in partes])


def especializar(base):
    cfg = deepcopy(base.config)
    indices = [cfg.label2id[n] for n in CLASES]
    cfg.id2label = dict(enumerate(CLASES))
    cfg.label2id = {n:i for i,n in cfg.id2label.items()}
    nuevo = AutoModelForObjectDetection.from_config(cfg)
    estado = base.state_dict()
    for clave, tensor in list(estado.items()):
        if "class_embed." in clave and tensor.shape[0] == base.config.num_labels:
            filas = indices.copy()
            # El canal 0 de la cabeza de propuestas predice presencia de objeto.
            if cfg.two_stage and int(clave.split(".")[-2]) == cfg.decoder_layers:
                filas[0] = 0
            estado[clave] = tensor[filas]
    nuevo.load_state_dict(estado, strict=True)
    nuevo.requires_grad_(False)
    nuevo.model.decoder.layers[-1].requires_grad_(True)
    ultimo = cfg.decoder_layers - 1
    nuevo.class_embed[ultimo].requires_grad_(True)
    nuevo.bbox_embed[ultimo].requires_grad_(True)
    if hasattr(nuevo.model, "query_position_embeddings"):
        nuevo.model.query_position_embeddings.requires_grad_(True)
    return nuevo


def epoca(modelo, datos, dispositivo, optim=None, scaler=None):
    entrenando = optim is not None
    modelo.train(entrenando)
    if not any(p.requires_grad for p in modelo.model.backbone.parameters()):
        modelo.model.backbone.eval()
    total = 0.
    for lote in tqdm(datos, desc="Entrenar" if entrenando else "Validar"):
        labels = [{k:v.to(dispositivo) for k,v in a.items()} for a in lote.pop("labels")]
        with torch.set_grad_enabled(entrenando), torch.autocast(
                dispositivo.type, enabled=dispositivo.type == "cuda", dtype=torch.float16):
            salida = modelo(**{k:v.to(dispositivo) for k,v in lote.items()}, labels=labels)
            loss = salida.loss
        if not math.isfinite(loss.item()):
            raise RuntimeError("Pérdida no finita. Revisa el lote y los hiperparámetros.")
        if entrenando:
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.)
            scaler.step(optim)
            scaler.update()
        total += loss.item() * len(labels)
    return total / len(datos.dataset)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for nombre, valor in [("train",512),("val",128),("epochs",5),("batch",2),
                          ("workers",2),("seed",42)]:
        p.add_argument("--"+nombre, type=int, default=valor)
    p.add_argument("--modo", choices=("parcial","completo"), default="parcial")
    args = p.parse_args()
    if min(args.train,args.val,args.epochs,args.batch) < 1 or args.workers < 0:
        p.error("Usa tamaños y épocas positivos y workers >= 0.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", dev)
    raiz = ROOT / "data" / "lvis"
    run = ROOT / "runs" / "detr" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run.mkdir(parents=True)
    processor = AutoImageProcessor.from_pretrained(BASE, size=SIZE)
    modelo = especializar(AutoModelForObjectDetection.from_pretrained(BASE)).to(dev)
    if args.modo == "completo":
        modelo.requires_grad_(True)
    (run/"configuracion.json").write_text(json.dumps(dict(vars(args),
        inicializacion="facebook/deformable-detr-detic", dataset="LVIS v1"), indent=2))
    parametros = [p for p in modelo.parameters() if p.requires_grad]
    n = sum(p.numel() for p in parametros)
    total = sum(p.numel() for p in modelo.parameters())
    print(f"Entrenables: {n:,}/{total:,} ({100*n/total:.2f} %)")
    loaders = {}
    for split, limite in [("train", args.train), ("val", args.val)]:
        registros = preparar(raiz, split, limite, args.seed)
        counts = [0]*len(CLASES)
        for r in registros:
            for a in r["objetos"]:
                counts[a["category_id"]] += 1
        (run / f"{split}.json").write_text(json.dumps({
            "ids":[r["id"] for r in registros],"cajas":dict(zip(CLASES,counts))}), encoding="utf8")
        ausentes = [n for n,c in zip(CLASES,counts) if not c]
        print(split, len(registros), "imágenes. Clases sin ejemplos:", ausentes)
        loaders[split] = DataLoader(Fotos(registros, raiz), batch_size=args.batch,
            shuffle=split=="train", num_workers=args.workers, collate_fn=Lotes(processor))
    optim = torch.optim.AdamW(parametros, lr=1e-5 if args.modo == "completo" else 1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type=="cuda")
    mejor = float("inf")
    historial = []
    for i in range(args.epochs):
        train = epoca(modelo, loaders["train"], dev, optim, scaler)
        val = epoca(modelo, loaders["val"], dev)
        historial.append(dict(epoca=i+1,train_loss=train,val_loss=val))
        (run/"historial.json").write_text(json.dumps(historial,indent=2))
        print(f"Época {i+1}: train={train:.4f}, val={val:.4f}")
        if val < mejor:
            mejor = val
            modelo.save_pretrained(run/"mejor")
            processor.save_pretrained(run/"mejor")
    puntero = ROOT/"models"/"detr_actual.txt"
    puntero.parent.mkdir(exist_ok=True)
    tmp = run/"seleccion.tmp"
    tmp.write_text((run/"mejor").relative_to(ROOT).as_posix())
    tmp.replace(puntero)
    print("Modelo listo para FruitVision:", run/"mejor")


if __name__ == "__main__":
    main()
