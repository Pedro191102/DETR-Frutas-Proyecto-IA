"""Ajuste supervisado de YOLOE-26s con cajas YOLO."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="YAML del dataset anotado")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error("No se encuentra el YAML del dataset.")
    if min(args.epochs, args.batch, args.imgsz) < 1:
        parser.error("epochs, batch e imgsz deben ser positivos.")
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPETrainer

    model = YOLOE("yoloe-26s.yaml").load("yoloe-26s-seg.pt")
    model.train(data=str(args.data.resolve()), trainer=YOLOEPETrainer,
                epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                device=args.device, workers=0, project=str(ROOT / "runs"),
                name="ajuste", exist_ok=False)
    print("Mejor checkpoint:", model.trainer.best)


if __name__ == "__main__":
    main()
