from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from .model_catalog import CUSTOM_WEIGHTS, get_detector_spec


@dataclass
class LoadedDetector:
    key: str
    label: str
    backend: str
    model: Any
    processor: Any | None
    device: str
    names: dict[int, str]


def _normalise_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        output = {}
        for key, value in names.items():
            try:
                output[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return output
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def load_pretrained_detector(key: str, device: str | None = None) -> LoadedDetector:
    spec = get_detector_spec(key)
    if spec.backend == "detr":
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(device).isdigit():
            device = f"cuda:{device}"
        processor = AutoImageProcessor.from_pretrained(spec.repo_id, local_files_only=True)
        model = AutoModelForObjectDetection.from_pretrained(spec.repo_id, local_files_only=True).to(device).eval()
        return LoadedDetector(key, spec.label, "detr", model, processor, device,
                              _normalise_names(model.config.id2label))
    from ultralytics import YOLOE
    if spec.prompt_classes:
        model = YOLOE(spec.repo_id)
        model.set_classes(list(spec.prompt_classes))
    else:
        model = YOLOE(str(CUSTOM_WEIGHTS))
    names = _normalise_names(model.names)
    device = device or ("0" if torch.cuda.is_available() else "cpu")
    return LoadedDetector(key, spec.label, spec.backend, model, None, device, names)


@torch.inference_mode()
def detect_pretrained_image(
    image: Image.Image,
    detector: LoadedDetector,
    threshold: float = 0.25,
) -> list[dict[str, Any]]:
    """Return detections in the shared FruitVision schema."""
    image = image.convert("RGB")
    found: list[dict[str, Any]] = []

    if detector.backend == "detr":
        from .lvis_clases import CLASES
        inputs = detector.processor(images=image, return_tensors="pt").to(detector.device)
        outputs = detector.model(**inputs)
        # Seleccionar alimentos antes del top-k evita que otras clases ocupen sus plazas.
        ids = [i for i, name in detector.names.items() if name in CLASES]
        from torchvision.ops import box_convert, nms
        # Una clase por consulta y una sola caja por pieza para el conteo energético.
        scores, class_indices = outputs.logits[0, :, ids].sigmoid().max(dim=-1)
        accepted = scores >= threshold
        scores = scores[accepted]
        classes = class_indices[accepted]
        boxes = box_convert(outputs.pred_boxes[0, accepted], "cxcywh", "xyxy").clamp(0, 1)
        keep = nms(boxes, scores, 0.75)
        boxes = (boxes[keep] * boxes.new_tensor([image.width, image.height]*2)).tolist()
        scores = scores[keep].tolist()
        labels = [ids[i] for i in classes[keep].tolist()]
    else:
        result = detector.model.predict(source=image, conf=float(threshold), imgsz=960,
                                       device=detector.device, verbose=False)[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.tolist()
        scores = result.boxes.conf.tolist()
        labels = result.boxes.cls.tolist()
    for box, score, label_value in zip(boxes, scores, labels):
        if not threshold <= score <= 1.0:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        if x2 <= x1 or y2 <= y1:
            continue
        label_id = int(label_value)
        found.append({
            "label_id": label_id,
            "label": detector.names.get(label_id, f"class_{label_id}"),
            "score": float(score),
            "bbox": [x1, y1, x2, y2],
            "center_x_px": (x1 + x2) / 2,
            "center_y_px": (y1 + y2) / 2,
            "confidence_pct": float(score) * 100,
            "width_px": x2 - x1,
            "height_px": y2 - y1,
            "alternatives": [],
        })
    found.sort(key=lambda item: item["score"], reverse=True)
    return found


