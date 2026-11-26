from __future__ import annotations

import argparse
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from fruitvision.detection import detect_pretrained_image, load_pretrained_detector  # noqa: E402
from fruitvision.model_catalog import DETECTORS  # noqa: E402
from fruitvision.panel import dibujar_panel, resumen
from fruitvision.utils import write_json_atomic  # noqa: E402


class SharedFrames:
    """Single-slot, latest-frame exchange. Intermediate camera frames are discarded."""
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.detections: list[dict[str, Any]] = []
        self.detection_frame_id = 0
        self.detection_time = 0.0
        self.inference_ms = 0.0
        self.fps = 0.0
        self.error: str | None = None

    def publish_frame(self, frame) -> None:
        with self.lock:
            self.frame = frame
            self.frame_id += 1

    def snapshot_frame(self):
        with self.lock:
            return (None if self.frame is None else self.frame.copy(), self.frame_id)

    def publish_detections(self, detections, frame_id: int, elapsed_ms: float) -> None:
        with self.lock:
            self.detections = detections
            self.detection_frame_id = frame_id
            self.detection_time = time.perf_counter()
            self.inference_ms = elapsed_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FruitVision latest-frame live camera inference")
    parser.add_argument("--model-key", required=True, choices=sorted(DETECTORS))
    parser.add_argument("--control", type=Path)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--inference-interval", type=float, default=0.18,
                        help="Minimum seconds between predictions. 0 means as fast as inference allows.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-box-age", type=float, default=0.9,
                        help="Hide boxes older than this amount of seconds to avoid displaying stale positions.")
    return parser.parse_args()


def draw_overlay(frame, detections, *, age_seconds: float | None, inference_ms: float, confidence: float):
    output = frame.copy()
    height, width = output.shape[:2]
    thickness = max(2, round(width / 640))
    font_scale = max(0.33, min(0.48, width / 3000))
    for detection in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in detection["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(output, (x1, y1), (x2, y2), (40, 220, 80), thickness)
        cx, cy = detection["center_x_px"], detection["center_y_px"]
        cv2.circle(output, (round(cx), round(cy)), 2, (0, 255, 255), -1)
        caption = f"{detection['label']} {detection['score']:.1%} ({cx:.0f},{cy:.0f}) {detection['width_px']:.0f}x{detection['height_px']:.0f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        left = max(0, min(x1, width - text_width - 8))
        top = max(0, y1 - text_height - baseline - 8)
        cv2.rectangle(output, (left, top), (left + text_width + 8, top + text_height + baseline + 6), (20, 120, 50), -1)
        cv2.putText(output, caption, (left + 4, top + text_height + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(output, (0, 0), (width, 74), (24, 24, 24), -1)
    cv2.putText(output, f"DETR LIVE | objetos: {len(detections)} | inferencia: {inference_ms:.0f} ms",
                (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)
    age_label = "--" if age_seconds is None else f"{age_seconds * 1000:.0f} ms"
    cv2.putText(output, f"Actualizacion: {age_label} | conf >= {confidence:.0%} | Q: salir | S: guardar captura",
                (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (205, 205, 205), 1, cv2.LINE_AA)
    return output


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence debe estar entre 0 y 1")
    if args.inference_interval < 0:
        raise SystemExit("--inference-interval no puede ser negativo")

    print(f"Cargando detector {args.model_key}...", flush=True)
    detector = load_pretrained_detector(args.model_key)
    print(f"Detector cargado en {detector.device}. Iniciando cámara {args.camera}...", flush=True)

    if os.name == "nt":
        camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(args.camera, cv2.CAP_ANY)
    else:
        camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara con índice {args.camera}. Comprueba permisos y que no esté ocupada.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    shared = SharedFrames()
    stop_event = threading.Event()

    def capture_loop() -> None:
        previous = time.perf_counter()
        frame_count = 0
        try:
            while not stop_event.is_set():
                ok, frame = camera.read()
                if not ok:
                    shared.error = "La cámara dejó de entregar fotogramas."
                    time.sleep(0.02)
                    continue
                shared.publish_frame(frame)
                frame_count += 1
                now = time.perf_counter()
                if now - previous >= 1.0:
                    with shared.lock:
                        shared.fps = frame_count / (now - previous)
                    frame_count = 0
                    previous = now
        except Exception as exc:
            shared.error = f"Error de captura: {exc}"
            stop_event.set()

    errores_hilos = []
    active_confidence = args.confidence

    def inference_loop() -> None:
        last_frame_id = -1
        last_inference_start = 0.0
        try:
            while not stop_event.is_set():
                now = time.perf_counter()
                if args.inference_interval and now - last_inference_start < args.inference_interval:
                    time.sleep(min(0.01, args.inference_interval / 4))
                    continue
                frame, frame_id = shared.snapshot_frame()
                if frame is None or frame_id == last_frame_id:
                    time.sleep(0.002)
                    continue

                # This is intentionally a snapshot of the newest frame, not a FIFO queue.
                # Frames captured while inference runs replace the previous snapshot.
                last_frame_id = frame_id
                last_inference_start = time.perf_counter()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                started = time.perf_counter()
                predictions = detect_pretrained_image(pil_image, detector, threshold=active_confidence)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                shared.publish_detections(predictions, frame_id, elapsed_ms)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            errores_hilos.append(exc)
            shared.error = f"Error de inferencia: {type(exc).__name__}: {exc}"
            stop_event.set()

    capture_thread = threading.Thread(target=capture_loop, name="FruitVisionCapture", daemon=True)
    inference_thread = threading.Thread(target=inference_loop, name="FruitVisionInference", daemon=True)
    capture_thread.start()
    inference_thread.start()

    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    capture_dir = ROOT / "outputs" / "live_sessions" / session_stamp / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)
    cv2.namedWindow("DETR LIVE - Q salir / S captura", cv2.WINDOW_NORMAL)
    print("Cámara activa. Q/Esc cierra; S guarda la captura actual.", flush=True)

    next_control_check = 0.0
    last_display_tick = time.perf_counter()
    try:
        while not stop_event.is_set():
            frame, frame_id = shared.snapshot_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            with shared.lock:
                detections = list(shared.detections)
                detection_time = shared.detection_time
                detection_frame_id = shared.detection_frame_id
                inference_ms = shared.inference_ms
                camera_fps = shared.fps
                error = shared.error
            now = time.perf_counter()
            if args.control and now >= next_control_check:
                next_control_check = now + 0.15
                try:
                    new_value = float(json.loads(args.control.read_text(encoding="utf-8"))["confidence"])
                    if 0 <= new_value <= 1:
                        active_confidence = new_value
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            age = now - detection_time if detection_time else float("inf")
            visible_detections = [d for d in detections if d["score"] >= active_confidence] if age <= args.max_box_age else []
            preview = draw_overlay(
                frame, visible_detections,
                age_seconds=age if age != float("inf") else None,
                inference_ms=inference_ms,
                confidence=active_confidence,
            )
            if age > args.max_box_age:
                cv2.putText(preview, "Esperando nueva deteccion...", (16, 105), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 210, 255), 2, cv2.LINE_AA)
            cv2.putText(preview, f"Camara: {camera_fps:.1f} FPS | fotograma #{frame_id}",
                        (16, preview.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            if error:
                cv2.putText(preview, error[:100], (16, 138), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1, cv2.LINE_AA)
            alimentos = resumen(visible_detections)
            preview = np.hstack((dibujar_panel(alimentos, preview.shape[0]), preview))
            cv2.imshow("DETR LIVE - Q salir / S captura", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                capture_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                original_path = capture_dir / f"{capture_stamp}_original.jpg"
                annotated_path = capture_dir / f"{capture_stamp}_annotated.jpg"
                metadata_path = capture_dir / f"{capture_stamp}.json"
                cv2.imwrite(str(original_path), frame)
                cv2.imwrite(str(annotated_path), preview)
                write_json_atomic(metadata_path, {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "model_key": args.model_key,
                    "model_label": detector.label,
                    "confidence_threshold": active_confidence,
                    "camera_index": args.camera,
                    "camera_frame_id": frame_id,
                    "inference_frame_id": detection_frame_id,
                    "detection_age_ms": round(age * 1000, 2) if age != float("inf") else None,
                    "inference_ms": round(inference_ms, 2),
                    "camera_fps": round(camera_fps, 2),
                    "detections": visible_detections,
                    "calorias_por_alimento": [dict(alimento=n, cantidad=c, kcal=k) for n,c,k in alimentos],
                    "kcal_subtotal": sum(k for _,_,k in alimentos if k is not None),
                    "kcal_total_completo": all(k is not None for _,_,k in alimentos),
                    "warning": "Captura en vivo: los resultados de detección corresponden al último fotograma procesado, no necesariamente al fotograma más reciente.",
                })
                print(f"Captura guardada: {original_path.relative_to(ROOT)}", flush=True)

            # Avoid a busy-spin in HighGUI while keeping the preview responsive.
            elapsed = time.perf_counter() - last_display_tick
            if elapsed < 0.001:
                time.sleep(0.001 - elapsed)
            last_display_tick = time.perf_counter()
    finally:
        stop_event.set()
        capture_thread.join(timeout=2.0)
        inference_thread.join(timeout=2.0)
        camera.release()
        cv2.destroyAllWindows()
        print("Sesión de cámara finalizada.", flush=True)
    return 1 if errores_hilos else 0


if __name__ == "__main__":
    raise SystemExit(main())
