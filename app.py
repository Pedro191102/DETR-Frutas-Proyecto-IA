from __future__ import annotations

import hashlib
import gc
import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from fruitvision.detection import detect_pretrained_image, load_pretrained_detector  # noqa: E402
from fruitvision.estimation import load_reference, estimate_selected_rows  # noqa: E402
from fruitvision.model_catalog import DETECTORS, get_detector_spec  # noqa: E402
from fruitvision.utils import write_json_atomic  # noqa: E402

st.set_page_config(page_title="DETR", page_icon="🍎", layout="wide")
st.title("DETR")
st.caption("Detección de frutas y verduras con modelos preentrenados, cámara en vivo y análisis de resultados")


@st.cache_resource(show_spinner=False)
def get_detector(model_key: str):
    return load_pretrained_detector(model_key)


def release_cached_detectors() -> None:
    """Free Streamlit's cached model copies before starting the camera subprocess."""
    get_detector.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def draw_detections(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", max(10, min(16, canvas.width // 110)))
    except OSError:
        font = ImageFont.load_default()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle((x1, y1, x2, y2), outline=(0, 220, 90), width=max(2, canvas.width // 350))
        cx, cy = (x1+x2)/2, (y1+y2)/2
        draw.ellipse((cx-2, cy-2, cx+2, cy+2), fill="yellow")
        label = f"{det['label']} {det['score']:.1%}  ({cx:.0f},{cy:.0f}) {x2-x1:.0f}x{y2-y1:.0f}"
        text_top = max(0, y1 - 17)
        left = max(0, min(x1, canvas.width - draw.textlength(label, font=font)))
        box = draw.textbbox((left, text_top), label, font=font)
        draw.rectangle(box, fill=(0, 100, 45))
        draw.text((left, text_top), label, fill="white", font=font)
    return canvas


def safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if value is None or (isinstance(value, float) and pd.isna(value)):
                item[key] = None
            elif hasattr(value, "item"):
                item[key] = value.item()
            else:
                item[key] = value
        cleaned.append(item)
    return cleaned


def save_single_test(
    image: Image.Image,
    original_bytes: bytes,
    original_name: str,
    detections: list[dict[str, Any]],
    reviewed_rows: list[dict[str, Any]],
    *,
    model_key: str,
    threshold: float,
    source_url: str = "",
) -> Path:
    tests_root = ROOT / "outputs" / "inference_tests"
    tests_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(original_name).stem).strip("._-") or "image"
    image_hash = hashlib.sha256(original_bytes).hexdigest()
    run_dir = tests_root / f"{stamp}_{safe_name}_{image_hash[:8]}_{uuid.uuid4().hex[:4]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    suffix = Path(original_name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".img"
    (run_dir / f"original{suffix}").write_bytes(original_bytes)
    draw_detections(image, detections).save(run_dir / "predictions.png")
    write_json_atomic(run_dir / "detections.json", detections)
    clean_review = clean_records(reviewed_rows)
    reviewed_visual = []
    for row in clean_review:
        if not row.get("Incluir", False) or row.get("Estado") == "Dudosa":
            continue
        try:
            box = [float(row[k]) for k in ("X1_px", "Y1_px", "X2_px", "Y2_px")]
        except (TypeError, ValueError, KeyError):
            continue
        if box[2] > box[0] and box[3] > box[1]:
            reviewed_visual.append({
                "bbox": box,
                "label": str(row.get("Clase", "")),
                "score": float(row.get("Confianza", 0.0)) / 100.0,
            })
    draw_detections(image, reviewed_visual).save(run_dir / "reviewed_predictions.png")
    write_json_atomic(run_dir / "reviewed_detections.json", clean_review)
    pd.DataFrame(clean_review).to_csv(run_dir / "reviewed_detections.csv", index=False, encoding="utf-8-sig")
    spec = get_detector_spec(model_key)
    write_json_atomic(run_dir / "test_manifest.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "original_filename": original_name,
        "image_sha256": image_hash,
        "image_width": image.width,
        "image_height": image.height,
        "model_key": model_key,
        "model_label": spec.label,
        "model_repository": spec.repo_id,
        "model_license": spec.license,
        "source_url": source_url.strip() or None,
        "confidence_threshold": threshold,
        "raw_detection_count": len(detections),
        "reviewed_detection_count": len(clean_review),
        "is_ground_truth": False,
    })
    return run_dir


def run_and_time(image: Image.Image, model_key: str, threshold: float):
    detector = get_detector(model_key)
    inference_started = time.perf_counter()
    detections = detect_pretrained_image(image, detector, threshold=threshold)
    elapsed = (time.perf_counter() - inference_started) * 1000.0
    return detector, detections, elapsed


def result_table(detections: list[dict[str, Any]], image_size: tuple[int, int]) -> pd.DataFrame:
    width, height = image_size
    rows = []
    for index, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["bbox"]
        rows.append({
            "ID": index,
            "Clase": det["label"],
            "Confianza": round(det["score"] * 100, 1),
            "X1_px": round(x1, 1),
            "Y1_px": round(y1, 1),
            "X2_px": round(x2, 1),
            "Y2_px": round(y2, 1),
            "Centro_X_px": round(det["center_x_px"], 1),
            "Centro_Y_px": round(det["center_y_px"], 1),
            "Ancho_px": round(x2 - x1, 1),
            "Alto_px": round(y2 - y1, 1),
            "Incluir": True,
            "Estado": "Completa",
            "Fracción": None,
            "Peso_manual_g": None,
        })
    return pd.DataFrame(rows, columns=[
        "ID", "Clase", "Confianza", "X1_px", "Y1_px", "X2_px", "Y2_px", "Centro_X_px", "Centro_Y_px", "Ancho_px", "Alto_px",
        "Incluir", "Estado", "Fracción", "Peso_manual_g",
    ])


with st.sidebar:
    st.header("Detector")
    model_keys = list(DETECTORS)
    model_labels = {key: DETECTORS[key].label for key in model_keys}
    proceso_actual = st.session_state.get("live_camera_process")
    camara_activa = proceso_actual is not None and proceso_actual.poll() is None
    selected_model_key = st.selectbox(
        "Modelo de detección",
        model_keys,
        disabled=camara_activa,
        index=model_keys.index("detr_propio" if "detr_propio" in model_keys else "detr_lvis"),
        format_func=lambda key: model_labels[key],
        help="Esta selección elige pesos y etiquetas del detector; no selecciona un dataset de entrenamiento.",
    )
    if camara_activa:
        st.caption("Detén la cámara para cambiar de detector.")
    selected_spec = get_detector_spec(selected_model_key)
    st.caption(selected_spec.description)
    confidence = st.slider(
        "Confianza mínima", 1, 100, 25, 1, format="%d%%",
        help="Un umbral bajo recupera más candidatos, pero puede producir falsos positivos. Revisa las cajas y etiquetas antes de aceptarlas.",
    )
    confidence /= 100
    active = st.session_state.get("live_camera_process")
    if active is not None and active.poll() is None:
        write_json_atomic(st.session_state["camera_control"], {"confidence": confidence})
    workflow = st.radio(
        "Área de trabajo",
        ["Cámara en vivo", "Prueba de imagen"],
        index=0,
        key="workflow_area_v110",
    )
    st.divider()
    st.caption(selected_spec.dataset)
    st.caption(f"Licencia: {selected_spec.license}")

if workflow == "Cámara en vivo":
    st.subheader("Cámara en vivo")
    st.write("El vídeo se captura continuamente. La inferencia trabaja sobre el fotograma más reciente disponible y descarta los intermedios: así no se forma una cola que vaya acumulando retraso.")
    st.info("La vista de cámara se abrirá en una ventana OpenCV de Windows. Selecciona el detector antes de iniciarla.")

    col1, col2, col3 = st.columns(3)
    with col1:
        inference_interval = st.select_slider(
            "Intervalo mínimo entre inferencias",
            options=[0.0, 0.10, 0.15, 0.18, 0.25, 0.33, 0.50],
            value=0.18,
            format_func=lambda value: "Máxima velocidad" if value == 0 else f"{value:.2f} s",
            help="0 permite inferir tan rápido como el modelo pueda. 0.18 s limita la inferencia a unas 5–6 actualizaciones/s.",
        )
    with col2:
        camera_index = st.number_input("Índice de cámara", min_value=0, max_value=10, value=0, step=1)
    with col3:
        resolution = st.selectbox("Resolución de captura", ["640 × 480", "1280 × 720", "1920 × 1080"], index=1)
    width, height = [int(part.strip()) for part in resolution.replace("×", "x").split("x")]

    process = st.session_state.get("live_camera_process")
    running = process is not None and process.poll() is None
    if running:
        st.success(f"Cámara activa · {st.session_state['camera_label']} · umbral solicitado {confidence:.0%}")
        st.caption("Pulsa Q o Esc en la ventana de cámara para salir, o detén el proceso desde aquí.")
        if st.button("Detener cámara", type="primary"):
            process.terminate()
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                process.kill()
            st.session_state["camera_control"].unlink(missing_ok=True)
            st.session_state.pop("live_camera_process", None)
            st.rerun()
    else:
        if process is not None:
            if process.returncode:
                st.error("La cámara terminó con un error. Abre el registro de esta sesión para ver la causa.")
            st.session_state["camera_control"].unlink(missing_ok=True)
        if st.button("Cargar modelo e iniciar cámara", type="primary"):
            script = ROOT / "scripts" / "live_camera.py"
            log_path = ROOT / "logs" / "live_camera.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            control = ROOT / "logs" / f"control_{uuid.uuid4().hex}.json"
            write_json_atomic(control, {"confidence": confidence})
            args = [
                sys.executable, str(script),
                "--model-key", selected_model_key,
                "--confidence", str(confidence), "--control", str(control),
                "--inference-interval", str(inference_interval),
                "--camera", str(int(camera_index)),
                "--width", str(width), "--height", str(height),
            ]
            try:
                # The camera worker loads the selected detector itself. Release any
                # image/comparison models cached by this Streamlit process first.
                release_cached_detectors()
                log_handle = log_path.open("w", encoding="utf-8")
                kwargs = {"cwd": str(ROOT), "stdout": log_handle, "stderr": subprocess.STDOUT}
                if os.name == "nt":
                    kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    kwargs["start_new_session"] = True
                process = subprocess.Popen(args, **kwargs)
                log_handle.close()
                st.session_state["camera_control"] = control
                st.session_state["camera_label"] = selected_spec.label
                st.session_state["live_camera_process"] = process
                st.success("Proceso de cámara iniciado. La descarga del modelo puede tardar la primera vez.")
                st.rerun()
            except Exception as exc:
                st.error(f"No se pudo iniciar la cámara: {exc}")

    log_path = ROOT / "logs" / "live_camera.log"
    if log_path.is_file():
        with st.expander("Registro de la última sesión"):
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            st.code("\n".join(lines[-35:]) or "Sin registros todavía.")
    st.markdown("**Controles de la ventana de cámara:** `Q` / `Esc` salir · `S` guardar una fotografía original, otra anotada y su JSON.")
    st.markdown("**Estrategia anti-retraso:** un hilo lee la cámara continuamente; otro procesa solo el fotograma nuevo más reciente. Si la inferencia tarda, se descartan fotogramas intermedios en vez de encolarlos.")
    st.stop()

# Single-image workflow
st.subheader("Prueba individual de imágenes")
st.write("Sube una fotografía. El detector seleccionado ejecutará inferencia sin entrenar ni modificar sus pesos.")
source_url = st.text_input("URL de procedencia (opcional)", placeholder="https://sitio-web/imagen.jpg")
uploaded = st.file_uploader("Selecciona una fotografía", type=["jpg", "jpeg", "png", "webp"], key="single_upload")
if uploaded is None:
    st.info("Selecciona una imagen para ejecutar la inferencia.")
    st.stop()

original_bytes = uploaded.getvalue()
try:
    image = Image.open(__import__("io").BytesIO(original_bytes)).convert("RGB")
except Exception as exc:
    st.error(f"No se pudo abrir la imagen: {exc}")
    st.stop()

with st.spinner(f"Cargando {selected_spec.label} y analizando la imagen..."):
    try:
        detector, detections, inference_ms = run_and_time(image, selected_model_key, confidence)
    except Exception as exc:
        st.error(f"No se pudo ejecutar el detector: {type(exc).__name__}: {exc}")
        st.stop()
overlay = draw_detections(image, detections)

left, right = st.columns([1.35, 1])
with left:
    st.subheader("Resultado visual")
    st.image(overlay, use_container_width=True)
    st.caption(f"{uploaded.name} · {image.width} × {image.height} px")
with right:
    st.subheader("Resumen")
    st.metric("Objetos detectados", len(detections))
    st.metric("Categorías distintas", len({item['label'] for item in detections}))
    st.metric("Tiempo de inferencia", f"{inference_ms:.1f} ms")
    device_label = (f"GPU CUDA {detector.device}" if str(detector.device).isdigit()
                    else str(detector.device).upper())
    st.caption(f"Ejecutado en {device_label} · {selected_spec.label}")

st.divider()
st.subheader("Revisión manual")
st.caption("Corrige clases o coordenadas, excluye objetos y registra el estado/peso. Los cambios se guardan separados de la predicción original.")
review_df = result_table(detections, image.size)
if not review_df.empty:
    edited = st.data_editor(
        review_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "ID": st.column_config.NumberColumn(disabled=True, width="small"),
            "Confianza": st.column_config.NumberColumn(disabled=True, format="%.1f%%"),
            "Centro_X_px": st.column_config.NumberColumn(disabled=True),
            "Centro_Y_px": st.column_config.NumberColumn(disabled=True),
            "Ancho_px": st.column_config.NumberColumn(disabled=True),
            "Alto_px": st.column_config.NumberColumn(disabled=True),
            "Incluir": st.column_config.CheckboxColumn(default=True),
            "Estado": st.column_config.SelectboxColumn(options=["Completa", "Parcial", "Cortada", "Dudosa"], required=True),
            "Fracción": st.column_config.NumberColumn(min_value=0.01, max_value=1.0, step=0.05),
            "Peso_manual_g": st.column_config.NumberColumn(min_value=0.1, step=1.0),
        },
        key=f"review_{hashlib.sha256(original_bytes).hexdigest()[:12]}_{selected_model_key}",
    )
else:
    edited = review_df

reviewed_rows = edited.to_dict(orient="records")
for row in reviewed_rows:
    for axis in ("X", "Y"):
        first, last = (safe_float(row.get(f"{axis}{i}_px")) for i in (1, 2))
        row[f"Centro_{axis}_px"] = (first + last) / 2 if first is not None and last is not None else None
    row["Ancho_px"] = max(0.0, (safe_float(row.get("X2_px")) or 0) - (safe_float(row.get("X1_px")) or 0))
    row["Alto_px"] = max(0.0, (safe_float(row.get("Y2_px")) or 0) - (safe_float(row.get("Y1_px")) or 0))
accepted_rows = []
reviewed_visual = []
for row in reviewed_rows:
    if not bool(row.get("Incluir", False)) or row.get("Estado") == "Dudosa":
        continue
    try:
        x1, y1, x2, y2 = [float(row[key]) for key in ("X1_px", "Y1_px", "X2_px", "Y2_px")]
    except (TypeError, ValueError, KeyError):
        continue
    if x2 <= x1 or y2 <= y1:
        continue
    reviewed_visual.append({
        "bbox": [x1, y1, x2, y2],
        "label": str(row.get("Clase", "")),
        "score": float(row.get("Confianza", 0)) / 100,
    })
    accepted_rows.append({
        "label": str(row.get("Clase", "")).strip(),
        "include": True,
        "portion_status": row.get("Estado", "Completa"),
        "portion_fraction": safe_float(row.get("Fracción")),
        "manual_weight_g": safe_float(row.get("Peso_manual_g")),
    })

counts = Counter(row["label"] for row in accepted_rows)
st.subheader("Conteo revisado")
if counts:
    st.dataframe(pd.DataFrame([{"Producto": name, "Cantidad": count} for name, count in sorted(counts.items())]), use_container_width=True, hide_index=True)
else:
    st.info("Todavía no hay detecciones incluidas en el conteo revisado.")
st.subheader("Vista con revisión aplicada")
st.image(draw_detections(image, reviewed_visual), use_container_width=True)

try:
    reference = load_reference(ROOT / "config" / "nutrition_reference.csv")
    estimates = estimate_selected_rows(accepted_rows, reference)
except Exception as exc:
    st.error(f"No se pudo leer la tabla nutricional: {exc}")
    estimates = None

st.subheader("Estimación complementaria")
st.caption("No se deduce el peso a partir de píxeles. Se usan pesos medios de referencia o valores manuales.")
if estimates is not None:
    c1, c2 = st.columns(2)
    with c1:
        if estimates["weight_complete"]:
            st.metric("Peso estimado total", f"{estimates['total_weight_g']:.1f} g")
        else:
            st.metric("Subtotal de pesos conocidos", f"{estimates['known_weight_subtotal_g']:.1f} g")
            st.warning("Falta peso de referencia, fracción o peso manual para uno o más objetos.")
    with c2:
        st.metric("Objetos incluidos", len(accepted_rows))
    nutrient_names = {
        "kcal_100g": "Energía (kcal)", "protein_g_100g": "Proteínas (g)",
        "fat_g_100g": "Grasas (g)", "carbs_g_100g": "Carbohidratos (g)",
        "fiber_g_100g": "Fibra (g)", "water_g_100g": "Agua (g)",
    }
    st.dataframe(pd.DataFrame([
        {"Nutriente": label, "Estimación": round(estimates["nutrients"][key], 2)
         if estimates["nutrients"][key] is not None else "Sin datos completos"}
        for key, label in nutrient_names.items()
    ]), use_container_width=True, hide_index=True)
    if not reference["weight_avg_g"].notna().any():
        st.info("La plantilla nutricional está vacía. Complétala con valores documentados de TPCA/USDA y pesos promedio medidos.")

if st.button("Guardar prueba de imagen", type="primary"):
    saved_dir = save_single_test(
        image, original_bytes, uploaded.name, detections, reviewed_rows,
        model_key=selected_model_key, threshold=confidence, source_url=source_url,
    )
    st.success(f"Prueba guardada en `{saved_dir.relative_to(ROOT)}`")
