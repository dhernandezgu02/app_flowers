#!/usr/bin/env python3
"""
Captura fotos desde la cámara de la Jetson y las sube a Google Cloud Storage.

Modos de operación:
    interval  - captura cada N segundos (comportamiento original)
    conveyor  - detecta flores en banda transportadora y captura solo la foto
                más nítida de cada flor (sin motion-blur)

Uso:
    python capture_upload.py                              # intervalo cada 3s
    python capture_upload.py --mode conveyor              # modo banda inteligente
    python capture_upload.py --mode conveyor --debug      # muestra ventana en vivo
    python capture_upload.py --mode conveyor --exposure 5 # exposure manual (ms)
    python capture_upload.py --camera-type csi            # cámara CSI Jetson
"""

import argparse
import datetime
import os
import sys
import time

import cv2
import numpy as np
from google.cloud import storage
from google.oauth2 import service_account

# ── Configuración ────────────────────────────────────────────────────────────
CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), "green-alchemy-301821-9753e8366e05.json")
BUCKET_NAME      = "bucket_flower"
GCS_FOLDER       = "fotos"
CAPTURE_INTERVAL = 3          # segundos entre capturas (modo interval)
NUM_FOTOS        = 5          # fotos totales (modo interval)
CAMERA_INDEX     = 0
IMAGE_FORMAT     = "jpg"
JPEG_QUALITY     = 92

# ── Parámetros modo conveyor ──────────────────────────────────────────────────
# Zona central (ROI) donde se evalúa si hay una flor: fracción del frame
ROI_X1, ROI_Y1 = 0.25, 0.20   # esquina superior-izquierda  (25 %, 20 %)
ROI_X2, ROI_Y2 = 0.75, 0.80   # esquina inferior-derecha    (75 %, 80 %)

MOTION_THRESHOLD  = 25         # cambio de pixel para detección de movimiento
MOTION_MIN_AREA   = 3000       # área mínima (px²) para considerar que hay objeto
BURST_FRAMES      = 12         # frames a capturar en ráfaga cuando se detecta flor
SHARPNESS_MIN     = 80.0       # varianza Laplaciana mínima para aceptar foto
COOLDOWN_SECS     = 2.0        # segundos de espera entre capturas de flores
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  Cámara
# ══════════════════════════════════════════════════════════════════════════════

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
    exposure_time=8000,     # microsegundos — valor bajo = exposure corto → menos blur
):
    """Pipeline GStreamer para cámara CSI en Jetson (IMX219 / IMX477).
    exposure_time en µs: 8000 ≈ 8 ms  →  congela objetos a ~1 m/s con margen.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} "
        f"exposuretimerange='{exposure_time} {exposure_time}' "
        f"gainrange='4 4' ispdigitalgainrange='1 1' ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
        f"format=NV12, framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )


def open_camera(camera_type: str, exposure_ms: float = 0) -> cv2.VideoCapture:
    if camera_type == "csi":
        exposure_us = int(exposure_ms * 1000) if exposure_ms > 0 else 8000
        pipeline = gstreamer_pipeline(exposure_time=exposure_us)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        if exposure_ms > 0:
            # Deshabilita auto-exposición y fija valor manual
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # 1 = manual en V4L2
            cap.set(cv2.CAP_PROP_EXPOSURE, -round(exposure_ms))  # log2 en algunas APIs

    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la cámara (tipo={camera_type}). "
            "Verifica que esté conectada y que tengas permisos."
        )
    return cap


# ══════════════════════════════════════════════════════════════════════════════
#  Nitidez (sharpness)
# ══════════════════════════════════════════════════════════════════════════════

def sharpness_score(frame: np.ndarray, roi: tuple | None = None) -> float:
    """Varianza del Laplaciano en la ROI.
    Valor alto → imagen nítida.  Valor bajo → imagen borrosa / motion-blur.
    """
    if roi:
        x1, y1, x2, y2 = roi
        region = frame[y1:y2, x1:x2]
    else:
        region = frame
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def best_frame_from_burst(frames: list[np.ndarray], roi: tuple | None = None
                          ) -> tuple[np.ndarray, float]:
    """Devuelve el frame más nítido de la ráfaga y su score."""
    scores = [sharpness_score(f, roi) for f in frames]
    best_idx = int(np.argmax(scores))
    return frames[best_idx], scores[best_idx]


# ══════════════════════════════════════════════════════════════════════════════
#  GCS
# ══════════════════════════════════════════════════════════════════════════════

def upload_to_gcs(image_bytes: bytes, filename: str, bucket: storage.Bucket) -> str:
    blob_name = f"{GCS_FOLDER}/{filename}" if GCS_FOLDER else filename
    blob = bucket.blob(blob_name)
    blob.upload_from_string(image_bytes, content_type=f"image/{IMAGE_FORMAT}")
    return f"gs://{bucket.name}/{blob_name}"


def frame_to_bytes(frame: np.ndarray) -> bytes:
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    success, buf = cv2.imencode(f".{IMAGE_FORMAT}", frame, encode_params)
    if not success:
        raise RuntimeError("Error al codificar la imagen.")
    return buf.tobytes()


def build_filename(prefix: str = "foto") -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:22]
    return f"{prefix}_{ts}.{IMAGE_FORMAT}"


# ══════════════════════════════════════════════════════════════════════════════
#  Modo intervalo (original)
# ══════════════════════════════════════════════════════════════════════════════

def run_interval_mode(cap, bucket, num_fotos: int, interval: int):
    print(f"[INFO] Modo intervalo: {num_fotos} fotos cada {interval}s")
    h, w = _frame_size(cap)
    roi = _roi_pixels(w, h)

    for i in range(1, num_fotos + 1):
        # Descarta frames acumulados para que el AE se estabilice
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[WARN] No se pudo leer frame {i}", file=sys.stderr)
        else:
            score = sharpness_score(frame, roi)
            print(f"[{i}/{num_fotos}] Nitidez={score:.1f}", end="  ")
            if score < SHARPNESS_MIN:
                print("RECHAZADA (borrosa) — no se sube")
            else:
                filename = build_filename()
                uri = upload_to_gcs(frame_to_bytes(frame), filename, bucket)
                print(f"Subida: {uri}  ({len(frame_to_bytes(frame))//1024} KB)")

        if i < num_fotos:
            time.sleep(interval)

    print("[INFO] Listo.")


# ══════════════════════════════════════════════════════════════════════════════
#  Modo banda transportadora (conveyor) ← NUEVO
# ══════════════════════════════════════════════════════════════════════════════

def run_conveyor_mode(cap, bucket, debug: bool = False, max_fotos: int = 0):
    """Detecta flores en la banda y sube solo la foto más nítida de cada flor.

    Algoritmo:
    1. Mantiene fondo con MOG2 (robusto a cambios de iluminación).
    2. Cuando detecta objeto en ROI → dispara ráfaga de BURST_FRAMES frames.
    3. Selecciona el frame con mayor varianza Laplaciana (el más nítido).
    4. Si supera SHARPNESS_MIN lo sube a GCS.
    5. Cooldown de COOLDOWN_SECS para no re-disparar por la misma flor.
    """
    print("[INFO] Modo conveyor iniciado. Esperando flores... (Ctrl+C para salir)")

    h, w = _frame_size(cap)
    roi = _roi_pixels(w, h)
    x1, y1, x2, y2 = roi

    # Sustractor de fondo adaptativo (detecta objetos en movimiento)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=120, varThreshold=50, detectShadows=False
    )

    # Calentamiento: alimenta el modelo de fondo sin banda vacía al inicio
    print("[INFO] Calibrando fondo (2s)...")
    t_cal = time.time()
    while time.time() - t_cal < 2.0:
        ret, frame = cap.read()
        if ret:
            bg_subtractor.apply(frame)

    fotos_subidas = 0
    last_capture  = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # ── Detección de movimiento en ROI ────────────────────────────
            roi_frame  = frame[y1:y2, x1:x2]
            fg_mask    = bg_subtractor.apply(roi_frame)

            # Limpieza morfológica para eliminar ruido
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

            # Área del objeto detectado en la ROI
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            max_area = max((cv2.contourArea(c) for c in contours), default=0)
            flower_present = max_area >= MOTION_MIN_AREA

            # ── Visualización debug ───────────────────────────────────────
            if debug:
                vis = frame.copy()
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"Nitidez zona: {sharpness_score(frame, roi):.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                cv2.putText(vis, f"Area mov: {max_area:.0f}", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255) if flower_present else (200, 200, 200), 2)
                cv2.imshow("Conveyor — ESC para salir", vis)
                if cv2.waitKey(1) == 27:
                    break

            # ── Cooldown activo → esperar ─────────────────────────────────
            if time.time() - last_capture < COOLDOWN_SECS:
                continue

            # ── Flor detectada → ráfaga ───────────────────────────────────
            if not flower_present:
                continue

            print(f"[INFO] Flor detectada (área={max_area:.0f} px²) → ráfaga de {BURST_FRAMES} frames...")
            burst = []
            for _ in range(BURST_FRAMES):
                ret_b, f_b = cap.read()
                if ret_b and f_b is not None:
                    burst.append(f_b)

            if not burst:
                print("[WARN] Ráfaga vacía — se omite", file=sys.stderr)
                continue

            best, score = best_frame_from_burst(burst, roi)
            print(f"       Mejor frame: nitidez={score:.1f}  "
                  f"({len(burst)} candidatos)")

            last_capture = time.time()

            if score < SHARPNESS_MIN:
                print(f"       RECHAZADA — nitidez {score:.1f} < umbral {SHARPNESS_MIN}")
                continue

            # ── Subir la mejor foto ───────────────────────────────────────
            filename  = build_filename("flor")
            img_bytes = frame_to_bytes(best)
            uri       = upload_to_gcs(img_bytes, filename, bucket)
            fotos_subidas += 1
            print(f"[OK]   Subida #{fotos_subidas}: {uri}  "
                  f"({len(img_bytes)//1024} KB)  nitidez={score:.1f}")

            if max_fotos and fotos_subidas >= max_fotos:
                print(f"[INFO] Alcanzado límite de {max_fotos} fotos.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Detenido por el usuario.")
    finally:
        if debug:
            cv2.destroyAllWindows()
        print(f"[INFO] Total fotos subidas: {fotos_subidas}")


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _frame_size(cap: cv2.VideoCapture) -> tuple[int, int]:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return h, w


def _roi_pixels(w: int, h: int) -> tuple[int, int, int, int]:
    return (
        int(w * ROI_X1), int(h * ROI_Y1),
        int(w * ROI_X2), int(h * ROI_Y2),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Captura y sube fotos a GCS desde la Jetson."
    )
    parser.add_argument("--mode", choices=["interval", "conveyor"], default="interval",
                        help="'interval': fotos periódicas  |  'conveyor': banda inteligente (default: interval)")
    parser.add_argument("--interval", type=int, default=CAPTURE_INTERVAL,
                        help=f"[interval] Segundos entre capturas (default: {CAPTURE_INTERVAL})")
    parser.add_argument("--num-fotos", type=int, default=NUM_FOTOS,
                        help=f"Máximo de fotos a tomar (default: {NUM_FOTOS}, 0=sin límite en conveyor)")
    parser.add_argument("--camera-type", choices=["usb", "csi"], default="usb",
                        help="'usb' (default) o 'csi' (CSI/MIPI Jetson)")
    parser.add_argument("--exposure", type=float, default=0,
                        help="Exposure manual en ms (e.g. 5 ms congela flores en banda). "
                             "0 = auto (default). Recomendado: 3–8 ms para banda rápida.")
    parser.add_argument("--sharpness-min", type=float, default=SHARPNESS_MIN,
                        help=f"Varianza Laplaciana mínima para aceptar foto (default: {SHARPNESS_MIN})")
    parser.add_argument("--burst", type=int, default=BURST_FRAMES,
                        help=f"[conveyor] Frames por ráfaga (default: {BURST_FRAMES})")
    parser.add_argument("--cooldown", type=float, default=COOLDOWN_SECS,
                        help=f"[conveyor] Segundos entre capturas (default: {COOLDOWN_SECS})")
    parser.add_argument("--debug", action="store_true",
                        help="[conveyor] Muestra ventana de video en vivo con ROI y métricas")
    args = parser.parse_args()

    # Aplica parámetros CLI a globals para que las funciones los usen
    global SHARPNESS_MIN, BURST_FRAMES, COOLDOWN_SECS
    SHARPNESS_MIN = args.sharpness_min
    BURST_FRAMES  = args.burst
    COOLDOWN_SECS = args.cooldown

    # ── Credenciales GCS ──────────────────────────────────────────────────
    if not os.path.exists(CREDENTIALS_FILE):
        sys.exit(f"[ERROR] No se encontró el archivo de credenciales: {CREDENTIALS_FILE}")

    creds  = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE)
    client = storage.Client(credentials=creds, project=creds.project_id)

    try:
        bucket = client.get_bucket(BUCKET_NAME)
        print(f"[OK] Bucket: gs://{BUCKET_NAME}")
    except Exception as e:
        sys.exit(f"[ERROR] No se pudo acceder al bucket '{BUCKET_NAME}': {e}")

    # ── Abre cámara ───────────────────────────────────────────────────────
    print(f"[INFO] Abriendo cámara ({args.camera_type})"
          + (f" — exposure manual {args.exposure} ms" if args.exposure else " — exposure auto") + "...")
    try:
        cap = open_camera(args.camera_type, args.exposure)
    except RuntimeError as e:
        sys.exit(f"[ERROR] {e}")

    print(f"[INFO] Resolución: "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
          f"@ {int(cap.get(cv2.CAP_PROP_FPS))} fps")

    # ── Ejecuta modo seleccionado ─────────────────────────────────────────
    try:
        if args.mode == "conveyor":
            run_conveyor_mode(cap, bucket, debug=args.debug,
                              max_fotos=args.num_fotos)
        else:
            run_interval_mode(cap, bucket, args.num_fotos, args.interval)
    finally:
        cap.release()
        print("[INFO] Cámara liberada.")


if __name__ == "__main__":
    main()
