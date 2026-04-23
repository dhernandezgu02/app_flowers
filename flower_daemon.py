#!/usr/bin/env python3
"""
flower_daemon.py — Daemon de captura de flores para Jetson.

Corre indefinidamente. Activo lunes–sábado, 09:00–17:00.
- Detecta flores en banda (MOG2 + ráfaga + nitidez Laplaciana).
- Sube la mejor foto a GCS; si falla, guarda en PENDING_DIR.
- Fuera de horario duerme hasta el próximo inicio de jornada.

Uso:
    python flower_daemon.py
    python flower_daemon.py --camera-type csi
    python flower_daemon.py --debug
"""

import argparse
import datetime
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from google.cloud import storage
from google.oauth2 import service_account

# ── Configuración ─────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
CREDENTIALS_FILE = BASE_DIR / "green-alchemy-301821-9753e8366e05.json"
BUCKET_NAME      = "bucket_flower"
GCS_FOLDER       = "fotos"
PENDING_DIR      = BASE_DIR / "pending"
LOG_FILE         = BASE_DIR / "flower_daemon.log"

CAMERA_INDEX     = 0
JPEG_QUALITY     = 92
IMAGE_FORMAT     = "jpg"

# Horario activo: lunes (0) – sábado (5), 09:00 – 17:00
WORK_START = datetime.time(9, 0)
WORK_END   = datetime.time(17, 0)
WORK_DAYS  = {0, 1, 2, 3, 4, 5}

# Parámetros conveyor
ROI_X1, ROI_Y1 = 0.25, 0.20
ROI_X2, ROI_Y2 = 0.75, 0.80
MOTION_MIN_AREA = 3_000
BURST_FRAMES    = 12
SHARPNESS_MIN   = 80.0
COOLDOWN_SECS   = 2.0
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Ventana de horario
# ══════════════════════════════════════════════════════════════════════════════

def in_work_window() -> bool:
    now = datetime.datetime.now()
    return now.weekday() in WORK_DAYS and WORK_START <= now.time() < WORK_END


def seconds_until_next_window() -> float:
    """Segundos hasta el próximo inicio de jornada (busca hasta 8 días adelante)."""
    now = datetime.datetime.now()
    candidate = now.replace(
        hour=WORK_START.hour, minute=WORK_START.minute, second=0, microsecond=0
    )
    if candidate <= now:
        candidate += datetime.timedelta(days=1)

    for _ in range(8):
        if candidate.weekday() in WORK_DAYS:
            break
        candidate += datetime.timedelta(days=1)

    return max(0.0, (candidate - now).total_seconds())


# ══════════════════════════════════════════════════════════════════════════════
#  Cámara
# ══════════════════════════════════════════════════════════════════════════════

def gstreamer_pipeline(sensor_id=0, cap_w=1920, cap_h=1080,
                       disp_w=1280, disp_h=720,
                       framerate=60, flip=0, exposure_us=8000):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} "
        f"exposuretimerange='{exposure_us} {exposure_us}' "
        f"gainrange='4 4' ispdigitalgainrange='1 1' ! "
        f"video/x-raw(memory:NVMM), width={cap_w}, height={cap_h}, "
        f"format=NV12, framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width={disp_w}, height={disp_h}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )


def open_camera(camera_type: str) -> cv2.VideoCapture:
    if camera_type == "csi":
        cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    else:
        # El OpenCV de Jetson usa GStreamer internamente incluso para USB.
        # No forzamos resolución — dejamos que la cámara reporte la suya.
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara (tipo={camera_type})")
    return cap


# ══════════════════════════════════════════════════════════════════════════════
#  Imagen
# ══════════════════════════════════════════════════════════════════════════════

def sharpness_score(frame: np.ndarray, roi: tuple) -> float:
    x1, y1, x2, y2 = roi
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def best_frame(frames: list, roi: tuple) -> tuple:
    scores = [sharpness_score(f, roi) for f in frames]
    idx = int(np.argmax(scores))
    return frames[idx], scores[idx]


def encode_frame(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(
        f".{IMAGE_FORMAT}", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError("Error al codificar imagen")
    return buf.tobytes()


def build_filename() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:22]
    return f"flor_{ts}.{IMAGE_FORMAT}"


# ══════════════════════════════════════════════════════════════════════════════
#  GCS
# ══════════════════════════════════════════════════════════════════════════════

def upload_to_gcs(img_bytes: bytes, filename: str, bucket) -> bool:
    if bucket is None:
        return False
    try:
        blob = bucket.blob(f"{GCS_FOLDER}/{filename}")
        blob.upload_from_string(
            img_bytes, content_type=f"image/{IMAGE_FORMAT}", timeout=15
        )
        return True
    except Exception as exc:
        log.warning("GCS error: %s", exc)
        return False


def save_pending(img_bytes: bytes, filename: str):
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    (PENDING_DIR / filename).write_bytes(img_bytes)
    log.info("Sin internet — guardada localmente: %s", filename)


# ══════════════════════════════════════════════════════════════════════════════
#  Sync de pendientes (se corre automáticamente al terminar la jornada)
# ══════════════════════════════════════════════════════════════════════════════

def sync_pending(bucket):
    """Sube a GCS todas las fotos guardadas localmente en PENDING_DIR."""
    if not PENDING_DIR.exists():
        return
    photos = sorted(
        p for p in PENDING_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not photos:
        return

    log.info("── Sync automático: %d fotos pendientes ─────────────────────", len(photos))

    if bucket is None:
        log.warning("Sin cliente GCS — sync pospuesto al próximo ciclo.")
        return

    ok = fail = 0
    for photo in photos:
        blob_name = f"{GCS_FOLDER}/{photo.name}"
        try:
            # Evita re-subir si ya está en GCS
            if bucket.blob(blob_name).exists():
                photo.unlink()
                ok += 1
                continue
            bucket.blob(blob_name).upload_from_filename(
                str(photo), content_type="image/jpeg", timeout=30
            )
            photo.unlink()
            ok += 1
            log.info("[sync OK] %s  %d KB", photo.name, photo.stat().st_size // 1024 if photo.exists() else 0)
        except Exception as exc:
            fail += 1
            log.warning("[sync FAIL] %s — %s", photo.name, exc)

    log.info("── Sync completo: %d subidas, %d fallidas ───────────────────", ok, fail)
    try:
        PENDING_DIR.rmdir()   # borra la carpeta si quedó vacía
    except OSError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  Sesión de captura (una jornada)
# ══════════════════════════════════════════════════════════════════════════════

def run_session(bucket, camera_type: str, debug: bool):
    """Corre el bucle de detección hasta que el horario laboral termine."""
    log.info("── Inicio de sesión (%s) ──────────────────────────────────", camera_type)

    try:
        cap = open_camera(camera_type)
    except RuntimeError as exc:
        log.error("%s — reintentando en 60 s", exc)
        time.sleep(60)
        return

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    roi = (int(w * ROI_X1), int(h * ROI_Y1),
           int(w * ROI_X2), int(h * ROI_Y2))
    x1, y1, x2, y2 = roi
    log.info("Cámara: %dx%d @ %.0f fps", w, h, fps)

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=120, varThreshold=50, detectShadows=False
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    log.info("Calibrando fondo (2 s)...")
    t_cal = time.time()
    while time.time() - t_cal < 2.0:
        cap.read()

    last_capture = 0.0
    fotos_subidas = 0
    fotos_pendientes = 0

    try:
        while in_work_window():
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Detección de movimiento en ROI
            fg = bg_sub.apply(frame[y1:y2, x1:x2])
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            max_area = max((cv2.contourArea(c) for c in contours), default=0)

            if debug:
                vis = frame.copy()
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"Area: {max_area:.0f}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
                cv2.imshow("Daemon — ESC para salir", vis)
                if cv2.waitKey(1) == 27:
                    break

            if time.time() - last_capture < COOLDOWN_SECS:
                continue
            if max_area < MOTION_MIN_AREA:
                continue

            # Ráfaga y selección del frame más nítido
            burst = [cap.read()[1] for _ in range(BURST_FRAMES)]
            burst = [f for f in burst if f is not None]
            if not burst:
                continue

            frame_best, score = best_frame(burst, roi)
            last_capture = time.time()

            if score < SHARPNESS_MIN:
                log.debug("Nitidez %.1f < umbral — descartada", score)
                continue

            filename  = build_filename()
            img_bytes = encode_frame(frame_best)

            if upload_to_gcs(img_bytes, filename, bucket):
                fotos_subidas += 1
                log.info("[GCS ↑] %s  nitidez=%.1f  %d KB",
                         filename, score, len(img_bytes) // 1024)
            else:
                fotos_pendientes += 1
                save_pending(img_bytes, filename)

    except KeyboardInterrupt:
        log.info("Interrupción de usuario en sesión")
        raise
    finally:
        cap.release()
        if debug:
            cv2.destroyAllWindows()
        log.info(
            "── Fin de sesión — subidas: %d  pendientes locales: %d ──",
            fotos_subidas, fotos_pendientes,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def build_gcs_bucket():
    """Construye el cliente GCS. Solo necesita leer el JSON local — no requiere internet."""
    if not CREDENTIALS_FILE.exists():
        # Sin credenciales no podemos subir nunca, pero sí podemos guardar localmente.
        # El daemon arranca igual; upload_to_gcs fallará y guardará en pending/.
        log.warning("Credenciales no encontradas: %s — todas las fotos irán a pending/",
                    CREDENTIALS_FILE)
        return None
    try:
        creds  = service_account.Credentials.from_service_account_file(
            str(CREDENTIALS_FILE)
        )
        client = storage.Client(credentials=creds, project=creds.project_id)
        return client.bucket(BUCKET_NAME)
    except Exception as exc:
        log.warning("No se pudo inicializar cliente GCS: %s — fotos irán a pending/", exc)
        return None


def main():
    parser = argparse.ArgumentParser(description="Daemon captura flores — Jetson")
    parser.add_argument("--camera-type", choices=["usb", "csi"], default="usb")
    parser.add_argument("--debug", action="store_true",
                        help="Muestra ventana de video con ROI en tiempo real")
    args = parser.parse_args()

    log.info("═══════════════════════════════════════════")
    log.info("  Flower Daemon iniciado")
    log.info("  Bucket  : gs://%s/%s/", BUCKET_NAME, GCS_FOLDER)
    log.info("  Horario : lun–sáb  %s – %s", WORK_START, WORK_END)
    log.info("  Pending : %s", PENDING_DIR)
    log.info("═══════════════════════════════════════════")

    bucket = build_gcs_bucket()

    try:
        while True:
            if in_work_window():
                run_session(bucket, args.camera_type, args.debug)
                # La sesión terminó (fin de horario o error de cámara).
                # Refresca cliente GCS por si ahora hay internet.
                bucket = build_gcs_bucket()
                # Sube automáticamente las fotos pendientes al terminar la jornada.
                if not in_work_window():
                    sync_pending(bucket)
            else:
                wait = seconds_until_next_window()
                log.info("Fuera de horario. Próxima jornada en %.0f min (%.1f h)",
                         wait / 60, wait / 3600)
                time.sleep(wait)
    except KeyboardInterrupt:
        log.info("Daemon detenido manualmente.")


if __name__ == "__main__":
    main()
