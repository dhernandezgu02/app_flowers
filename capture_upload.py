#!/usr/bin/env python3
"""
Captura fotos desde la cámara de la Jetson y las sube a Google Cloud Storage.
Uso:
    python capture_upload.py                        # captura continua cada 60s
    python capture_upload.py --interval 30          # cada 30 segundos
    python capture_upload.py --once                 # una sola foto y sale
    python capture_upload.py --camera-type csi      # usa cámara CSI (Raspberry Pi cam)
"""

import argparse
import datetime
import os
import sys
import time

import cv2
from google.cloud import storage
from google.oauth2 import service_account

# ── Configuración ────────────────────────────────────────────────────────────
CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), "green-alchemy-301821-9753e8366e05.json")
BUCKET_NAME      = "bucket_flower"
GCS_FOLDER       = "fotos"                  # carpeta dentro del bucket (puede estar vacío "")
CAPTURE_INTERVAL = 3                         # segundos entre capturas
NUM_FOTOS        = 5                         # cantidad de fotos a tomar
CAMERA_INDEX     = 0                         # 0 = primer dispositivo USB
IMAGE_FORMAT     = "jpg"
JPEG_QUALITY     = 90
# ─────────────────────────────────────────────────────────────────────────────


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    """Pipeline GStreamer para cámara CSI en Jetson (e.g. IMX219)."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
        f"format=NV12, framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )


def open_camera(camera_type: str) -> cv2.VideoCapture:
    if camera_type == "csi":
        pipeline = gstreamer_pipeline()
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la cámara (tipo={camera_type}). "
            "Verifica que la cámara esté conectada y los permisos del dispositivo."
        )
    return cap


def capture_image(cap: cv2.VideoCapture) -> bytes:
    """Captura un frame y lo devuelve como bytes JPEG."""
    # Descarta algunos frames para que el auto-exposición se estabilice
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("No se pudo leer el frame de la cámara.")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    success, buffer = cv2.imencode(f".{IMAGE_FORMAT}", frame, encode_params)
    if not success:
        raise RuntimeError("Error al codificar la imagen.")
    return buffer.tobytes()


def upload_to_gcs(image_bytes: bytes, filename: str, bucket: storage.Bucket) -> str:
    """Sube bytes de imagen al bucket y devuelve la URI gs://."""
    blob_name = f"{GCS_FOLDER}/{filename}" if GCS_FOLDER else filename
    blob = bucket.blob(blob_name)
    blob.upload_from_string(image_bytes, content_type=f"image/{IMAGE_FORMAT}")
    uri = f"gs://{bucket.name}/{blob_name}"
    return uri


def build_filename() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"foto_{ts}.{IMAGE_FORMAT}"


def main():
    parser = argparse.ArgumentParser(description="Captura y sube fotos a GCS desde la Jetson.")
    parser.add_argument("--interval", type=int, default=CAPTURE_INTERVAL,
                        help=f"Segundos entre capturas (default: {CAPTURE_INTERVAL})")
    parser.add_argument("--num-fotos", type=int, default=NUM_FOTOS,
                        help=f"Cantidad de fotos a tomar (default: {NUM_FOTOS})")
    parser.add_argument("--camera-type", choices=["usb", "csi"], default="usb",
                        help="Tipo de cámara: 'usb' (default) o 'csi' (CSI/MIPI)")
    args = parser.parse_args()

    # Valida credenciales
    if not os.path.exists(CREDENTIALS_FILE):
        sys.exit(f"[ERROR] No se encontró el archivo de credenciales: {CREDENTIALS_FILE}")

    # Autenticación GCS
    creds = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE)
    client = storage.Client(credentials=creds, project=creds.project_id)

    try:
        bucket = client.get_bucket(BUCKET_NAME)
        print(f"[OK] Bucket encontrado: gs://{BUCKET_NAME}")
    except Exception as e:
        sys.exit(f"[ERROR] No se pudo acceder al bucket '{BUCKET_NAME}': {e}")

    # Abre cámara
    print(f"[INFO] Abriendo cámara ({args.camera_type})...")
    try:
        cap = open_camera(args.camera_type)
    except RuntimeError as e:
        sys.exit(f"[ERROR] {e}")

    print(f"[INFO] Cámara lista. Tomando {args.num_fotos} fotos...")
    try:
        for i in range(1, args.num_fotos + 1):
            try:
                image_bytes = capture_image(cap)
                filename    = build_filename()
                uri         = upload_to_gcs(image_bytes, filename, bucket)
                print(f"[{i}/{args.num_fotos}] Subida: {uri}  ({len(image_bytes)//1024} KB)")
            except Exception as e:
                print(f"[WARN] Error en foto {i}: {e}", file=sys.stderr)

            if i < args.num_fotos:
                time.sleep(args.interval)

        print("[INFO] Listo.")
    except KeyboardInterrupt:
        print("\n[INFO] Detenido por el usuario.")
    finally:
        cap.release()
        print("[INFO] Cámara liberada.")


if __name__ == "__main__":
    main()
