#!/usr/bin/env python3
"""
manual_capture.py — Captura manual y sube a GCS (modo consola).

Uso:
    python manual_capture.py                  # captura 1 foto y sube
    python manual_capture.py --count 5        # captura 5 fotos seguidas
    python manual_capture.py --loop           # captura en bucle (Enter = foto, Ctrl+C = salir)
    python manual_capture.py --file foto.jpg  # sube un archivo existente sin cámara
    python manual_capture.py --camera 1       # usa otra cámara (default: 0)
"""

import argparse
import datetime
import sys
from pathlib import Path

import cv2
from google.cloud import storage
from google.oauth2 import service_account

# ── Configuración ─────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
CREDENTIALS_FILE = BASE_DIR / "green-alchemy-301821-9753e8366e05.json"
BUCKET_NAME      = "bucket_flower"
GCS_FOLDER       = "fotos"
JPEG_QUALITY     = 92
WARMUP_FRAMES    = 8   # frames descartados para que la cámara estabilice exposición
# ─────────────────────────────────────────────────────────────────────────────


def build_filename() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:22]
    return f"flor_{ts}.jpg"


def connect_gcs():
    if not CREDENTIALS_FILE.exists():
        sys.exit(f"[ERROR] Credenciales no encontradas:\n  {CREDENTIALS_FILE}")
    creds  = service_account.Credentials.from_service_account_file(str(CREDENTIALS_FILE))
    client = storage.Client(credentials=creds, project=creds.project_id)
    print(f"[OK] GCS conectado → gs://{BUCKET_NAME}/{GCS_FOLDER}/")
    return client.bucket(BUCKET_NAME)


def upload(img_bytes: bytes, filename: str, bucket) -> str:
    blob = bucket.blob(f"{GCS_FOLDER}/{filename}")
    blob.upload_from_string(img_bytes, content_type="image/jpeg")
    return f"gs://{BUCKET_NAME}/{GCS_FOLDER}/{filename}"


def capture_frame(cap) -> bytes:
    # Descarta frames acumulados para que AE se estabilice
    for _ in range(WARMUP_FRAMES):
        cap.read()
    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("No se pudo leer el frame de la cámara.")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen.")
    return buf.tobytes()


def do_capture_and_upload(cap, bucket) -> str:
    img_bytes = capture_frame(cap)
    filename  = build_filename()
    uri = upload(img_bytes, filename, bucket)
    kb  = len(img_bytes) // 1024
    print(f"[OK] {filename}  ({kb} KB)\n     → {uri}")
    return uri


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"[ERROR] No se pudo abrir la cámara (índice {index}).")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Cámara {index} abierta: {w}x{h}")
    return cap


def mode_single(cap, bucket, count: int):
    total = 0
    for i in range(1, count + 1):
        print(f"[{i}/{count}] Capturando...", end=" ", flush=True)
        try:
            do_capture_and_upload(cap, bucket)
            total += 1
        except Exception as exc:
            print(f"ERROR — {exc}", file=sys.stderr)
    print(f"\n[INFO] Fotos subidas: {total}/{count}")


def mode_loop(cap, bucket):
    print("[INFO] Modo bucle — Enter = capturar   Ctrl+C = salir")
    total = 0
    try:
        while True:
            input("\nPresiona Enter para capturar...")
            print("Capturando...", end=" ", flush=True)
            try:
                do_capture_and_upload(cap, bucket)
                total += 1
            except Exception as exc:
                print(f"ERROR — {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print(f"\n[INFO] Saliendo. Fotos subidas: {total}")


def mode_file(path: Path, bucket):
    if not path.exists():
        sys.exit(f"[ERROR] Archivo no encontrado: {path}")
    img_bytes = path.read_bytes()
    filename  = build_filename()
    print(f"Subiendo {path.name}...", end=" ", flush=True)
    uri = upload(img_bytes, filename, bucket)
    kb  = len(img_bytes) // 1024
    print(f"OK  ({kb} KB)\n     → {uri}")


def main():
    parser = argparse.ArgumentParser(
        description="Captura manual y sube a GCS (mismo formato que la Jetson)."
    )
    parser.add_argument("--file",   type=Path, default=None,
                        help="Sube un archivo existente sin abrir la cámara")
    parser.add_argument("--camera", type=int,  default=0,
                        help="Índice de cámara (default: 0)")
    parser.add_argument("--count",  type=int,  default=1,
                        help="Número de fotos a capturar seguidas (default: 1)")
    parser.add_argument("--loop",   action="store_true",
                        help="Captura en bucle: Enter = foto, Ctrl+C = salir")
    args = parser.parse_args()

    bucket = connect_gcs()

    if args.file:
        mode_file(args.file, bucket)
        return

    cap = open_camera(args.camera)
    try:
        if args.loop:
            mode_loop(cap, bucket)
        else:
            mode_single(cap, bucket, args.count)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
