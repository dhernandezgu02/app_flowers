#!/usr/bin/env python3
"""
flower_sync.py — Sube fotos pendientes al bucket GCS.

Escanea la carpeta 'pending/', intenta subir cada foto a GCS
y borra las que se subieron correctamente. Las fallidas se
quedan en la carpeta para el próximo intento.

Uso:
    python flower_sync.py
    python flower_sync.py --dry-run    # muestra qué haría sin subir ni borrar
    python flower_sync.py --verbose    # imprime cada foto con más detalle
"""

import argparse
import logging
import sys
from pathlib import Path

from google.cloud import storage
from google.oauth2 import service_account

# ── Configuración ─────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
CREDENTIALS_FILE = BASE_DIR / "green-alchemy-301821-9753e8366e05.json"
BUCKET_NAME      = "bucket_flower"
GCS_FOLDER       = "fotos"
PENDING_DIR      = BASE_DIR / "pending"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def collect_pending() -> list[Path]:
    if not PENDING_DIR.exists():
        return []
    return sorted(
        p for p in PENDING_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def already_in_gcs(filename: str, bucket) -> bool:
    """Verifica si el blob ya existe en GCS (evita re-subir duplicados)."""
    return bucket.blob(f"{GCS_FOLDER}/{filename}").exists()


def main():
    parser = argparse.ArgumentParser(
        description="Sube fotos pendientes de la Jetson a GCS"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Muestra qué haría sin subir ni borrar nada"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Muestra detalles de cada foto"
    )
    args = parser.parse_args()

    pending = collect_pending()

    if not pending:
        log.info("No hay fotos pendientes en %s", PENDING_DIR)
        return

    total_size = sum(p.stat().st_size for p in pending)
    log.info("Fotos pendientes: %d  (%.1f MB total)",
             len(pending), total_size / 1_048_576)

    if args.dry_run:
        for p in pending:
            log.info("  [dry-run] %s  (%d KB)", p.name, p.stat().st_size // 1024)
        return

    # Conectar GCS
    if not CREDENTIALS_FILE.exists():
        log.error("Credenciales no encontradas: %s", CREDENTIALS_FILE)
        sys.exit(1)

    creds  = service_account.Credentials.from_service_account_file(
        str(CREDENTIALS_FILE)
    )
    client = storage.Client(credentials=creds, project=creds.project_id)
    bucket = client.bucket(BUCKET_NAME)

    ok = fail = skip = 0

    for i, photo in enumerate(pending, 1):
        prefix = f"[{i}/{len(pending)}]"

        # Si el archivo ya está en GCS (subida parcial anterior), solo borra local
        if already_in_gcs(photo.name, bucket):
            photo.unlink()
            skip += 1
            log.info("%s Ya estaba en GCS — borrado local: %s", prefix, photo.name)
            continue

        try:
            blob = bucket.blob(f"{GCS_FOLDER}/{photo.name}")
            blob.upload_from_filename(
                str(photo),
                content_type="image/jpeg",
                timeout=30,
            )
            size_kb = photo.stat().st_size // 1024
            photo.unlink()
            ok += 1
            if args.verbose:
                log.info("%s [OK] %s  %d KB → gs://%s/%s/%s",
                         prefix, photo.name, size_kb,
                         BUCKET_NAME, GCS_FOLDER, photo.name)
            else:
                log.info("%s [OK] %s  %d KB", prefix, photo.name, size_kb)

        except Exception as exc:
            fail += 1
            log.warning("%s [FAIL] %s — %s", prefix, photo.name, exc)

    log.info("─────────────────────────────────────────")
    log.info("Sync completo:")
    log.info("  Subidas exitosas : %d", ok)
    log.info("  Ya estaban en GCS: %d", skip)
    log.info("  Fallidas         : %d  (siguen en %s)", fail, PENDING_DIR)
    if fail == 0 and PENDING_DIR.exists():
        # Borra la carpeta si quedó vacía
        try:
            PENDING_DIR.rmdir()
            log.info("Carpeta pending eliminada (vacía).")
        except OSError:
            pass


if __name__ == "__main__":
    main()
