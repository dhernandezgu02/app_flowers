#!/usr/bin/env python3
"""
Galería web para visualizar y descargar fotos de flores desde GCS.

Instalar:  pip install -r requirements.txt
Ejecutar:  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
Abrir:     http://localhost:8000
"""

import io
import re
import sys
import datetime
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    HTMLResponse, StreamingResponse, RedirectResponse, JSONResponse
)
from fastapi.templating import Jinja2Templates
from google.cloud import storage
from google.oauth2 import service_account
from PIL import Image

# ── Configuración ─────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"   # pon aquí tu service account key
BUCKET_NAME      = "bucket_flower"
GCS_FOLDER       = "fotos"       # carpeta de fotos originales dentro del bucket
THUMB_FOLDER     = "thumbs"      # carpeta de thumbnails (se crea automáticamente)
THUMB_MAX_W      = 420
THUMB_MAX_H      = 315
SIGNED_URL_TTL   = datetime.timedelta(hours=3)
BLOBS_CACHE_TTL  = 30            # segundos entre re-listados de GCS

MONTHS_ES   = ["Ene","Feb","Mar","Abr","May","Jun","Jul",
                "Ago","Sep","Oct","Nov","Dic"]
WEEKDAYS_ES = ["Lunes","Martes","Miercoles","Jueves",
                "Viernes","Sabado","Domingo"]

# ── Conexión GCS ──────────────────────────────────────────────────────────
if not CREDENTIALS_FILE.exists():
    sys.exit(
        f"[ERROR] Coloca tu archivo de credenciales en:\n  {CREDENTIALS_FILE}\n"
        "Puedes copiar el .json que ya tienes en la Jetson."
    )

_creds  = service_account.Credentials.from_service_account_file(str(CREDENTIALS_FILE))
_client = storage.Client(credentials=_creds, project=_creds.project_id)
_bucket = _client.bucket(BUCKET_NAME)

# ── Caches en memoria ─────────────────────────────────────────────────────
_blobs_cache: tuple[list, datetime.datetime] | None = None
_url_cache: dict[str, tuple[str, datetime.datetime]] = {}

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Flower Gallery", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ══════════════════════════════════════════════════════════════════════════
#  Helpers GCS
# ══════════════════════════════════════════════════════════════════════════

def _list_photo_blobs() -> list:
    global _blobs_cache
    now = datetime.datetime.utcnow()
    if _blobs_cache and (now - _blobs_cache[1]).total_seconds() < BLOBS_CACHE_TTL:
        return _blobs_cache[0]

    prefix = f"{GCS_FOLDER}/" if GCS_FOLDER else ""
    blobs  = [
        b for b in _client.list_blobs(BUCKET_NAME, prefix=prefix)
        if re.search(r"\.(jpg|jpeg|png)$", b.name, re.IGNORECASE)
    ]
    _blobs_cache = (blobs, now)
    return blobs


def _parse_date(filename: str) -> Optional[str]:
    m = re.search(r"(\d{8})_\d{6}", filename)
    return m.group(1) if m else None


def _parse_time(filename: str) -> str:
    m = re.search(r"\d{8}_(\d{6})", filename)
    if not m:
        return "—"
    t = m.group(1)
    return f"{t[:2]}:{t[2:4]}:{t[4:]}"


def _day_labels(day_str: str) -> tuple[str, str]:
    try:
        dt      = datetime.datetime.strptime(day_str, "%Y%m%d")
        label   = f"{dt.day} {MONTHS_ES[dt.month - 1]} {dt.year}"
        weekday = WEEKDAYS_ES[dt.weekday()]
        return label, weekday
    except ValueError:
        return day_str, ""


def _signed_url(blob_name: str) -> str:
    now = datetime.datetime.utcnow()
    if blob_name in _url_cache:
        url, exp = _url_cache[blob_name]
        if exp > now + datetime.timedelta(minutes=15):
            return url
    blob = _bucket.blob(blob_name)
    url  = blob.generate_signed_url(
        expiration=SIGNED_URL_TTL, method="GET", version="v4"
    )
    _url_cache[blob_name] = (url, now + SIGNED_URL_TTL)
    return url


def _get_or_create_thumb(orig_blob_name: str) -> str:
    """
    Genera thumbnail la primera vez y lo guarda en GCS bajo thumbs/.
    Las veces siguientes solo devuelve el URL firmado (sin re-procesar).
    """
    filename   = orig_blob_name.split("/")[-1]
    thumb_name = f"{THUMB_FOLDER}/{filename}"
    thumb_blob = _bucket.blob(thumb_name)

    if not thumb_blob.exists():
        raw = _bucket.blob(orig_blob_name).download_as_bytes()
        img = Image.open(io.BytesIO(raw))
        img.thumbnail((THUMB_MAX_W, THUMB_MAX_H), Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=78, optimize=True)
        buf.seek(0)
        thumb_blob.upload_from_file(buf, content_type="image/jpeg")

    return _signed_url(thumb_name)


# ══════════════════════════════════════════════════════════════════════════
#  Rutas
# ══════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/days")
async def get_days():
    blobs  = _list_photo_blobs()
    counts: dict[str, int] = {}
    for blob in blobs:
        day = _parse_date(blob.name.split("/")[-1])
        if day:
            counts[day] = counts.get(day, 0) + 1

    days = []
    for day_str, count in sorted(counts.items(), reverse=True):
        label, weekday = _day_labels(day_str)
        days.append({"date": day_str, "count": count,
                     "label": label, "weekday": weekday})
    return JSONResponse(days)


@app.get("/api/photos/{day}")
async def get_photos(day: str):
    if not re.match(r"^\d{8}$", day):
        raise HTTPException(400, "Formato de fecha inválido (esperado YYYYMMDD)")

    result = []
    for blob in _list_photo_blobs():
        name = blob.name.split("/")[-1]
        if _parse_date(name) != day:
            continue
        result.append({
            "filename":  name,
            "blob_name": blob.name,
            "url":       _signed_url(blob.name),
            "thumb_url": f"/thumb/{blob.name}",
            "time":      _parse_time(name),
            "size":      blob.size or 0,
        })

    result.sort(key=lambda x: x["filename"])
    return JSONResponse(result)


@app.get("/thumb/{blob_path:path}")
async def get_thumbnail(blob_path: str):
    """Genera thumbnail en GCS si no existe y redirige al URL firmado."""
    try:
        url = _get_or_create_thumb(blob_path)
        return RedirectResponse(url, status_code=302)
    except Exception as exc:
        raise HTTPException(500, f"Error generando thumbnail: {exc}")


# IMPORTANTE: ruta más específica ANTES que la genérica ───────────────────
@app.get("/download/photo/{blob_path:path}")
async def download_photo(blob_path: str):
    """Redirige al URL firmado de GCS para descarga directa."""
    url = _signed_url(blob_path)
    return RedirectResponse(url, status_code=302)


@app.get("/download/{day}")
async def download_day_zip(day: str):
    """Descarga todas las fotos del día como ZIP (se construye en memoria)."""
    if not re.match(r"^\d{8}$", day):
        raise HTTPException(400, "Formato de fecha inválido")

    blobs = [
        b for b in _list_photo_blobs()
        if _parse_date(b.name.split("/")[-1]) == day
    ]
    if not blobs:
        raise HTTPException(404, "No hay fotos para este día")

    def _generate():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            for blob in blobs:
                zf.writestr(blob.name.split("/")[-1], blob.download_as_bytes())
        yield buf.getvalue()

    return StreamingResponse(
        _generate(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="flores_{day}.zip"'},
    )
