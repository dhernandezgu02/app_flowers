#!/usr/bin/env python3
"""
Galería web para visualizar y descargar fotos de flores desde GCS.

Instalar:  pip install -r requirements.txt
Ejecutar:  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
Abrir:     http://localhost:8000
"""

import asyncio
import io
import numpy as np
import json
import os
import re
import sys
import base64
import datetime
import zipfile
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import (
    HTMLResponse, StreamingResponse, RedirectResponse, JSONResponse
)
from google.cloud import storage
from google.oauth2 import service_account
from PIL import Image
from pydantic import BaseModel

# ── Configuración ─────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"   # pon aquí tu service account key
BUCKET_NAME      = "bucket_flower"
GCS_FOLDER       = "fotos"       # carpeta de fotos originales dentro del bucket
THUMB_FOLDER     = "thumbs"      # carpeta de thumbnails (se crea automáticamente)
THUMB_MAX_W      = 420
THUMB_MAX_H      = 315
ANALYSIS_MAX_W   = 960   # resolución mayor para análisis de enfermedades
ANALYSIS_MAX_H   = 720
SIGNED_URL_TTL   = datetime.timedelta(hours=3)
BLOBS_CACHE_TTL  = 30            # segundos entre re-listados de GCS
ANALYSIS_CACHE_FILE = BASE_DIR / "analysis_cache.json"
CLAUDE_MODEL        = "claude-sonnet-4-6"
REFERENCES_DIR      = BASE_DIR / "references" / "botrytis"
MAX_REFS_IN_PROMPT  = 3    # cuántas fotos de referencia incluir por análisis
BATCH_DELAY_SECS    = 0.8  # pausa entre fotos en análisis en lote (rate limit)

MONTHS_ES   = ["Ene","Feb","Mar","Abr","May","Jun","Jul",
                "Ago","Sep","Oct","Nov","Dic"]
WEEKDAYS_ES = ["Lunes","Martes","Miercoles","Jueves",
                "Viernes","Sabado","Domingo"]

# ── Conexión GCS ──────────────────────────────────────────────────────────
_gcp_json = os.environ.get("GCP_CREDENTIALS")
if _gcp_json:
    _creds = service_account.Credentials.from_service_account_info(json.loads(_gcp_json))
elif CREDENTIALS_FILE.exists():
    _creds = service_account.Credentials.from_service_account_file(str(CREDENTIALS_FILE))
else:
    sys.exit(
        "[ERROR] No se encontraron credenciales GCS.\n"
        "  - Define la variable de entorno GCP_CREDENTIALS con el JSON de la service account.\n"
        f"  - O coloca el archivo en: {CREDENTIALS_FILE}\n"
    )

_client = storage.Client(credentials=_creds, project=_creds.project_id)
_bucket = _client.bucket(BUCKET_NAME)

# ── Anthropic (Vision API para detección de enfermedades) ─────────────────
_ANTHROPIC_KEY    = os.environ.get("ANTHROPIC_API_KEY")
_anthropic_client = anthropic.Anthropic(api_key=_ANTHROPIC_KEY) if _ANTHROPIC_KEY else None

_BOTRYTIS_PROMPT_BASE = (
    "Eres un inspector de calidad de Gypsophila paniculata para exportación a Europa.\n\n"
    "INSPECCIONA LA IMAGEN y reporta TODAS las anomalías que encuentres:\n\n"
    "1. BOTRYTIS CINEREA en tallos (indicador principal):\n"
    "   · Sano → verde brillante uniforme\n"
    "   · Infectado → grisáceo, blanquecino, parduzco, manchas opacas o decoloración\n"
    "   · Describe zona exacta y extensión del daño\n\n"
    "2. TALLOS ROTOS O DAÑADOS:\n"
    "   · ¿Hay tallos partidos, doblados, aplastados o con lesiones físicas?\n"
    "   · ¿Dónde exactamente en el ramo?\n\n"
    "3. FLORES:\n"
    "   · Sanas → blanco puro, abiertas, sueltas\n"
    "   · Problemas → grisáceas/amarillas, apelmazadas, secas, moho visible\n\n"
    "4. OTRAS ANOMALÍAS:\n"
    "   · Plagas, manchas inusuales, daño mecánico general, calidad del ramo\n\n"
    "CLASIFICACIÓN FINAL:\n"
    "   'sano'       → sin signos de enfermedad ni daño relevante\n"
    "   'sospechoso' → alguna anomalía dudosa que requiere revisión manual\n"
    "   'botrytis'   → señales claras de Botrytis cinerea\n"
    "   ⚠️ ANTE CUALQUIER DUDA → 'sospechoso'\n\n"
    "Responde SOLO con este JSON (sin markdown):\n"
    '{"status":"sano","confidence":"alta",'
    '"hallazgos":["hallazgo 1","hallazgo 2","hallazgo 3"],'
    '"anomalias":["tallo roto en zona X","daño mecánico en..."],'
    '"resumen":"diagnóstico en 2-3 oraciones",'
    '"descripcion":"descripción técnica completa para registro de calidad, mínimo 3 oraciones"}\n\n'
    "  anomalias → lista de problemas físicos/mecánicos (vacía [] si no hay)\n"
    "  descripcion → descripción detallada de TODO lo que ves en la imagen"
)

_BOTRYTIS_PROMPT_WITH_REFS = (
    "Eres un inspector de calidad de Gypsophila paniculata para exportación a Europa.\n\n"
    "Las primeras imágenes son casos CONFIRMADOS CON BOTRYTIS CINEREA del mismo proveedor.\n"
    "Compara la última imagen con esas referencias al clasificar.\n\n"
) + _BOTRYTIS_PROMPT_BASE[_BOTRYTIS_PROMPT_BASE.index("INSPECCIONA"):]

_BOTRYTIS_PROMPT = _BOTRYTIS_PROMPT_BASE  # alias para compatibilidad

# ── Imágenes de referencia (few-shot) ────────────────────────────────────
_reference_b64: list[str] = []

def _load_reference_images() -> None:
    global _reference_b64
    if not REFERENCES_DIR.exists():
        return
    loaded = []
    for p in sorted(REFERENCES_DIR.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        try:
            img = Image.open(p)
            img.thumbnail((640, 480), Image.LANCZOS)
            if img.mode != "RGB":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=82)
            loaded.append(base64.standard_b64encode(buf.getvalue()).decode())
        except Exception as e:
            print(f"[refs] Error cargando {p.name}: {e}")
    _reference_b64 = loaded
    print(f"[refs] {len(_reference_b64)} imágenes de referencia Botrytis cargadas.")

_load_reference_images()


def _color_context(stats: dict | None) -> str:
    if not stats:
        return ""
    return (
        f"\n\n[ANÁLISIS DE COLOR PREVIO — HSV automático]\n"
        f"  Verde sano detectado: {stats.get('pct_verde_sano', '?')}% de los píxeles de tallo\n"
        f"  Zona sospechosa (grisácea/parduzca): {stats.get('pct_sospechoso', '?')}%\n"
        f"  Diagnóstico preliminar por color: {stats.get('assessment', '?').upper()}\n"
        "Usa estos datos como contexto adicional para tu análisis visual.\n"
    )


def _claude_messages(img_b64: str, color_stats: dict | None = None) -> list:
    refs = _reference_b64[:MAX_REFS_IN_PROMPT]
    content: list = []
    if refs:
        for i, r in enumerate(refs, 1):
            content.append({"type": "text", "text": f"REFERENCIA {i} — Botrytis confirmada:"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": r}})
        content.append({"type": "text", "text": "\nIMAGEN A ANALIZAR (compara con las referencias):"})
        prompt = _BOTRYTIS_PROMPT_WITH_REFS
    else:
        prompt = _BOTRYTIS_PROMPT_BASE
    content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}})
    content.append({"type": "text", "text": prompt + _color_context(color_stats)})
    return [{"role": "user", "content": content}]


def _extract_plant_mask(v: np.ndarray, s: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Extrae la región de la planta (tallo + flor) usando operaciones morfológicas.
    1. Candidatos: cualquier píxel que NO sea fondo oscuro ni cinta saturada.
    2. Dilatación morfológica para conectar tallos y flores cercanas.
    3. Relleno de huecos internos.
    4. Mantiene solo el componente conectado más grande.
    Devuelve máscara booleana (True = planta).
    """
    from scipy import ndimage

    mask_dark    = v < 0.10
    mask_bg_sat  = (s > 0.42) & ((h < 42) | (h > 158))
    candidates   = ~mask_dark & ~mask_bg_sat   # todo lo que podría ser planta

    # Dilatación con elemento 25×25 para cerrar los huecos entre tallos
    struct = np.ones((25, 25), dtype=bool)
    closed = ndimage.binary_closing(candidates, structure=struct, iterations=2)
    filled = ndimage.binary_fill_holes(closed)

    # Quedarse solo con el componente más grande (la planta principal)
    labeled, n_comp = ndimage.label(filled)
    if n_comp == 0:
        return candidates   # fallback: sin planta detectada, usar candidatos
    sizes  = np.bincount(labeled.ravel())
    sizes[0] = 0            # ignorar fondo (etiqueta 0)
    largest  = sizes.argmax()
    return labeled == largest


def _analyze_stem_colors(raw_bytes: bytes) -> dict:
    """
    Análisis HSV restringido a la región de la planta extraída morfológicamente.
    - Verde brillante  → sano
    - Grisáceo/blanquecino → sospechoso Botrytis
    - Parduzco/amarillento → sospechoso Botrytis
    """
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img.thumbnail((960, 720), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0

    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    diff = maxc - minc

    v = maxc
    s = np.where(maxc > 1e-6, diff / maxc, 0.0)

    h = np.zeros(arr.shape[:2], dtype=np.float32)
    m = (maxc == r) & (diff > 1e-6)
    h[m] = ((g[m] - b[m]) / diff[m]) % 6 * 60
    m = (maxc == g) & (diff > 1e-6)
    h[m] = (b[m] - r[m]) / diff[m] * 60 + 120
    m = (maxc == b) & (diff > 1e-6)
    h[m] = (r[m] - g[m]) / diff[m] * 60 + 240

    # ── Extracción de la región de planta ─────────────────────────────────
    try:
        plant_mask = _extract_plant_mask(v, s, h)
    except Exception:
        plant_mask = np.ones(arr.shape[:2], dtype=bool)   # fallback: imagen completa

    # ── Máscaras de tejido (solo dentro de la planta) ─────────────────────
    # Flores blancas: excluidas del análisis pero sí forman parte de la planta
    mask_flowers = (s < 0.22) & (v > 0.78) & plant_mask

    # Tallo sano: verde dentro de la planta, excluyendo flores
    mask_green = (
        (h >= 45) & (h <= 168) & (s >= 0.18) & (v >= 0.12)
        & plant_mask & ~mask_flowers
    )
    # Sospechoso: grisáceo en tallo
    mask_gray = (
        (s < 0.20) & (v >= 0.12) & (v < 0.78)
        & plant_mask & ~mask_flowers
    )
    # Sospechoso: parduzco/amarillento (S media, H fuera del verde)
    mask_brown = (
        ((h < 45) | (h > 168)) & (s >= 0.15) & (s < 0.45) & (v >= 0.12)
        & plant_mask & ~mask_flowers
    )
    mask_suspect = mask_gray | mask_brown

    # ── Estadísticas sobre tejido de tallo ────────────────────────────────
    mask_tissue = mask_green | mask_suspect
    total    = max(int(np.sum(mask_tissue)), 1)
    n_green  = int(np.sum(mask_green))
    n_suspect= int(np.sum(mask_suspect))

    pct_green   = round(n_green   / total * 100, 1)
    pct_suspect = round(n_suspect / total * 100, 1)

    assessment = (
        "botrytis"    if pct_suspect > 35 or pct_green < 25 else
        "sospechoso"  if pct_suspect > 15 or pct_green < 50 else
        "sano"
    )

    # ── Zonas sospechosas (cuadrícula 3×3 dentro de la planta) ────────────
    img_h, img_w = arr.shape[:2]
    row_names = ["Superior", "Media", "Inferior"]
    col_names = ["izquierda", "centro", "derecha"]
    ROW_N, COL_N = 3, 3
    r_step = img_h // ROW_N
    c_step = img_w // COL_N

    zonas: list[dict] = []
    for ri in range(ROW_N):
        for ci in range(COL_N):
            y1_ = ri * r_step
            y2_ = (ri + 1) * r_step if ri < ROW_N - 1 else img_h
            x1_ = ci * c_step
            x2_ = (ci + 1) * c_step if ci < COL_N - 1 else img_w
            z_tissue  = mask_tissue[y1_:y2_, x1_:x2_]
            z_suspect = mask_suspect[y1_:y2_, x1_:x2_]
            if int(np.sum(z_tissue)) < 500:
                continue
            pct_z = round(int(np.sum(z_suspect)) / max(int(np.sum(z_tissue)), 1) * 100, 1)
            if pct_z > 8:
                severity = "alta" if pct_z > 30 else "media" if pct_z > 15 else "baja"
                zonas.append({
                    "zona": f"{row_names[ri]} {col_names[ci]}",
                    "pct":  pct_z, "severity": severity,
                    "bbox": [x1_, y1_, x2_, y2_],
                })

    zonas.sort(key=lambda z: z["pct"], reverse=True)
    for i, z in enumerate(zonas, 1):
        z["num"] = i

    # ── Visualización ─────────────────────────────────────────────────────
    vis = arr.copy()
    # Fondo (fuera de la planta): desaturar + oscurecer levemente
    bg = ~plant_mask
    vis[bg] = arr[bg] * 0.25 + 0.05

    # Overlay semitransparente en tejido vegetal
    ag, ar = 0.65, 0.65
    vis[mask_green,  0] = arr[mask_green, 0]  * (1-ag) + 0.08 * ag
    vis[mask_green,  1] = arr[mask_green, 1]  * (1-ag) + 0.90 * ag
    vis[mask_green,  2] = arr[mask_green, 2]  * (1-ag) + 0.15 * ag
    vis[mask_suspect,0] = arr[mask_suspect, 0]* (1-ar) + 0.95 * ar
    vis[mask_suspect,1] = arr[mask_suspect, 1]* (1-ar) + 0.05 * ar
    vis[mask_suspect,2] = arr[mask_suspect, 2]* (1-ar) + 0.05 * ar
    # Las flores se mantienen con color original (blancas, reconocibles)

    vis_img = Image.fromarray((vis * 255).astype(np.uint8))
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(vis_img)
    iw, ih = vis_img.size

    try:
        lbl_font = ImageFont.load_default(size=16)
        num_font = ImageFont.load_default(size=22)
    except TypeError:
        lbl_font = num_font = ImageFont.load_default()

    sev_rgb = {"alta": (248,113,113), "media": (251,191,36), "baja": (74,222,128)}
    for z in zonas:
        x1_, y1_, x2_, y2_ = z["bbox"]
        col = sev_rgb[z["severity"]]
        draw.rectangle([x1_-1, y1_-1, x2_, y2_], outline=(255,255,255), width=1)
        draw.rectangle([x1_, y1_, x2_-1, y2_-1], outline=col, width=4)
        lx, ly = x1_+6, y1_+6
        draw.rectangle([lx-3, ly-3, lx+20, ly+22], fill=(0,0,0))
        draw.rectangle([lx-3, ly-3, lx+20, ly+22], outline=col, width=2)
        draw.text((lx, ly), str(z["num"]), fill=col, font=num_font)

    bar_h = 34
    draw.rectangle([(0, ih-bar_h), (iw, ih)], fill=(10,10,10))
    c = {"sano":(52,211,153),"sospechoso":(251,191,36),"botrytis":(248,113,113)}.get(assessment,(200,200,200))
    draw.text((10, ih-bar_h+9),
              f"Verde sano: {pct_green}%   Sospechoso: {pct_suspect}%   → {assessment.upper()}",
              fill=c, font=lbl_font)

    buf = io.BytesIO()
    vis_img.save(buf, "JPEG", quality=85)
    vis_b64 = base64.standard_b64encode(buf.getvalue()).decode()

    return {
        "pct_verde_sano": pct_green,
        "pct_sospechoso": pct_suspect,
        "assessment":     assessment,
        "visualization":  vis_b64,
        "zonas":          zonas,
        "n_green":        n_green,
        "n_suspect":      n_suspect,
        "total":          total,
    }


def _resize_for_analysis(raw: bytes) -> str:
    """Redimensiona bytes de imagen a resolución de análisis y devuelve base64."""
    img = Image.open(io.BytesIO(raw))
    img.thumbnail((ANALYSIS_MAX_W, ANALYSIS_MAX_H), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88, optimize=True)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def _get_analysis_image_b64(blob_name: str) -> str:
    raw = _bucket.blob(blob_name).download_as_bytes()
    return _resize_for_analysis(raw)


def _parse_analysis(text: str) -> dict:
    """Parsea la respuesta JSON del modelo, tolerando bloques markdown."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text.strip())
    try:
        data = json.loads(text.strip())
        if "hallazgos" not in data:
            data["hallazgos"] = [data.pop("notes", "Sin observaciones adicionales")]
        return data
    except json.JSONDecodeError:
        status = ("botrytis" if "botrytis" in text.lower() else
                  "sano"     if "sano"     in text.lower() else "sospechoso")
        return {
            "status": status,
            "confidence": "baja",
            "hallazgos": [text[:300]] if text else ["No se pudo interpretar la respuesta"],
            "resumen": "Clasificación automática por texto — revisar manualmente",
        }


# ── Caches en memoria ─────────────────────────────────────────────────────
_blobs_cache: tuple[list, datetime.datetime] | None = None
_url_cache: dict[str, tuple[str, datetime.datetime]] = {}
_analysis_cache: dict[str, dict] = {}        # blob_name → resultado Claude


def _load_analysis_cache() -> None:
    global _analysis_cache
    if not ANALYSIS_CACHE_FILE.exists():
        return
    try:
        data = json.loads(ANALYSIS_CACHE_FILE.read_text(encoding="utf-8"))
        _analysis_cache = data.get("claude", data) if isinstance(data, dict) and "claude" in data else data
        print(f"[cache] Cargados {len(_analysis_cache)} análisis Claude.")
    except Exception as e:
        print(f"[cache] No se pudo cargar analysis_cache.json: {e}")


def _save_analysis_cache() -> None:
    try:
        ANALYSIS_CACHE_FILE.write_text(
            json.dumps({"claude": _analysis_cache}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[cache] No se pudo guardar analysis_cache.json: {e}")


_load_analysis_cache()

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Flower Gallery", docs_url=None, redoc_url=None)
_INDEX_HTML = (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")


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
async def index():
    html = (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


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
            "filename":        name,
            "blob_name":       blob.name,
            "url":             _signed_url(blob.name),
            "thumb_url":       f"/thumb/{blob.name}",
            "time":            _parse_time(name),
            "size":            blob.size or 0,
            "analysis": _analysis_cache.get(blob.name),
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


class AnalyzeRequest(BaseModel):
    blob_name: str
    force: bool = False


@app.post("/api/analyze")
async def analyze_photo(req: AnalyzeRequest):
    """Analiza una foto con Claude Sonnet. Usa cache persistente salvo force=True."""
    blob_name = req.blob_name

    if not req.force and blob_name in _analysis_cache:
        return JSONResponse({**_analysis_cache[blob_name], "cached": True})

    if not _anthropic_client:
        raise HTTPException(503, "ANTHROPIC_API_KEY no configurado en el servidor")

    raw      = _bucket.blob(blob_name).download_as_bytes()
    img_b64  = _resize_for_analysis(raw)
    color    = _analyze_stem_colors(raw)
    response = _anthropic_client.messages.create(
        model=CLAUDE_MODEL, max_tokens=1024,
        messages=_claude_messages(img_b64, color)
    )
    result = _parse_analysis(response.content[0].text.strip())
    result["color"] = {k: v for k, v in color.items() if k != "visualization"}
    _analysis_cache[blob_name] = result
    _save_analysis_cache()
    return JSONResponse(result)


@app.post("/api/upload-analyze")
async def upload_and_analyze(file: UploadFile = File(...)):
    """Analiza una imagen subida manualmente con Claude (sin guardar en GCS)."""
    raw = await file.read()
    img = Image.open(io.BytesIO(raw))
    img.thumbnail((ANALYSIS_MAX_W, ANALYSIS_MAX_H), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88, optimize=True)
    img_b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    result: dict = {"filename": file.filename, "claude": None}

    if _anthropic_client:
        try:
            resp = _anthropic_client.messages.create(
                model=CLAUDE_MODEL, max_tokens=1024,
                messages=_claude_messages(img_b64)
            )
            result["claude"] = _parse_analysis(resp.content[0].text.strip())
        except Exception as exc:
            result["claude"] = {"status": "error", "confidence": "baja", "hallazgos": [str(exc)[:120]], "resumen": "Error en análisis"}

    return JSONResponse(result)


@app.get("/api/color-viz/{blob_path:path}")
async def color_viz(blob_path: str):
    """Devuelve imagen con overlay de análisis de color HSV."""
    try:
        raw    = _bucket.blob(blob_path).download_as_bytes()
        result = _analyze_stem_colors(raw)
        img_bytes = base64.standard_b64decode(result["visualization"])
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg",
                                 headers={"Cache-Control": "max-age=3600"})
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/color-stats/{blob_path:path}")
async def color_stats(blob_path: str):
    """Devuelve stats de color sin la imagen (más ligero)."""
    try:
        raw    = _bucket.blob(blob_path).download_as_bytes()
        result = _analyze_stem_colors(raw)
        return JSONResponse({k: v for k, v in result.items() if k != "visualization"})
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/refs/status")
async def refs_status():
    """Estado de las imágenes de referencia cargadas."""
    return JSONResponse({"count": len(_reference_b64), "dir": str(REFERENCES_DIR)})


@app.post("/api/refs/reload")
async def refs_reload():
    """Recarga las imágenes de referencia desde disco."""
    _load_reference_images()
    return JSONResponse({"count": len(_reference_b64)})


# ── Análisis en lote ──────────────────────────────────────────────────────
_batch_jobs: dict[str, dict] = {}  # day → {total, done, errors, running}


async def _run_batch_job(day: str, blobs: list) -> None:
    job = _batch_jobs[day]
    for blob in blobs:
        if not job["running"]:
            break
        try:
            img_b64 = _get_analysis_image_b64(blob.name)
            if _anthropic_client and blob.name not in _analysis_cache:
                resp = _anthropic_client.messages.create(
                    model=CLAUDE_MODEL, max_tokens=1024,
                    messages=_claude_messages(img_b64)
                )
                _analysis_cache[blob.name] = _parse_analysis(resp.content[0].text.strip())
            job["done"] += 1
            if job["done"] % 5 == 0:
                _save_analysis_cache()
        except Exception as e:
            job["errors"] += 1
            print(f"[batch] Error {blob.name}: {e}")
        await asyncio.sleep(BATCH_DELAY_SECS)

    _save_analysis_cache()
    job["running"] = False


@app.post("/api/analyze-batch/{day}")
async def analyze_batch_start(day: str, background_tasks: BackgroundTasks, force: bool = False):
    if not re.match(r"^\d{8}$", day):
        raise HTTPException(400, "Formato de fecha inválido")

    blobs = [b for b in _list_photo_blobs() if _parse_date(b.name.split("/")[-1]) == day]
    pending = [b for b in blobs if force or b.name not in _analysis_cache]

    if not pending:
        return JSONResponse({"status": "all_done", "total": len(blobs), "pending": 0})

    if day in _batch_jobs and _batch_jobs[day].get("running"):
        return JSONResponse({"status": "already_running", **_batch_jobs[day]})

    _batch_jobs[day] = {"total": len(pending), "done": 0, "errors": 0, "running": True}
    background_tasks.add_task(_run_batch_job, day, pending)
    return JSONResponse({"status": "started", "total": len(pending)})


@app.get("/api/analyze-batch/{day}/status")
async def analyze_batch_status(day: str):
    return JSONResponse(_batch_jobs.get(day, {"running": False, "done": 0, "total": 0, "errors": 0}))


@app.post("/api/analyze-batch/{day}/cancel")
async def analyze_batch_cancel(day: str):
    if day in _batch_jobs:
        _batch_jobs[day]["running"] = False
    return JSONResponse({"cancelled": True})


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
