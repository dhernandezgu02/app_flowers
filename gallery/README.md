# FlowerGallery — Servidor Web

App web para ver y descargar las fotos capturadas por la Jetson.

## Setup

```bash
# 1. Copiar credenciales de GCS
cp ../green-alchemy-301821-9753e8366e05.json credentials.json

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Correr el servidor
uvicorn app:app --host 0.0.0.0 --port 8000

# 4. Abrir en el navegador
http://localhost:8000
```

## Notas

- **Thumbnails**: se generan automáticamente la primera vez que se abre un día
  y se guardan en GCS bajo `thumbs/`. Las siguientes cargas son instantáneas.
- **ZIP**: el botón "Descargar día" genera un ZIP con todas las fotos del día.
- **Credenciales**: el archivo `credentials.json` debe ser el mismo service account
  que tiene acceso al bucket `bucket_flower`.
