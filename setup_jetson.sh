#!/bin/bash
# setup_jetson.sh
# Crea el ambiente virtual e instala dependencias para la Jetson.
# Uso: bash setup_jetson.sh

set -e

VENV_DIR="venv"
PYTHON="python3"

echo "======================================"
echo " Setup app_flowers - Jetson"
echo "======================================"

# Dependencias del sistema necesarias para OpenCV y GStreamer
echo "[1/4] Instalando dependencias del sistema..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    libgstreamer1.0-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    libcamera-dev 2>/dev/null || true

# Crear ambiente virtual
echo "[2/4] Creando ambiente virtual en ./${VENV_DIR}..."
$PYTHON -m venv $VENV_DIR --system-site-packages
# --system-site-packages permite acceder al OpenCV optimizado de JetPack

# Activar ambiente virtual
source $VENV_DIR/bin/activate

# Actualizar pip
echo "[3/4] Actualizando pip..."
pip install --upgrade pip --quiet

# Instalar dependencias Python
echo "[4/4] Instalando dependencias Python..."
pip install \
    "google-cloud-storage>=2.0.0" \
    "google-auth>=2.0.0"

# Verificar OpenCV
echo ""
echo "Verificando OpenCV..."
python -c "import cv2; print(f'  OpenCV {cv2.__version__} OK')" || \
    pip install "opencv-python>=4.5.0"

echo ""
echo "======================================"
echo " Instalación completa."
echo " Para activar el ambiente:"
echo "   source ${VENV_DIR}/bin/activate"
echo ""
echo " Para ejecutar:"
echo "   python capture_upload.py"
echo "   python capture_upload.py --camera-type csi"
echo "   python capture_upload.py --num-fotos 10 --interval 5"
echo "======================================"
