#!/bin/bash
# deploy.sh — jalankan di server Ubuntu setelah file di-copy
set -e

echo "======================================"
echo "  Smart Waste Classifier — Deploy"
echo "======================================"

# ── 1. Install dependencies sistem ──────────────────────────────
echo "[1/7] Install nginx, python3, pip..."
apt-get install -y nginx python3 python3-pip python3-venv

# ── 2. Siapkan folder ───────────────────────────────────────────
echo "[2/7] Siapkan folder..."
mkdir -p /opt/smart-waste/backend
mkdir -p /var/www/smart-waste

# ── 3. Copy backend ─────────────────────────────────────────────
echo "[3/7] Copy backend files..."
cp -r /tmp/smart-waste/backend/* /opt/smart-waste/backend/
cp -r /tmp/smart-waste/backend/model /opt/smart-waste/backend/

# ── 4. Setup Python venv & install deps ─────────────────────────
echo "[4/7] Setup Python virtual environment..."
cd /opt/smart-waste/backend
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

# ── 5. Copy frontend build ──────────────────────────────────────
echo "[5/7] Copy frontend build..."
cp -r /tmp/smart-waste/frontend/dist/* /var/www/smart-waste/
chown -R www-data:www-data /var/www/smart-waste
chown -R www-data:www-data /opt/smart-waste

# ── 6. Setup nginx ──────────────────────────────────────────────
echo "[6/7] Setup nginx..."
cp /tmp/smart-waste/nginx.conf /etc/nginx/sites-available/smart-waste
ln -sf /etc/nginx/sites-available/smart-waste /etc/nginx/sites-enabled/smart-waste
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl restart nginx
systemctl enable nginx

# ── 7. Setup systemd service ────────────────────────────────────
echo "[7/7] Setup backend service..."
cp /tmp/smart-waste/smart-waste.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable smart-waste
systemctl restart smart-waste

echo ""
echo "======================================"
echo "  Deploy selesai!"
echo "  Backend : http://127.0.0.1:8000"
echo "  Frontend: http://localhost"
echo "======================================"
