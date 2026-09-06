#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "[1/5] Installing dependencies..."
pip install -r requirements.txt || {
    echo "Retrying with mirror..."
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
    pip install -r requirements.txt
}

echo "[2/5] Preparing .env..."
[ -f .env ] || cp .env.example .env

echo "[3/5] Creating local admin if needed..."
python3 scripts/create_local_admin.py --email admin@example.com --password StrongPass123! --name "Local Admin"

echo "[4/5] Running pre-flight smoke check..."
python3 scripts/check_setup.py

echo "[5/5] Starting API on http://127.0.0.1:8000 ..."
(sleep 4 && xdg-open http://127.0.0.1:8000/ 2>/dev/null || open http://127.0.0.1:8000/ 2>/dev/null || echo "Open http://127.0.0.1:8000/") &
python3 scripts/run_local.py
