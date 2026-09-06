@echo off
setlocal
cd /d %~dp0

echo [1/5] Installing dependencies...
py -m pip install -r requirements.txt
if errorlevel 1 (
    echo Retrying with mirror...
    py -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    py -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
    py -m pip install -r requirements.txt || exit /b 1
)

echo [2/5] Preparing .env...
if not exist .env copy .env.example .env >nul

echo [3/5] Creating local admin if needed...
py scripts\create_local_admin.py --email admin@example.com --password StrongPass123! --name "Local Admin" || exit /b 1

echo [4/5] Running pre-flight smoke check...
py scripts\check_setup.py || exit /b 1

echo [5/5] Starting API on http://127.0.0.1:8000 ...
start "" /min cmd /c "timeout /t 4 >nul & start http://127.0.0.1:8000/"
py scripts\run_local.py
