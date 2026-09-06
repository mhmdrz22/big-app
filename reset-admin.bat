@echo off
setlocal
cd /d %~dp0
echo Resetting local admin credentials to admin@example.com / StrongPass123! ...
py scripts\create_local_admin.py --email admin@example.com --password StrongPass123! --name "Local Admin"
echo.
echo Now open http://127.0.0.1:8000/admin and log in with those credentials.
pause
