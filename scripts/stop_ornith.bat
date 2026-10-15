@echo off
setlocal enabledelayedexpansion

echo ==================================================
echo   Stopping llama-server (Ornith)
echo ==================================================

tasklist /FI "IMAGENAME eq llama-server.exe" 2>nul | find /I /N "llama-server.exe" >nul
if not errorlevel 1 (
    echo [INFO] Stopping running llama-server process...
    taskkill /F /T /IM llama-server.exe >nul 2>&1
    echo [INFO] llama-server stopped successfully.
) else (
    echo [INFO] llama-server is not currently running.
)

echo ==================================================
