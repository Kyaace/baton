@echo off
setlocal

echo ==================================================
echo   Stopping System 1 - Ollama Server
echo ==================================================

if "%BATON_OLLAMA_URL%"=="" (
    set "OLLAMA_URL=http://127.0.0.1:11434"
) else (
    set "OLLAMA_URL=%BATON_OLLAMA_URL%"
)

:: 1. Check if custom stop hook command is configured
if defined BATON_OLLAMA_STOP_CMD (
    echo [INFO] Executing custom stop hook: %BATON_OLLAMA_STOP_CMD%
    cmd /c "%BATON_OLLAMA_STOP_CMD%"
    goto :verify
)

:: 2. Standard process termination fallback
echo [INFO] Terminating Ollama processes...
taskkill /F /T /IM ollama.exe >nul 2>&1
taskkill /F /T /IM "ollama app.exe" >nul 2>&1
taskkill /F /T /IM llama-server.exe >nul 2>&1
echo [SUCCESS] Process termination sent.

:verify
ping 127.0.0.1 -n 2 >nul
curl.exe -s --connect-timeout 1 "%OLLAMA_URL%/api/version" >nul 2>&1
if errorlevel 1 (
    echo [SUCCESS] Verified: System 1 Ollama server is offline.
) else (
    echo [WARNING] Ollama server is still responding at %OLLAMA_URL%.
)

echo ==================================================
