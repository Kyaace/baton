@echo off
setlocal enabledelayedexpansion

echo ==================================================
echo   Starting System 1 - Ollama Server
echo ==================================================

if "%BATON_OLLAMA_URL%"=="" (
    set "OLLAMA_URL=http://127.0.0.1:11434"
) else (
    set "OLLAMA_URL=%BATON_OLLAMA_URL%"
)

:: 1. Check if already online and responsive
curl.exe -s --connect-timeout 1 "%OLLAMA_URL%/api/version" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Ollama server is already running and reachable at %OLLAMA_URL%
    goto :done
)

:: 2. Check if custom start hook command is configured
if defined BATON_OLLAMA_START_CMD (
    echo [INFO] Executing custom start hook: %BATON_OLLAMA_START_CMD%
    cmd /c "%BATON_OLLAMA_START_CMD%"
    goto :wait_loop
)

:: 3. Locate Ollama binary
set "OLLAMA_EXE="
where ollama >nul 2>&1
if not errorlevel 1 (
    set "OLLAMA_EXE=ollama"
) else if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
) else (
    echo [ERROR] ollama.exe was not found in PATH or in %LOCALAPPDATA%\Programs\Ollama\
    exit /b 1
)

echo [INFO] Launching Ollama daemon (minimized background window)...
start "Ollama Server" /min "!OLLAMA_EXE!" serve

:wait_loop
:: Wait up to 15 seconds for server to respond
echo [INFO] Waiting for server at %OLLAMA_URL% to initialize...
for /L %%i in (1,1,15) do (
    ping 127.0.0.1 -n 2 >nul
    curl.exe -s --connect-timeout 1 "%OLLAMA_URL%/api/version" >nul 2>&1
    if not errorlevel 1 (
        echo [SUCCESS] Ollama server is online and listening at %OLLAMA_URL%
        goto :done
    )
)

echo [WARNING] Ollama process launched, but server healthcheck timed out at %OLLAMA_URL%.
exit /b 1

:done
echo ==================================================
