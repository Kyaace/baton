@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "ACTION=%~1"

if "%ACTION%"=="" set "ACTION=status"

if /I "%ACTION%"=="start" goto :do_start
if /I "%ACTION%"=="stop" goto :do_stop
if /I "%ACTION%"=="restart" goto :do_restart
if /I "%ACTION%"=="status" goto :do_status

echo Usage: %~nx0 {start^|stop^|restart^|status}
exit /b 1

:do_start
call "%SCRIPT_DIR%start_ollama.bat"
goto :eof

:do_stop
call "%SCRIPT_DIR%stop_ollama.bat"
goto :eof

:do_restart
echo [INFO] Restarting System 1 / Ollama server...
call "%SCRIPT_DIR%stop_ollama.bat"
timeout /t 2 /nobreak >nul
call "%SCRIPT_DIR%start_ollama.bat"
goto :eof

:do_status
echo ==================================================
echo   System 1 / Ollama Server Status
echo ==================================================
if "%BATON_OLLAMA_URL%"=="" (
    set "OLLAMA_URL=http://127.0.0.1:11434"
) else (
    set "OLLAMA_URL=%BATON_OLLAMA_URL%"
)
echo Endpoint: %OLLAMA_URL%
if defined BATON_OLLAMA_START_CMD echo Start Hook: %BATON_OLLAMA_START_CMD%
if defined BATON_OLLAMA_STOP_CMD  echo Stop Hook:  %BATON_OLLAMA_STOP_CMD%
curl.exe -s --connect-timeout 1 "%OLLAMA_URL%/api/version" >nul 2>&1
if not errorlevel 1 (
    echo Health:   ONLINE [responding]
) else (
    echo Health:   OFFLINE
)
echo ==================================================
goto :eof
