@echo off
setlocal enabledelayedexpansion

:: 1. Active virtual environment
if defined VIRTUAL_ENV (
    call :check_python "%VIRTUAL_ENV%\Scripts\python.exe"
    if not errorlevel 1 goto :found_python
)

:: 2. Repository-local virtual environment (.venv or venv)
call :check_python "%~dp0.venv\Scripts\python.exe"
if not errorlevel 1 goto :found_python
call :check_python "%~dp0venv\Scripts\python.exe"
if not errorlevel 1 goto :found_python

:: 3. Active Conda environment
if defined CONDA_PREFIX (
    call :check_python "%CONDA_PREFIX%\python.exe"
    if not errorlevel 1 goto :found_python
)

:: 4. System PATH python (verifies executable works, filtering out WindowsApps 0-byte stubs)
for /f "delims=" %%I in ('where python 2^>nul') do (
    call :check_python "%%I"
    if not errorlevel 1 goto :found_python
)

:: 5. Anaconda3 system installation
call :check_python "C:\ProgramData\Anaconda3\python.exe"
if not errorlevel 1 goto :found_python

:: 6. Windows Python launcher (py.exe)
call :check_python "py"
if not errorlevel 1 goto :found_python

echo [ERROR] No working Python interpreter found!
echo Please ensure Python is installed and accessible, or activate a virtual environment.
exit /b 1

:check_python
set "CANDIDATE=%~1"
if "%CANDIDATE%"=="" exit /b 1
if exist "%CANDIDATE%" (
    "%CANDIDATE%" -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "PY_EXE=%CANDIDATE%"
        exit /b 0
    )
) else (
    where "%CANDIDATE%" >nul 2>&1
    if not errorlevel 1 (
        "%CANDIDATE%" -c "import sys" >nul 2>&1
        if not errorlevel 1 (
            set "PY_EXE=%CANDIDATE%"
            exit /b 0
        )
    )
)
exit /b 1

:found_python
set "PYTHONPATH=%~dp0;%PYTHONPATH%"

:: When running chat, ensure llama-server router is running on port 8080 instead of ollama
if /i not "%~1"=="chat" goto :run_cli

netstat -ano 2>nul | findstr :8080 | findstr LISTENING >nul 2>&1
if not errorlevel 1 goto :run_cli

echo [INFO] llama-server router is not running on port 8080.
echo [INFO] Booting up llama-server Multi-Model Router (Ornith + Qwen3)...
start "llama-server Router" /min cmd /c "%~dp0scripts\run_router.bat serve > %~dp0router_server.log 2>&1"
timeout /t 2 /nobreak >nul 2>&1

:run_cli
"%PY_EXE%" -m baton.cli %*
