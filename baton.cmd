@echo off
setlocal
set "PY_EXE=C:\ProgramData\Anaconda3\python.exe"
if not exist "%PY_EXE%" (
    where python >nul 2>&1 && set "PY_EXE=python"
)

set "PYTHONPATH=%~dp0;%PYTHONPATH%"
"%PY_EXE%" -m baton.cli %*
