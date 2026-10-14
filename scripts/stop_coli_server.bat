@echo off
setlocal

echo ==================================================
echo   Stopping Colibri Server and Freeing Memory
echo ==================================================

:: 1. Stop WSL Colibri instance if active
wsl -e /mnt/e/dev/Antigravity/local_llm/colibri/c/coli stop 2>nul

:: 2. Stop native Windows Colibri instance if active
if exist "E:\dev\Antigravity\local_llm\coli.cmd" (
    call "E:\dev\Antigravity\local_llm\coli.cmd" stop %* 2>nul
)

echo ==================================================
