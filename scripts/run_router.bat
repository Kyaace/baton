@echo off
setlocal enabledelayedexpansion
title Multi-Model Router (Ornith + Qwen3) - llama-server

REM ==============================================================================
REM Script: run_router.bat
REM Purpose: Windows launcher for llama-server Multi-Model Router with dynamic
REM          VRAM/RAM swapping between Ornith-1.5-35B-A3B and Qwen3:14B.
REM ==============================================================================

:: Locate llama-server binary
where llama-server.exe >nul 2>&1
if errorlevel 1 (
    if exist "C:\llama.cpp\llama-server.exe" (
        set "PATH=C:\llama.cpp;!PATH!"
    ) else (
        echo [ERROR] llama-server.exe not found in PATH or in C:\llama.cpp\
        exit /b 1
    )
)

:: Locate preset configuration file
set "PRESET_FILE=%BATON_ROUTER_PRESET%"
if "%PRESET_FILE%"=="" (
    if exist "%~dp0..\models_preset.ini" (
        set "PRESET_FILE=%~dp0..\models_preset.ini"
    ) else if exist "%~dp0models_preset.ini" (
        set "PRESET_FILE=%~dp0models_preset.ini"
    ) else if exist "E:\dev\Antigravity\local_llm\models_preset.ini" (
        set "PRESET_FILE=E:\dev\Antigravity\local_llm\models_preset.ini"
    )
)

if not exist "%PRESET_FILE%" (
    echo [ERROR] Preset configuration not found: %PRESET_FILE%
    exit /b 1
)

:: Port configuration (default 8080)
set "PORT=%BATON_ROUTER_PORT%"
if "%PORT%"=="" set "PORT=8080"

:: Handle 'serve' command argument passed by orchestrator
set "EXTRA_ARGS="
:parse_args
if "%~1"=="" goto run
if /i "%~1"=="serve" (
    shift
    goto parse_args
)
set "EXTRA_ARGS=!EXTRA_ARGS! %1"
shift
goto parse_args

:run
echo ==============================================================================
echo   Launching llama-server Multi-Model Router (Dynamic VRAM/RAM Swapping)
echo ==============================================================================
echo   Preset:     %PRESET_FILE%
echo   Max VRAM:   1 model active at a time (--models-max 1)
echo   Models:     - Ornith-1.5-35B-A3B (MoE: VRAM dense + RAM experts, MTP)
echo               - Qwen3:14B (Dense: 100% VRAM offloaded)
echo   Endpoint:   http://127.0.0.1:%PORT%/v1
echo ==============================================================================

llama-server.exe ^
  --models-preset "%PRESET_FILE%" ^
  --models-max 1 ^
  --host 0.0.0.0 ^
  --port %PORT% !EXTRA_ARGS!
