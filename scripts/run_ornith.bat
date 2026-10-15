@echo off
setlocal enabledelayedexpansion
title Ornith-1.5-35B-A3B (llama-server)

REM ==============================================================================
REM Script: run_ornith.bat
REM Purpose: Windows launcher for Ornith-1.5-35B-A3B MoE on llama-server with
REM          CUDA acceleration (RTX 3060 dense + 128GB RAM MoE experts) and MTP.
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

:: Model path resolution
set "MODEL_PATH=C:\colibri_models\Ornith-1.5-35B-A3B\Ornith-1.5-35B-A3B-Q4_K_M.gguf"
if not "%BATON_ORNITH_MODEL%"=="" set "MODEL_PATH=%BATON_ORNITH_MODEL%"
if not "%BATON_SYSTEM2_MODEL_PATH%"=="" set "MODEL_PATH=%BATON_SYSTEM2_MODEL_PATH%"

if not exist "%MODEL_PATH%" (
    echo [ERROR] Ornith model file not found at %MODEL_PATH%
    exit /b 1
)

:: Port configuration (default 8000 for seamless drop-in Baton compatibility)
set "PORT=%BATON_SYSTEM2_PORT%"
if "%PORT%"=="" set "PORT=8000"

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
echo   Launching Ornith-1.5-35B-A3B on llama-server (CUDA)
echo ==============================================================================
echo   Model:       %MODEL_PATH%
echo   VRAM Dense:  -ngl 99 (RTX 3060 12GB)
echo   RAM Experts: -ot ".ffn_.*_exps.=CPU" (System RAM MoE routing)
echo   Speculative: --spec-type draft-mtp (Multi-Token Prediction)
echo   Context:     -c 32768 (32,768 tokens)
echo   Endpoint:    http://127.0.0.1:%PORT%/v1
echo ==============================================================================

llama-server.exe ^
  -m "%MODEL_PATH%" ^
  -ngl 99 ^
  -ot ".ffn_.*_exps.=CPU" ^
  --spec-type draft-mtp ^
  -c 32768 ^
  --host 0.0.0.0 ^
  --port %PORT% !EXTRA_ARGS!
