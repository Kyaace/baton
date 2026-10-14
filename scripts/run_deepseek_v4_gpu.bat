@echo off
setlocal enabledelayedexpansion

set MODE=%~1
if "%MODE%"=="" set MODE=serve

shift
set EXTRA_ARGS=
:parse_args
if "%~1"=="" goto run
set EXTRA_ARGS=%EXTRA_ARGS% %~1
shift
goto parse_args

:run
if exist "%~dp0run_deepseek_v4_gpu.sh" (
    set "SH_PATH=%~dp0run_deepseek_v4_gpu.sh"
    set "SH_PATH=!SH_PATH:\=/!"
    set "SH_PATH=/mnt/!SH_PATH:~0,1!!SH_PATH:~2!"
    wsl -e !SH_PATH! %MODE% %EXTRA_ARGS%
) else (
    wsl -e /mnt/e/dev/Antigravity/local_llm/scripts/run_deepseek_v4_gpu.sh %MODE% %EXTRA_ARGS%
)
