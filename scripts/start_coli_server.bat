@echo off
setlocal enabledelayedexpansion

echo ==================================================
echo   Starting Colibri DeepSeek-V4 Server
echo ==================================================

:: Prefer run_deepseek_v4_gpu.bat if WSL CUDA is available
if exist "%~dp0run_deepseek_v4_gpu.bat" (
    call "%~dp0run_deepseek_v4_gpu.bat" serve %*
    goto :done
)

if exist "E:\dev\Antigravity\local_llm\scripts\start_coli_server.bat" (
    call "E:\dev\Antigravity\local_llm\scripts\start_coli_server.bat" %*
    goto :done
)

echo [ERROR] No Colibri start script found!
exit /b 1

:done
echo ==================================================
