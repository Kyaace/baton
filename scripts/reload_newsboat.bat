@echo off
echo ==================================================
echo   Reloading Newsboat RSS Feeds (WSL)
echo ==================================================
wsl.exe -d Ubuntu-24.04 newsboat -x reload
if %ERRORLEVEL% equ 0 (
    echo [INFO] Newsboat RSS cache successfully updated in baton\data\rss\cache.db
) else (
    echo [ERROR] Failed to reload Newsboat feeds.
)
