#!/usr/bin/env bash
# Reloads Newsboat feeds in WSL and updates baton/data/rss/cache.db
set -e

echo "[INFO] Reloading Newsboat RSS feeds..."
newsboat -x reload
echo "[INFO] Cache updated successfully."
