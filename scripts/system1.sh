#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-status}"

case "$ACTION" in
    start)
        "$SCRIPT_DIR/start_ollama.sh"
        ;;
    stop)
        "$SCRIPT_DIR/stop_ollama.sh"
        ;;
    restart)
        "$SCRIPT_DIR/stop_ollama.sh"
        sleep 2
        "$SCRIPT_DIR/start_ollama.sh"
        ;;
    status)
        OLLAMA_URL="${BATON_OLLAMA_URL:-http://127.0.0.1:11434}"
        echo "=================================================="
        echo "  System 1 / Ollama Server Status"
        echo "=================================================="
        echo "Endpoint: $OLLAMA_URL"
        [ -n "$BATON_OLLAMA_START_CMD" ] && echo "Start Hook: $BATON_OLLAMA_START_CMD"
        [ -n "$BATON_OLLAMA_STOP_CMD" ] && echo "Stop Hook:  $BATON_OLLAMA_STOP_CMD"
        if curl -s --connect-timeout 1 "$OLLAMA_URL/api/version" >/dev/null 2>&1; then
            echo "Health:   ONLINE [responding]"
        else
            echo "Health:   OFFLINE"
        fi
        echo "=================================================="
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
