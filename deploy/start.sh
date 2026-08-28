#!/usr/bin/env bash
# fusion-store fs-serve 启停脚本（F-OPS-6 部署配置）
# 用法：./deploy/start.sh start|stop|status|restart
# 默认 home=~/.fusion-store，port=11463，bind=127.0.0.1
# 环境变量覆盖：FS_HOME / FUSION_STORE_PORT / FS_BIND / FS_AUTH_TOKEN

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BINARY="$PROJECT_DIR/target/release/fusion-store"

HOME_DIR="${FS_HOME:-$HOME/.fusion-store}"
PORT="${FUSION_STORE_PORT:-11463}"
BIND="${FS_BIND:-127.0.0.1}"
PID_FILE="$HOME_DIR/fs-serve.pid"
LOG_FILE="$HOME_DIR/fs-serve.log"

cmd="${1:-}"
case "$cmd" in
    start)
        if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "[fusion-store] already running (pid $(cat "$PID_FILE"))"
            exit 0
        fi
        mkdir -p "$HOME_DIR"
        if [[ ! -x "$BINARY" ]]; then
            echo "[fusion-store] binary not found at $BINARY, run: cargo build --release -p fs-serve -p fs-cli"
            exit 1
        fi
        echo "[fusion-store] starting serve --home $HOME_DIR --port $PORT --bind $BIND"
        "$BINARY" serve \
            --home "$HOME_DIR" \
            --port "$PORT" \
            --bind "$BIND" \
            >"$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        sleep 0.5
        if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "[fusion-store] started (pid $(cat "$PID_FILE")), log: $LOG_FILE"
        else
            echo "[fusion-store] failed to start, see $LOG_FILE"
            rm -f "$PID_FILE"
            exit 1
        fi
        ;;
    stop)
        if [[ ! -f "$PID_FILE" ]]; then
            echo "[fusion-store] not running (no pid file)"
            exit 0
        fi
        pid="$(cat "$PID_FILE")"
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            for _ in 1 2 3 4 5; do
                kill -0 "$pid" 2>/dev/null || break
                sleep 0.5
            done
            if kill -0 "$pid" 2>/dev/null; then
                echo "[fusion-store] graceful stop timeout, sending SIGKILL"
                kill -9 "$pid"
            fi
            echo "[fusion-store] stopped (pid $pid)"
        else
            echo "[fusion-store] process $pid not alive, cleaning pid file"
        fi
        rm -f "$PID_FILE"
        ;;
    status)
        if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "[fusion-store] running (pid $(cat "$PID_FILE")), port $PORT"
            curl -s "http://$BIND:$PORT/health" || echo "[fusion-store] health check failed"
        else
            echo "[fusion-store] not running"
            exit 1
        fi
        ;;
    restart)
        "$0" stop || true
        "$0" start
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac
