#!/bin/bash

# MCP Server 관리 스크립트
#   ./mcp.sh start    서버 시작
#   ./mcp.sh stop     서버 종료
#   ./mcp.sh restart  서버 재시작
#   ./mcp.sh status   서버 상태 확인

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PID_FILE="$SCRIPT_DIR/mcp-server.pid"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/mcp-server.log"
VENV="$SCRIPT_DIR/venv/bin/activate"
PYTHON="$SCRIPT_DIR/venv/bin/python3"
APP="$SCRIPT_DIR/main.py"

mkdir -p "$LOG_DIR"

# ── 색상 ──────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            return 0  # 실행 중
        fi
    fi
    return 1  # 중지 상태
}

start() {
    if is_running; then
        PID=$(cat "$PID_FILE")
        echo -e "${YELLOW}[mcp-server] 이미 실행 중입니다. (PID: $PID)${NC}"
        return 1
    fi

    if [ ! -f "$VENV" ]; then
        echo -e "${RED}[mcp-server] venv를 찾을 수 없습니다: $VENV${NC}"
        return 1
    fi

    echo -e "${GREEN}[mcp-server] 시작 중...${NC}"
    cd "$SCRIPT_DIR"
    source "$VENV"
    nohup "$PYTHON" "$APP" >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    PID=$(cat "$PID_FILE")

    # 정상 기동 확인 (최대 5초 대기)
    for i in $(seq 1 5); do
        sleep 1
        if is_running; then
            echo -e "${GREEN}[mcp-server] 시작 완료 (PID: $PID)${NC}"
            return 0
        fi
    done

    echo -e "${RED}[mcp-server] 시작 실패. 로그를 확인하세요: $LOG_FILE${NC}"
    rm -f "$PID_FILE"
    return 1
}

stop() {
    if ! is_running; then
        echo -e "${YELLOW}[mcp-server] 실행 중이 아닙니다.${NC}"
        rm -f "$PID_FILE"
        return 0
    fi

    PID=$(cat "$PID_FILE")
    echo -e "${GREEN}[mcp-server] 종료 중... (PID: $PID)${NC}"
    kill "$PID"

    # 최대 10초 대기 후 강제 종료
    for i in $(seq 1 10); do
        sleep 1
        if ! kill -0 "$PID" 2>/dev/null; then
            rm -f "$PID_FILE"
            echo -e "${GREEN}[mcp-server] 종료 완료${NC}"
            return 0
        fi
    done

    echo -e "${RED}[mcp-server] 정상 종료 실패. 강제 종료합니다. (PID: $PID)${NC}"
    kill -9 "$PID" 2>/dev/null
    rm -f "$PID_FILE"
    echo -e "${GREEN}[mcp-server] 강제 종료 완료${NC}"
}

restart() {
    echo -e "${GREEN}[mcp-server] 재시작 중...${NC}"
    stop
    sleep 1
    start
}

status() {
    if is_running; then
        PID=$(cat "$PID_FILE")
        echo -e "${GREEN}[mcp-server] 실행 중 (PID: $PID)${NC}"
    else
        echo -e "${RED}[mcp-server] 중지 상태${NC}"
    fi
}

# ─────────────────────────────────────────
# 명령어 분기
# ─────────────────────────────────────────
case "$1" in
    start)   start   ;;
    stop)    stop    ;;
    restart) restart ;;
    status)  status  ;;
    *)
        echo "사용법: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac