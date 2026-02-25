#!/usr/bin/env bash
set -euo pipefail

# Run qllmd + qllm-serve and execute the tests using a real GGUF model.
# Usage: make run
# Optionally set MODEL=/path/to/model.gguf

ROOT=$(cd "$(dirname "$0")/.." && pwd)
QLLMD_BIN="$ROOT/bin/qllmd"
SERVE_BIN="$ROOT/bin/qllm-serve"
LOGDIR=/tmp
QLLMD_LOG="$LOGDIR/qllmd.log"
SERVE_LOG="$LOGDIR/qllm-serve.log"
QLLMD_PID_FILE="$LOGDIR/qllmd.pid"
SERVE_PID_FILE="$LOGDIR/qllm-serve.pid"

MODEL_PATH="${MODEL:-}"

find_default_model() {
    # Try a few likely places for a gguf model in the user's cache or repo
    local m
    m=$(ls -d "$HOME/.cache/huggingface/hub/models--"*/snapshots/*/*.gguf 2>/dev/null | head -n1 || true)
    if [ -n "$m" ]; then echo "$m"; return 0; fi
    m=$(ls -d "$HOME/.cache/huggingface/hub/models--"*/*.gguf 2>/dev/null | head -n1 || true)
    if [ -n "$m" ]; then echo "$m"; return 0; fi
    # fallback: search repo for a named Phi-3 snapshot (best-effort)
    m=$(find "$ROOT" -maxdepth 6 -type f -name "*Phi-3*.gguf" 2>/dev/null | head -n1 || true)
    if [ -n "$m" ]; then echo "$m"; return 0; fi
    return 1
}

cleanup() {
    echo "[run-integration] Cleaning up..."
    if [ -f "$SERVE_PID_FILE" ]; then
        kill "$(cat $SERVE_PID_FILE)" 2>/dev/null || true
        rm -f "$SERVE_PID_FILE"
    fi
    if [ -f "$QLLMD_PID_FILE" ]; then
        kill "$(cat $QLLMD_PID_FILE)" 2>/dev/null || true
        rm -f "$QLLMD_PID_FILE"
    fi
}

trap cleanup EXIT

if [ -z "$MODEL_PATH" ]; then
    echo "[run-integration] MODEL not set, searching for a model..."
    MODEL_PATH=$(find_default_model || true)
fi

if [ -z "$MODEL_PATH" ]; then
    echo "[run-integration] No .gguf model found. Set MODEL=/path/to/model.gguf and retry." >&2
    exit 2
fi

echo "[run-integration] Using model: $MODEL_PATH"

echo "[run-integration] Starting qllmd... (logs -> $QLLMD_LOG)"
rm -f "$QLLMD_LOG" "$SERVE_LOG" || true
# Preserve existing LD_LIBRARY_PATH if set
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$ROOT/lib"
export LD_LIBRARY_PATH
"$QLLMD_BIN" -d -p 4242 "$MODEL_PATH" &> "$QLLMD_LOG" &
echo $! > "$QLLMD_PID_FILE"

echo "[run-integration] waiting for qllmd to respond to 'info' (timeout 120s)"
RETRY=0
MAX=120
while true; do
    if python3 - <<PY >/dev/null 2>&1
import socket,sys
try:
    s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1',4242)); s.sendall(b'info\n'); data=s.recv(65536); s.close();
    sys.stdout.write(data.decode('utf-8',errors='ignore'))
except Exception as e:
    sys.exit(1)
PY
    then
        echo "[run-integration] qllmd responded"
        break
    fi
    RETRY=$((RETRY+1))
    if [ $RETRY -ge $MAX ]; then
        echo "[run-integration] qllmd did not respond within timeout; check $QLLMD_LOG" >&2
        exit 3
    fi
    sleep 1
done

echo "[run-integration] Starting qllm-serve... (logs -> $SERVE_LOG)"
"$SERVE_BIN" --port 8002 --qllmd-port 4242 &> "$SERVE_LOG" &
echo $! > "$SERVE_PID_FILE"

sleep 1

echo "[run-integration] Running tests (tests/Makefile run)"
set -x
make -C tests run
RC=$?
set +x

if [ $RC -eq 0 ]; then
    echo "[run-integration] Tests passed"
else
    echo "[run-integration] Tests failed (rc=$RC). Check $SERVE_LOG and $QLLMD_LOG" >&2
fi

exit $RC
