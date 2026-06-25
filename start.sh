#!/bin/bash
set -e

echo "[start] Launching NB-Whisper server (model load may take ~30s)..."
python3 /app/whisper_server.py &
WHISPER_PID=$!

echo "[start] Waiting for Whisper server to be ready..."
for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:8765/health > /dev/null 2>&1; then
    echo "[start] Whisper server is ready after ${i}s."
    break
  fi
  if ! kill -0 "$WHISPER_PID" 2>/dev/null; then
    echo "[start] ERROR: Whisper server process exited unexpectedly." >&2
    exit 1
  fi
  sleep 1
done

if ! curl -sf http://127.0.0.1:8765/health > /dev/null 2>&1; then
  echo "[start] ERROR: Whisper server did not become ready within 180s." >&2
  exit 1
fi

echo "[start] Starting Node.js server..."
exec node /app/server.js
