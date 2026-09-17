#!/usr/bin/env bash
# Start/stop the macOS `say` TTS shim on 127.0.0.1:$SAY_SHIM_PORT (default 9999).
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
port="${SAY_SHIM_PORT:-9999}"
export SAY_SHIM_DEFAULT_VOICE="${SAY_SHIM_DEFAULT_VOICE:-Yuna (Premium)}"
python_bin="${SAY_SHIM_PYTHON:-$here/../opentalking/.venv/bin/python}"
log="$here/shim.log"
pidfile="$here/shim.pid"

stop() {
  local pids
  pids="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "stopping shim on :$port (pid $pids)"
    kill $pids 2>/dev/null || true
    for _ in $(seq 1 20); do
      lsof -nP -iTCP:"$port" -sTCP:LISTEN -t >/dev/null 2>&1 || break
      sleep 0.2
    done
    kill -9 $pids 2>/dev/null || true
  else
    echo "shim not running on :$port"
  fi
  rm -f "$pidfile"
}

start() {
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "port $port already in use; run '$0 restart'" >&2
    exit 1
  fi
  ( cd "$here" && SAY_SHIM_PORT="$port" nohup "$python_bin" server.py >"$log" 2>&1 & echo $! >"$pidfile" )
  for _ in $(seq 1 40); do
    if curl -sf --max-time 2 "http://127.0.0.1:$port/health" >/dev/null; then
      echo "shim up on :$port (pid $(cat "$pidfile"))"
      curl -s "http://127.0.0.1:$port/health"; echo
      return 0
    fi
    sleep 0.25
  done
  echo "shim failed to start; see $log" >&2
  tail -20 "$log" >&2
  exit 1
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status)
    curl -s --max-time 2 "http://127.0.0.1:$port/health" && echo || echo "not running on :$port" ;;
  *) echo "usage: $0 {start|stop|restart|status}" >&2; exit 2 ;;
esac
