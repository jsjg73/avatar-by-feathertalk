#!/usr/bin/env bash
# Start / stop FlashHead Studio on 127.0.0.1:${STUDIO_PORT:-8300}.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
port="${STUDIO_PORT:-8300}"
# Loopback by default: the page has no auth, and anyone who reaches it can spend
# GPU time, upload images and pop Finder windows. STUDIO_HOST=0.0.0.0 opens it
# to the LAN (e.g. to watch from a phone) — do that knowingly.
host="${STUDIO_HOST:-127.0.0.1}"
py=/Users/kjs0703/playground/opentalk/opentalking/.venv/bin/python
log="$here/studio.log"

stop() {
  pids="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)"
  [ -n "$pids" ] && { echo "stopping :$port ($pids)"; kill $pids 2>/dev/null || true; sleep 1; } || echo "not running"
}
start() {
  lsof -nP -iTCP:"$port" -sTCP:LISTEN -t >/dev/null 2>&1 && { echo "port $port busy; use '$0 restart'"; exit 1; }
  ( cd "$here" && nohup "$py" -m uvicorn server:app --host "$host" --port "$port" >"$log" 2>&1 & )
  for _ in $(seq 1 40); do curl -sf "http://127.0.0.1:$port/api/config" >/dev/null && { echo "studio up: http://127.0.0.1:$port (bound to $host)"; exit 0; }; sleep 0.25; done
  echo "failed to start; see $log"; tail -20 "$log"; exit 1
}
case "${1:-start}" in start) start;; stop) stop;; restart) stop; start;; *) echo "usage: $0 {start|stop|restart}"; exit 2;; esac
