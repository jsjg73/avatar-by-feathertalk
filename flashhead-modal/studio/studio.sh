#!/usr/bin/env bash
# Start / stop FlashHead Studio on 127.0.0.1:${STUDIO_PORT:-8300}.
#
# Always start it through this script rather than by hand. A server launched as a
# plain `nohup … &` job from an interactive shell stays in that terminal's job
# control and can be stopped by it, which reads exactly like a hang.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
port="${STUDIO_PORT:-8300}"
# Loopback by default: the page has no auth, and anyone who reaches it can spend
# GPU time, upload images and pop Finder windows. STUDIO_HOST=0.0.0.0 opens it
# to the LAN (e.g. to watch from a phone) — do that knowingly.
host="${STUDIO_HOST:-127.0.0.1}"
py=/Users/kjs0703/playground/opentalk/opentalking/.venv/bin/python
log="$here/studio.log"

# A stopped (STAT "T") server does not run its signal handlers, so a plain
# SIGTERM is queued and never acted on: the process stays alive, keeps the port,
# and the next start fails with "address already in use". Resume it first, give
# it a second to exit cleanly, then take what is left with SIGKILL.
stop() {
  pids="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)"
  [ -z "$pids" ] && { echo "not running"; return; }
  echo "stopping :$port ($pids)"
  kill -CONT $pids 2>/dev/null || true
  kill $pids 2>/dev/null || true
  for _ in $(seq 1 12); do
    sleep 0.25
    pids="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)"
    [ -z "$pids" ] && return
  done
  echo "  still holding the port; SIGKILL ($pids)"
  kill -9 $pids 2>/dev/null || true
  sleep 1
}
start() {
  lsof -nP -iTCP:"$port" -sTCP:LISTEN -t >/dev/null 2>&1 && { echo "port $port busy; use '$0 restart'"; exit 1; }
  # `( ... & )` orphans the process group, and stdin from /dev/null removes the
  # other way a background job gets stopped (SIGTTIN on a terminal read). Both
  # matter: a stopped server is indistinguishable from a hung one — connections
  # are accepted by the kernel, nothing answers, no CPU is used, no log is written.
  ( cd "$here" && nohup "$py" -m uvicorn server:app --host "$host" --port "$port" </dev/null >>"$log" 2>&1 & )
  for _ in $(seq 1 40); do
    if curl -sf "http://127.0.0.1:$port/api/config" >/dev/null; then
      pid="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -1)"
      echo "studio up: http://127.0.0.1:$port (bound to $host, pid $pid)"
      echo "  응답이 없으면: ps -o stat= -p $pid  (T 면 정지) · kill -USR1 $pid → studio.log 에 스레드 스택"
      exit 0
    fi
    sleep 0.25
  done
  echo "failed to start; see $log"; tail -20 "$log"; exit 1
}
case "${1:-start}" in start) start;; stop) stop;; restart) stop; start;; *) echo "usage: $0 {start|stop|restart}"; exit 2;; esac
