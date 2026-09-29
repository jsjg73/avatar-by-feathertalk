#!/bin/bash
# W1 of docs/live-serving monitoring plan: a fully independent watcher.
# Never touches gateway.py or live_server.py -- only observes them from
# outside via ps/nvidia-smi/df, so it can be started (or killed) without any
# risk to sessions currently in flight.
#
# Appends one line to $ALERT_LOG only when something CHANGES (not every
# tick) -- a 24/7 loop that logs "still fine" every 30s would drown the one
# line that matters.
#
# Usage: nohup ./monitor_watch.sh > /root/monitor_watch.log 2>&1 & disown

ALERT_LOG=/root/monitor_alerts.log
EXPECTED_WORKERS=24
INTERVAL_S=30

prev_gateway=""
prev_workers=""
prev_gpu_err=""

while true; do
  ts=$(date -Iseconds)

  gateway_alive=$(pgrep -f "python gateway.py" >/dev/null 2>&1 && echo yes || echo no)
  worker_count=$(pgrep -c -f "python.*live_server.py" 2>/dev/null || echo 0)
  gpu_err=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader 2>/dev/null)
  disk_pct=$(df /root 2>/dev/null | tail -1 | awk '{gsub("%","",$5); print $5}')
  mem_avail_gb=$(free -g 2>/dev/null | awk '/^Mem:/{print $7}')

  if [ -n "$prev_gateway" ] && [ "$gateway_alive" != "$prev_gateway" ]; then
    echo "$ts gateway_alive: $prev_gateway -> $gateway_alive" >> "$ALERT_LOG"
  fi
  prev_gateway="$gateway_alive"

  if [ -n "$prev_workers" ] && [ "$worker_count" != "$prev_workers" ]; then
    echo "$ts worker_count: $prev_workers -> $worker_count (expected $EXPECTED_WORKERS)" >> "$ALERT_LOG"
  fi
  prev_workers="$worker_count"

  if [ -n "$prev_gpu_err" ] && [ "$gpu_err" != "$prev_gpu_err" ]; then
    echo "$ts gpu_ecc_errors: $prev_gpu_err -> $gpu_err" >> "$ALERT_LOG"
  fi
  prev_gpu_err="$gpu_err"

  if [ -n "$disk_pct" ] && [ "$disk_pct" -ge 90 ] 2>/dev/null; then
    echo "$ts disk_usage_high: ${disk_pct}%" >> "$ALERT_LOG"
  fi
  if [ -n "$mem_avail_gb" ] && [ "$mem_avail_gb" -le 4 ] 2>/dev/null; then
    echo "$ts mem_low: ${mem_avail_gb}GB available" >> "$ALERT_LOG"
  fi

  sleep "$INTERVAL_S"
done
