#!/usr/bin/env bash
# Everything that has to be measured on a rented box, in one command.
#
# The box is metered, so the order is by value-per-minute, not by narrative:
# the correctness gate first (a wrong batch makes every later number a lie),
# then the one question that changes the hardware choice, then the runs that
# only refine what is already known. Each step writes its own log and a failure
# does not stop the rest — a box that dies halfway should still have paid for
# something.
#
#   bash measure.sh            # ~35 min
#   bash measure.sh quick      # ~12 min: the gate and the batch curve only
set -uo pipefail
cd "$(dirname "$0")"
[ -f /root/flashhead.env ] && . /root/flashhead.env
PY="${PYTHON:-python3}"
MODE="${1:-full}"

CARD=$("$PY" -c "import torch;print(torch.cuda.get_device_name(0).replace(' ','_'))" 2>/dev/null || echo unknown)
OUT="bench/$CARD-$(date -u +%m%d-%H%M)"
mkdir -p "$OUT"
echo "== $CARD  →  $OUT"

step() {                       # step <name> <minutes-budget> <cmd...>
    local name=$1 budget=$2; shift 2
    echo; echo "── $name  (예상 ${budget}분)  $(date -u +%T)"
    if "$@" > "$OUT/$name.log" 2>&1; then
        echo "   ✓ $OUT/$name.log"
    else
        echo "   ✗ 실패 — $OUT/$name.log 의 마지막 줄:"
        tail -4 "$OUT/$name.log" | sed 's/^/     /'
    fi
}

# 0. provenance. Every number below is about this exact card and driver, and
#    the last survey compared an H200 against an "H100" it thought it had.
{ nvidia-smi; echo; "$PY" - <<'PYP'
import torch
p = torch.cuda.get_device_properties(0)
print(f"{p.name}  {p.total_memory/1e9:.0f} GB  sm{p.major}.{p.minor}  SM {p.multi_processor_count}")
print("torch", torch.__version__, "cuda", torch.version.cuda)
try:
    import flash_attn; print("flash_attn", flash_attn.__version__)
except Exception as e: print("flash_attn 없음 → SDPA:", type(e).__name__)
PYP
} > "$OUT/card.txt" 2>&1
cat "$OUT/card.txt" | tail -5

# 1. the gate: is the batched wav2vec each session's own audio?
step embed-correctness 3 "$PY" check_embed.py --batch 8

# 2. the batch curve, and with it the number that decides the card:
#    a slot is 0.96 s, so anything under 0.48 s at B=1 earns budget 2.
step batch-curve 8 "$PY" spike_batch.py --batches 1,2,4,8

[ "$MODE" = quick ] && { echo; echo "== quick 종료 · $OUT"; exit 0; }

# 3. the interview as actually specified: 7.07 s of avatar speech per 42 s turn.
#    Residency timing is on — re-measuring it after the ring buffer and the
#    batched embedding is the other thing left open.
step interview-1 4 env FLASHHEAD_RESIDENCY_TIMING=1 \
    "$PY" run_local.py --audio inputs/q_avatar_7s.wav --image inputs/newscaster.png \
    --sessions 1 --minutes 3 --gap-s 42 --jitter 0

# 4. how many of those fit. The sweep's own verdict gates on freeze-out, wait
#    p95 and video lag; the gantt is what makes a failure legible.
step sweep 18 env FLASHHEAD_RESIDENCY_TIMING=1 \
    "$PY" run_local.py --audio inputs/q_avatar_7s.wav --image inputs/newscaster.png \
    --sweep 1,4,8,16,24 --minutes 3 --gap-s 42 --jitter 0.5 \
    --out-json "$OUT/sweep.json"

echo; echo "── 간트"
for L in "$OUT"/interview-1.log "$OUT"/sweep.log; do
    [ -f "$L" ] && "$PY" gantt.py "$L" --title "$CARD $(basename "$L" .log)" 2>&1 | sed 's/^/   /'
done

echo; echo "== 끝. 가져갈 것: $OUT"
ls -la "$OUT"
