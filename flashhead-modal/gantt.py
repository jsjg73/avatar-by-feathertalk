#!/usr/bin/env python3
"""한 실행의 세션 타임라인을 간트로 그린다.

`docs/capacity/report.py` 는 스윕 전체를 세션 수 축에 놓고 "몇 세션까지 되나"를
본다. 이 도구는 그 반대다 — **한 번의 실행 안에서** 세션들이 언제 말했고 언제
밀렸는지를 시간 축에 그린다. 평균으로 뭉개면 사라지는 것, 즉 발화가 겹치는
순간과 그때 누가 기다렸는지가 여기서만 보인다.

입력은 `run_local.py` 의 실행 로그다. 렌더러가 청크마다 남기는 줄을 읽는다:

    [aa2d] [  11.7s] live: chunk 12 speech gen 0.77s behind 0.00s queued 5.8s

로그만 읽으므로 실행 쪽을 고칠 필요가 없고, 이미 저장해 둔 옛 로그에도 쓸 수 있다.

    python gantt.py run.log -o gantt.html
    python gantt.py /tmp/n16.log            # 같은 이름의 .html 로 저장

주의: 무음 청크는 20개마다만 기록된다(로그가 폭발하지 않게). 그래서 "조용함"은
그리지 않고, 발화와 대기만 칠한 뒤 나머지를 세션이 살아 있던 구간으로 남긴다.
"""

import argparse
import html
import pathlib
import re
import statistics
from collections import defaultdict

SLOT_S = 0.96          # 한 청크가 차지하는 시간; 막대 하나의 너비
LINE = re.compile(
    r"\[\s*(?P<sid>[0-9a-f]{4})\]\s*\[\s*(?P<t>[\d.]+)s\]\s*live: chunk (?P<k>\d+) "
    r"(?P<kind>speech|deferred|silence) gen (?P<gen>[\d.]+)s behind (?P<behind>[\d.]+)s"
)


SWEEP_HEAD = re.compile(r"^\s*(\d+) 세션\s*$")


def sections(text: str) -> list:
    """스윕 로그를 지점별로 나눈다 — [(제목, 본문), …].

    A sweep runs its points one after another and **each point restarts the
    clock at zero**, because the timestamps are per shipper. Drawing the whole
    file as one chart therefore lays five separate runs on top of each other and
    reports concurrency that never happened. The points are separated by the
    banner `run_local.sweep` prints, so split on that and draw each on its own
    axis. A file without the banner is one run, and comes back as one section.
    """
    out, cur, title = [], [], None
    for line in text.splitlines():
        m = SWEEP_HEAD.match(line)
        if m:
            if cur and title:
                out.append((title, "\n".join(cur)))
            title, cur = f"{m.group(1)} 세션", []
            continue
        cur.append(line)
    if title and cur:
        out.append((title, "\n".join(cur)))
    return out or [("", text)]


def parse(path: pathlib.Path) -> dict:
    """로그에서 세션별 (시각, 종류) 이벤트를 모은다."""
    return parse_text(path.read_text(errors="ignore"))


def parse_text(text: str) -> dict:
    ev: dict = defaultdict(list)
    behind: dict = defaultdict(list)
    for raw in text.splitlines():
        m = LINE.search(raw)
        if not m:
            continue
        sid, t = m["sid"], float(m["t"])
        ev[sid].append((t, m["kind"]))
        behind[sid].append(float(m["behind"]))
    return {"events": dict(ev), "behind": dict(behind)}


def spans(events: list, kind: str) -> list:
    """같은 종류가 연달아 나온 청크를 하나의 구간으로 합친다."""
    out: list = []
    for t, k in sorted(events):
        if k != kind:
            continue
        if out and t - out[-1][1] <= SLOT_S * 1.6:      # 바로 이어지는 청크
            out[-1][1] = t + SLOT_S
        else:
            out.append([t, t + SLOT_S])
    return out


def concurrency(events_by_sid: dict, t0: float, t1: float) -> list:
    """슬롯마다 몇 세션이 동시에 발화했는지."""
    # round, not int: the timestamps are k * SLOT_S in floating point, and
    # truncating drops a slot boundary now and then, folding two slots into one
    # and reporting twice the concurrency that actually happened.
    bins: dict = defaultdict(int)
    for evs in events_by_sid.values():
        for t, k in evs:
            if k == "speech":
                bins[round((t - t0) / SLOT_S)] += 1
    n = max(1, round((t1 - t0) / SLOT_S) + 1)
    return [bins.get(i, 0) for i in range(n)]


def waits(events: list) -> list:
    """대기 구간의 길이들 — 말할 준비가 된 뒤 실제로 말하기까지."""
    # A run of deferrals ends either at a different kind or at a gap in the log.
    # Silence is only written every 20th chunk, so waiting for the kind to change
    # would join separate waits across the quiet stretch between them — the first
    # version of this reported 38 s waits where the real ones were under 8 s.
    runs, cur, last = [], 0.0, None
    for t, k in sorted(events):
        gap = last is not None and t - last > SLOT_S * 1.6
        if k != "deferred" or gap:
            if cur:
                runs.append(cur)
            cur = 0.0
        if k == "deferred":
            cur += SLOT_S
        last = t
    if cur:
        runs.append(cur)
    return runs


def chart(data: dict, title: str) -> str:
    """한 판의 카드 하나. 문서 전체는 page() 가 감싼다."""
    ev = data["events"]
    if not ev:
        raise SystemExit("로그에서 청크 줄을 찾지 못했다 — run_local.py 로그가 맞는지 확인할 것")
    sids = sorted(ev, key=lambda s: min(t for t, _ in ev[s]))
    all_t = [t for e in ev.values() for t, _ in e]
    t0, t1 = min(all_t), max(all_t) + SLOT_S
    span_s = max(1.0, t1 - t0)

    conc = concurrency(ev, t0, t1)
    all_waits = [w for s in sids for w in waits(ev[s])]
    all_behind = [b for bs in data["behind"].values() for b in bs]
    speech_slots = sum(1 for c in conc if c)

    W, ROW, GAP, PAD_L = 1000, 18, 4, 76
    H_rows = len(sids) * (ROW + GAP)
    H_conc = 70
    x = lambda t: PAD_L + (t - t0) / span_s * (W - PAD_L - 12)      # noqa: E731

    rows = []
    for i, sid in enumerate(sids):
        y = i * (ROW + GAP)
        bars = [f'<rect class="life" x="{x(min(t for t,_ in ev[sid])):.1f}" y="{y + ROW/2 - 1:.1f}" '
                f'width="{x(max(t for t,_ in ev[sid]) + SLOT_S) - x(min(t for t,_ in ev[sid])):.1f}" height="2"/>']
        for a, b in spans(ev[sid], "deferred"):
            bars.append(f'<rect class="wait" x="{x(a):.1f}" y="{y:.1f}" '
                        f'width="{max(1.0, x(b)-x(a)):.1f}" height="{ROW}"><title>대기 {b-a:.1f}s</title></rect>')
        for a, b in spans(ev[sid], "speech"):
            bars.append(f'<rect class="talk" x="{x(a):.1f}" y="{y:.1f}" '
                        f'width="{max(1.0, x(b)-x(a)):.1f}" height="{ROW}"><title>발화 {b-a:.1f}s</title></rect>')
        rows.append(f'<text class="lbl" x="{PAD_L-8}" y="{y + ROW - 4}">{sid}</text>' + "".join(bars))

    cw = (W - PAD_L - 12) / max(1, len(conc))
    cmax = max(conc) or 1
    cbars = "".join(
        f'<rect class="conc" x="{PAD_L + i*cw:.1f}" y="{H_conc - c/cmax*H_conc:.1f}" '
        f'width="{max(0.8, cw):.1f}" height="{c/cmax*H_conc:.1f}"><title>{c}개 동시</title></rect>'
        for i, c in enumerate(conc) if c)

    step = 30 if span_s <= 180 else 60 if span_s <= 600 else 300
    axis = "".join(
        f'<line class="grid" x1="{x(t0+s):.1f}" y1="0" x2="{x(t0+s):.1f}" y2="{H_rows + H_conc + 24}"/>'
        f'<text class="tick" x="{x(t0+s):.1f}" y="{H_rows + H_conc + 38}">{int(s)}s</text>'
        for s in range(0, int(span_s) + 1, step))

    def stat(v, f="{:.2f}"):
        return f.format(v) if v else "—"

    return f"""<h2 class=sec>{html.escape(title)}</h2>
<p class=sub>{len(sids)}세션 · {span_s:.0f}초 · 슬롯 {SLOT_S}s</p>
<div class=card>
  <div class=kpi>
    <div>세션<b>{len(sids)}</b></div>
    <div>최대 동시 발화<b>{max(conc) if conc else 0}</b></div>
    <div>발화 슬롯 평균<b>{stat(sum(conc)/speech_slots if speech_slots else 0)}</b></div>
    <div>대기 p95<b>{stat(statistics.quantiles(all_waits, n=20)[18] if len(all_waits) > 19 else (max(all_waits) if all_waits else 0), "{:.1f}s")}</b></div>
    <div>영상 지연 최대<b>{stat(max(all_behind) if all_behind else 0, "{:.2f}s")}</b></div>
  </div>

  <h2>세션별 타임라인</h2>
  <svg viewBox="0 0 {W} {H_rows + H_conc + 46}">
    {axis}
    {"".join(rows)}
    <g transform="translate(0,{H_rows + 16})">
      <text class="lbl" x="{PAD_L-8}" y="{H_conc}">동시</text>
      {cbars}
    </g>
  </svg>
  <div class=key>
    <span><i style="background:var(--talk)"></i>발화 (GPU 점유)</span>
    <span><i style="background:var(--wait)"></i>대기 (말할 준비가 됐는데 슬롯을 못 받음)</span>
    <span><i style="background:var(--conc)"></i>슬롯별 동시 발화 수</span>
  </div>
</div>
"""


STYLE = """
:root{{--paper:#F6F7F9;--surface:#FFF;--ink:#141A24;--muted:#5C6672;--faint:#8A929C;
--line:#DDE1E7;--grid:#E7EAEE;--talk:#2F6FB0;--wait:#C77B24;--conc:#8FA8C4;
--sans:"IBM Plex Sans KR",ui-sans-serif,system-ui,-apple-system,"Apple SD Gothic Neo",sans-serif;
--mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--paper:#0F141B;--surface:#161D26;
--ink:#E6E9EE;--muted:#98A2AE;--faint:#6B7684;--line:#27303C;--grid:#1E2733;--talk:#6EA8FE;
--wait:#E0A33C;--conc:#3D5570}}}}
body{{margin:0;padding:28px;background:var(--paper);color:var(--ink);font-family:var(--sans)}}
.wrap{{max-width:1060px;margin:0 auto}}
h1{{font-size:18px;margin:0 0 4px}}
.sub{{color:var(--muted);font-size:13px;margin:0 0 18px;font-family:var(--mono)}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:18px;margin-bottom:16px}}
.kpi{{display:flex;gap:26px;flex-wrap:wrap;margin-bottom:16px}}
.kpi div{{font-size:13px;color:var(--muted)}}
.kpi b{{display:block;font-size:20px;color:var(--ink);font-family:var(--mono);font-weight:600}}
svg{{display:block;width:100%;height:auto;overflow:visible}}
.lbl{{font:11px var(--mono);fill:var(--faint);text-anchor:end}}
.tick{{font:10px var(--mono);fill:var(--faint);text-anchor:middle}}
.grid{{stroke:var(--grid);stroke-width:1}}
.life{{fill:var(--line)}}
.talk{{fill:var(--talk)}} .wait{{fill:var(--wait)}} .conc{{fill:var(--conc)}}
.key{{display:flex;gap:16px;font-size:12px;color:var(--muted);margin-top:10px;align-items:center}}
.key i{{width:12px;height:12px;border-radius:2px;display:inline-block;margin-right:5px;vertical-align:-2px}}
h2{{font-size:13px;color:var(--muted);margin:22px 0 6px;font-weight:600}}
"""


def page(title: str, cards: list) -> str:
    return (f'<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title>'
            f"<style>{STYLE}.sec{{font-size:16px;margin:26px 0 2px}}</style>"
            f"<div class=wrap><h1>{html.escape(title)}</h1>" + "".join(cards) + "</div>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=pathlib.Path)
    ap.add_argument("-o", "--out", type=pathlib.Path, default=None)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    out = a.out or a.log.with_suffix(".html")
    secs = sections(a.log.read_text(errors="ignore"))
    title = a.title or a.log.name
    cards = []
    for name, body in secs:
        d = parse_text(body)
        if not d["events"]:          # a point that never logged a chunk
            continue
        cards.append(chart(d, name or title))
    if not cards:
        raise SystemExit("로그에서 청크 줄을 찾지 못했다 — run_local.py 로그가 맞는지 확인할 것")
    out.write_text(page(title, cards), encoding="utf-8")
    print(f"{out}  ({len(cards)}개 구간)", flush=True)


if __name__ == "__main__":
    main()
