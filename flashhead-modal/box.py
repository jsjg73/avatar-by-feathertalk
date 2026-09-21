#!/usr/bin/env python3
"""Renting a GPU box, seeing what is running, and stopping it — per vendor.

`renderer.py` pushed *where the code runs* behind the `Host` seam. This pushes
*how the box is obtained and released* behind a second one, `Provider`. They are
separate problems: Host is transport inside a run, Provider is the meter.

    python box.py list                          every vendor, whole account
    python box.py offers vast 4090              what can be rented, before paying
    python box.py rent vast 4090 --name flashhead
    python box.py ssh vast:1234567              how to get on it
    python box.py stop lightning:01jq...
    python box.py stop --all-billable           dry run unless --yes

Why `list` walks partitions instead of names
--------------------------------------------
On 2026-09-21 a Studio nobody was looking at had been running 34.6 hours and had
cost $13.48. It was not one of the two this project created, and every earlier
"stopped, verified" check had been a name lookup on those two. So nothing here
looks up a name: `list` enumerates the account's *partitions* — Modal
environments, Lightning teamspaces including the ones reached only through an
organisation — and reports whatever is inside. The resource that gets missed is
precisely the one this project did not create.

`stop` takes a `Box` that `list` returned, never a bare id. That is deliberate:
it makes "what can be stopped" a subset of "what was found", so the account-wide
sweep is on the path of every stop rather than beside it.

Credentials are never handled here. Both vendors authenticate out of band —
`modal setup`, `lightning login` — which the account owner runs themselves. This
module only uses what is already on the machine, and says plainly when it isn't.

The two vendors are not symmetric and that is left visible:

    Lightning   a box you rent by the hour. rent → start, stop → stop. The meter
                runs while it is up, whether or not anything is using it.
    Modal       no box. `rent` deploys the app so the GPU can be summoned; the
                meter starts when something calls it and stops on scaledown.
                What bills is a *container*, which is why containers are listed
                as their own rows.
    vast.ai     a marketplace. You do not ask for a 4090, you bid on one named
                machine, so `offers` is a verb. Stopping keeps billing the disk
                and forfeits the GPU, so `stop` destroys — see VastProvider.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

RUNNING = {"running", "pending", "starting"}


@dataclass
class Box:
    """One thing on a vendor that can cost money."""

    vendor: str
    id: str
    name: str
    kind: str          # hardware, in the vendor's own words
    state: str         # running | pending | stopped | ...
    resource: str      # container / app / studio / job
    where: str         # the partition it lives in — an environment, a teamspace
    billing: bool      # is the meter running right now
    note: str = ""
    # The live SDK handle, when the vendor has one. Kept off repr/compare: it is
    # a capability, not data, and it is what lets stop() act on exactly the
    # object the sweep found.
    handle: object = field(default=None, repr=False, compare=False)

    @property
    def ref(self) -> str:
        return f"{self.vendor}:{self.id}"


class Provider:
    name = "?"

    def preflight(self) -> str | None:
        """None if usable, else what the account owner must do first.

        Never returns or writes a credential.
        """
        raise NotImplementedError

    def list(self) -> list[Box]:
        """Everything on the account that can bill — not just what we created."""
        raise NotImplementedError

    def rent(self, kind: str, name: str, **kw) -> Box:
        raise NotImplementedError

    def stop(self, box: Box) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------- Modal
class ModalProvider(Provider):
    name = "modal"

    def _exe(self) -> str | None:
        # `--json` on the CLI is a documented surface; the equivalent Python
        # calls are private. Prefer the stable contract.
        exe = os.environ.get("MODAL_BIN") or shutil.which("modal")
        if exe:
            return exe
        # The usual case is an unactivated venv: modal sits next to whichever
        # interpreter can import it.
        cand = os.path.join(os.path.dirname(sys.executable), "modal")
        return cand if os.path.exists(cand) else None

    def preflight(self) -> str | None:
        if not self._exe():
            return "modal CLI 이 없습니다 — pip install modal (또는 MODAL_BIN 에 경로를 주세요)"
        if not (os.path.exists(os.path.expanduser("~/.modal.toml"))
                or os.environ.get("MODAL_TOKEN_ID")):
            return "Modal 에 로그인되어 있지 않습니다 — 직접 `modal setup` 을 실행하세요"
        return None

    def _json(self, *args: str) -> list[dict]:
        out = subprocess.run([self._exe(), *args, "--json"],
                             capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            raise RuntimeError((out.stderr or out.stdout).strip()[:400])
        return json.loads(out.stdout or "[]")

    def list(self) -> list[Box]:
        boxes: list[Box] = []
        # Environments partition a workspace. Listing only the active one is the
        # same mistake as looking up two Studios by name.
        envs = [e["name"] for e in self._json("environment", "list")] or ["main"]
        for env in envs:
            for a in self._json("app", "list", "-e", env):
                tasks = int(a.get("tasks") or 0)
                boxes.append(Box(
                    vendor=self.name, id=a["app_id"], name=a.get("description") or "—",
                    kind="—", state=(a.get("state") or "unknown").lower(),
                    resource="app", where=env, billing=tasks > 0,
                    note=f"컨테이너 {tasks}개" if tasks else "",
                ))
            for c in self._json("container", "list", "-e", env):
                boxes.append(Box(
                    vendor=self.name, id=c["container_id"], name=c.get("app_name") or "—",
                    kind="—",   # the CLI does not report the GPU; do not guess it
                    state="running", resource="container", where=env, billing=True,
                    note=f"시작 {c.get('start_time') or '?'}",
                ))
        return boxes

    def rent(self, kind: str, name: str, app: str = "app.py", **kw) -> Box:
        """Deploy, so the GPU can be summoned. Nothing bills until it is called.

        `kind` reaches the deploy through FLASHHEAD_GPU, which is the same knob
        `app.py` reads for `@app.cls(gpu=...)`.
        """
        env = dict(os.environ, FLASHHEAD_GPU=kind)
        out = subprocess.run([self._exe(), "deploy", app], env=env,
                             capture_output=True, text=True, timeout=1800)
        if out.returncode != 0:
            raise RuntimeError((out.stderr or out.stdout).strip()[-800:])
        return Box(vendor=self.name, id=name or app, name=app, kind=kind,
                   state="deployed", resource="app", where=env.get("MODAL_ENVIRONMENT", "main"),
                   billing=False, note="호출될 때까지 과금 없음")

    def stop(self, box: Box) -> None:
        if box.resource == "app":
            cmd = [self._exe(), "app", "stop", box.id, "-y", "-e", box.where]
        elif box.resource == "container":
            cmd = [self._exe(), "container", "stop", box.id, "-y"]
        else:
            raise RuntimeError(f"Modal 에서 정지할 수 없는 종류입니다: {box.resource}")
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if out.returncode != 0:
            raise RuntimeError((out.stderr or out.stdout).strip()[:400])


# ------------------------------------------------------------ Lightning
class LightningProvider(Provider):
    name = "lightning"

    def preflight(self) -> str | None:
        try:
            import lightning_sdk  # noqa: F401
        except ImportError:
            return "lightning_sdk 가 없습니다 — pip install lightning-sdk"
        if not (os.path.exists(os.path.expanduser("~/.lightning/credentials.json"))
                or os.environ.get("LIGHTNING_API_KEY")):
            return "Lightning 에 로그인되어 있지 않습니다 — 직접 `lightning login` 을 실행하세요"
        return None

    def _teamspaces(self) -> list:
        """Every teamspace the credentials can reach, deduped.

        `user.teamspaces` is only what the user personally owns. A teamspace
        reached through an organisation is invisible to it — and a Studio in one
        would be invisible to any check built on it. Both are walked here.
        """
        from lightning_sdk.utils.resolve import _get_authed_user

        user = _get_authed_user()
        seen: dict[str, object] = {t.id: t for t in user.teamspaces}
        for org in user.organizations:
            for t in org.teamspaces:
                seen.setdefault(t.id, t)
        return list(seen.values())

    def list(self) -> list[Box]:
        boxes: list[Box] = []
        for ts in self._teamspaces():
            for s in ts.studios:
                state = str(getattr(s.status, "name", s.status)).lower()
                try:
                    kind = str(getattr(s.machine, "name", s.machine) or "—")
                except Exception:                    # not running: no machine yet
                    kind = "—"
                boxes.append(Box(
                    vendor=self.name, id=s.id, name=s.name, kind=kind, state=state,
                    resource="studio", where=ts.name, billing=state in RUNNING,
                    handle=s,
                ))
            for j in ts.jobs:
                state = str(getattr(j.status, "name", j.status)).lower()
                boxes.append(Box(
                    vendor=self.name, id=j.id, name=j.name,
                    kind=str(getattr(j.machine, "name", j.machine) or "—"),
                    state=state, resource="job", where=ts.name,
                    billing=state in RUNNING, handle=j,
                ))
        return boxes

    def rent(self, kind: str, name: str, teamspace: str | None = None,
             cloud: str | None = None, interruptible: bool = False, **kw) -> Box:
        from lightning_sdk import Machine, Studio

        spaces = self._teamspaces()
        ts = next((t for t in spaces if t.name == teamspace), None) if teamspace else (
            spaces[0] if spaces else None)
        if ts is None:
            raise RuntimeError(f"teamspace 를 찾지 못했습니다 (있는 것: {[t.name for t in spaces]})")
        # An unknown name is passed through: the SDK treats it as a custom
        # instance type, which is how the baremetal names (gpu-h100-1x) work.
        machine = getattr(Machine, kind.upper().replace("-", "_"), kind)
        st = Studio(name=name, teamspace=ts, cloud=cloud, create_ok=True)
        st.start(machine=machine, interruptible=interruptible)
        return Box(vendor=self.name, id=st.id, name=st.name,
                   kind=str(getattr(st.machine, "name", st.machine) or kind),
                   state=str(getattr(st.status, "name", st.status)).lower(),
                   resource="studio", where=ts.name, billing=True, handle=st)

    def stop(self, box: Box) -> None:
        if box.handle is None:
            raise RuntimeError("이 Box 에는 핸들이 없습니다 — list 로 찾은 것만 정지할 수 있습니다")
        box.handle.stop()


# ---------------------------------------------------------------- vast.ai
class VastProvider(Provider):
    """A marketplace, not a cloud. Three differences are load-bearing here.

    *Renting is bidding on one specific machine.* There is no "give me a 4090";
    there is offer #50080431 in Japan at $0.335/hr with that host's disk, uplink
    and driver. So `offers` is a first-class verb — the choice is the work, and
    it has to be visible before any money moves.

    *Stopping does not stop the meter.* A stopped instance keeps billing its
    disk ($0.15–0.22/GB·mo) and gives up its GPU with no promise of getting it
    back. Both halves of that make `stop` here mean **destroy**: the state a
    stopped instance preserves is not worth paying to keep, and a machine that
    cost $13.48 for doing nothing is why this file exists at all.

    *The driver is the host's, not ours.* `cuda_max_good` below 12.4 fails at
    `import torch`, after the box is rented and billing. It is filtered in the
    search, not discovered on the box.

    Credentials: an API key in ~/.vast_key (chmod 600) or $VAST_API_KEY. It is
    read, never printed, never passed on a command line.
    """

    name = "vast"
    API = "https://console.vast.ai/api/v0"
    KEY_FILE = os.path.expanduser("~/.vast_key")
    SSH_KEY = os.path.expanduser("~/.ssh/vast_flashhead")

    # Short names to what the marketplace calls them. A bare "4090" should not
    # silently miss the 4090D, and "H100" spans two very different cards.
    CARDS = {
        "4090": ["RTX 4090", "RTX 4090D"],
        "4080": ["RTX 4080", "RTX 4080S"],
        "5090": ["RTX 5090"],            # sm_120: needs CUDA 12.8+, not our cu124 stack
        "L4": ["L4"],
        "L40S": ["L40S", "L40"],
        "A100": ["A100 SXM4", "A100 PCIE"],
        "H100": ["H100 SXM", "H100 PCIE", "H100 NVL"],
        "6000Ada": ["RTX 6000Ada"],
        "5000Ada": ["RTX 5000Ada"],
        "A6000": ["RTX A6000"],
    }
    # Asia, nearest first. Vast reports geolocation as "Japan, JP".
    NEAR = ["KR", "JP", "TW", "HK", "SG", "CN", "VN", "TH", "MY", "IN"]

    def _key(self) -> str | None:
        k = os.environ.get("VAST_API_KEY")
        if k:
            return k.strip()
        try:
            with open(self.KEY_FILE) as f:
                return f.read().strip() or None
        except OSError:
            return None

    def _api(self, path: str, method: str = "GET", body: dict | None = None,
             auth: bool = True) -> dict:
        import urllib.error
        import urllib.request

        data = json.dumps(body).encode() if body is not None else None
        headers = {"Accept": "application/json", "Content-Type": "application/json",
                   # Cloudflare 403s urllib's default agent on some vendors; this
                   # costs nothing and removed a whole class of confusion on Novita.
                   "User-Agent": "curl/8.7.1"}
        if auth:
            k = self._key()
            if not k:
                raise RuntimeError("API 키가 없습니다")
            headers["Authorization"] = f"Bearer {k}"
        url = (f"https://console.vast.ai/api/{path[3:]}" if path.startswith("../")
               else f"{self.API}/{path.lstrip('/')}")
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            try:
                j = json.loads(raw)
            except ValueError:
                raise RuntimeError(f"HTTP {e.code}: {raw[:300]}") from None
            # Vast answers a bad key with 404 + error:auth_error, which reads as
            # "no such endpoint" unless you have seen it before. Say what it is.
            if j.get("error") == "auth_error":
                raise RuntimeError(f"API 키가 거부됐습니다 ({j.get('msg')}) — {self.KEY_FILE} 확인") from None
            raise RuntimeError(j.get("msg") or raw[:300]) from None

    # -- searching ---------------------------------------------------------
    def offers(self, card: str | None = None, geo: str | None = "asia",
               max_price: float | None = None, limit: int = 12,
               verified: bool = False) -> list[dict]:
        """Rentable offers that can actually run this workload, cheapest first.

        The filters are the preconditions, not preferences: below CUDA 12.4
        torch will not import, below ~60 GB the 8 GB of weights plus the CUDA
        wheels do not fit, and with no direct port the studio cannot be reached.
        """
        import urllib.parse
        import urllib.request

        q: dict = {
            "rentable": {"eq": True}, "rented": {"eq": False},
            "num_gpus": {"eq": 1},
            "cuda_max_good": {"gte": 12.4},
            "disk_space": {"gte": 60},
            "direct_port_count": {"gte": 2},
            "inet_down": {"gte": 200},
            "reliability2": {"gte": 0.95},
            "order": [["dph_total", "asc"]], "type": "on-demand", "limit": 64,
        }
        if verified:
            q["verified"] = {"eq": True}
        if max_price:
            q["dph_total"] = {"lte": max_price}

        # The geography goes into the query, not into a filter afterwards. The
        # endpoint caps every response at 64 rows whatever `limit` says, and the
        # rows it keeps are the cheapest globally — so filtering after the fetch
        # lets cheap US offers crowd Korea out of the answer entirely. The first
        # version of this reported "no Korean 4090" while five were listed.
        want = None
        if geo and geo != "any":
            want = self.NEAR if geo == "asia" else [g.strip().upper() for g in geo.split(",")]
            q["geolocation"] = {"in": want}

        names = self.CARDS.get(card, [card]) if card else [None]
        out: list[dict] = []
        for n in names:
            qq = dict(q)
            if n:
                qq["gpu_name"] = {"eq": n}
            url = f"https://cloud.vast.ai/api/v0/bundles/?q={urllib.parse.quote(json.dumps(qq))}"
            req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                       "User-Agent": "curl/8.7.1"})
            with urllib.request.urlopen(req, timeout=90) as r:
                out += json.load(r).get("offers", [])

        # Nearest country first, then price. Latency to Korea is the thing the
        # money cannot buy back, so it outranks a few cents.
        if want:
            out.sort(key=lambda o: (want.index(self._cc(o)) if self._cc(o) in want else 99,
                                    o["dph_total"]))
        else:
            out.sort(key=lambda o: o["dph_total"])
        return out[:limit]

    @staticmethod
    def _cc(o: dict) -> str:
        return ((o.get("geolocation") or "?").split(",")[-1]).strip().upper()

    # -- the account -------------------------------------------------------
    def preflight(self) -> str | None:
        if not self._key():
            return (f"vast.ai API 키가 없습니다 — console.vast.ai → Account → API Keys 에서 만들어\n"
                    f"      직접 저장하세요:  umask 077; pbpaste > {self.KEY_FILE}\n"
                    f"      (키를 채팅에 붙여넣지 마세요)")
        try:
            self._api("users/current/")
        except Exception as e:                        # noqa: BLE001
            return str(e)
        return None

    def balance(self) -> tuple[float, float]:
        u = self._api("users/current/")
        return float(u.get("credit") or 0.0), float(u.get("balance") or 0.0)

    def _instances(self) -> list[dict]:
        """Every instance on the account, following the pages.

        `/api/v0/instances/` answers 410 now; only this path moved to v1, while
        users/ssh/bundles are still v0. And v1 pages — a sweep that reads one
        page and stops is the same bug as a sweep that looks up two names.
        """
        rows, token, guard = [], None, 0
        while guard < 50:
            guard += 1
            path = "../v1/instances/" + (f"?next_token={token}" if token else "")
            r = self._api(path)
            rows += r.get("instances") or []
            token = r.get("next_token")
            if not token:
                break
        return rows

    def list(self) -> list[Box]:
        rows = self._instances()
        boxes = []
        for i in rows:
            st = (i.get("actual_status") or i.get("cur_state") or "unknown").lower()
            live = st in ("running", "loading", "created", "starting")
            gpu = f"{i.get('num_gpus', 1)}x {i.get('gpu_name', '?')}"
            dph = float(i.get("dph_total") or 0)
            disk = float(i.get("storage_cost") or 0)   # $/GB/month on this host
            gb = float(i.get("disk_space") or 0)
            # A stopped instance is still a bill. That is the whole reason this
            # column is not just `state == running`.
            note = (f"${dph:.3f}/hr · stop=삭제" if live else
                    f"정지 상태지만 디스크 {gb:.0f}GB 과금 중 (~${disk*gb/730:.4f}/hr) — 삭제해야 멈춥니다")
            boxes.append(Box(
                vendor=self.name, id=str(i["id"]), name=i.get("label") or f"#{i['id']}",
                kind=gpu, state=st, resource="instance",
                where=(i.get("geolocation") or "?"), billing=True, note=note,
                handle=i,
            ))
        return boxes

    # -- renting -----------------------------------------------------------
    def _onstart(self) -> str:
        """provision.sh with setup.sh carried inside it.

        setup.sh stays the only description of the environment; this just makes
        it reachable from a machine that has none of our files yet.
        """
        import base64
        import gzip

        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "setup.sh"), "rb") as f:
            # gzip first: the field has an unknown length limit and 5.7 KB of
            # shell is 7.7 KB of base64 but only 2.7 KB compressed.
            b64 = base64.b64encode(gzip.compress(f.read(), 9)).decode()
        with open(os.path.join(here, "provision.sh")) as f:
            body = f.read()
        return f"export SETUP_B64='{b64}'\n{body}"

    def _ssh_pub(self) -> str:
        p = self.SSH_KEY + ".pub"
        if not os.path.exists(p):
            raise RuntimeError(
                f"{p} 가 없습니다 — 빌린 박스 전용 키를 먼저 만드세요:\n"
                f"  ssh-keygen -t ed25519 -N '' -C flashhead-rented-gpu -f {self.SSH_KEY}")
        with open(p) as f:
            return f.read().strip()

    def rent(self, kind: str, name: str, offer: int | None = None,
             geo: str = "asia", image: str = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
             disk: int = 80, max_price: float | None = None, dry_run: bool = False,
             **kw) -> Box:
        pub = self._ssh_pub()
        if offer is None:
            found = self.offers(kind, geo=geo, max_price=max_price, limit=1)
            if not found:
                raise RuntimeError(f"{kind} 오퍼를 찾지 못했습니다 (geo={geo}) — "
                                   f"`python box.py offers vast {kind} --geo any` 로 넓혀보세요")
            offer = found[0]["id"]
            pick = found[0]
        else:
            # Look the pinned offer up rather than inventing a placeholder: the
            # price printed back must be the price about to be charged.
            import urllib.parse
            import urllib.request
            u = ("https://cloud.vast.ai/api/v0/bundles/?q="
                 + urllib.parse.quote(json.dumps({"id": {"eq": offer}, "limit": 1})))
            req = urllib.request.Request(u, headers={"Accept": "application/json",
                                                     "User-Agent": "curl/8.7.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                got = json.load(r).get("offers") or []
            pick = got[0] if got else {"id": offer, "gpu_name": kind,
                                       "geolocation": "?", "dph_total": 0.0}

        body = {
            "client_id": "me", "image": image, "disk": float(disk),
            "label": name, "runtype": "ssh", "onstart": self._onstart(),
            "env": {"-p 8000:8000": "1"},   # the studio's HTTP port, if it is wanted
            "target_state": "running", "cancel_unavail": True,
            "image_login": None, "use_jupyter_lab": False,
        }
        if dry_run:
            # A malformed body is answered by the marketplace taking the offer
            # and then failing, so the request gets read before it is sent once.
            shown = dict(body, onstart=f"<{len(body['onstart'])} bytes: provision.sh + setup.sh>")
            print(f"PUT {self.API}/asks/{offer}/\n"
                  + json.dumps(shown, indent=2, ensure_ascii=False))
            print(f"\n오퍼 {offer}: {pick.get('gpu_name')} ${pick.get('dph_total', 0):.3f}/hr "
                  f"· {pick.get('geolocation')}\n보내지 않았습니다 (--dry-run).")
            return Box(vendor=self.name, id="dry-run", name=name,
                       kind=str(pick.get("gpu_name", kind)), state="dry-run",
                       resource="instance", where=str(pick.get("geolocation", "?")),
                       billing=False, note="보내지 않음")

        r = self._api(f"asks/{offer}/", "PUT", body)
        if not r.get("success"):
            raise RuntimeError(r.get("msg") or str(r)[:300])
        iid = str(r.get("new_contract"))

        # `PUT /asks/` has no field for a key, so it goes on the account. The
        # per-instance attach would be narrower but it hangs off the retired v0
        # instances path. Registering the same key twice is a no-op.
        try:
            if not any(pub.split()[1] in (k.get("public_key") or "")
                       for k in (self._api("ssh/") or [])):
                self._api("ssh/", "POST", {"ssh_key": pub})
        except Exception as e:                        # noqa: BLE001
            print(f"⚠ SSH 키 등록 실패 ({e}) — console.vast.ai 에서 직접 넣으세요")

        return Box(vendor=self.name, id=iid, name=name,
                   kind=f"1x {pick.get('gpu_name', kind)}", state="loading",
                   resource="instance", where=pick.get("geolocation", "?"),
                   billing=True, note=f"${pick.get('dph_total', 0):.3f}/hr · 디스크 {disk}GB",
                   handle=None)

    def ssh(self, box: Box) -> tuple[str, int] | None:
        """(host, port) for a direct connection, once the box has one."""
        i = box.handle or next((x for x in self._instances() if str(x["id"]) == box.id), None)
        if not i:
            return None
        ports = i.get("ports") or {}
        mapped = (ports.get("22/tcp") or [{}])[0].get("HostPort")
        ip = i.get("public_ipaddr")
        if ip and mapped:
            return ip.strip(), int(mapped)
        if i.get("ssh_host") and i.get("ssh_port"):   # Vast's proxy; slower but works
            return i["ssh_host"], int(i["ssh_port"])
        return None

    def stop(self, box: Box) -> None:
        """Destroy. On vast.ai a stopped instance still bills disk and may not
        get its GPU back, so there is no state here worth paying to keep."""
        r = self._api(f"instances/{box.id}/", "DELETE", {})
        if not r.get("success", True):
            raise RuntimeError(r.get("msg") or str(r)[:300])


PROVIDERS: dict[str, type[Provider]] = {p.name: p for p in (ModalProvider, LightningProvider, VastProvider)}


# ------------------------------------------------------------------ CLI
def sweep(names: list[str]) -> tuple[list[Box], list[tuple[str, str]]]:
    """Every vendor asked for. A vendor that cannot answer is reported, not raised —
    one broken credential must not hide another vendor's running box."""
    boxes, problems = [], []
    for n in names:
        p = PROVIDERS[n]()
        why = p.preflight()
        if why:
            problems.append((n, why))
            continue
        try:
            boxes += p.list()
        except Exception as e:                        # noqa: BLE001
            problems.append((n, f"{type(e).__name__}: {e}"))
    return boxes, problems


def _pad(s: str, n: int) -> str:
    """Pad to n terminal columns. Hangul is double-width, so len() would leave
    the money column ragged in the one view that has to be read at a glance."""
    from unicodedata import east_asian_width

    w = sum(2 if east_asian_width(c) in "WF" else 1 for c in s)
    return s + " " * max(0, n - w)


def show(boxes: list[Box], problems: list[tuple[str, str]]) -> None:
    if boxes:
        w = max(len(b.name) for b in boxes)
        head = ("벤더", 11), ("종류", 11), ("이름", w + 2), ("하드웨어", 17), ("상태", 11)
        print("  " + "".join(_pad(h, n) for h, n in head) + "위치")
        for b in sorted(boxes, key=lambda b: (not b.billing, b.vendor, b.name)):
            cells = (b.vendor, 11), (b.resource, 11), (b.name, w + 2), (b.kind, 17), (b.state, 11)
            print(("● " if b.billing else "· ") + "".join(_pad(c, n) for c, n in cells)
                  + b.where + (f"   {b.note}" if b.note else ""))
    else:
        print("  (아무것도 없음)")

    live = [b for b in boxes if b.billing]
    print()
    if live:
        print(f"● 지금 과금 중: {len(live)}개")
        for b in live:
            print(f"    python box.py stop {b.ref}")
    elif problems:
        print(f"● 확인된 범위에서는 과금 중인 것 없음 — 그러나 {len(problems)}개 벤더를 "
              f"보지 못했습니다. '없음'이 아닙니다.")
    else:
        print("● 지금 과금 중인 것: 없음")
    for n, why in problems:
        print(f"⚠ {n}: {why}")
    print("\n이 표가 못 보는 것: Lightning 팀스페이스 스토리지(일 단위), Modal Volume,"
          "\n그리고 이 머신의 자격증명으로 닿지 않는 계정."
          "\nvast 는 정지 상태도 디스크가 과금되므로 ● 로 센다 — 삭제해야 멈춘다.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="계정 전체에서 돈이 나갈 수 있는 것 전부")
    pl.add_argument("--vendor", choices=list(PROVIDERS), action="append")
    pl.add_argument("--json", action="store_true")

    pr = sub.add_parser("rent", help="박스를 빌린다 (Modal 은 배포한다)")
    pr.add_argument("vendor", choices=list(PROVIDERS))
    pr.add_argument("kind", help="H100 / L4 / gpu-h100-1x …")
    pr.add_argument("--name", default="flashhead")
    pr.add_argument("--teamspace", default=None)
    pr.add_argument("--cloud", default=None, help="lightning: lightning-baremetal 등")
    pr.add_argument("--interruptible", action="store_true")
    pr.add_argument("--app", default="app.py", help="modal: 배포할 파일")
    pr.add_argument("--offer", type=int, default=None, help="vast: 오퍼 번호를 직접 지정")
    pr.add_argument("--geo", default="asia", help="vast: asia | any | KR,JP")
    pr.add_argument("--image", default="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    pr.add_argument("--disk", type=int, default=80, help="vast: GB (가중치 8GB + 휠)")
    pr.add_argument("--max-price", type=float, default=None, help="vast: $/hr 상한")
    pr.add_argument("--dry-run", action="store_true", help="vast: 보낼 요청만 보여준다")

    po = sub.add_parser("offers", help="빌릴 수 있는 것을 본다 (vast: 돈 안 듦, 키도 필요 없음)")
    po.add_argument("vendor", choices=["vast"])
    po.add_argument("card", nargs="?", default=None, help="4090 / L4 / A100 / H100 …")
    po.add_argument("--geo", default="asia", help="asia | any | KR,JP")
    po.add_argument("--max-price", type=float, default=None)
    po.add_argument("--limit", type=int, default=12)
    po.add_argument("--verified", action="store_true", help="데이터센터 검증 호스트만")

    pk = sub.add_parser("ssh", help="빌린 박스에 붙는 명령을 뽑는다")
    pk.add_argument("ref", help="벤더:아이디")

    ps = sub.add_parser("stop", help="정지한다 — list 가 찾은 것만")
    ps.add_argument("ref", nargs="*", help="벤더:아이디")
    ps.add_argument("--all-billing", action="store_true", help="과금 중인 것 전부")
    ps.add_argument("--yes", action="store_true", help="--all-billing 을 실제로 실행")

    a = ap.parse_args()

    if a.cmd == "list":
        boxes, problems = sweep(a.vendor or list(PROVIDERS))
        if a.json:
            print(json.dumps([{k: v for k, v in b.__dict__.items() if k != "handle"}
                              for b in boxes], ensure_ascii=False, indent=2))
            return
        show(boxes, problems)
        return

    if a.cmd == "offers":
        p = PROVIDERS[a.vendor]()
        offs = p.offers(a.card, geo=a.geo, max_price=a.max_price,
                        limit=a.limit, verified=a.verified)
        if not offs:
            sys.exit("조건에 맞는 오퍼가 없습니다 — --geo any 로 넓혀보세요")
        print(f"  {'오퍼':>10}  {'카드':<14}{'$/hr':>7}  {'위치':<20}{'cuda':<7}"
              f"{'↓Mbps':>7}{'디스크':>7}  검증")
        for o in offs:
            print(f"  {o['id']:>10}  {o['gpu_name']:<14}{o['dph_total']:>7.3f}  "
                  f"{(o.get('geolocation') or '?'):<20}{o['cuda_max_good']:<7}"
                  f"{int(o.get('inet_down', 0)):>7}{int(o['disk_space']):>6}GB  "
                  f"{'예' if o.get('verified') else '아니오'}")
        c = offs[0]
        print(f"\n  가장 싼 것으로 빌리기:\n"
              f"    python box.py rent vast {a.card or c['gpu_name']} --offer {c['id']} --name flashhead")
        print(f"  1시간 ${c['dph_total']:.3f} · 3시간 ${c['dph_total']*3:.2f}")
        return

    if a.cmd == "ssh":
        vendor, _, iid = a.ref.partition(":")
        p = PROVIDERS[vendor]()
        if not hasattr(p, "ssh"):
            sys.exit(f"{vendor} 는 ssh 접속 정보를 이 도구로 주지 않습니다")
        b = next((x for x in p.list() if x.id == iid), None)
        if b is None:
            sys.exit(f"찾지 못했습니다: {a.ref}")
        where = p.ssh(b)
        if not where:
            sys.exit(f"아직 접속 정보가 없습니다 (상태 {b.state}) — 잠시 뒤 다시 보세요")
        host, port = where
        key = VastProvider.SSH_KEY
        print(f"  ssh -p {port} -i {key} -o StrictHostKeyChecking=accept-new root@{host}")
        print(f"  scp -P {port} -i {key} renderer.py run_local.py root@{host}:/root/")
        print(f"\n  준비됐는지:  ssh ... 'cat /root/PROVISIONED 2>/dev/null || tail -5 /root/provision.log'")
        return

    if a.cmd == "rent":
        p = PROVIDERS[a.vendor]()
        # A dry run sends nothing, so it must not require a credential — the
        # moment you most want to read the request is before there is a key.
        why = None if getattr(a, "dry_run", False) else p.preflight()
        if why:
            sys.exit(f"⚠ {why}")
        b = p.rent(a.kind, a.name, teamspace=a.teamspace, cloud=a.cloud,
                   interruptible=a.interruptible, app=a.app, offer=a.offer,
                   geo=a.geo, image=a.image, disk=a.disk, max_price=a.max_price,
                   dry_run=a.dry_run)
        if b.state == "dry-run":
            return
        print(f"● {b.ref} · {b.kind} · {b.state} · {b.where}"
              + (f"\n  {b.note}" if b.note else ""))
        print(f"  끝나면:  python box.py stop {b.ref}")
        return

    # stop — always resolve against a fresh account-wide sweep, so the thing
    # being stopped is the thing the sweep sees, and so a stop is never the
    # first time the account gets looked at.
    vendors = list(PROVIDERS)
    boxes, problems = sweep(vendors)
    for n, why in problems:
        print(f"⚠ {n}: {why}")
    if a.all_billing:
        targets = [b for b in boxes if b.billing]
        if not targets:
            print("과금 중인 것이 없습니다.")
            return
        if not a.yes:
            print("정지 대상 (실행하려면 --yes):")
            for b in targets:
                print(f"  {b.ref}  {b.resource} {b.name} · {b.kind} · {b.where}")
            return
    else:
        by_ref = {b.ref: b for b in boxes}
        targets = []
        for r in a.ref:
            if r not in by_ref:
                print(f"⚠ 찾지 못함: {r}")
                continue
            targets.append(by_ref[r])
        if not targets:
            sys.exit("정지할 대상이 없습니다. `python box.py list` 로 확인하세요.")

    for b in targets:
        try:
            PROVIDERS[b.vendor]().stop(b)
            print(f"✓ 정지: {b.ref}  {b.resource} {b.name}")
        except Exception as e:                        # noqa: BLE001
            print(f"✗ 실패: {b.ref}  {type(e).__name__}: {e}")

    print("\n정지 후 확인:")
    show(*sweep(vendors))


if __name__ == "__main__":
    main()
