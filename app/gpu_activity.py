"""What each GPU tenant is doing, in its own terms, and a per-card record of who ran.

The card meters say how full a card is and how busy; the tenant table says who holds it. Neither
says what the work IS. A chat model has tokens per second, an image pipeline has a queue and a
render time, and a card at 100% is the same reading for both. So the overview's card panels lead
with three things per tenant:

  footprint   VRAM held, measured by the kernel or nvidia-smi (gpu_procs)
  activity    whether it ran kernels in the last few seconds (gpu_procs, never VRAM)
  meaning     the tool's own account of the work, from a small adapter per kind of tool

Only ComfyUI has an adapter so far: its /api/jobs lists what is running, what is queued and how
long recent jobs took, which is everything a card panel needs. LLM backends are described by the
speed probe the hero already uses. Anything else is still shown, with footprint and activity only.

The timeline is the other half. A card's util sparkline says the card was busy; it cannot say for
whom. Each sample here keeps every tenant's compute share, so the card panel can draw who used the
card over the last fifteen minutes, which is what a box shared between a chat model and an image
pipeline actually needs to know.
"""
from __future__ import annotations

import json
import os
import statistics
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field

from . import hw
from .utils import docker_client

# ---------------------------------------------------------------------------------------------
# Tenant kinds. Colour follows the KIND, not the process: a second image tool should read as an
# image tool at a glance, and a restarted container (new pid) must not repaint itself.

KINDS = ("llm", "image", "audio", "other")
_AUDIO_HINTS = ("tts", "chatterbox", "audio", "whisper", "asr", "voice", "kokoro", "piper", "breeze")
_IMAGE_HINTS = ("comfy", "flux", "diffusion", "sdxl", "image")


def kind_of(label: str, is_backend: bool) -> str:
    """llm for one of our backends, else a guess from the container name. Names only: a kind
    is a colour and a word on the panel, never a decision anything acts on."""
    if is_backend:
        return "llm"
    low = (label or "").lower()
    if any(h in low for h in _IMAGE_HINTS):
        return "image"
    if any(h in low for h in _AUDIO_HINTS):
        return "audio"
    return "other"


# ---------------------------------------------------------------------------------------------
# ComfyUI adapter

_COMFY_PORT = 8188
_COMFY_URL_TTL_S = 60.0
_COMFY_TTL_S = 1.0
_COMFY_TIMEOUT_S = 1.5
_RECENT_N = 10

_lock = threading.Lock()
_comfy_url: tuple[float, str] = (0.0, "")
_comfy_cache: dict[str, tuple[float, "ComfyStatus"]] = {}


@dataclass
class ComfyJob:
    id: str
    status: str                  # pending | in_progress | completed | failed
    created_ms: int = 0
    start_ms: int = 0
    end_ms: int = 0
    filename: str = ""
    subfolder: str = ""
    width: int = 0
    height: int = 0

    @property
    def duration_s(self) -> float:
        if self.start_ms and self.end_ms and self.end_ms >= self.start_ms:
            return (self.end_ms - self.start_ms) / 1000.0
        return 0.0


@dataclass
class ComfyStatus:
    ok: bool
    url: str = ""
    error: str = ""
    running: ComfyJob | None = None
    pending: int = 0
    recent: list[ComfyJob] = field(default_factory=list)   # newest first, finished only

    @property
    def busy(self) -> bool:
        return self.running is not None

    @property
    def elapsed_s(self) -> float:
        if not self.running or not self.running.start_ms:
            return 0.0
        return max(0.0, time.time() - self.running.start_ms / 1000.0)

    @property
    def typical_s(self) -> float:
        """Median of the recent successful jobs. A median, because one job that had to load a
        model from disk takes several times the usual and would drag a mean with it."""
        d = [j.duration_s for j in self.recent if j.status == "completed" and j.duration_s > 0]
        return statistics.median(d) if d else 0.0

    @property
    def progress_pct(self) -> int | None:
        """Elapsed against typical, capped below 100 so a slow job never reads as finished.
        None without a typical to compare to. An ESTIMATE: ComfyUI reports step progress only
        over the submitting client's websocket, so nothing else can read the real figure."""
        t = self.typical_s
        if not self.running or not t:
            return None
        return min(95, int(100 * self.elapsed_s / t))

    @property
    def last(self) -> ComfyJob | None:
        return self.recent[0] if self.recent else None

    @property
    def failed_recent(self) -> int:
        return sum(1 for j in self.recent if j.status == "failed")


def _default_gateway() -> str:
    """This container's default route, i.e. the docker host as seen from inside. A published
    port binds every host address, so it answers here even across compose projects, where the
    container's own IP would not (docker isolates one bridge network from another)."""
    try:
        with open("/proc/net/route", encoding="ascii") as f:
            for line in f.readlines()[1:]:
                parts = line.split()
                if len(parts) > 2 and parts[1] == "00000000":
                    g = int(parts[2], 16)
                    return ".".join(str((g >> s) & 0xFF) for s in (0, 8, 16, 24))
    except (OSError, ValueError):
        pass
    return ""


def _discover_comfy_url(container: str) -> str:
    """COMFY_URL when set; otherwise the container's published 8188 on the docker host."""
    env = os.environ.get("COMFY_URL", "").strip()
    if env:
        return env.rstrip("/")
    client = docker_client()
    if client is None or not container:
        return ""
    try:
        ports = (client.containers.get(container).attrs.get("NetworkSettings") or {}).get("Ports") or {}
    except Exception:  # noqa: BLE001 - a container that went away has no URL
        return ""
    for binding in ports.get(f"{_COMFY_PORT}/tcp") or []:
        port = binding.get("HostPort")
        host = _default_gateway()
        if port and host:
            return f"http://{host}:{port}"
    return ""


def comfy_url(container: str) -> str:
    global _comfy_url
    now = time.time()
    with _lock:
        if _comfy_url[1] and now - _comfy_url[0] < _COMFY_URL_TTL_S:
            return _comfy_url[1]
    url = _discover_comfy_url(container)
    with _lock:
        _comfy_url = (now, url)
    return url


def _get_json(url: str):
    with urllib.request.urlopen(url, timeout=_COMFY_TIMEOUT_S) as r:
        return json.loads(r.read().decode("utf-8"))


def _job(d: dict) -> ComfyJob:
    prev = d.get("preview_output") or {}
    meta = prev.get("metadata") or {}
    return ComfyJob(
        id=str(d.get("id") or ""), status=str(d.get("status") or ""),
        created_ms=int(d.get("create_time") or 0),
        start_ms=int(d.get("execution_start_time") or 0),
        end_ms=int(d.get("execution_end_time") or 0),
        filename=str(prev.get("filename") or ""), subfolder=str(prev.get("subfolder") or ""),
        width=int(meta.get("width") or 0), height=int(meta.get("height") or 0))


def parse_comfy(active: dict, done: dict, url: str = "") -> ComfyStatus:
    """Two /api/jobs pages (unfinished, finished) as one status. Split out for the tests."""
    live = [_job(j) for j in (active.get("jobs") or [])]
    running = [j for j in live if j.status == "in_progress"]
    recent = [_job(j) for j in (done.get("jobs") or [])][:_RECENT_N]
    # Oldest first: with one GPU worker there is one, and if a build ever runs two the one that
    # has been going longest is the one a progress estimate is most wrong about.
    run = min(running, key=lambda j: j.start_ms or j.created_ms) if running else None
    if run and not run.start_ms:
        # ComfyUI stamps execution_start_time from the history messages, which exist only once a
        # job finishes, so a running job has none (measured: in_progress carries create_time and
        # nothing else). One worker runs one job at a time, so it started when it was queued or
        # when the job before it ended, whichever was later. Exact to the worker's own handoff.
        prev_end = max((j.end_ms for j in recent if j.end_ms), default=0)
        run.start_ms = max(run.created_ms, prev_end)
    return ComfyStatus(
        ok=True, url=url, running=run,
        pending=sum(1 for j in live if j.status == "pending"), recent=recent)


def comfy_status(container: str) -> ComfyStatus:
    """The ComfyUI tenant's queue and recent jobs, cached for a second. Never raises."""
    now = time.time()
    with _lock:
        hit = _comfy_cache.get(container)
        if hit and now - hit[0] < _COMFY_TTL_S:
            return hit[1]
    url = comfy_url(container)
    if not url:
        st = ComfyStatus(ok=False, error="no published port 8188 (set COMFY_URL)")
    else:
        try:
            active = _get_json(f"{url}/api/jobs?status=in_progress,pending")
            done = _get_json(f"{url}/api/jobs?status=completed,failed&limit={_RECENT_N}")
            st = parse_comfy(active, done, url)
        except Exception as e:  # noqa: BLE001 - an unreachable API is a line on the panel
            st = ComfyStatus(ok=False, url=url, error=type(e).__name__)
    with _lock:
        _comfy_cache[container] = (now, st)
    return st


# ---------------------------------------------------------------------------------------------
# Per-card timeline: who was working on each card, sampled once a second.

_TIMELINE_MAX = 900          # 15 minutes at 1 s
_TIMELINE_INTERVAL_S = 1.0
_timeline: dict[int, deque] = {}
_kinds_seen: dict[str, str] = {}
_started = False


@dataclass
class CardSample:
    ts: float
    util: float
    vram_gb: float
    share: dict[str, int]    # tenant label -> compute %, only tenants that ran


def _cards() -> list:
    return hw.host_nvidia_cards() or hw._backend_cards()


def record_once() -> None:
    from .fit import _gpu_tenants   # local: fit imports services, which imports half the app
    cards = _cards()
    if not cards:
        return
    tenants = _gpu_tenants()
    now = time.time()
    for c in cards:
        share: dict[str, int] = {}
        for t in tenants:
            cu = t.cu_by_card.get(c.index, 0)
            if cu and t.cu_total:
                share[t.label] = share.get(t.label, 0) + round(100.0 * cu / t.cu_total)
            if t.per_card_gb.get(c.index):
                _kinds_seen[t.label] = kind_of(t.label, not t.foreign)
        with _lock:
            buf = _timeline.setdefault(c.index, deque(maxlen=_TIMELINE_MAX))
            buf.append(CardSample(now, c.util_pct, c.vram_used_gb, share))


def _loop() -> None:
    while True:
        try:
            record_once()
        except Exception:  # noqa: BLE001 - the timeline is cosmetic; never kill the thread
            pass
        time.sleep(_TIMELINE_INTERVAL_S)


def start() -> None:
    global _started
    if _started:
        return
    _started = True
    threading.Thread(target=_loop, daemon=True, name="gpu-activity").start()


def timeline(card: int) -> list[CardSample]:
    with _lock:
        return list(_timeline.get(card) or ())


def kind_for(label: str) -> str:
    return _kinds_seen.get(label, "other")


def _smooth(points: list[tuple[float, float]], lo: float, hi: float) -> str:
    """Catmull-Rom through the points as cubic Beziers, control points clamped to [lo, hi] so a
    curve between two zeros never dips below the baseline it is drawn on."""
    if len(points) < 2:
        return ""
    clamp = lambda v: min(hi, max(lo, v))  # noqa: E731
    out = [f"M{points[0][0]:.2f},{points[0][1]:.2f}"]
    for i in range(len(points) - 1):
        p0 = points[i - 1] if i else points[i]
        p1, p2 = points[i], points[i + 1]
        p3 = points[i + 2] if i + 2 < len(points) else p2
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, clamp(p1[1] + (p2[1] - p0[1]) / 6))
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, clamp(p2[1] - (p3[1] - p1[1]) / 6))
        out.append(f"C{c1[0]:.2f},{c1[1]:.2f} {c2[0]:.2f},{c2[1]:.2f} {p2[0]:.2f},{p2[1]:.2f}")
    return " ".join(out)


_W, _H, _PAD = 100.0, 32.0, 1.5


def sparklines(samples: list[CardSample], buckets: int = 60) -> dict:
    """The timeline as one smooth line per tenant, plus the card's own util, in a 100 x 32 box.

    Each column is a MEAN over its samples: NVIDIA's pmon reading is a 2 s sample, so the raw
    1 s series is mostly repeats and zeros, and a mean says how much of that stretch of time the
    tenant had the card. At least two samples per column, or a short window (after a restart)
    draws a comb. Fixed 0-100% scale, so a line's height means the same thing every frame.

    Lines, not a stack: tenants rarely overlap on one card, and when they do a line each says
    who was doing what without one band's height depending on the band beneath it.
    """
    if len(samples) < 4:
        return {"lines": [], "card": None, "span_s": 0}
    buckets = max(2, min(buckets, len(samples) // 2))
    t0, t1 = samples[0].ts, samples[-1].ts
    span = max(1.0, t1 - t0)
    col = lambda s: min(buckets - 1, int((s.ts - t0) / span * buckets))  # noqa: E731
    counts = [0] * buckets
    util = [0.0] * buckets
    per: dict[str, list[float]] = {}
    for s in samples:
        i = col(s)
        counts[i] += 1
        util[i] += s.util
        for label, pct in s.share.items():
            per.setdefault(label, [0.0] * buckets)[i] += pct
    xs = [i * _W / (buckets - 1) for i in range(buckets)]
    y = lambda v: _H - _PAD - min(100.0, max(0.0, v)) / 100.0 * (_H - 2 * _PAD)  # noqa: E731

    def shape(vals: list[float]) -> dict:
        means = [v / c if c else 0.0 for v, c in zip(vals, counts)]
        pts = [(x, y(m)) for x, m in zip(xs, means)]
        line = _smooth(pts, _PAD, _H - _PAD)
        return {"line": line, "area": f"{line} L{_W:.2f},{_H:.2f} L0,{_H:.2f} Z",
                "now": round(means[-1]), "peak": round(max(means)),
                "end_y": round(pts[-1][1], 2)}

    lines = []
    for label, vals in per.items():
        sh = shape(vals)
        if sh["peak"] < 1:
            continue
        lines.append({"label": label, "kind": kind_for(label), **sh})
    lines.sort(key=lambda ln: -ln["peak"])
    return {"lines": lines, "card": shape(util), "span_s": int(span)}


def curve(values, top: float | None = None, points: int = 60) -> dict | None:
    """Any series as the overview's one chart shape: a smooth line and the area under it, in a
    100 x 32 box. None for fewer than three values, so a template can test it directly.

    `top` pins the scale (a limit, a total, 100%) so the height means the same thing every
    frame; without it the series is drawn against its own peak, which is right only for rates
    with no meaningful ceiling (tok/s), and the caller should then print that peak. Long series
    are averaged down to `points` columns: the line is drawn at card width, and 240 points in
    a few hundred pixels is noise rather than detail.
    """
    vals = [float(v) for v in (values or []) if v is not None]
    if len(vals) < 3:
        return None
    if len(vals) > points:
        k = len(vals) / points
        chunks = [vals[int(i * k):int((i + 1) * k)] or [vals[int(i * k)]] for i in range(points)]
        vals = [sum(c) / len(c) for c in chunks]
    ceil = top if top and top > 0 else (max(vals) or 1.0)
    n = len(vals)
    y = lambda v: _H - _PAD - min(1.0, max(0.0, v / ceil)) * (_H - 2 * _PAD)  # noqa: E731
    pts = [(i * _W / (n - 1), y(v)) for i, v in enumerate(vals)]
    line = _smooth(pts, _PAD, _H - _PAD)
    return {"line": line, "area": f"{line} L{_W:.2f},{_H:.2f} L0,{_H:.2f} Z",
            "last": vals[-1], "peak": max(vals)}
