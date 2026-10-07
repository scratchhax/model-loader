from __future__ import annotations

import asyncio
import configparser
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import docker
import httpx
from docker.errors import APIError, DockerException, NotFound

from . import ini
from . import diagnose
from .config import settings
from .utils import human_bytes, shard_key


# ---------- models directory ----------


@dataclass(frozen=True)
class ModelShape:
    """MoE or dense, read from the GGUF header.

    Worth surfacing because it decides what offloading costs, and the two are not close. In a
    DENSE model every weight is read for every token, so a layer moved to system RAM is paid on
    every token. In an MoE only the routed experts are read - 8 of 128, or 10 of 512 - so most
    of what sits in RAM is untouched on any given token. Measured here: Qwen3.8-Flash-Next keeps
    ~60 GB of experts in DDR4-2667 and still generates at 17 tok/s; a dense model with that much
    on the same memory would be under 1.

    It also says whether --cpu-moe / --n-cpu-moe will do anything at all. On a dense model they
    are silently inert.
    """
    arch: str = ""
    expert_count: int = 0
    expert_used: int = 0
    # True when the weights carry their own MTP layers and llama.cpp can draft against the
    # target file with no separate head. Read from {arch}.nextn_predict_layers, the same key
    # autoconfig treats as load-or-fatal for MTP.
    internal_mtp: bool = False

    @property
    def known(self) -> bool:
        return bool(self.arch)

    @property
    def is_moe(self) -> bool:
        return self.expert_count > 0

    @property
    def label(self) -> str:
        if not self.known:
            return ""
        if not self.is_moe:
            return "dense"
        return f"MoE {self.expert_count}x{self.expert_used}" if self.expert_used else \
               f"MoE {self.expert_count}"


def model_shape(path: Path) -> ModelShape:
    """Read architecture + expert counts from a GGUF header.

    gguf_meta.read_raw already caches on (path, mtime, size) and only reads the head of the
    file, so calling this per row on a page render is cheap after the first hit.
    """
    from . import gguf_meta
    try:
        kv = gguf_meta.read_raw(path)
    except Exception:  # noqa: BLE001 - an unreadable header is "unknown", never an error page
        return ModelShape()
    if isinstance(kv, dict) and isinstance(kv.get("kv"), dict):
        kv = kv["kv"]
    if not isinstance(kv, dict):
        return ModelShape()
    arch = str(kv.get("general.architecture") or "")
    if not arch:
        return ModelShape()

    def _int(suffix: str) -> int:
        v = kv.get(f"{arch}.{suffix}")
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    return ModelShape(arch=arch, expert_count=_int("expert_count"),
                      expert_used=_int("expert_used_count"),
                      internal_mtp=_int("nextn_predict_layers") > 0)


# Trailing quantisation token, with the publisher prefixes that ride in front of it.
# Anchored to the end so a name that merely CONTAINS a quant-looking run is left alone.
_QUANT_RE = re.compile(
    r"[-_.](?:(?:UD|i1|ud|unsloth)-)?"
    r"(I?Q\d+(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:_MOE)?|TQ\d_\d)$"
)


def split_quant(stem: str) -> tuple[str, str]:
    """("Qwen3.8-27B-UD-Q4_K_M") -> ("Qwen3.8-27B", "UD-Q4_K_M").

    The quant is the only part of a GGUF filename that varies between copies of the SAME
    model, so stripping it is what lets three files collapse into one family heading. When
    nothing matches, the whole stem is the family and the quant is empty - which is the right
    answer for a file that genuinely has no quant in its name, not a parse failure.
    """
    m = _QUANT_RE.search(stem)
    if not m:
        return stem, ""
    return stem[:m.start()], stem[m.start() + 1:]


def canonical_families(families: "Iterable[str]") -> dict[str, str]:
    """{family: the family it belongs under} once variants are folded onto their base.

    Stripping the quant leaves one family per FINETUNE, which splits things that are plainly
    the same model: Qwen3.8-Flash-Next, -GSQ-RCO and -Uncensored became three headings of one
    file each. The useful heading is the base model, with its variants under it.

    The rule is the files themselves, not a word list. A family folds onto a shorter one when
    that shorter one is a token-boundary prefix of it AND is itself a family present on disk.
    So -GSQ-RCO and -Uncensored join Qwen3.8-Flash-Next because that base is here, while
    gemma-4-12b-it and gemma-4-E4B-it-qat stay apart because no "gemma-4" base is - and they
    SHOULD stay apart, being different model sizes rather than variants of one. A word list
    would have had to know that "GSQ-RCO" names a compression method, and would be wrong again
    on the next publisher's suffix.

    Shortest matching prefix wins, which is the same answer as folding repeatedly. The one
    consequence worth knowing: delete the base and its variants separate again, because then
    the base is not something you have.
    """
    known = set(families)
    out: dict[str, str] = {}
    for f in known:
        parts = f.split("-")
        out[f] = f
        for i in range(1, len(parts)):
            prefix = "-".join(parts[:i])
            if prefix in known:
                out[f] = prefix
                break
    return out


def largest_card_gb() -> float:
    """VRAM of the single biggest card we can see, 0 when nothing reports one.

    One card is the threshold that actually matters for placement: under it a model is a
    whole-card tenant, over it it has to be split or spilled. Summed VRAM is the wrong number
    for that question - two 32 GB cards do not hold a 40 GB model the way one 64 GB card would.
    """
    from . import hw
    best = 0.0
    for name in _effective_container_names():
        st = hw.stats_for(name)
        if not (st.ok and st.gpu):
            continue
        cards = st.gpu.cards or []
        if cards:
            best = max([best] + [float(c.vram_total_gb or 0) for c in cards])
        else:
            best = max(best, float(st.gpu.vram_total_gb or 0))
    return best


# Fractions of ONE card. Below the lower one a model can share a card with something else and
# both still have room for their KV; above a whole card it cannot be a single-card tenant at all.
_TIER_SHARE = 0.4


def size_tier(total_bytes: int, card_gb: float) -> tuple[str, str, str]:
    """(key, heading, chip) for how this file has to be placed on THIS box.

    Two labels because the same fact reads differently in the two places it appears. As a
    group heading it is a sentence you read once - "Needs more than one GPU" - and as a chip
    on every row it has to be a word you skim past, so it is "2+ GPUs". A chip carrying the
    long form made every sharded row wrap onto a second line.

    Weights only. KV and compute buffers are not in it and they are not small - the autoconfig
    panel is where that sum is done properly. This is a shelf label, so it answers "roughly
    where does this one live"; the tooltip says it is weights alone.
    """
    if card_gb <= 0:
        return "unknown", "Size unknown", ""
    gb = total_bytes / 1024 ** 3
    if gb > card_gb:
        return "multi", "Needs more than one GPU", "2+ GPUs"
    if gb > card_gb * _TIER_SHARE:
        return "single", "Fits one GPU", "1 GPU"
    return "share", "Shares a GPU", "part of a GPU"

@dataclass
class GgufEntry:
    display_name: str        # shown to user (base name, without shard suffix if grouped)
    parts: list[Path]        # one entry, or many shards sorted by index
    total_bytes: int
    mtime: float             # newest part's mtime
    is_sharded: bool
    subdir: str = ""         # empty for flat files, subdir name for shard groups in a subdir
    is_companion: bool = False   # True for mmproj / other files referenced from a main model's section
    aliases: list[str] = field(default_factory=list)  # ini section names that reference this
    # If this is a main model with a companion mmproj in the same subdir, these fields
    # describe the companion. Standalone companion entries are hidden from listings.
    companion_name: str = ""
    companion_bytes: int = 0
    companion_parts: list[Path] = field(default_factory=list)

    @property
    def human_size(self) -> str:
        return human_bytes(self.total_bytes)

    @property
    def modified(self) -> str:
        return datetime.fromtimestamp(self.mtime, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")

    @property
    def stem(self) -> str:
        # e.g. "gemma-4-12b-it-Q4_K_M.gguf" -> "gemma-4-12b-it-Q4_K_M"
        return self.display_name[:-5] if self.display_name.lower().endswith(".gguf") else self.display_name

    @property
    def model_id(self) -> str:
        """The id llama-server serves this file under.

        That is its ini section name when it has one — which is NOT necessarily the filename
        stem, since a section can be renamed to give the model a short API id — and otherwise
        the stem, which is what llama-server falls back to.
        """
        if self.stem in self.aliases:
            return self.stem      # a section named after the file: the natural id
        return self.aliases[0] if self.aliases else self.stem

    @property
    def route_key(self) -> str:
        """URL-safe key for /model/<key> routes; for subdir'd entries this includes the subdir."""
        return f"{self.subdir}/{self.display_name}" if self.subdir else self.display_name

    @property
    def first_shard_rel(self) -> str:
        """Relative path (from models_dir) to the first shard/file. Used in ini `model = ...`."""
        first = self.parts[0].name
        return f"{self.subdir}/{first}" if self.subdir else first


@dataclass
class DiskInfo:
    total: int
    free: int
    used_pct: float

    @property
    def total_h(self) -> str: return human_bytes(self.total)
    @property
    def free_h(self) -> str: return human_bytes(self.free)


@dataclass
class ModelsDirSnapshot:
    path: Path
    exists: bool
    error: str | None = None
    disk: DiskInfo | None = None
    ggufs: list[GgufEntry] = field(default_factory=list)
    ini_aliases: list[str] = field(default_factory=list)
    ini_present: bool = False


def read_ini_aliases(ini_path: Path) -> list[str]:
    if not ini_path.exists():
        return []
    try:
        cp = configparser.ConfigParser(strict=False)
        cp.read(ini_path, encoding="utf-8")
        return cp.sections()
    except (OSError, configparser.Error):
        return []


def snapshot_models_dir() -> ModelsDirSnapshot:
    path = settings.models_dir
    snap = ModelsDirSnapshot(path=path, exists=path.exists())
    if not path.exists():
        snap.error = "directory not found (is the volume mounted?)"
        return snap

    try:
        u = shutil.disk_usage(path)
        snap.disk = DiskInfo(total=u.total, free=u.free, used_pct=round(100 * (u.total - u.free) / u.total, 1))
    except OSError as e:
        snap.error = f"disk usage failed: {e}"

    aliases = read_ini_aliases(settings.models_ini_path)
    snap.ini_aliases = aliases
    # file -> section, so a section renamed to a short API id still owns its GGUF
    by_file = ini.sections_by_file()
    snap.ini_present = settings.models_ini_path.exists()

    # collect gguf files, grouping shards. Also descend one level into subdirs
    # (multi-shard downloads land in /models/<base>/ to keep the top level tidy).
    # Key = (subdir, shard_base). subdir="" for flat.
    groups: dict[tuple[str, str], list[Path]] = {}
    try:
        for p in path.iterdir():
            if p.is_file() and p.suffix.lower() == ".gguf":
                base, _, _ = shard_key(p.name)
                groups.setdefault(("", base), []).append(p)
            elif p.is_dir() and not p.name.startswith("."):
                try:
                    subparts = list(p.iterdir())
                except OSError:
                    continue
                for sp in subparts:
                    if sp.is_file() and sp.suffix.lower() == ".gguf":
                        base, _, _ = shard_key(sp.name)
                        groups.setdefault((p.name, base), []).append(sp)
    except OSError as e:
        snap.error = f"listing failed: {e}"
        return snap

    entries: list[GgufEntry] = []
    for (subdir, base), parts in groups.items():
        parts.sort(key=lambda pp: pp.name)
        total = sum(pp.stat().st_size for pp in parts)
        mtime = max(pp.stat().st_mtime for pp in parts)
        stem_no_ext = base[:-5] if base.lower().endswith(".gguf") else base
        rel = f"{subdir}/{parts[0].name}" if subdir else parts[0].name
        matched = by_file.get(rel) or [a for a in aliases if a == stem_no_ext]
        # Use the same predicate the ini layer uses to decide what may become its own section,
        # rather than a second, narrower rule. This one only tested for mmproj, so speculative
        # draft heads - Qwen MTP files, generic -draft- - were listed as standalone models even
        # though nothing can be done with them: they cannot be run or configured alone, and they
        # are already referenced from their main model's section via `model-draft`.
        is_comp = ini._is_companion(base)
        entries.append(GgufEntry(
            display_name=base,
            parts=parts,
            total_bytes=total,
            mtime=mtime,
            is_sharded=len(parts) > 1,
            subdir=subdir,
            is_companion=is_comp,
            aliases=matched,
        ))

    # Fold companion entries into their same-subdir main model.
    # A companion is a file that only makes sense paired with its main model: a multimodal
    # projector (mmproj) or a speculative draft head (MTP). You can't run one alone, can't
    # configure it, can't do anything with it. So we hide it from the list and expose it as a
    # badge on the main model instead. Un-paired companions (one sitting in a subdir with no
    # main model) stay visible so they can be cleaned up.
    main_by_subdir: dict[str, GgufEntry] = {
        e.subdir: e for e in entries if not e.is_companion and e.subdir
    }
    surviving: list[GgufEntry] = []
    for e in entries:
        if e.is_companion and e.subdir and e.subdir in main_by_subdir:
            main = main_by_subdir[e.subdir]
            # ACCUMULATE. A subdir can hold several projectors (mmproj-BF16 + mmproj-F32 is
            # a common upload pattern). Assigning here instead of appending meant the second
            # one overwrote the first, so deleting the model stranded a projector on disk and
            # left the subdir non-empty -- which then silently defeated the rmdir below.
            main.companion_name = (f"{main.companion_name}, {e.display_name}"
                                   if main.companion_name else e.display_name)
            main.companion_bytes += e.total_bytes
            main.companion_parts.extend(e.parts)
            continue  # drop the standalone companion row
        surviving.append(e)
    # sort: (remaining) companions after main models, alpha within group
    surviving.sort(key=lambda e: (e.subdir.lower(), e.is_companion, e.display_name.lower()))
    snap.ggufs = surviving
    return snap


def delete_gguf(display_name: str, subdir: str = "") -> tuple[bool, str, int]:
    """Delete a GGUF (and all shards). Returns (ok, message, bytes_freed).
    If subdir is empty, matches only flat files with that display_name; otherwise the entry in that subdir."""
    snap = snapshot_models_dir()
    if snap.error and not snap.ggufs:
        return False, snap.error, 0
    # match by display_name AND subdir (allows same base filename to exist both flat and in a subdir)
    match = next((g for g in snap.ggufs if g.display_name == display_name and g.subdir == subdir), None)
    if match is None and subdir == "":
        # fall back: match any subdir if only display_name given (bulk-delete callers pass just display_name)
        match = next((g for g in snap.ggufs if g.display_name == display_name), None)
    if match is None:
        return False, f"not found: {display_name}", 0
    freed = 0
    removed: list[str] = []

    # Whole-directory delete when this model is the only model in its subdir.
    #
    # Downloads land one model per directory, and everything beside the weights there is
    # support material for THAT model: projectors (often more than one), chat_template.jinja,
    # tokenizer.model. Deleting file-by-file means anything not explicitly enumerated is
    # stranded -- and a single leftover keeps the directory non-empty, so the tidy-up rmdir
    # below silently does nothing and the orphans persist invisibly. Measured on a real
    # deletion: 3.3 GB of projectors left behind.
    #
    # Guarded on there being no OTHER main GGUF present, so a directory someone has put two
    # models into degrades to per-file deletion rather than taking the neighbour with it.
    if match.subdir:
        subpath = settings.models_dir / match.subdir
        own = {p.resolve() for p in list(match.parts) + list(match.companion_parts)}
        try:
            others = [
                q for q in subpath.iterdir()
                if q.is_file() and q.suffix.lower() == ".gguf"
                and not ini._is_companion(q.name) and q.resolve() not in own
            ]
        except OSError:
            others = []
        if not others and subpath.is_dir():
            try:
                for q in sorted(subpath.rglob("*")):
                    if q.is_file():
                        freed += q.stat().st_size
                        removed.append(q.name)
                shutil.rmtree(subpath)
                return True, f"deleted {len(removed)} file(s), removed {match.subdir}/", freed
            except OSError as e:
                return False, f"failed to remove {match.subdir}/: {e}", freed

    # Fallback: shared directory, or a flat file with no directory of its own.
    paths_to_remove: list[Path] = list(match.parts) + list(match.companion_parts)
    for p in paths_to_remove:
        try:
            size = p.stat().st_size
            p.unlink()
            freed += size
            removed.append(p.name)
        except OSError as e:
            return False, f"failed to delete {p.name}: {e}. removed so far: {removed}", freed
    # if this entry lived in a subdir, remove the dir when empty
    if match.subdir:
        subpath = settings.models_dir / match.subdir
        try:
            if subpath.is_dir() and not any(subpath.iterdir()):
                subpath.rmdir()
        except OSError:
            pass
    return True, f"deleted {len(removed)} file(s)", freed


# ---------- docker / containers ----------

def _docker_client() -> docker.DockerClient | None:
    try:
        return docker.from_env()
    except DockerException:
        return None


@dataclass
class ContainerInfo:
    name: str
    found: bool
    status: str | None = None
    image: str | None = None
    id: str | None = None


# ---------- llama backends: richer view + restart + probe ----------

# error-of-last-restart per container, shown in the card until it clears
_restart_errors: dict[str, str] = {}
_restart_errors_lock = threading.Lock()


@dataclass
class LlamaBackend:
    name: str
    found: bool
    status: str = ""           # running | exited | restarting | not_found | error
    image: str = ""
    short_id: str = ""
    started_at: str = ""
    uptime: str = ""
    host_ports: list[str] = field(default_factory=list)
    internal_port: int | None = None
    loaded_model: str | None = None
    probe_error: str | None = None
    # A genuine load failure reported by the router — a model whose /v1/models status is an
    # error value, with its message. Distinct from probe_error (a probe/HTTP problem) and from
    # mere idleness: under --models-preset with --models-max 1, "nothing loaded" is the normal
    # resting state, so nothing-loaded must never read as a failure. None = no reported failure.
    load_failed: str | None = None
    # The model the router is bringing up right now, and how long it has been at it. Both None /
    # 0.0 whenever nothing is loading, which is almost always. loading_since is wall-clock
    # seconds measured by US, not by the router: /v1/models reports the state but not when it
    # started, so the clock begins the first time a probe sees it. The probe runs every 1.5 s
    # (_HERO_BACKENDS_TTL), so the figure can read up to ~1.5 s short of the truth - fine for a
    # counter whose job is to prove the box is busy, not to time anything.
    loading_model: str | None = None
    loading_since: float = 0.0
    # A model the server still owns but whose VRAM it has released under --sleep-idle-seconds.
    # Never folded into loaded_model: see _probe_loaded_model for why that would stop sleep
    # working at all. The next request reloads it, measured at 8.2-9.4 s on this box.
    sleeping_model: str | None = None
    last_restart_error: str | None = None
    # --sleep-idle-seconds: after this many seconds without a request the server releases its
    # model's VRAM and reloads on the next one. 0 = the flag is absent, which is llama.cpp's
    # default and means the model holds its allocation until something evicts it.
    sleep_idle_s: int = 0
    # Whether it is sleeping RIGHT NOW. Only meaningful when sleep_idle_s > 0.
    asleep: bool = False
    # True when launched with --models-preset/--models-dir, i.e. it serves models.ini and can
    # swap models. False for a -m server such as the Orpheus TTS backend, which serves exactly
    # one file and is not a chat endpoint.
    router: bool = True
    # "rocm" | "cuda" | "cpu" | "unknown", from the image tag. Whether this backend can touch
    # the cards at all, which is what decides who leads the overview.
    vendor: str = ""
    # Which inference stack this is: "llama" for every llama.cpp image, or whatever the
    # container declares in its `ai-lab.engine` label. Strata is the first non-llama one.
    #
    # It exists because almost everything this app knows is llama.cpp-specific: models.ini,
    # autoconfig's KV and compute-buffer budget, the argv vocabulary, --models-max eviction.
    # A backend of another engine is a real backend - it holds a card, serves an OpenAI API
    # and answers /v1/models - but none of that machinery describes it, so the code that
    # would otherwise size it or write its arguments has to be able to ask.
    engine: str = "llama"


def _parse_started_at(iso: str) -> tuple[str, str]:
    # docker returns e.g. "2025-08-11T00:51:00.123456789Z"
    if not iso or iso.startswith("0001"):
        return "", ""
    try:
        # trim nanoseconds to microseconds
        core, _, frac = iso.partition(".")
        if frac:
            frac = frac.rstrip("Z")[:6]
            iso_norm = f"{core}.{frac}+00:00"
        else:
            iso_norm = core.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso_norm)
    except ValueError:
        return iso, ""
    local = dt.astimezone().strftime("%Y-%m-%d %H:%M")
    delta = datetime.now(timezone.utc) - dt
    secs = int(delta.total_seconds())
    if secs < 60:
        up = f"{secs}s"
    elif secs < 3600:
        up = f"{secs // 60}m"
    elif secs < 86400:
        up = f"{secs // 3600}h {(secs % 3600) // 60}m"
    else:
        up = f"{secs // 86400}d {(secs % 86400) // 3600}h"
    return local, up


def _extract_ports(attrs: dict) -> tuple[list[str], int | None]:
    """Return (['8082->8080/tcp'], 8080)."""
    ports_map = (attrs.get("NetworkSettings") or {}).get("Ports") or {}
    result: list[str] = []
    internal: int | None = None
    for cont_port, bindings in ports_map.items():
        # cont_port like "8080/tcp"
        try:
            internal = int(cont_port.split("/")[0])
        except ValueError:
            pass
        if not bindings:
            continue
        for b in bindings:
            hp = b.get("HostPort")
            if hp:
                result.append(f"{hp}->{cont_port}")
    return result, internal


def _api_key_from_attrs(attrs: dict) -> str:
    """The llama-server --api-key, if the container was started with one.

    Compose `command:` lands in Config.Cmd, an image entrypoint in Config.Entrypoint; scan both
    so a key set either way is found. Returns "" when the server runs open (the common LAN case).
    """
    cfg = attrs.get("Config") or {}
    argv = [str(t) for t in (cfg.get("Entrypoint") or [])] + [str(t) for t in (cfg.get("Cmd") or [])]
    for i, tok in enumerate(argv):
        if tok == "--api-key" and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith("--api-key="):
            return tok.split("=", 1)[1]
    return ""


def browser_endpoints(browser_host: str) -> list[dict]:
    """Per-backend, browser-reachable /v1 base URL + api key, for client snippets.

    Probing uses docker-internal http://<container>:<port>, which the browser cannot resolve.
    Here we take the published host port that llama-server's API port is bound to and pair it
    with the address the browser actually used to reach Model Loader, so a snippet works from
    any machine on the LAN rather than only from inside the compose network.
    """
    client = _docker_client()
    if client is None or not browser_host:
        return []
    out: list[dict] = []
    for name in _effective_container_names():
        try:
            attrs = client.containers.get(name).attrs or {}
        except (NotFound, DockerException):
            continue
        host_ports, internal = _extract_ports(attrs)
        host_port = None
        if internal is not None:
            suffix = f"{internal}/"
            for hp in host_ports:  # each is "8082->8080/tcp"
                left, _, cont = hp.partition("->")
                if cont.startswith(suffix) and left.isdigit():
                    host_port = left
                    break
        if host_port is None:  # no binding matched the API port; fall back to first numeric
            for hp in host_ports:
                left, _, _ = hp.partition("->")
                if left.isdigit():
                    host_port = left
                    break
        if host_port is None:
            continue
        out.append({
            "name": name,
            "base_url": f"http://{browser_host}:{host_port}/v1",
            "api_key": _api_key_from_attrs(attrs),
        })
    return out


def _load_failure(status: dict) -> tuple[bool, int | None]:
    """(failed, exit_code) for one model's /v1/models status object.

    Measured on this llama.cpp build by loading a section whose model file does not exist:

        before the attempt : {"value": "unloaded"}
        after it failed    : {"value": "unloaded", "exit_code": 1, "failed": true}

    There is no distinct error `value` and no message; the model returns to unloaded and gains a
    flag, which persists. A model evicted normally reads plain {"value": "unloaded"} with neither
    key, so the flag is specific to failure and an ordinary model swap never trips it. The check
    this replaced looked for an error in `value`, which this router never sends, so the dashboard
    alarm could not fire. A non-zero exit code alone is also treated as a failure, since a clean
    stop leaves no exit code at all.
    """
    raw = status.get("exit_code")
    try:
        code = int(raw) if raw is not None else None
    except (TypeError, ValueError):
        code = None
    return (status.get("failed") is True or (code is not None and code != 0)), code


async def _probe_loaded_model(
    container_name: str, internal_port: int | None
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """(loaded_model_csv, probe_error, load_failed, loading_model, sleeping_model).

    loaded_model  — comma list of ids whose status is 'loaded'.
    probe_error   — a problem talking to the endpoint (HTTP/JSON), or an idle-state note.
    load_failed   — the first model whose last load failed, as "name (exit N)"; None otherwise.
                    The only field the dashboard alarm keys off: a real failure, not the normal
                    unloaded state. See _load_failure for what the router actually reports.
    loading_model — the model the router is bringing up RIGHT NOW, or None.
    sleeping_model — a model held under --sleep-idle-seconds: the server still owns it and will
                    reload it on the next request, but its VRAM is released. Measured: 50.1 GB
                    awake, 7.3 GB asleep on this box.

    sleeping is kept OUT of loaded_model on purpose, and it is not a stylistic choice. Callers
    treat loaded_model as "there is something here worth probing", and the probe is /slots -
    which both resets the idle timer and wakes a sleeper. Folding the two together would mean
    the overview page, polling every 500 ms, silently prevented any model from ever sleeping
    and woke the one that already had. Display and liveness have to stay separate fields.

    The last of those is why this function was wrong for as long as it has existed. Measured on
    b11206 by polling this endpoint at 120 ms through a real swap:

        0.00  gemma-4-31B-it-Q6_K=loaded
        1.11  Qwen3.8-27B-UD-Q4_K_M=loading
        10.51 Qwen3.8-27B-UD-Q4_K_M=loaded

    The outgoing model flips straight to 'unloaded' and the incoming one sits on 'loading', so
    for the whole window NOTHING reports as loaded. Collecting only 'loaded' therefore made a
    model swap indistinguishable from an idle box, and the overview said "nothing loaded" for
    the entire load - 9.6 s on a 27B, 40 s on gemma-4-31B at Q6_K. The status was always on the
    wire; it was being thrown away here.
    """
    if internal_port is None:
        return None, None, None, None, None
    url = f"http://{container_name}:{internal_port}/v1/models"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(2.0, read=3.0)) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return None, f"HTTP {r.status_code}", None, None, None
            data = r.json()
            items = data.get("data") or []
            if not items:
                return None, "no models configured", None, None, None
            loaded_ids: list[str] = []
            failure: str | None = None
            loading: str | None = None
            sleeping: str | None = None
            for it in items:
                mid = str(it.get("id") or "")
                # A ROUTER reports status.value per model because it loads and evicts them. A
                # server started with -m has no status field at all, because it has nothing to
                # load or unload: the one model named on its command line is resident for the
                # life of the process. Reading an absent status as "not loaded" made Orpheus
                # TTS report "1 configured, none loaded" while it was actively speaking, and
                # left the overview claiming nothing was running whenever it was the only
                # backend holding a model. Its id is the container path, so show the stem.
                if "status" not in it:
                    stem = mid.rsplit("/", 1)[-1]
                    if stem.lower().endswith(".gguf"):
                        stem = stem[:-5]
                    if stem:
                        loaded_ids.append(stem)
                    continue
                status = it.get("status") or {}
                val = str(status.get("value") or "").lower()
                if val == "loaded":
                    loaded_ids.append(mid)
                elif val == "sleeping":
                    sleeping = mid
                elif val == "loading":
                    # At --models-max 1 there can only be one, and it flips within ~0.3 s of the
                    # request - early enough to cover the eviction of its predecessor, so this
                    # one field spans the whole episode the user is staring at.
                    loading = mid
                elif failure is None:
                    failed, code = _load_failure(status)
                    if failed:
                        failure = f"{mid} (exit {code})" if code is not None else mid
            if failure is not None:
                return (", ".join(i for i in loaded_ids if i) or None, None, failure,
                        loading, sleeping)
            if loaded_ids:
                return ", ".join(i for i in loaded_ids if i), None, None, loading, sleeping
            # "none loaded" is the honest note only when nothing is on its way in and nothing is
            # merely asleep. Either of those read as an idle box, which is the confusion this
            # whole function keeps being wrong about.
            if loading or sleeping:
                return None, None, None, loading, sleeping
            return None, f"{len(items)} configured, none loaded", None, None, None
    except (httpx.HTTPError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}", None, None, None


def _engine_label(c) -> str:
    """The engine a container declares in its `ai-lab.engine` label, lowercased, or "".

    Backends were found purely by image name, which holds exactly as long as every backend is
    a llama.cpp tag. Strata is the first one that is not: a different stack, its own locally
    built image, and nothing in models.ini applying to it. An explicit label is the
    declaration. Sniffing for "strata" in an image name would be the same substring guess one
    rung further down, and image tags are not ours to depend on.
    """
    try:
        labels = ((c.attrs or {}).get("Config") or {}).get("Labels") or {}
    except DockerException:
        return ""
    if not isinstance(labels, dict):
        return ""
    return str(labels.get("ai-lab.engine") or "").strip().lower()


def discover_llama_containers() -> list[dict]:
    """Return metadata for every container that is a backend: a llama.cpp:server-* image, or
    any container declaring an `ai-lab.engine` label. Includes stopped containers so the user
    can see and start them.
    """
    client = _docker_client()
    if client is None:
        return []
    out: list[dict] = []
    try:
        containers = client.containers.list(all=True)
    except DockerException:
        return []
    for c in containers:
        try:
            tags = c.image.tags or [c.image.short_id]
            img = (tags[0] if tags else "").lower()
        except DockerException:
            img = ""
        engine = _engine_label(c)
        if not engine and "ghcr.io/ggml-org/llama.cpp" not in img and "llama.cpp" not in img:
            continue
        if c.name == "model-loader":
            continue
        engine = engine or "llama"
        vendor = "rocm" if "rocm" in img else "cuda" if "cuda" in img else ("cpu" if "server" in img else "unknown")
        # Router or single-model server? A router is given --models-preset/--models-dir and
        # serves whatever models.ini holds; a server given -m serves exactly one file and has
        # no relationship to models.ini at all. They are both llama.cpp:server images, so the
        # image tag cannot tell them apart - only the command line can. Orpheus TTS is the
        # case that forced the distinction: a -m server on port 5006 that this app would
        # otherwise offer to register with Open WebUI as a chat endpoint, at :8080.
        try:
            cmd = (c.attrs or {}).get("Config", {}).get("Cmd") or []
        except DockerException:
            cmd = []
        joined = " ".join(str(t) for t in cmd) if isinstance(cmd, list) else str(cmd)
        # Only a llama.cpp image can be a router. Another engine has no models.ini to serve
        # and no eviction to do, so it is a one-model server by construction - the same shape
        # as a `-m` llama server, which is exactly what `router = False` already means here.
        is_router = engine == "llama" and ("--models-preset" in joined or "--models-dir" in joined)
        out.append({"name": c.name, "image": img, "vendor": vendor,
                    "router": is_router, "engine": engine})
    return out


def engine_for(name: str) -> str:
    """The engine of one backend by container name; "llama" when nothing says otherwise.

    Defaulting to "llama" rather than "" is deliberate: every caller is llama.cpp machinery
    asking "is this mine?", and a container that has gone missing between the snapshot and
    the question should not silently acquire a new engine. A backend declares its way OUT of
    llama, never into it.
    """
    for d in discover_llama_containers():
        if d["name"] == name:
            return str(d.get("engine") or "llama")
    return "llama"


def _effective_container_names() -> list[str]:
    """Union of LLAMA_CONTAINERS env (retained even if not currently running) and any
    live-discovered llama.cpp containers on the docker socket. Order: env names first, then
    newly-discovered ones. Duplicates removed while preserving order."""
    whitelist = settings.llama_container_names
    discovered = [c["name"] for c in discover_llama_containers()]
    seen: set[str] = set()
    result: list[str] = []
    for n in list(whitelist) + discovered:
        if n and n not in seen:
            seen.add(n)
            result.append(n)
    return result


# What each backend reported as loaded on the last probe, so SYNCHRONOUS callers can ask.
# The router's /v1/models is the only authority on this and reaching it is async; every page
# that draws a VRAM meter also renders the hero, which probes, so this is refreshed in the
# same request rather than being a cache with a life of its own.
_LOADED_BY_BACKEND: dict[str, str] = {}
_LOADED_LOCK = threading.Lock()


# When each backend's current load was FIRST seen, keyed by (backend, model). The model is
# part of the key so that a swap straight into another swap restarts the clock instead of
# inheriting the previous model's start time. Pruned on every observation, so it holds at
# most one entry per backend and never needs a sweep.
_LOADING_SINCE: dict[tuple[str, str], float] = {}


def _note_loading(backend: str, model: str | None) -> float:
    """Seconds this backend has been loading `model`. 0.0 when it is not loading."""
    with _LOADED_LOCK:
        for k in [k for k in _LOADING_SINCE
                  if k[0] == backend and (model is None or k[1] != model)]:
            del _LOADING_SINCE[k]
        if not model:
            return 0.0
        t0 = _LOADING_SINCE.setdefault((backend, model), time.time())
    return max(0.0, time.time() - t0)


def last_loaded_ids(name: str) -> set[str]:
    """Model ids the last probe saw loaded on this backend. Empty when never probed."""
    with _LOADED_LOCK:
        csv = _LOADED_BY_BACKEND.get(name, "")
    return {p.strip() for p in csv.split(",") if p.strip()}


def _sleep_idle_seconds(attrs: dict) -> int:
    """The container's --sleep-idle-seconds, or 0 when absent or disabled.

    Read off the container's own command line rather than from a setting of ours, because the
    command line is what llama-server is actually running. llama.cpp spells "off" as -1 and as
    the flag simply not being there; both come back 0 here so callers have one thing to test.
    """
    cmd = (attrs.get("Config") or {}).get("Cmd") or []
    if not isinstance(cmd, list):
        return 0
    for i, tok in enumerate(cmd):
        if tok == "--sleep-idle-seconds" and i + 1 < len(cmd):
            try:
                v = int(str(cmd[i + 1]).strip())
            except (TypeError, ValueError):
                return 0
            return v if v > 0 else 0
        if isinstance(tok, str) and tok.startswith("--sleep-idle-seconds="):
            try:
                v = int(tok.split("=", 1)[1].strip())
            except (TypeError, ValueError):
                return 0
            return v if v > 0 else 0
    return 0


# llama-server brackets a sleep with exactly these two, and nothing else reports the state:
# /props and /v1/models read identically asleep or awake, so the log is the only signal.
_RE_SLEEP_MARK = re.compile(r"(entering|exiting) sleeping state")


def _is_asleep(container) -> bool:
    """True when the last sleep marker in the log says it went to sleep and never came back.

    Deliberately NOT probed over HTTP. Waking is what a request does - /slots alone both
    resets the idle timer and wakes a sleeper - so a liveness check would be the thing that
    prevents the sleep it is trying to observe.
    """
    try:
        raw = container.logs(tail=400, stdout=True, stderr=True)
    except Exception:  # noqa: BLE001 - a log we cannot read is not a sleeping model
        return False
    last = ""
    for line in raw.decode("utf-8", errors="replace").splitlines():
        m = _RE_SLEEP_MARK.search(line)
        if m:
            last = m.group(1)
    return last == "entering"


async def snapshot_llama_backends() -> list[LlamaBackend]:
    client = _docker_client()
    effective = _effective_container_names()
    if client is None:
        return [LlamaBackend(name=n, found=False, status="docker unreachable") for n in effective]

    out: list[LlamaBackend] = []
    probe_targets: list[tuple[int, str, int | None]] = []  # (idx, name, internal_port)
    _discovered = discover_llama_containers()
    _routers = {d["name"]: bool(d.get("router")) for d in _discovered}
    _vendors = {d["name"]: str(d.get("vendor") or "") for d in _discovered}
    _engines = {d["name"]: str(d.get("engine") or "llama") for d in _discovered}

    for i, name in enumerate(effective):
        b = LlamaBackend(name=name, found=False, status="not_found")
        with _restart_errors_lock:
            b.last_restart_error = _restart_errors.get(name)
        try:
            c = client.containers.get(name)
            b.found = True
            b.status = c.status
            b.image = (c.image.tags or [c.image.short_id])[0]
            b.short_id = c.short_id
            attrs = c.attrs or {}
            state = (attrs.get("State") or {})
            started, up = _parse_started_at(state.get("StartedAt", ""))
            b.started_at = started
            b.uptime = up
            b.host_ports, b.internal_port = _extract_ports(attrs)
            b.sleep_idle_s = _sleep_idle_seconds(attrs)
            b.router = _routers.get(name, True)
            b.vendor = _vendors.get(name, "")
            b.engine = _engines.get(name, "llama")
            if b.sleep_idle_s > 0 and b.status == "running":
                b.asleep = _is_asleep(c)
        except NotFound:
            pass
        except DockerException as e:
            b.status = f"error: {e}"
        out.append(b)
        if b.found and b.status == "running":
            probe_targets.append((i, name, b.internal_port))

    if probe_targets:
        results = await asyncio.gather(
            *[_probe_loaded_model(n, p) for _, n, p in probe_targets],
            return_exceptions=False,
        )
        for (i, _n, _p), (loaded, err, failed, loading, sleeping) in zip(probe_targets, results):
            out[i].loaded_model = loaded
            out[i].probe_error = err
            out[i].load_failed = failed
            out[i].loading_model = loading
            out[i].loading_since = _note_loading(out[i].name, loading)
            out[i].sleeping_model = sleeping
            # The router's own word beats the log scrape. _is_asleep reads the last sleep marker
            # out of `docker logs`, which is a guess that goes stale the moment the log rotates
            # or the marker scrolls past the tail; this is the server stating its current status.
            if sleeping:
                out[i].asleep = True
            with _LOADED_LOCK:
                _LOADED_BY_BACKEND[out[i].name] = loaded or ""
    return out


# ---------- engine switching ----------

# Two inference engines cannot share these cards. A llama.cpp router with --models-max 1 holds
# its model until something evicts it, and Strata sizes its expert cache against free VRAM at
# startup, so whichever starts second gets the scraps - or, as measured on this box, dies on
# "mtp: buffers do not fit (0 MiB of 32624 MiB VRAM free on this GPU)". Making them exclusive
# is the only arrangement where either one gets the hardware it was configured for.
#
# The switch stops every OTHER BACKEND that was given a GPU, plus any non-backend GPU tenant
# that is actually holding VRAM right now.
#
# The two halves have different tests for a reason. A backend is stopped even when it holds
# nothing, because holding nothing is its idle state and the next request would take the cards
# straight back. A tenant is stopped only when it is measurably in the way, because plenty of
# containers are handed /dev/kfd and allocate nothing - gpu-monitor is one, and the first
# version of this stopped it, SIGKILLed it when its shutdown ran past the grace period, and
# left the box without its monitoring sidecar to free zero bytes.

# What the current switch is doing, for the panel to render. One at a time by construction -
# the lock is held for the whole run, so a second request is rejected rather than interleaved.
_switch_lock = threading.Lock()
_switch_state: dict = {}


def _gpu_device_container(attrs: dict) -> bool:
    """Whether docker was told to give this container a GPU.

    Read from the device list rather than from what is currently on the cards, because a
    stopped container holds no VRAM and appears in no kernel table - and a stopped container
    is exactly what the switch has to reason about.

    /dev/kfd is the AMD compute device. DeviceRequests covers nvidia-container-toolkit, which
    is how this box was wired before the AMD swap and is cheap to keep honouring.
    """
    hc = (attrs or {}).get("HostConfig") or {}
    paths = {str(d.get("PathOnHost", "")) for d in (hc.get("Devices") or [])}
    if "/dev/kfd" in paths:
        return True
    return bool(hc.get("DeviceRequests"))


def gpu_holders() -> list[dict]:
    """Every container that was given a GPU: [{name, engine, backend, running, status}].

    `engine` is "" for a container that is not a backend at all (chatterbox, comfyui), which
    the switch still has to stop - they hold VRAM just the same.
    """
    client = _docker_client()
    if client is None:
        return []
    backends = {d["name"]: str(d.get("engine") or "llama") for d in discover_llama_containers()}
    out: list[dict] = []
    try:
        containers = client.containers.list(all=True)
    except DockerException:
        return []
    for c in containers:
        if c.name == "model-loader":
            continue
        try:
            attrs = c.attrs or {}
        except DockerException:
            continue
        if not _gpu_device_container(attrs):
            continue
        out.append({"name": c.name, "engine": backends.get(c.name, ""),
                    "backend": c.name in backends,
                    "running": c.status == "running", "status": c.status})
    out.sort(key=lambda r: r["name"])
    return out


def gpu_engines() -> list[dict]:
    """The engines that have at least one GPU backend: [{engine, names, running}].

    `running` means every one of that engine's GPU backends is up, so the switch can render a
    single active/inactive state per engine rather than per container. With one backend each -
    which is what this box has - the distinction never comes up, but a second GPU router would
    otherwise show as "active" while half of it was down.
    """
    by: dict[str, list[dict]] = {}
    for row in gpu_holders():
        if row["backend"] and row["engine"]:
            by.setdefault(row["engine"], []).append(row)
    return [{"engine": e, "names": [r["name"] for r in rows],
             "running": all(r["running"] for r in rows)}
            for e, rows in sorted(by.items())]


def engine_switch_state() -> dict:
    """A snapshot of the running (or last) switch, for the panel. {} when none has run.

    No lock: the worker only ever replaces whole values through dict.update, so a copy taken
    here is a consistent-enough view for a status line, and taking the switch lock to read it
    would block the panel for the whole sixty seconds the switch is allowed to run.
    """
    return dict(_switch_state)


def _vram_by_container() -> dict[str, float]:
    """{container name: GB of VRAM its processes hold right now}.

    From the kernel's own per-process accounting under /sys/class/kfd, which is world-readable
    from any container. This app has no /dev/kfd and cannot run rocm-smi, and the container it
    would otherwise exec into for that is often the one about to be stopped.
    """
    from . import gpu_procs
    out: dict[str, float] = {}
    try:
        for t in gpu_procs.tenants():
            if t.container:
                out[t.container] = out.get(t.container, 0.0) + t.total_gb
    except Exception:  # noqa: BLE001 - an unreadable topology must not wedge the switch
        return {}
    return out


def _vram_held_gb(exclude: set[str]) -> float:
    """GB of VRAM held by processes belonging to containers NOT in `exclude`.

    Read from the kernel's own per-process accounting under /sys/class/kfd, which is
    world-readable from any container - this app has no /dev/kfd and cannot run rocm-smi, and
    the container it would normally exec into for that is the one being stopped.
    """
    from . import gpu_procs
    try:
        return sum(t.total_gb for t in gpu_procs.tenants()
                   if (t.container or "") not in exclude and t.total_gb > 0.005)
    except Exception:  # noqa: BLE001 - an unreadable topology must not wedge the switch
        return 0.0


def _wait_ready(name: str, timeout: float = 420.0) -> tuple[bool, str]:
    """Poll one backend's /v1/models until it answers 200. (ready, note).

    `docker start` returning is not "serving", and the gap is not small: Strata reads ~55 GB of
    experts into RAM before its port opens - measured at 90 s here - and a llama router is 10-40 s
    depending on the model. Reporting the switch done at `docker start` meant clicking the button,
    seeing "done", finding nothing on the port and concluding it had not worked. That is exactly
    what happened.

    /v1/models rather than /health, because it is the one endpoint both engines answer the same
    way: llama.cpp's /health returns 503 "Loading model" while loading (and curl does not fail on
    a 503, so a naive poll measures a model mid-load), while Strata does not open the port at all
    until the model is in. A 200 from /v1/models means serving on both.
    """
    client = _docker_client()
    if client is None:
        return False, "docker unreachable"
    try:
        attrs = client.containers.get(name).attrs or {}
    except (NotFound, DockerException) as e:
        return False, f"{name}: {e}"
    _, port = _extract_ports(attrs)
    if port is None:
        return True, f"{name} exposes no API port — not waiting"
    url = f"http://{name}:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=httpx.Timeout(3.0)) as c:
                if c.get(url).status_code == 200:
                    return True, ""
        except httpx.HTTPError:
            pass
        _switch_state.update(phase="loading",
                             msg=f"{name} up {int(time.time() - _switch_state.get('t0', time.time()))}s, "
                                 "port not answering yet")
        time.sleep(2.0)
    return False, f"{name} started but {url} did not answer within {int(timeout)}s"


def activate_engine(engine: str) -> tuple[bool, str]:
    """Make `engine` the only thing holding the cards. Fire-and-forget; progress in _switch_state.

    Returns immediately because the whole sequence takes tens of seconds - stopping a router
    with 30 GB of weights, waiting for the driver to actually release it, then a start that
    reads ~55 GB of experts into RAM. A request that blocked on all that would time out in the
    browser long before it finished, and the GPU strip is already polling, so the panel shows
    the phases on its own.
    """
    engines = {e["engine"]: e for e in gpu_engines()}
    if engine not in engines:
        return False, f"no GPU backend of engine '{engine}'"
    if not _switch_lock.acquire(blocking=False):
        cur = _switch_state.get("engine") or "another engine"
        return False, f"already switching to {cur}"

    targets = set(engines[engine]["names"])
    live = _vram_by_container()
    others = []
    skipped = []
    for r in gpu_holders():
        if r["name"] in targets or not r["running"]:
            continue
        if r["backend"] or live.get(r["name"], 0.0) > 0.005:
            others.append(r)
        else:
            skipped.append(r["name"])
    _switch_state.clear()
    _switch_state.update(engine=engine, phase="stopping", msg="", t0=time.time(),
                         stopped=[r["name"] for r in others], skipped=skipped, started=[])

    def _run() -> None:
        try:
            from . import gpu_procs
            for r in others:
                ok, msg = gpu_procs.set_container_running(r["name"], start=False)
                if not ok:
                    _switch_state.update(phase="failed", msg=f"could not stop {r['name']}: {msg}")
                    return

            # Wait for the DRIVER to release it, not merely for the process to exit. Strata
            # sizes its expert cache against free VRAM at startup and llama.cpp's own fitter
            # does the same; starting the moment docker returns means reading a stale figure
            # and silently getting a fraction of the card. Measured on this box, the drop is
            # not instant after the process is gone.
            deadline = time.time() + 60
            while time.time() < deadline:
                held = _vram_held_gb(targets)
                if held < 0.5:
                    break
                _switch_state.update(phase="draining", msg=f"{held:.1f} GB still held")
                time.sleep(1.0)
            else:
                _switch_state.update(
                    phase="failed",
                    msg=f"{_vram_held_gb(targets):.1f} GB still on the cards after 60 s — "
                        "something outside docker is holding them")
                return

            _switch_state.update(phase="starting", msg="")
            for name in sorted(targets):
                ok, msg = gpu_procs.set_container_running(name, start=True)
                if not ok:
                    _switch_state.update(phase="failed", msg=f"could not start {name}: {msg}")
                    return
                _switch_state["started"].append(name)
            # Not done until it ANSWERS. See _wait_ready: `docker start` returning is 90 s short
            # of serving on Strata, and reporting done there is what made a working switch look
            # like a broken button.
            for name in sorted(targets):
                ready, note = _wait_ready(name)
                if not ready:
                    # "slow", not "failed": the container is up and may still come good. Saying
                    # it failed would send someone looking for a crash that has not happened.
                    _switch_state.update(phase="slow", msg=note)
                    return
            _switch_state.update(phase="done", msg="", t1=time.time())
        except Exception as e:  # noqa: BLE001 - a thread that dies silently leaves the panel lying
            _switch_state.update(phase="failed", msg=f"{type(e).__name__}: {e}")
        finally:
            _switch_lock.release()

    threading.Thread(target=_run, daemon=True).start()
    stopping = ", ".join(r["name"] for r in others) or "nothing"
    note = f" (left {', '.join(skipped)} alone — holding no VRAM)" if skipped else ""
    return True, (f"switching to {engine}: stopping {stopping}, then starting "
                  f"{', '.join(sorted(targets))}{note}")


def set_backend_running(name: str, start: bool) -> tuple[bool, str]:
    """Start or stop one backend. The control the Containers page was missing.

    Restart was the only lifecycle button a backend had, which is fine until two engines have
    to take turns on the same cards: freeing them meant an ssh session, and a control that
    needs a terminal is not a control. Stopping is also the ONLY way to make a llama router
    give up its model - at --models-max 1 it holds it until a request for another one arrives,
    and there is no unload endpoint on this build.
    """
    if name not in _effective_container_names():
        return False, f"{name} is not a configured backend"
    from . import gpu_procs
    return gpu_procs.set_container_running(name, start=start)


def restart_llama_backend(name: str) -> tuple[bool, str]:
    """Fire-and-forget restart. Errors captured in _restart_errors."""
    if name not in _effective_container_names():
        return False, "container not in configured list"
    client = _docker_client()
    if client is None:
        return False, "docker unreachable"
    try:
        c = client.containers.get(name)
    except NotFound:
        return False, "container not found"
    except DockerException as e:
        return False, f"docker error: {e}"

    with _restart_errors_lock:
        _restart_errors.pop(name, None)

    def _do() -> None:
        try:
            c.restart(timeout=15)
        except (DockerException, APIError) as e:
            with _restart_errors_lock:
                _restart_errors[name] = f"{type(e).__name__}: {e}"

    threading.Thread(target=_do, daemon=True, name=f"restart-{name}").start()
    return True, "restart queued"


def _fit_backends() -> dict[str, float]:
    """{backend_name: total VRAM GiB} for everything we can actually plan against.

    Prefers the live probe over settings.gpu_vram_map. The static map is an optional
    override and is empty by default — relying on it alone silently disabled the fit
    chips entirely once the hardcoded example values were removed.
    """
    from . import hw
    out: dict[str, float] = {}
    for name in _effective_container_names():
        vram = float(settings.gpu_vram_map.get(name, 0) or 0)
        if vram <= 0:
            st = hw.stats_for(name)
            if st.ok and st.gpu and st.gpu.vram_total_gb > 0:
                vram = float(st.gpu.vram_total_gb)
        if vram > 0:
            out[name] = vram
    return out


# Above this, a CPU-resident DENSE model is technically loadable and practically unusable.
# Calibrated on measured tok/s on a dual-channel DDR5 box: a 6.6 GB dense model managed
# 4.4 tok/s, implying roughly 29 GB/s of effective read bandwidth, so 30 GB of dense weights
# lands near 1 tok/s.
#
# KNOWN LIMITATION: this is size-based, and size is the wrong axis for MoE. A mixture-of-
# experts model reads only its active experts per token, so a 30 GB MoE can be several times
# faster than a 30 GB dense one — measured here, a 15.8 GB MoE beat a 6.6 GB dense by 2x.
# Telling them apart needs expert_count/expert_used_count from the GGUF header, which the
# chips do not have (they receive a file size and nothing else). The tooltip says so rather
# than pretending the number applies to both.
_CPU_CRAWL_GB = 30.0


def _cpu_backends() -> list[str]:
    """Names of discovered llama containers with no GPU — they run on the CPU.

    _fit_backends() keys on VRAM and so cannot see these at all. They are still real places
    to run a model: the constraint is system RAM rather than VRAM, and ctx-size is the only
    fit lever, since ngl, n-cpu-moe and tensor-split all presuppose a GPU.
    """
    names = set(_effective_container_names())
    out: list[str] = []
    for d in discover_llama_containers():
        n = d.get("name") or ""
        if n in names and (d.get("vendor") or "").lower() in ("cpu", "", "unknown"):
            from . import hw
            st = hw.stats_for(n)
            if not (st.ok and st.gpu and st.gpu.vram_total_gb > 0):
                out.append(n)
    return out


def vram_fit_chips(size_bytes: int) -> list[dict]:
    """Per-backend fit verdicts: {'name', 'vram_gb', 'verdict', 'ratio_pct'}.

    verdict is one of {fits, tight, oom, impossible}.

    `oom` and `impossible` are genuinely different answers and must not look alike. `oom`
    means "not on the GPU alone" — offload moves layers into system RAM and it runs, slower.
    `impossible` means the model exceeds VRAM **plus** RAM, so there is nowhere for those
    layers to go and no setting rescues it. Rendering both as a red cross against a backend
    name tells the user the GPU is the constraint, when for `impossible` the machine is.

    An impossible model returns a SINGLE machine-level chip rather than one per backend,
    because naming backends implies picking a different one would help.
    """
    from . import hw
    gb = size_bytes / (1024 ** 3)

    # usable, not total: the OS reserve is already spoken for (see hw.usable_ram_gb).
    ram = hw.usable_ram_gb()
    pooled = max((v for v in _fit_backends().values() if v > 0), default=0.0)
    if ram > 0 and pooled > 0 and gb > (pooled + ram):
        return [{
            "name": "this machine",
            "vram_gb": round(pooled, 1),
            "host_ram_gb": round(ram, 1),
            "verdict": "impossible",
            "ratio_pct": round(gb / (pooled + ram) * 100),
            "needs_gb": round(gb),
            "ceiling_gb": round(pooled + ram),
        }]

    out: list[dict] = []
    for name, vram in _fit_backends().items():
        if vram <= 0:
            continue
        ratio = gb / vram
        if ratio < 0.65:
            verdict = "fits"
        elif ratio < 0.9:
            verdict = "tight"
        else:
            verdict = "oom"
        out.append({
            "name": name,
            "vram_gb": vram,
            "verdict": verdict,
            "ratio_pct": round(ratio * 100),
        })

    # CPU backends are sized against usable RAM. "Fits" and "usable" diverge sharply here:
    # generation is RAM-bandwidth-bound, so a large dense model can occupy memory perfectly
    # well and still produce well under a token per second. A plain green tick would be
    # true and misleading, so anything past a threshold gets its own 'slow' verdict.
    for name in _cpu_backends():
        if ram <= 0:
            continue
        ratio = gb / ram
        if ratio >= 1.0:
            verdict = "oom"
        elif gb >= _CPU_CRAWL_GB:
            verdict = "slow"
        elif ratio < 0.75:
            verdict = "fits"
        else:
            verdict = "tight"
        out.append({
            "name": name,
            "vram_gb": ram,
            "verdict": verdict,
            "ratio_pct": round(ratio * 100),
            "is_cpu": True,
        })
    return out


@dataclass
class SlotSpeed:
    """One slot's own figures. There is no fixed number of these - see slot_density()."""
    index: int = 0
    state: str = "idle"        # "generating" | "prefill" | "idle"
    gen_tps: float = 0.0       # this slot's own rate, not a share of the total
    decoded: int = 0           # tokens this slot has produced for its current task
    ctx_used: int = 0          # tokens held by this slot
    ctx_total: int = 0         # this slot's share of the KV pool (ctx-size / parallel)
    ctx_cached: int = 0        # of ctx_used, how many came from cache
    prefill_pct: int = 0       # progress through this slot's prompt, 0-100
    # The numerator and denominator that percentage came from. A bare percentage cannot tell a
    # 400-token prompt from an 80,000-token one, and those take 0.3 s and 70 s respectively on
    # this box - the difference between "it is about to answer" and "go and get a coffee".
    prefill_done: int = 0      # prompt tokens read so far, cache hits included
    prefill_total: int = 0     # the whole prompt this slot is working through
    # The prompt this slot's CURRENT task actually arrived with: cache hits plus processed.
    # Deliberately not n_prompt_tokens, which keeps growing while the model generates - it is
    # the slot's whole context, not the prompt. Dividing the (frozen) cache count by the
    # (growing) context made a perfectly cached conversation appear to shed cache as it
    # answered: measured on one task, n_prompt_tokens ran 47,626 -> 51,042 over 90 s while
    # n_prompt_tokens_processed sat still at 46,844.
    prompt_tokens: int = 0
    task_id: int | None = None

    @property
    def ctx_pct(self) -> int:
        """How full this slot is, for the strip's bar. 0 when the budget is unknown."""
        if self.ctx_total <= 0:
            return 0
        return max(0, min(100, int(round(100.0 * self.ctx_used / self.ctx_total))))


@dataclass
class InferenceSpeed:
    """How fast the loaded model is working right now, for the card's speedometer.

    The aggregate fields describe the MACHINE: gen_tps is the sum over generating slots, and
    ctx_used/ctx_total describe the whole KV pool. With parallel = 1, which is most sections,
    that is identical to the single slot's own figures. With more, reporting one slot's rate as
    though it were the machine's was simply wrong - measured on a parallel = 3 model, two slots
    generated at 38.5 and 27.4 tok/s simultaneously (a persistent 40% gap, because speculative
    decoding accepts more drafts on code than on prose), so no single slot's number described
    either the conversation you were watching or the box. Per-slot detail lives in `slots`.
    """
    model: str = ""
    state: str = "idle"        # "generating" | "prefill" | "idle" - busiest slot wins
    gen_tps: float = 0.0       # tokens/s out, summed over generating slots
    gen_tokens: int = 0        # tokens produced in the run these rates came from
    prefill_tps: float = 0.0   # tokens/s in
    prefill_pct: int = 0       # progress through the current prompt, 0-100
    prompt_tokens: int = 0     # the prompts the busy slots arrived with, cache + processed.
                               # The honest denominator for "how much came from cache".
    prefill_done: int = 0      # prompt tokens read so far, summed over prefilling slots
    prefill_total: int = 0     # the whole prompt, summed the same way. 0 when only the log
                               # knows, because the log's total is progress-derived guesswork
    ctx_used: int = 0          # tokens held across all slots
    ctx_total: int = 0         # the whole KV pool: per-slot ctx x slot count
    ctx_cached: int = 0        # of ctx_used, how many came from cache instead of being re-read
    live: bool = False         # rates describe work happening NOW, not the last run
    slots: list = field(default_factory=list)   # list[SlotSpeed], one per live slot


def slot_density(n: int) -> str:
    """How much room each slot row gets, chosen from the slot count alone.

    The strip has to survive `parallel = 8` without pushing the hero off the page, and a fixed
    row height cannot do that: 8 comfortable rows measured 128px, far more than the 24px
    sparkline they sit beside. So rows shrink and labels abbreviate as the count grows, and
    slot_columns() wraps them - every slot stays visible at any count, nothing truncates or
    scrolls. 8 slots land at 4 rows of 16px, which is about the height of one readout.
    """
    if n <= 4:
        return "roomy"      # ~22px rows, "433 / 128,768"
    if n <= 8:
        return "tight"      # ~16px rows, "34%"
    return "dense"          # ~13px rows, "34%"


def slot_columns(n: int) -> int:
    """How many columns to wrap the slot strip into, so its HEIGHT stays roughly flat.

    Without this the strip grows linearly and a future parallel = 16 would be 208px of hero.
    Wrapping keeps it between 4 and 6 rows tall at every count a homelab will realistically
    use: 8 slots become 2 x 4, 16 become 3 x 6.
    """
    if n <= 4:
        return 1
    if n <= 12:
        return 2
    return 3


# llama-server prints these at its default log level, every ~3 s while a slot works. There is no
# metrics endpoint to ask instead: /metrics needs a flag this deployment does not pass, and
# /slots carries state but no counters or rates. So the numbers come from the log, and the
# state comes from /slots - each from the source that actually has it.
#
#   ... n_gen =    806, tg =  26.02 t/s, tg_3s =  25.99 t/s
#   ... prompt processing, n_tokens =   7105, progress = 1.00, t =   4.70 s / 1511.80 tokens per second
_RE_TG = re.compile(r"n_gen\s*=\s*(\d+),\s*tg\s*=\s*([\d.]+) t/s"
                    r"(?:,\s*tg_3s\s*=\s*([\d.]+))?")
_RE_PP = re.compile(r"prompt processing, n_tokens\s*=\s*(\d+), progress\s*=\s*([\d.]+), "
                    r"t\s*=\s*[\d.]+ s / ([\d.]+) tokens per second")
_RE_TASK = re.compile(r"task\s+(\d+)\s*\|")
# Slot occupancy, straight from the log. A task is in flight between its launch line and its
# release line, which is all that is needed to decide whether anything is running:
#   ... I slot launch_slot_: id  2 | task 11269 | processing task, is_child = 0
#   ... I slot      release: id  2 | task 11269 | stop processing: n_tokens = 83416, truncated = 0
_RE_LAUNCH = re.compile(r"launch_slot_:\s*id\s+(\d+)\s*\|\s*task\s+(\d+)")
_RE_RELEASE = re.compile(r"release:\s*id\s+(\d+)\s*\|\s*task\s+(\d+)")

# One read per container per interval, shared by every caller in that window. The card polls
# every 2 s and reading a log tail is the expensive half, so this keeps a burst of renders from
# multiplying docker calls.
# Two different clocks. The slot read is ~1 ms so it happens on every poll; the log read is
# ~11 ms and its numbers only change every few seconds anyway, so it is throttled. An earlier
# version cached the whole result for 1.8 s, which meant polling faster just redrew stale text.
_SPEED_TTL = 0.35          # floor on full recomputation, to survive a burst of requests
_RATES_TTL = 1.5           # how often the log tail is re-read
_speed_cache: dict[str, tuple[float, "InferenceSpeed | None"]] = {}
_rates_cache: dict[str, tuple[float, dict]] = {}
_GEN_EMA_ALPHA = 0.4       # smoothing for the token-delta rate; raw 500 ms deltas twitch
# (when, token count, task) of the previous sample, so a rate can be derived from how fast the
# slot's token count is moving. llama only logs a generation timing line every ~3 s, which left
# the first seconds of a reply looking like it was still reading the prompt.
# Keyed "container#slot", never by container alone, so the slot count is never baked in: a
# section can go from parallel = 1 to 8 and back and these just grow and shrink with it.
# _prune_slot_state drops keys for slots that no longer exist, so a step down leaves nothing
# behind to be read as a live slot later.
_speed_prev: dict[str, tuple[float, int, int | None, bool]] = {}
_gen_ema: dict[str, float] = {}      # smoothed token-delta rate, per slot


def _slot_key(container_name: str, index: int) -> str:
    return f"{container_name}#{index}"


def _slot_speed(container_name: str, s: dict, r: dict, now: float) -> "SlotSpeed":
    """One slot's figures, and its own EMA state. Called once per slot per sample.

    Each slot keeps a separate smoother because their rates genuinely differ: measured on a
    parallel = 3 section, two slots decoding 400 tokens each in the same batch held 38.5 and
    27.4 tok/s for seven seconds straight. Speculative decoding is why - the draft head is
    accepted far more often on code than on prose - so one shared EMA would blur a real signal.
    """
    index = int(s.get("id") or 0)
    key = _slot_key(container_name, index)
    ctx_used = int(s.get("n_prompt_tokens") or 0)
    ctx_total = int(s.get("n_ctx") or 0)
    cached_tok = int(s.get("n_prompt_tokens_cache") or 0)
    processed = int(s.get("n_prompt_tokens_processed") or 0)
    task_id = s.get("id_task")
    task_id = int(task_id) if isinstance(task_id, (int, float)) else None
    nt = s.get("next_token")
    nt0 = nt[0] if isinstance(nt, list) and nt and isinstance(nt[0], dict) else {}
    decoded = int(nt0.get("n_decoded") or 0)

    if not s.get("is_processing"):
        # A finished run must not smooth into the next one on this slot.
        _speed_prev.pop(key, None)
        _gen_ema.pop(key, None)
        return SlotSpeed(index=index, state="idle", ctx_used=ctx_used,
                         ctx_total=ctx_total, ctx_cached=cached_tok)

    # One decoded token is the switch out of prefill. See the n_prompt_tokens note below.
    state = "generating" if decoded else "prefill"
    prefill_pct = 0
    prefill_done = prefill_total = 0
    if state == "prefill" and ctx_used:
        # Cache hits count as read: they are tokens the slot no longer has to process, and
        # excluding them would make a 95%-cached prompt crawl from 0 while finishing instantly.
        prefill_done, prefill_total = cached_tok + processed, ctx_used
        prefill_pct = min(100, int(round(100.0 * prefill_done / prefill_total)))

    gen_tps = 0.0
    prev = _speed_prev.get(key)
    if state == "generating" and prev and prev[2] == task_id and prev[3]:
        dt, dn = now - prev[0], decoded - prev[1]
        if 0.25 < dt < 10.0 and dn > 0:
            raw = dn / dt
            last = _gen_ema.get(key)
            gen_tps = (raw if last is None
                       else _GEN_EMA_ALPHA * raw + (1 - _GEN_EMA_ALPHA) * last)
            _gen_ema[key] = gen_tps
    elif state == "generating" and r.get("open_task") == task_id:
        # First generating sample for this task, so there is no delta yet. The log's figure
        # describes only the newest open task, so at most one slot can borrow it.
        gen_tps = r["cur_gen_tps"]
    _speed_prev[key] = (now, decoded, task_id, state == "generating")

    return SlotSpeed(index=index, state=state, gen_tps=gen_tps, decoded=decoded,
                     ctx_used=ctx_used, ctx_total=ctx_total, ctx_cached=cached_tok,
                     prefill_pct=prefill_pct, prefill_done=prefill_done,
                     prefill_total=prefill_total, prompt_tokens=cached_tok + processed,
                     task_id=task_id)


def _prune_slot_state(container_name: str, live: set[int]) -> None:
    """Forget per-slot state for slots this backend no longer has."""
    prefix = f"{container_name}#"
    keep = {_slot_key(container_name, i) for i in live}
    for store in (_speed_prev, _gen_ema):
        for k in [k for k in store if k.startswith(prefix) and k not in keep]:
            del store[k]

# /slots is the ONLY llama endpoint this app must not poll while a model is idle. It is served as
# a task on the server's own queue, which is the same queue the idle detector watches, so polling
# it both resets `--sleep-idle-seconds` and wakes an already-sleeping model. Measured on an
# R9700 against a 20 s threshold: polling /slots every 500 ms held VRAM at 25.63 GiB for 45 s and
# it never slept, and polling it once asleep took VRAM straight back from 22.51 to 25.61 GiB.
# /health and /props both slept normally under the same polling, and /v1/models is router state
# that is never proxied to the child, so those three are safe at any rate.
#
# So occupancy comes from the log instead, and /slots is touched only while a task is genuinely
# in flight - at which point the model was never going to sleep anyway, so the poll is free.
_PROPS_TTL = 60.0                    # n_ctx and slot count only change when a model reloads
# Both of these are keyed by "container/model", not container alone: the router swaps models
# under one container, and a 128k context read from the model before the swap must not be shown
# against the one after it.
_props_cache: dict[str, tuple[float, dict]] = {}
# Last per-slot context figures seen while busy, so the strip survives going idle without a
# /slots read. One (ctx_used, ctx_total) per slot, in index order; the length IS the slot count.
_last_slots: dict[str, list[tuple[int, int]]] = {}
# A task the log still calls open but /slots has reported finished. Remembering it stops a
# missing release line - a crash, or a tail that scrolled - from polling /slots forever.
_settled_task: dict[str, int] = {}

# Rolling generation-speed history for the dashboard sparkline. 240 samples at the hero's 500 ms
# poll is about two minutes, which is long enough to see a request start, run and finish.
_TPS_HISTORY_MAX = 240
_TPS_MIN_GAP_S = 0.25                # two tabs polling at once must not double-sample
_tps_history: dict[str, list[tuple[float, float]]] = {}


def tps_history(container_name: str) -> list[float]:
    """Recent generation speeds for a backend, oldest first. Empty until something has run."""
    return [v for _ts, v in _tps_history.get(container_name, [])]


def _record_tps(container_name: str, value: float) -> None:
    hist = _tps_history.setdefault(container_name, [])
    now = time.time()
    if hist and (now - hist[-1][0]) < _TPS_MIN_GAP_S:
        return
    hist.append((now, round(float(value or 0.0), 2)))
    if len(hist) > _TPS_HISTORY_MAX:
        del hist[:len(hist) - _TPS_HISTORY_MAX]


async def _slot_states(container_name: str, internal_port: int, model_id: str) -> list[dict]:
    """Every slot the loaded model has, in index order, or [] if they cannot be read.

    All of them, not just the working one: one response already carries the lot, so reporting
    per-slot detail costs no extra HTTP. It used to return slots[0] unconditionally, which
    reported an idle slot for every request that landed anywhere else - measured at 38% of tasks
    (5,304 of 13,922) on Qwen3.8-27B-UD-Q4_K_M, each one a gap in the speedometer and a zero in
    the history. The caller no longer has to pick one at all.
    """
    url = f"http://{container_name}:{internal_port}/slots"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(1.5, read=2.0)) as client:
            r = await client.get(url, params={"model": model_id})
        if r.status_code != 200:
            return []
        slots = r.json()
        if not isinstance(slots, list):
            return []
        out = [s for s in slots if isinstance(s, dict)]
        # Index order, because the strip renders in this order and llama is not required to
        # return them sorted. Slots without an id sort last rather than crashing the sort.
        out.sort(key=lambda s: s.get("id") if isinstance(s.get("id"), int) else 1 << 30)
        return out
    except (httpx.HTTPError, ValueError, KeyError, IndexError):
        return []


async def _props_state(container_name: str, internal_port: int, model_id: str) -> dict:
    """Static per-load facts: the real n_ctx, the slot count, and whether the model is asleep.

    Safe to poll at any rate - see the note by _PROPS_TTL. It is also the right source for the
    context a single conversation actually gets, which is NOT what /v1/models reports. The argv
    there carries `--ctx-size`, the total KV pool; llama-server divides that pool by `parallel`
    and rounds up to a multiple of 128 to get the per-slot context, which is what `n_ctx` means
    here and what the slot objects report. On Qwen3.8-27B-UD-Q4_K_M, ctx-size 386000 with
    parallel 3 gives 128768 (386000/3 = 128666.7, rounded up), and the load log says so outright:
    "n_slots = 3, n_ctx_slot = 128768". Both numbers are real, they answer different questions,
    so the panel compares per-slot usage against the per-slot total.
    """
    key = f"{container_name}/{model_id}"
    cached = _props_cache.get(key)
    if cached and (time.time() - cached[0]) < _PROPS_TTL:
        return cached[1]
    out: dict = {}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(1.5, read=2.0)) as client:
            r = await client.get(f"http://{container_name}:{internal_port}/props",
                                 params={"model": model_id})
        if r.status_code == 200:
            d = r.json()
            gen = d.get("default_generation_settings") or {}
            out = {"n_ctx": int(gen.get("n_ctx") or 0),
                   "total_slots": int(d.get("total_slots") or 0),
                   "is_sleeping": bool(d.get("is_sleeping"))}
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        out = {}
    _props_cache[key] = (time.time(), out)
    return out


def _rates_from_log(name: str) -> dict:
    """Occupancy and throughput from the current run's log tail.

    Two things come out of one read. `open_task` is the newest task that has a launch line and no
    matching release, which is how this app knows whether anything is running WITHOUT asking
    /slots and waking the model. And the rates, split by WHICH request they describe: every timing
    line names its task, so tying them to the open task is the difference between "reading the
    prompt at 1,985 tok/s" and a stale number from the request before it. An earlier version
    ignored the task id and reported the previous run's rates - and its finished 100% progress
    bar - as if they were current.

    Returns rates for the live task (cur_*) and, separately, the most recent of any task (last_*),
    which is what the card shows, dimmed, while the model sits idle.
    """
    out = {"cur_gen_tps": 0.0, "cur_gen_tokens": 0, "cur_pp_tps": 0.0, "cur_pp_pct": 0,
           "cur_pp_done": 0,
           "last_gen_tps": 0.0, "last_gen_tokens": 0, "last_pp_tps": 0.0, "newest": "",
           "open_task": None, "open_slot": None}
    # A wide tail, then the router's own chatter removed. It has to be both. While a slot
    # generates it logs a timing line only every ~3 s, so a long reply buries the
    # prompt-processing lines from the start of that same request; and in router mode llama.cpp
    # logs EVERY proxied request, including this app's own 500 ms /slots poll, at about two lines
    # a second, forever. At tail=300 that spam was 92% of the window (276 of 300 lines, measured
    # on llama-9700x2), leaving two generation timing lines and no prompt-processing lines at
    # all, so every rate here came back zero while the model was plainly working. The proxy lines
    # carry nothing this function wants, so they go before the scan rather than inside it.
    ok, text = container_logs(name, tail=1200, current_run_only=True)
    if not ok:
        return out
    lines = [ln for ln in text.splitlines() if "proxy_reques" not in ln]

    # Oldest first, pairing launches with releases per slot. Whatever is still unpaired at the end
    # is in flight. Sound within one window: a release is always newer than its own launch, so a
    # launch that is in the window can only be unpaired because the task really is still running.
    # Release lines with no launch in the window are from before it and are simply ignored.
    open_tasks: dict[int, int] = {}
    for line in lines:
        m = _RE_LAUNCH.search(line)
        if m:
            open_tasks[int(m.group(1))] = int(m.group(2))
            continue
        m = _RE_RELEASE.search(line)
        if m:
            slot_id, tid = int(m.group(1)), int(m.group(2))
            if open_tasks.get(slot_id) == tid:
                del open_tasks[slot_id]
    if open_tasks:
        # Several slots can run at once; the newest task is the one the panel follows.
        slot_id = max(open_tasks, key=lambda s: open_tasks[s])
        out["open_slot"], out["open_task"] = slot_id, open_tasks[slot_id]

    task_id = out["open_task"]
    for line in reversed(lines):
        m_task = _RE_TASK.search(line)
        if not m_task:
            continue
        mine = task_id is not None and int(m_task.group(1)) == task_id
        m = _RE_TG.search(line)
        if m:
            tps, toks = float(m.group(3) or m.group(2)), int(m.group(1))
            if not out["last_gen_tps"]:
                out["last_gen_tps"], out["last_gen_tokens"] = tps, toks
            if mine and not out["cur_gen_tps"]:
                out["cur_gen_tps"], out["cur_gen_tokens"] = tps, toks
                out["newest"] = out["newest"] or "gen"
            continue
        m = _RE_PP.search(line)
        if m:
            tps, pct = float(m.group(3)), int(round(float(m.group(2)) * 100))
            # n_tokens is CUMULATIVE, not the size of this chunk. Verified against a live task:
            #   n_tokens = 18665, progress = 0.23   ...   n_tokens = 57157, progress = 0.69
            # which implies the same ~82k prompt at both ends. Only the running count is taken
            # from here; the total is NOT, because progress is printed to two decimals and
            # dividing by it gives +/-6% at the start of a prompt. An exact total comes from
            # /slots or it does not come at all.
            done = int(m.group(1))
            if not out["last_pp_tps"]:
                out["last_pp_tps"] = tps
            if mine and not out["cur_pp_tps"]:
                out["cur_pp_tps"], out["cur_pp_pct"] = tps, pct
                out["cur_pp_done"] = done
                out["newest"] = out["newest"] or "prefill"
    return out


async def inference_speed(container_name: str, internal_port: int | None,
                          loaded_model: str | None) -> "InferenceSpeed | None":
    """Live throughput for a backend, or None when there is nothing to show.

    Cheap parts first: no loaded model means no speedometer, and that is the common case on an
    idle box. Both reads are skipped entirely in that state.
    """
    if not loaded_model or internal_port is None:
        return None
    cached = _speed_cache.get(container_name)
    if cached and (time.time() - cached[0]) < _SPEED_TTL:
        return cached[1]

    model_id = loaded_model.split(",")[0].strip()

    # The log read comes FIRST now, because it is what decides whether /slots may be touched.
    cached_rates = _rates_cache.get(container_name)
    if cached_rates and (time.time() - cached_rates[0]) < _RATES_TTL:
        r = cached_rates[1]
    else:
        r = await asyncio.to_thread(_rates_from_log, container_name)
        _rates_cache[container_name] = (time.time(), r)

    # Only ask /slots when a task is actually in flight. An idle model must be left alone or it
    # can never sleep, and asking would wake it outright.
    raw_slots: list[dict] = []
    open_task = r["open_task"]
    if open_task is None:
        # Nothing in flight. Drop any settled-task marker here rather than letting it persist:
        # task ids restart from 1 when the router reloads a child, so a marker left over from a
        # previous life could collide with a genuine new task and silence one request's readout.
        _settled_task.pop(container_name, None)
    elif _settled_task.get(container_name) != open_task:
        raw_slots = await _slot_states(container_name, internal_port, model_id)
        if not any(s.get("is_processing") for s in raw_slots):
            # /slots is authoritative. Record the disagreement so a release line that never
            # arrived does not keep this backend awake for the rest of the process's life.
            _settled_task[container_name] = open_task
            raw_slots = []

    now = time.time()
    ctx_key = f"{container_name}/{model_id}"
    slots: list[SlotSpeed] = []

    if raw_slots:
        _prune_slot_state(container_name, {int(s.get("id") or 0) for s in raw_slots})
        slots = [_slot_speed(container_name, s, r, now) for s in raw_slots]
        _last_slots[ctx_key] = [(sl.ctx_used, sl.ctx_total) for sl in slots]
    else:
        # Idle, and /slots was deliberately not read. Rebuild the strip from the last busy
        # sample - that KV is still resident, so the figures are real - or from /props when
        # there is no history yet. /props carries total_slots, so even a freshly started process
        # renders the right NUMBER of idle slots without touching /slots.
        remembered = _last_slots.get(ctx_key)
        if remembered:
            slots = [SlotSpeed(index=i, state="idle", ctx_used=u, ctx_total=t)
                     for i, (u, t) in enumerate(remembered)]
        else:
            p = await _props_state(container_name, internal_port, model_id)
            per_slot = int(p.get("n_ctx") or 0)
            count = max(1, int(p.get("total_slots") or 1))
            slots = [SlotSpeed(index=i, state="idle", ctx_total=per_slot)
                     for i in range(count)]
        _prune_slot_state(container_name, set())

    generating = [sl for sl in slots if sl.state == "generating"]
    prefilling = [sl for sl in slots if sl.state == "prefill"]
    busy = bool(generating or prefilling)
    state = "generating" if generating else ("prefill" if prefilling else "idle")

    # The machine's rate, not one slot's. Summing is the honest aggregate: slots decode in the
    # same batch, so their rates add up to what the box is actually producing.
    gen_tps = sum(sl.gen_tps for sl in generating)
    gen_tokens = sum(sl.decoded for sl in generating)
    if state == "generating" and not gen_tps:
        # First frame of a run, before any slot has two samples to difference.
        gen_tps, gen_tokens = r["cur_gen_tps"], gen_tokens or r["cur_gen_tokens"]
    if state == "idle":
        gen_tps, gen_tokens = r["last_gen_tps"], r["last_gen_tokens"]
        prefill_tps, prefill_pct = r["last_pp_tps"], 0
    else:
        prefill_tps, prefill_pct = r["cur_pp_tps"], r["cur_pp_pct"]
        if not prefill_pct and prefilling:
            prefill_pct = max(sl.prefill_pct for sl in prefilling)

    # Summed over prefilling slots for the same reason gen_tps is summed: the figure describes
    # the machine. Both are 0 off the prefill path, since `prefilling` is empty then.
    # Busy slots only. An idle slot still reports the context it is holding, and folding that
    # into the cache ratio mixes one conversation's cache with another conversation's size.
    prompt_tokens = sum(sl.prompt_tokens for sl in slots
                        if sl.state in ("prefill", "generating"))
    prefill_done = sum(sl.prefill_done for sl in prefilling)
    prefill_total = sum(sl.prefill_total for sl in prefilling)
    if not prefill_done and state == "prefill":
        # /slots had nothing - the log's running count, with no total to go with it.
        prefill_done = r["cur_pp_done"]

    # Pool-wide context: every slot's share added up, against the whole --ctx-size pool. With
    # parallel = 1 this is just the one slot, unchanged.
    ctx_used = sum(sl.ctx_used for sl in slots)
    ctx_cached = sum(sl.ctx_cached for sl in slots)
    per_slot_ctx = max((sl.ctx_total for sl in slots), default=0)
    ctx_total = per_slot_ctx * len(slots)

    # Idle samples are recorded as zero rather than skipped: the gaps between requests are part
    # of the shape, and a line drawn only from busy moments would imply continuous work.
    _record_tps(container_name, gen_tps if state == "generating" else 0.0)

    out = InferenceSpeed(
        model=model_id, state=state, gen_tps=gen_tps, gen_tokens=gen_tokens,
        prefill_tps=prefill_tps, prefill_pct=prefill_pct,
        prefill_done=prefill_done, prefill_total=prefill_total,
        prompt_tokens=prompt_tokens,
        ctx_used=ctx_used, ctx_total=ctx_total, ctx_cached=ctx_cached,
        live=busy, slots=slots,
    )
    _speed_cache[container_name] = (time.time(), out)
    return out


async def test_prompt(container_name: str, prompt: str, max_tokens: int = 256) -> dict:
    """Send a short chat completion to a container's llama-server. Returns dict with reply, tokens, elapsed_s, err."""
    client = _docker_client()
    internal_port: int | None = None
    if client is not None:
        try:
            c = client.containers.get(container_name)
            _, internal_port = _extract_ports(c.attrs or {})
        except (NotFound, DockerException):
            pass
    if internal_port is None:
        return {"ok": False, "err": "container not reachable"}

    # discover a loaded model id first
    loaded, err, _failed, loading, sleeping = await _probe_loaded_model(
        container_name, internal_port)
    if not loaded:
        if loading:
            return {"ok": False, "err": f"{loading} is still loading - try again in a moment"}
        if sleeping:
            # Deliberately not woken here: a test prompt is a diagnostic, and silently paying a
            # 9 s reload to answer one would hide the very state the caller wants to know about.
            loaded = sleeping
        return {"ok": False, "err": err or "no model loaded on this backend"}
    model_id = loaded.split(",")[0].strip()

    url = f"http://{container_name}:{internal_port}/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    import time as _time
    t0 = _time.time()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=180.0)) as ac:
            r = await ac.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        return {"ok": False, "err": f"{type(e).__name__}: {e}", "model": model_id}
    elapsed = _time.time() - t0

    reply = ""
    choices = data.get("choices") or []
    if choices:
        msg = (choices[0] or {}).get("message") or {}
        reply = msg.get("content") or ""
    usage = data.get("usage") or {}
    completion_toks = int(usage.get("completion_tokens") or 0)
    tps = round(completion_toks / elapsed, 1) if elapsed > 0 and completion_toks else None
    return {
        "ok": True,
        "model": model_id,
        "reply": reply,
        "completion_tokens": completion_toks,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "elapsed_s": round(elapsed, 2),
        "tokens_per_s": tps,
    }


_OWUI_DB = "/app/backend/data/webui.db"


def _has_webui_db(c) -> bool:
    """Definitive test: does this container actually hold OpenWebUI's database?"""
    try:
        r = c.exec_run(["test", "-f", _OWUI_DB])
        return r.exit_code == 0
    except (DockerException, APIError):
        return False


def _find_open_webui(client):
    """Locate the OpenWebUI container.

    Matching on `"open-webui" in <full image ref>` is too loose: an unrelated image from the
    same GitHub org — e.g. ghcr.io/open-webui/open-terminal — contains that string in its ORG
    path and would match first, after which every DB call fails with
    "unable to open database file" because that container has no webui.db.

    So: compare against the image's repository basename (not the org), prefer an exact
    container-name match, and confirm the database is actually present before accepting.
    """
    def _repo_basename(c) -> str:
        try:
            ref = ((c.image.tags or [""]) + [""])[0].lower()
        except DockerException:
            return ""
        ref = ref.split("@", 1)[0]                 # strip digest
        path = ref.rsplit(":", 1)[0]              # strip tag
        return path.rsplit("/", 1)[-1]            # image name only, no org/registry

    try:
        containers = list(client.containers.list(all=True))
    except DockerException:
        return None

    exact_name = [c for c in containers if (c.name or "").lower() in ("open-webui", "openwebui")]
    by_image = [c for c in containers if _repo_basename(c) in ("open-webui", "openwebui")]

    # Running instances first, then anything else; verify the DB before committing.
    for group in (exact_name, by_image):
        for c in sorted(group, key=lambda x: 0 if x.status == "running" else 1):
            if _has_webui_db(c):
                return c
    # Nothing verified — fall back to the best name/image guess so callers can still report
    # a sensible container name in their error message.
    return (exact_name or by_image or [None])[0]


def _openwebui_persisted_urls(ow) -> list[str]:
    """Read the actual URLs OpenWebUI is using — from its sqlite config, not the env var.
    Env is only a first-startup seed; persistent config in webui.db is the source of truth."""
    script = (
        "import sqlite3, json;"
        "c = sqlite3.connect('/app/backend/data/webui.db');"
        "r = c.execute(\"SELECT value FROM config WHERE key='openai.api_base_urls'\").fetchone();"
        "print(r[0] if r else '[]')"
    )
    try:
        r = ow.exec_run(["python3", "-c", script])
    except DockerException:
        return []
    if r.exit_code != 0:
        return []
    try:
        import json as _json
        return _json.loads(r.output.decode(errors="replace").strip() or "[]")
    except (ValueError, UnicodeDecodeError):
        return []


def _compose_service_of(client, container_name: str) -> str:
    """Return the compose service name of a container (from labels), or empty."""
    try:
        c = client.containers.get(container_name)
        return c.labels.get("com.docker.compose.service", "") or ""
    except (DockerException, NotFound):
        return ""


def openwebui_state() -> dict:
    """Report the URLs OpenWebUI currently has PERSISTED and which discovered llama backends are missing."""
    client = _docker_client()
    if client is None:
        return {"found": False, "reason": "docker unreachable", "current_urls": [], "missing_backends": []}
    ow = _find_open_webui(client)
    if ow is None:
        return {"found": False, "reason": "open-webui container not found", "current_urls": [], "missing_backends": []}

    current_urls = _openwebui_persisted_urls(ow)

    # A llama backend counts as "represented" if either its container-name URL or
    # its compose service-name URL is already in the list.
    discovered = discover_llama_containers()
    missing: list[dict] = []
    for d in discovered:
        # Single-model servers are not chat endpoints. Orpheus TTS is a llama.cpp container
        # serving SNAC audio codes on 5006; offering to add it to Open WebUI's model list
        # would register a URL that is wrong (:8080) for a service that is not a chat model.
        if not d.get("router"):
            continue
        candidate_urls = [f"http://{d['name']}:8080/v1"]
        svc = _compose_service_of(client, d["name"])
        if svc:
            candidate_urls.append(f"http://{svc}:8080/v1")
        if not any(u in current_urls for u in candidate_urls):
            missing.append({"name": d["name"], "vendor": d.get("vendor", ""), "url": candidate_urls[0]})

    # Detect stale entries — URLs pointing at compose-local hosts that no longer exist
    valid_hosts: set[str] = set()
    try:
        for c in client.containers.list(all=True):
            if c.name:
                valid_hosts.add(c.name)
            svc = c.labels.get("com.docker.compose.service") if c.labels else None
            if svc:
                valid_hosts.add(svc)
    except DockerException:
        pass
    from urllib.parse import urlparse
    stale: list[str] = []
    for u in current_urls:
        host = urlparse(u).hostname or ""
        looks_local = host and "." not in host and ":" not in host
        if looks_local and valid_hosts and host not in valid_hosts:
            stale.append(u)

    # Container facts + per-connection settings. Model Loader writes to this container's
    # database, so the page that does the writing should also show what it is writing to.
    image = ""
    short_id = ""
    ports: list[str] = []
    try:
        image = ((ow.image.tags or [ow.image.short_id]) or [""])[0]
        short_id = ow.short_id
        # Docker lists IPv4 and IPv6 bindings separately for the same mapping; dedupe so the
        # card doesn't show "3000→8080" twice.
        seen_ports: set[str] = set()
        for cport, binds in ((ow.attrs or {}).get("NetworkSettings", {}).get("Ports") or {}).items():
            for b in (binds or []):
                hp = b.get("HostPort")
                if not hp:
                    continue
                label = f"{hp}→{cport.split('/')[0]}"
                if label not in seen_ports:
                    seen_ports.add(label)
                    ports.append(label)
    except (DockerException, AttributeError, KeyError):
        pass

    # Pair each persisted URL with its connection config (prefix, enabled, model filter).
    cfgs = _openwebui_api_configs(ow)
    # Every llama backend serves the same models.ini, so a whitelist id that is not a section
    # name is one no backend can resolve. OpenWebUI still LISTS such an id — the whitelist is
    # what it renders — so the model appears in the picker and then fails with "model not
    # found" on first use. That is invisible from both ends unless something names it here.
    known_ids = set(ini.section_names())
    conns: list[dict] = []
    for i, u in enumerate(current_urls):
        c = cfgs.get(str(i)) or {}
        host = urlparse(u).hostname or ""
        wanted = c.get("model_ids") or []
        conns.append({
            "url": u,
            "host": host,
            "prefix_id": c.get("prefix_id") or "",
            "enabled": c.get("enable", True),
            "model_ids": wanted,
            "unknown_ids": [m for m in wanted if m not in known_ids],
            "live_ids": [m for m in wanted if m in known_ids],
            "stale": u in stale,
        })

    return {
        "found": True,
        "container_name": ow.name,
        "status": ow.status,
        "image": image,
        "short_id": short_id,
        "ports": ports,
        "current_urls": current_urls,
        "connections": conns,
        "companions": _openwebui_companions(client, ow, valid_hosts),
        "missing_backends": missing,
        "stale_urls": stale,
    }


def _openwebui_companions(client, ow, valid_hosts: set[str]) -> list[dict]:
    """Other services OpenWebUI depends on, and the containers behind them.

    Right now that means the terminal server (ghcr.io/open-webui/open-terminal), which
    OpenWebUI records under `terminal_server.connections`. It matters here for two reasons:
    it's part of a working OpenWebUI stack, and its image lives under the same GitHub org,
    which is exactly what previously made backend discovery grab the wrong container.
    """
    import json as _json
    from urllib.parse import urlparse as _urlparse

    script = (
        "import sqlite3, json;"
        "c=sqlite3.connect('/app/backend/data/webui.db');"
        "r=c.execute(\"SELECT value FROM config WHERE key='terminal_server.connections'\").fetchone();"
        "print(r[0] if r else '[]')"
    )
    try:
        res = ow.exec_run(["python3", "-c", script])
        raw = (res.output or b"").decode(errors="replace").strip() if res.exit_code == 0 else "[]"
        entries = _json.loads(raw or "[]")
    except (DockerException, APIError, ValueError):
        entries = []

    # Index running containers by name so each configured URL can be tied to a real container.
    by_name: dict[str, object] = {}
    try:
        for c in client.containers.list(all=True):
            if c.name:
                by_name[c.name] = c
    except DockerException:
        pass

    out: list[dict] = []
    for e in entries if isinstance(entries, list) else []:
        url = str(e.get("url") or "")
        host = _urlparse(url).hostname or ""
        c = by_name.get(host)
        image = ""
        status = "not found"
        if c is not None:
            status = getattr(c, "status", "") or ""
            try:
                image = ((c.image.tags or [c.image.short_id]) or [""])[0]
            except (DockerException, AttributeError):
                pass
        looks_local = bool(host) and "." not in host and ":" not in host
        out.append({
            "kind": "terminal server",
            "name": e.get("name") or host,
            "url": url,
            "host": host,
            "enabled": bool(e.get("enabled", True)),
            "container": host if c is not None else "",
            "image": image,
            "status": status,
            # Same stale rule as the llama backends: a compose-local host that no longer exists.
            "stale": looks_local and bool(valid_hosts) and host not in valid_hosts,
        })
    return out


def _openwebui_api_configs(ow) -> dict:
    """Read openai.api_configs out of webui.db. Best-effort: an empty dict just means the
    card shows URLs without their prefixes, never an error."""
    import json as _json
    script = (
        "import sqlite3, json;"
        "c=sqlite3.connect('/app/backend/data/webui.db');"
        "r=c.execute(\"SELECT value FROM config WHERE key='openai.api_configs'\").fetchone();"
        "print(r[0] if r else '{}')"
    )
    try:
        res = ow.exec_run(["python3", "-c", script])
        if res.exit_code != 0:
            return {}
        return _json.loads((res.output or b"").decode(errors="replace").strip() or "{}")
    except (DockerException, APIError, ValueError):
        return {}


def prune_openwebui_unknown_ids() -> list[str]:
    """Drop whitelist ids that no models.ini section provides, on every connection.

    Called after a rename, where the old id becomes unservable the moment the section changes
    name. Only ever REMOVES ids that cannot resolve, so it can't hide a working model; an
    empty whitelist means "offer everything", so a connection is never emptied to nothing —
    that would silently widen it instead of narrowing it.
    """
    st = openwebui_state()
    if not st.get("found"):
        return []
    cleaned: list[str] = []
    for c in st.get("connections") or []:
        dead = c.get("unknown_ids") or []
        keep = c.get("live_ids") or []
        if not dead or not keep:
            continue  # nothing dead, or pruning would empty the list into "offer everything"
        ok, _ = set_openwebui_model_filter(c["url"], keep)
        if ok:
            cleaned.extend(dead)
    return cleaned


def openwebui_capability_plan() -> dict[str, bool]:
    """{prefixed OpenWebUI model id -> supports vision}.

    A models.ini section is multimodal iff it declares `mmproj`. OpenWebUI stores capability
    per model id, and its ids carry the connection's prefix, so the same section served by two
    connections needs an entry for each.

    Only models a connection actually offers are included: an entry for a model the connection
    filters out would be a record for something the user can never select.
    """
    plan: dict[str, bool] = {}
    try:
        st = openwebui_state()
        if not st.get("found"):
            return {}
        # Vision means the projector encodes IMAGES, not merely that a projector exists --
        # llama.cpp uses the same --mmproj slot for audio encoders too.
        vision_by_section = {}
        for name in ini.section_names():
            mm = (ini.get_section(name) or {}).get("mmproj", "").strip()
            vision_by_section[name] = bool(mm) and "vision" in projector_modalities(mm)
        for c in st.get("connections") or []:
            if c.get("stale"):
                continue
            offered = c.get("model_ids") or list(vision_by_section)  # empty filter = all
            prefix = c.get("prefix_id") or ""
            for name in offered:
                if name not in vision_by_section:
                    continue  # stale whitelist id; prune handles those
                plan[f"{prefix}.{name}" if prefix else name] = vision_by_section[name]
    except (OSError, KeyError, AttributeError):
        return {}
    return plan


def sync_openwebui_capabilities() -> tuple[bool, str]:
    """Push per-model `capabilities.vision` into OpenWebUI, derived from models.ini.

    Without this OpenWebUI offers the image-upload control on every model, because it has no
    capability record for them. Picking a text-only model then fails at inference with
    "image input is not supported" -- the model id and the fact that it has a projector live
    in two different systems, and nothing warns you at selection time.

    Existing meta is MERGED, not replaced: capabilities a user set by hand in OpenWebUI's own
    model editor (web_search, code_interpreter, ...) survive. Only `vision` is authoritative
    here, because it is the only one derivable from the ini.
    """
    import json as _json
    plan = openwebui_capability_plan()
    if not plan:
        return False, "nothing to sync (no OpenWebUI connections, or no models.ini sections)"
    client = _docker_client()
    if client is None:
        return False, "docker unreachable"
    ow = _find_open_webui(client)
    if ow is None:
        return False, "open-webui not found"

    script = """
import sqlite3, json, os, time
c = sqlite3.connect('/app/backend/data/webui.db')
now = int(time.time())
plan = json.loads(os.environ['PLAN'])

row = c.execute('SELECT id FROM user WHERE role=? ORDER BY created_at LIMIT 1', ('admin',)).fetchone()
if row is None:
    row = c.execute('SELECT id FROM user ORDER BY created_at LIMIT 1').fetchone()
if row is None:
    print('NOUSER'); raise SystemExit(0)
uid = row[0]

changed = 0
for mid, vision in plan.items():
    cur = c.execute('SELECT meta FROM model WHERE id=?', (mid,)).fetchone()
    if cur is None:
        meta = {'capabilities': {'vision': bool(vision)}}
        name = mid.split('.', 1)[1] if '.' in mid else mid
        c.execute(
            'INSERT INTO model(id, user_id, base_model_id, name, params, meta, updated_at, created_at, is_active)'
            ' VALUES(?,?,?,?,?,?,?,?,1)',
            (mid, uid, None, name, '{}', json.dumps(meta), now, now))
        changed += 1
    else:
        try: meta = json.loads(cur[0]) if cur[0] else {}
        except Exception: meta = {}
        if not isinstance(meta, dict): meta = {}
        caps = meta.get('capabilities')
        if not isinstance(caps, dict): caps = {}
        if caps.get('vision') == bool(vision):
            continue
        caps['vision'] = bool(vision)
        meta['capabilities'] = caps
        c.execute('UPDATE model SET meta=?, updated_at=? WHERE id=?', (json.dumps(meta), now, mid))
        changed += 1
c.commit()
print('OK %d' % changed)
"""
    try:
        r = ow.exec_run(["python3", "-c", script], environment={"PLAN": _json.dumps(plan)})
    except (DockerException, APIError) as e:
        return False, f"exec into open-webui failed: {e}"
    out = (r.output or b"").decode(errors="replace").strip()
    if r.exit_code != 0 or not out.startswith("OK"):
        return False, f"capability sync failed: {out[:200]}"
    n = out.split()[1] if len(out.split()) > 1 else "0"
    vis = sum(1 for v in plan.values() if v)
    # No restart. Unlike openai.api_configs (PersistentConfig, read once at boot), the `model`
    # table is ordinary application data that OpenWebUI queries per request, so capability
    # changes are picked up on the next page load.
    if n == "0":
        return True, f"already in sync ({vis} of {len(plan)} vision-capable)"
    return True, f"{n} model(s) updated ({vis} of {len(plan)} vision-capable)"


def align_openwebui_capabilities() -> tuple[bool, str]:
    """Make OpenWebUI's per-model capabilities match models.ini, authoritatively.

    Differs from sync_openwebui_capabilities() in two ways:
      * OVERWRITES `capabilities.vision` on backend records instead of deferring to what is
        already there, so OpenWebUI's permissive default (vision on for everything) loses.
        Only that one key is touched -- other capabilities are left exactly as found.
      * DELETES backend records for model ids nothing serves any more -- the residue of
        deleted or renamed models, which otherwise keep claiming capabilities forever.

    Two categories are never touched:
      * Records with a base_model_id: those are workspace models the USER built in
        OpenWebUI (a custom name, system prompt and params on top of a base). Deleting one
        destroys real work that does not exist anywhere else.
      * Any record another model names as its base_model_id, even if the backend no longer
        serves it -- removing it would orphan the workspace model sitting on top.
    """
    import json as _json
    plan = openwebui_capability_plan()
    if not plan:
        return False, "nothing to align (no OpenWebUI connections, or no models.ini sections)"
    client = _docker_client()
    if client is None:
        return False, "docker unreachable"
    ow = _find_open_webui(client)
    if ow is None:
        return False, "open-webui not found"

    script = """
import sqlite3, json, os, time
c = sqlite3.connect('/app/backend/data/webui.db')
now = int(time.time())
plan = json.loads(os.environ['PLAN'])

row = c.execute('SELECT id FROM user WHERE role=? ORDER BY created_at LIMIT 1', ('admin',)).fetchone()
if row is None:
    row = c.execute('SELECT id FROM user ORDER BY created_at LIMIT 1').fetchone()
if row is None:
    print('NOUSER'); raise SystemExit(0)
uid = row[0]

# ids that are somebody's base -- protected from deletion
bases = {r[0] for r in c.execute('SELECT DISTINCT base_model_id FROM model WHERE base_model_id IS NOT NULL')}

updated = 0
for mid, vision in plan.items():
    cur = c.execute('SELECT meta, base_model_id FROM model WHERE id=?', (mid,)).fetchone()
    if cur is not None and cur[1]:
        continue                      # user workspace model; not ours to rewrite
    if cur is None:
        meta = {}
    else:
        try: meta = json.loads(cur[0]) if cur[0] else {}
        except Exception: meta = {}
        if not isinstance(meta, dict): meta = {}
    # Authoritative about `vision` ONLY. Replacing the whole capabilities block also
    # destroyed image_generation, web_search, code_interpreter and citations -- flags this
    # app knows nothing about and has no business clearing. Aligning vision used to switch
    # off image generation for every model it touched.
    caps = meta.get('capabilities')
    if not isinstance(caps, dict):
        caps = {}
    caps['vision'] = bool(vision)
    meta['capabilities'] = caps
    if cur is None:
        name = mid.split('.', 1)[1] if '.' in mid else mid
        c.execute('INSERT INTO model(id, user_id, base_model_id, name, params, meta, updated_at, created_at, is_active)'
                  ' VALUES(?,?,?,?,?,?,?,?,1)', (mid, uid, None, name, '{}', json.dumps(meta), now, now))
    else:
        c.execute('UPDATE model SET meta=?, updated_at=? WHERE id=?', (json.dumps(meta), now, mid))
    updated += 1

removed, kept = [], []
for mid, base in c.execute('SELECT id, base_model_id FROM model').fetchall():
    if base or mid in plan:
        continue                      # workspace model, or still served
    if mid in bases:
        kept.append(mid); continue    # another model is built on it
    c.execute('DELETE FROM model WHERE id=?', (mid,))
    removed.append(mid)
c.commit()
print('OK ' + json.dumps({'updated': updated, 'removed': removed, 'kept': kept}))
"""
    try:
        r = ow.exec_run(["python3", "-c", script], environment={"PLAN": _json.dumps(plan)})
    except (DockerException, APIError) as e:
        return False, f"exec into open-webui failed: {e}"
    out = (r.output or b"").decode(errors="replace").strip()
    if r.exit_code != 0 or not out.startswith("OK"):
        return False, f"align failed: {out[:200]}"
    try:
        res = _json.loads(out[3:])
    except ValueError:
        return True, out[:160]
    vis = sum(1 for v in plan.values() if v)
    msg = f"aligned {res['updated']} model(s) — {vis} vision-capable of {len(plan)}"
    if res["removed"]:
        msg += f"; removed {len(res['removed'])} stale record(s): " + ", ".join(res["removed"][:4])
    if res["kept"]:
        msg += f"; kept {len(res['kept'])} stale record(s) still used as a base model"
    return True, msg


def set_openwebui_model_filter(url: str, model_ids: list[str]) -> tuple[bool, str]:
    """Restrict which models one OpenWebUI connection exposes.

    OpenWebUI's `openai.api_configs[<index>].model_ids` is a whitelist: empty means "offer
    everything this endpoint reports", non-empty means "offer only these". Model ids are the
    RAW ids from the backend's /v1/models — the connection's prefix_id is applied by
    OpenWebUI afterwards for display, so it must not appear here.

    Useful when several backends share one models.ini but shouldn't all serve every model —
    e.g. a CPU backend that should only offer the small models it can actually run.
    """
    import json as _json
    client = _docker_client()
    if client is None:
        return False, "docker unreachable"
    ow = _find_open_webui(client)
    if ow is None:
        return False, "open-webui not found"

    script = """
import sqlite3, json, os, time
c = sqlite3.connect('/app/backend/data/webui.db')
now = int(time.time())

def _get(k, default):
    r = c.execute('SELECT value FROM config WHERE key=?', (k,)).fetchone()
    if not r:
        return default
    try: return json.loads(r[0])
    except Exception: return default

def _set(k, v):
    val = json.dumps(v)
    if c.execute('SELECT 1 FROM config WHERE key=?', (k,)).fetchone():
        c.execute('UPDATE config SET value=?, updated_at=? WHERE key=?', (val, now, k))
    else:
        c.execute('INSERT INTO config(key, value, updated_at) VALUES(?, ?, ?)', (k, val, now))

target = os.environ['TARGET_URL']
ids = json.loads(os.environ['MODEL_IDS'])
urls = _get('openai.api_base_urls', [])
cfgs = _get('openai.api_configs', {})
if target not in urls:
    print('NOTFOUND'); raise SystemExit(0)
idx = str(urls.index(target))
cfg = cfgs.get(idx) or {'enable': True, 'tags': [], 'prefix_id': '', 'model_ids': [],
                        'connection_type': 'external', 'auth_type': 'bearer', 'passthrough_params': []}
cfg['model_ids'] = ids
cfgs[idx] = cfg
_set('openai.api_configs', cfgs)
c.commit()
print('OK ' + str(len(ids)))
"""
    try:
        r = ow.exec_run(
            ["python3", "-c", script],
            environment={"TARGET_URL": url, "MODEL_IDS": _json.dumps(model_ids)},
        )
    except (DockerException, APIError) as e:
        return False, f"exec into open-webui failed: {e}"

    out = (r.output or b"").decode(errors="replace").strip()
    if r.exit_code != 0:
        return False, f"DB update failed (exit {r.exit_code}): {out[:200]}"
    if out.startswith("NOTFOUND"):
        return False, f"connection not found in OpenWebUI: {url}"

    # PersistentConfig only re-reads on boot, same as the endpoint sync.
    try:
        ow.restart(timeout=30)
    except (DockerException, APIError) as e:
        return False, f"filter saved but restart failed: {e}"
    n = len(model_ids)
    return True, (f"{url}: now offering {n} selected model(s)" if n
                  else f"{url}: now offering all models")


def toggle_openwebui_model(url: str, model_id: str, show: bool,
                           all_model_ids: list[str]) -> tuple[bool, str]:
    """Show/hide ONE model on ONE OpenWebUI connection.

    The wrinkle: model_ids == [] means "offer everything", not "offer nothing". So hiding a
    model on a connection that is currently unfiltered cannot just remove an entry — there
    are no entries. It has to materialise the full model list minus that one, which converts
    the connection from implicit-all to an explicit whitelist.

    That has a lasting consequence worth surfacing to the caller: once explicit, models
    downloaded later will NOT appear on that connection until they're added. Returns a
    message saying so, so the UI can warn rather than silently changing future behaviour.
    """
    cur = openwebui_state()
    if not cur.get("found"):
        return False, cur.get("reason") or "open-webui not found"
    conn = next((c for c in cur.get("connections", []) if c["url"] == url), None)
    if conn is None:
        return False, f"connection not found: {url}"

    existing = list(conn.get("model_ids") or [])
    was_implicit_all = not existing
    if show:
        if was_implicit_all:
            return True, f"{model_id} already visible on {conn['host']} (offering all models)"
        if model_id in existing:
            return True, f"{model_id} already visible on {conn['host']}"
        new_ids = existing + [model_id]
    else:
        if was_implicit_all:
            # Convert implicit-all into an explicit list so one model can be excluded.
            new_ids = [m for m in all_model_ids if m != model_id]
        else:
            new_ids = [m for m in existing if m != model_id]
            if not new_ids:
                # An empty list would silently mean "all models" — the opposite of hiding
                # the last one. Refuse rather than do the reverse of what was asked.
                return False, (f"cannot hide the last model on {conn['host']}: an empty list "
                               "means 'offer everything' in OpenWebUI. Disable the connection "
                               "instead if you want it to serve nothing.")

    ok, msg = set_openwebui_model_filter(url, new_ids)
    if ok and was_implicit_all and not show:
        msg += (f" — {conn['host']} now uses an explicit list, so models added later "
                "will not appear there until you enable them")
    return ok, msg


def sync_openwebui_endpoints() -> tuple[bool, str]:
    """Reconcile OpenWebUI's PERSISTED config (webui.db → openai.api_base_urls) with reality:
      1. Prune URLs whose target container/service no longer exists on the docker socket
      2. Add discovered llama backends that aren't in the list yet
    Preserves existing per-connection prefixes/tags for endpoints that stay.
    Then restarts open-webui so PersistentConfig re-hydrates from the updated DB.
    """
    import json as _json
    client = _docker_client()
    if client is None:
        return False, "docker unreachable"
    ow = _find_open_webui(client)
    if ow is None:
        return False, "open-webui not found"

    # Build the set of hostnames that are valid targets (container names + compose service names)
    valid_hosts: set[str] = set()
    try:
        for c in client.containers.list(all=True):
            if c.name:
                valid_hosts.add(c.name)
            svc = c.labels.get("com.docker.compose.service") if c.labels else None
            if svc:
                valid_hosts.add(svc)
    except DockerException:
        pass

    state = openwebui_state()
    missing = state.get("missing_backends") or []

    additions = [{"name": m["name"]} for m in missing]

    script = r"""
import sqlite3, json, os, time
from urllib.parse import urlparse
now = int(time.time())
c = sqlite3.connect('/app/backend/data/webui.db')

def _get(k, default):
    r = c.execute('SELECT value FROM config WHERE key=?', (k,)).fetchone()
    if not r: return default
    try: return json.loads(r[0])
    except Exception: return default

def _set(k, v):
    val = json.dumps(v)
    if c.execute('SELECT 1 FROM config WHERE key=?', (k,)).fetchone():
        c.execute('UPDATE config SET value=?, updated_at=? WHERE key=?', (val, now, k))
    else:
        c.execute('INSERT INTO config(key, value, updated_at) VALUES(?, ?, ?)', (k, val, now))

urls = _get('openai.api_base_urls', [])
keys = _get('openai.api_keys', [])
cfgs = _get('openai.api_configs', {})
valid_hosts = set(json.loads(os.environ.get('VALID_HOSTS', '[]')))

# ---- 1) Prune stale entries whose host isn't a live container / service ----
kept_urls, kept_keys, kept_cfgs = [], [], {}
removed = 0
for old_idx, u in enumerate(urls):
    host = urlparse(u).hostname or ''
    # Keep non-llama hosts (e.g. openrouter, real openai) — only prune if it LOOKS like a compose-local
    # llama host (unqualified, no dots, was in the DB but no matching container exists any more).
    looks_local = host and '.' not in host and ':' not in host
    if looks_local and valid_hosts and host not in valid_hosts:
        removed += 1
        continue
    new_idx = str(len(kept_urls))
    kept_urls.append(u)
    kept_keys.append(keys[old_idx] if old_idx < len(keys) else 'dummy')
    if str(old_idx) in cfgs:
        kept_cfgs[new_idx] = cfgs[str(old_idx)]

# ---- 2) Add missing entries ----
added = 0
for a in json.loads(os.environ.get('ADDITIONS', '[]')):
    url = 'http://' + a['name'] + ':8080/v1'
    if url in kept_urls:
        continue
    kept_urls.append(url)
    kept_keys.append('dummy')
    idx = str(len(kept_urls) - 1)
    pref = a['name'].replace('llama-', '').replace('_', '-').upper()[:12]
    kept_cfgs[idx] = {
        'enable': True,
        'tags': [],
        'prefix_id': pref,
        'model_ids': [],
        'connection_type': 'external',
        'auth_type': 'bearer',
        'passthrough_params': [],
    }
    added += 1

_set('openai.api_base_urls', kept_urls)
_set('openai.api_keys', kept_keys)
_set('openai.api_configs', kept_cfgs)
_set('openai.enable', True)

c.commit()
print('added=' + str(added) + ' removed=' + str(removed))
"""

    try:
        r = ow.exec_run(
            ["python3", "-c", script],
            environment={
                "ADDITIONS": _json.dumps(additions),
                "VALID_HOSTS": _json.dumps(sorted(valid_hosts)),
            },
        )
    except (DockerException, APIError) as e:
        return False, f"exec into open-webui failed: {e}"

    out = r.output.decode(errors="replace").strip() if r.output else ""
    if r.exit_code != 0:
        return False, f"DB update failed (exit {r.exit_code}): {out[:300]}"

    # Restart so PersistentConfig re-hydrates from the updated DB
    try:
        ow.restart(timeout=30)
    except (DockerException, APIError) as e:
        return False, f"DB updated but restart failed: {e}. Restart open-webui manually."

    if additions or "removed=0" not in out:
        return True, f"reconciled OpenWebUI ({out}) and restarted"
    return True, f"already in sync — no changes needed ({out})"


def _started_epoch(container) -> float | None:
    """When the container's current run began, as fractional epoch seconds. None if unknown.

    Docker reports StartedAt with nanoseconds ("2026-09-13T18:50:08.749054149Z"). Keeping the
    fraction matters: flooring to the second can pull in the previous run's last lines, which is
    exactly what reading "this run only" exists to exclude.
    """
    raw = str(((getattr(container, "attrs", None) or {}).get("State") or {}).get("StartedAt") or "")
    head, _, frac = raw.rstrip("Z").partition(".")
    try:
        base = datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None
    digits = "".join(ch for ch in frac if ch.isdigit())
    return base + (float("0." + digits) if digits else 0.0)


def container_logs(name: str, tail: int = 200, current_run_only: bool = False) -> tuple[bool, str]:
    """A container's recent log output.

    `current_run_only` limits the read to the present run. `docker logs` keeps history across
    restarts, so a plain tail reaches back into previous runs - fine for a human scrolling the log
    panel, wrong for anything that treats what it reads as the container's CURRENT state.
    """
    client = _docker_client()
    if client is None:
        return False, "docker unreachable"
    try:
        c = client.containers.get(name)
    except NotFound:
        return False, "container not found"
    except DockerException as e:
        return False, f"docker error: {e}"
    try:
        kw = {}
        if current_run_only:
            since = _started_epoch(c)
            if since is not None:
                kw["since"] = since
        raw = c.logs(tail=tail, stdout=True, stderr=True, timestamps=False, **kw)
    except DockerException as e:
        return False, f"log fetch failed: {e}"
    return True, raw.decode("utf-8", errors="replace")


def diagnose_container(name: str) -> tuple[bool, list[dict], str]:
    """Recognised failure signatures from a backend's recent log, each with a suggested fix.

    Reads a wider tail than the log panel's 200 lines: a failed load is often preceded by a lot
    of chatter, and the decisive line can sit well above the most recent noise. Returns
    (ok, findings, err); findings is empty-and-ok when the log simply holds no known signature.

    Reads the CURRENT run only. `docker logs` keeps history across restarts, and a 1200-line tail
    easily reaches back past one: after a preset was fixed and the router restarted, Diagnose
    kept reporting the old section's missing-projector error from hours before the restart.
    Bounding by the run is the right cut rather than treating the router's "listening" line as a
    success marker, because the router validates presets during boot and only THEN listens - so
    a startup preset error sits just before that line, and would be hidden the moment it happened.
    """
    ok, tail = container_logs(name, tail=1200, current_run_only=True)
    if not ok:
        return False, [], tail
    findings = [{"error": f.error, "hint": f.hint, "model": f.model} for f in diagnose.diagnose(tail)]
    return True, findings, ""


def openwebui_capability_state() -> dict[str, "bool | None"]:
    """{prefixed model id -> vision as OpenWebUI currently has it}. None = no record.

    Read side of the capability sync, so the UI can show whether OpenWebUI agrees with
    models.ini for a given model rather than offering a blind "align" that gives no
    indication whether anything was actually wrong. Skips workspace models (those with a
    base_model_id): their capabilities belong to the user, not to us.
    """
    import json as _json
    client = _docker_client()
    if client is None:
        return {}
    ow = _find_open_webui(client)
    if ow is None:
        return {}
    script = """
import sqlite3, json
c = sqlite3.connect('/app/backend/data/webui.db')
out = {}
for i, b, m in c.execute('SELECT id, base_model_id, meta FROM model'):
    if b:
        continue
    try:
        d = json.loads(m) if m else {}
    except Exception:
        d = {}
    out[i] = (d.get('capabilities') or {}).get('vision')
print(json.dumps(out))
"""
    try:
        r = ow.exec_run(["python3", "-c", script])
        if r.exit_code != 0:
            return {}
        return _json.loads((r.output or b"{}").decode(errors="replace").strip() or "{}")
    except (DockerException, APIError, ValueError):
        return {}


def align_openwebui_capability_for(section: str) -> tuple[bool, str]:
    """Align capabilities for ONE models.ini section, across every connection serving it.

    Same authoritative replace as the bulk align, scoped to a single model. Stale-record
    cleanup is deliberately NOT done here: a stale record has no section to hang a per-model
    control off, so removing those belongs to the sweep instead.
    """
    import json as _json
    full = openwebui_capability_plan()
    # endswith("." + section), not rsplit: section names contain dots of their own
    # (Qwen3.8-27B-Q4_K_M), so splitting on the last dot matches the wrong thing.
    plan = {mid: v for mid, v in full.items()
            if mid == section or mid.endswith("." + section)}
    if not plan:
        return False, f"{section}: not offered by any OpenWebUI connection"
    client = _docker_client()
    if client is None:
        return False, "docker unreachable"
    ow = _find_open_webui(client)
    if ow is None:
        return False, "open-webui not found"

    script = """
import sqlite3, json, os, time
c = sqlite3.connect('/app/backend/data/webui.db')
now = int(time.time())
plan = json.loads(os.environ['PLAN'])
row = c.execute('SELECT id FROM user WHERE role=? ORDER BY created_at LIMIT 1', ('admin',)).fetchone()
if row is None:
    row = c.execute('SELECT id FROM user ORDER BY created_at LIMIT 1').fetchone()
if row is None:
    print('NOUSER'); raise SystemExit(0)
uid = row[0]
n = 0
for mid, vision in plan.items():
    cur = c.execute('SELECT meta, base_model_id FROM model WHERE id=?', (mid,)).fetchone()
    if cur is not None and cur[1]:
        continue
    meta = {}
    if cur is not None:
        try:
            meta = json.loads(cur[0]) if cur[0] else {}
        except Exception:
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
    # Same rule as align_openwebui_capabilities(): own `vision`, leave the rest alone.
    caps = meta.get('capabilities')
    if not isinstance(caps, dict):
        caps = {}
    caps['vision'] = bool(vision)
    meta['capabilities'] = caps
    if cur is None:
        name = mid.split('.', 1)[1] if '.' in mid else mid
        c.execute('INSERT INTO model(id, user_id, base_model_id, name, params, meta, updated_at, created_at, is_active)'
                  ' VALUES(?,?,?,?,?,?,?,?,1)', (mid, uid, None, name, '{}', json.dumps(meta), now, now))
    else:
        c.execute('UPDATE model SET meta=?, updated_at=? WHERE id=?', (json.dumps(meta), now, mid))
    n += 1
c.commit()
print('OK %d' % n)
"""
    try:
        r = ow.exec_run(["python3", "-c", script], environment={"PLAN": _json.dumps(plan)})
    except (DockerException, APIError) as e:
        return False, f"exec into open-webui failed: {e}"
    out = (r.output or b"").decode(errors="replace").strip()
    if r.exit_code != 0 or not out.startswith("OK"):
        return False, f"align failed: {out[:200]}"
    vis = any(plan.values())
    return True, f"{section}: vision = {str(vis).lower()} on {', '.join(sorted(plan))}"


_PROJ_MODALITY_CACHE: dict[tuple, frozenset] = {}


def projector_modalities(mmproj_rel_or_abs: str) -> frozenset:
    """What a projector actually encodes: {'vision'}, {'audio'}, or both. Empty if unreadable.

    llama.cpp uses the same `--mmproj` slot for BOTH image and audio encoders (Qwen2-Audio,
    Ultravox and friends ship an audio projector). So the presence of an mmproj says the model
    is multimodal, not that it takes pictures. Deriving OpenWebUI's `vision` flag from the
    mere existence of a projector would offer image upload on an audio-only model -- the same
    class of wrong as offering it on a text-only one.

    The projector declares itself: a vision encoder carries clip.vision.* keys, an audio
    encoder clip.audio.*. Cached on (path, size, mtime) since this reads a file header.
    """
    raw_path = (mmproj_rel_or_abs or "").strip()
    if not raw_path:
        return frozenset()
    rel = raw_path.replace("/models/", "", 1).lstrip("/")
    p = settings.models_dir / rel
    try:
        st = p.stat()
    except OSError:
        return frozenset()
    key = (str(p), st.st_size, int(st.st_mtime))
    hit = _PROJ_MODALITY_CACHE.get(key)
    if hit is not None:
        return hit
    mods: set[str] = set()
    try:
        from . import gguf_meta
        kv = gguf_meta.read_raw(p)
        if isinstance(kv, dict) and "kv" in kv and isinstance(kv["kv"], dict):
            kv = kv["kv"]
        if isinstance(kv, dict):
            for k, v in kv.items():
                lk = str(k).lower()
                if lk.startswith("clip.vision.") or lk == "clip.has_vision_encoder" and v:
                    mods.add("vision")
                elif lk.startswith("clip.audio.") or lk == "clip.has_audio_encoder" and v:
                    mods.add("audio")
    except Exception:  # noqa: BLE001 -- a projector we cannot parse must not break the page
        return frozenset()
    out = frozenset(mods)
    if len(_PROJ_MODALITY_CACHE) > 32:
        _PROJ_MODALITY_CACHE.clear()
    _PROJ_MODALITY_CACHE[key] = out
    return out


def sections_with_speculative() -> dict[str, str]:
    """{section name -> draft head filename} for sections with speculative decoding wired up.

    Requires BOTH a draft model and a spec-type that is not "none": naming a head without
    selecting a type does nothing, and a type without a head cannot run.
    """
    out: dict[str, str] = {}
    try:
        for name in ini.section_names():
            sec = ini.get_section(name) or {}
            model = (sec.get("spec-draft-model") or "").strip()
            stype = (sec.get("spec-type") or "").strip().lower()
            if model and stype and stype != "none":
                out[name] = model.rsplit("/", 1)[-1]
    except (OSError, KeyError, AttributeError):
        return {}
    return out


def section_modalities() -> dict[str, list[str]]:
    """{section name -> sorted modalities its projector encodes}. Empty list = text-only.

    Separate from openwebui_capability_plan(), which reduces this to a single vision bool
    because that is all OpenWebUI models. Audio has no OpenWebUI capability flag at all, so
    an audio modality can be shown here but cannot be pushed anywhere — it is Model Loader's
    own display, not a setting.
    """
    out: dict[str, list[str]] = {}
    try:
        for name in ini.section_names():
            mm = (ini.get_section(name) or {}).get("mmproj", "").strip()
            out[name] = sorted(projector_modalities(mm)) if mm else []
    except (OSError, KeyError, AttributeError):
        return {}
    return out


# Vendors that mean "this backend has a GPU". Read from the image tag by
# discover_llama_containers, so it needs no live probe — which matters because this decision
# is made right after a download completes, possibly before the stats sampler has warmed up.
_GPU_VENDORS = {"cuda", "rocm", "vulkan", "nvidia", "amd"}


def assign_new_model_to_gpu(section: str) -> tuple[bool, str]:
    """Offer a newly-created section on every GPU-backed OpenWebUI connection.

    A connection with a non-empty model_ids list is an explicit whitelist, so a model added
    later is offered by nothing until someone ticks it — a new download lands in models.ini,
    works perfectly, and is invisible in the chat UI with no indication why. Defaulting it
    onto the GPU backends matches what anyone downloading a model actually wants.

    Deliberately narrow:
      * GPU connections only. A CPU backend should not silently inherit a 27B.
      * Connections already offering everything (empty whitelist) are skipped — they serve it
        already, and writing an explicit list would convert them to a whitelist, quietly
        changing behaviour for every FUTURE model.
      * Callers must only invoke this on CREATION. Re-running it on an ordinary save would
        undo a deliberate removal.
    """
    try:
        gpu_hosts = {d.get("name") for d in discover_llama_containers()
                     if (d.get("vendor") or "").lower() in _GPU_VENDORS}
    except DockerException:
        return False, "docker unreachable"
    if not gpu_hosts:
        return False, "no GPU-backed llama container discovered"

    st = openwebui_state()
    if not st.get("found"):
        return False, st.get("reason") or "open-webui not found"

    touched: list[str] = []
    for c in st.get("connections") or []:
        if c.get("stale") or c["host"] not in gpu_hosts:
            continue
        ids = list(c.get("model_ids") or [])
        if not ids:
            continue                      # already offers everything
        if section in ids:
            continue                      # nothing to do
        ok, _ = set_openwebui_model_filter(c["url"], ids + [section])
        if ok:
            touched.append(c.get("prefix_id") or c["host"])
    if not touched:
        return True, ""
    return True, f"offered on {', '.join(touched)}"
