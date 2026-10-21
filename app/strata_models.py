"""Finding, installing and selecting the models Strata can run.

This is the Strata half of what `hf` + `downloader` + `autoconfig` do for llama.cpp, and it is
a different shape on purpose, because the engine is a different shape.

llama.cpp runs any GGUF: search the Hub, download a file, point a server at it. Strata runs
ONE architecture - Qwen3.8-Flash-Next - out of an expert pack its own setup builds, and it
resolves a model by EXACT FILENAME against a closed table of families and quant tiers:

    setup.gguf_choice("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf") -> ("qwen", "IQ3_S")
    setup.gguf_choice("Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf") -> None

So a straight port of the llama search would return hundreds of repos of which approximately
none are installable, and the one repo that IS installable would be rejected for having an
extra word in its filename. Neither half of that is useful.

What this module does instead:

  * reads the catalog of (family, tier) pairs LIVE FROM THE ENGINE, so it tracks whatever
    version of Strata is in the image rather than a copy that rots here (`catalog()`);
  * classifies a Hub repo against that catalog, allowing the decorated names a community
    re-quant or fine-tune actually ships, and producing the exact rename each file needs
    (`classify_repo()`);
  * plans the download, reusing byte-identical shards already on disk (`plan_install()`).

WHY RENAMING IS LEGITIMATE HERE, AND WHERE IT STOPS BEING SO. The name is not a checksum. It
is how setup looks the file up, and for the three GSQ-RCO families there is no checksum at all
- only the `unsloth` family pins SHA-256 (FAMILIES["unsloth"]["sha256"]), because its files are
fetched by setup itself. A GSQ-RCO file from a mirror repo is the same quantization of the same
architecture in the same two-shard layout, so handing it to setup under the name setup expects
is telling the truth about what it is. What would NOT be legitimate is renaming a file whose
quantization Strata cannot read (a K-quant, Unsloth's UD-IQ3_XXS) into a name it can: that
produces a confident failure deep inside the pack builder instead of an honest refusal up
front, which is exactly why `classify_repo` checks the quant format marker and the shard
layout rather than trusting the tier token in the name.

WHAT "INSTALLED" MEANS, AND THE SLOT LIMIT THAT FALLS OUT OF IT. Strata names a prepared pack
and its install config after family tag + tier: `strata-iq3_s.json`, `strata-swift-iq3_s.json`,
`strata-unsloth-ud-iq4_xs.json`. Two different mirrors of the SAME tier therefore collide
unless they go into different family slots, and there are only as many slots as there are
families that accept GSQ-RCO files. That is a real ceiling on "how many re-tunes can I keep
installed at once", it is not something this app can lift, and `free_slots()` exists so the UI
can say so before a download starts rather than after 55 GB has landed.
"""

from __future__ import annotations

import json
import re
import shlex
import time
from pathlib import Path
from dataclasses import dataclass, field

# The architecture every Strata pack is. A repo that does not say this in its filenames is not
# a Qwen3.8-Flash-Next mirror, whatever its repo id claims.
ARCH_MARKER = "qwen3.8-flash-next"

# The quantization format marker the three ISTA-DASLab-style families use. Strata's pack tool
# reads GSQ-RCO tensors; a file without this marker is some other quantizer's output even when
# its tier token (IQ3_S) matches one of ours.
GSQ_MARKER = "gsq-rco"

# Families whose files are GSQ-RCO and carry no pinned hashes, i.e. the slots a mirror repo can
# legitimately be installed into. `unsloth` is deliberately absent: its files ARE hash-pinned,
# so a mirror renamed into that slot would fail verify_sha256 - correctly, and confusingly.
MIRROR_SLOTS = ("qwen", "swift", "coder")

_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)

_CATALOG_TTL_S = 300.0
_catalog_cache: tuple[float, dict] | None = None


@dataclass
class Tier:
    """One (family, tier) pair the engine will serve, with the numbers the UI needs to size it."""
    family: str
    model: str
    family_title: str
    family_by: str
    tag: str                      # "" | "swift-" | "coder-" | "unsloth-"
    download_gb: float
    ram_gb: int
    arena_gb: float
    experimental: bool
    vision: bool                  # whether images are allowed on this pair
    shards: int
    first_shard: str              # the exact filename setup will look for
    hash_pinned: bool
    canonical_repo: str = ""      # the Hub repo this family's own files come from

    @property
    def install_tag(self) -> str:
        """The pack/config name this pair occupies - the thing two installs can collide on."""
        return f"{self.tag}{self.model.lower()}"


@dataclass
class RepoMatch:
    """What a Hub repo can be installed as."""
    tier: Tier
    files: list[dict] = field(default_factory=list)   # {src, dst, size, sha}
    mmproj: dict | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(int(f.get("size") or 0) for f in self.files) + int((self.mmproj or {}).get("size") or 0)


def _docker_exec_json(container: str, code: str, timeout: int = 20) -> dict | None:
    """Run a snippet inside the engine container and parse its stdout as JSON.

    Through the Docker SDK, NOT a `docker` subprocess: this app runs in a container that has
    the daemon socket bind-mounted but no docker CLI on PATH, so shelling out returns
    FileNotFoundError and the catalog silently comes back empty - a page that renders fine and
    lists nothing, which is exactly how this was first shipped and caught.

    The catalog is read from the RUNNING ENGINE rather than duplicated here. Strata adds
    families and tiers between releases (UD-IQ4_XS stopped being experimental in 0.1.39), and a
    hardcoded copy here would keep offering what the image no longer has, or hide what it
    gained, with no error either way.
    """
    from .utils import docker_client, timed_shell
    cli = docker_client()
    if cli is None:
        return None
    try:
        c = cli.containers.get(container)
        if c.status != "running":
            return None
        # 0.2 (upstream's image) has its Python and setup.py under /opt/strata/terminal; 0.1.x
        # (ours) had a venv in /opt/strata. setup.py is imported by name, so cd to its folder.
        q = shlex.quote(code)
        cmd = (f"if [ -x /opt/strata/terminal/python/bin/python3 ]; then cd /opt/strata/terminal && "
               f"exec /opt/strata/terminal/python/bin/python3 -c {q}; "
               f"else exec /opt/strata/.venv/bin/python -c {q}; fi")
        rc, out = c.exec_run(timed_shell(cmd, timeout), demux=False)
    except Exception:  # noqa: BLE001 - a stopped or missing engine is "no catalog", not an error
        return None
    if rc != 0 or not out:
        return None
    try:
        return json.loads(out.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None


_CATALOG_CODE = r"""
import json, setup
out = {"families": {}, "models": {}}
for f, d in setup.FAMILIES.items():
    out["families"][f] = {
        "title": d.get("title", f), "by": d.get("by", ""), "tag": d.get("tag", ""),
        "vision": d.get("vision"), "shards": int(d.get("shards") or 2),
        "hash_pinned": bool(d.get("sha256")),
        "hf": d.get("hf", ""),
    }
for m, d in setup.MODELS.items():
    out["models"][m] = {
        "download_gb": float(d.get("download_gb") or 0), "ram_gb": int(d.get("ram_gb") or 0),
        "arena_gb": float(d.get("arena_gb") or 0),
        "experimental": bool(d.get("experimental")),
        "vision": d.get("vision"),
        "shards": int(d.get("shards") or 0),
        "families": list(d.get("families", ("qwen", "swift"))),
    }
out["files"] = {}
for f, d in setup.FAMILIES.items():
    for m in setup.MODELS:
        if f in setup.MODELS[m].get("families", ("qwen", "swift")):
            try:
                out["files"]["%s/%s" % (f, m)] = setup.model_file(d, m, 1)
            except Exception:
                pass
print(json.dumps(out))
"""


def catalog(container: str = "strata", force: bool = False) -> list[Tier]:
    """Every (family, tier) the engine in `container` will serve. Empty when it cannot be read.

    Empty is a legitimate answer the UI must render as "the engine is not running", NOT as "no
    models exist": the catalog lives inside the image, so it is unreadable exactly when Strata
    is stopped - which is most of the time on a box where the engines take turns.
    """
    global _catalog_cache
    now = time.time()
    if not force and _catalog_cache and now - _catalog_cache[0] < _CATALOG_TTL_S:
        raw = _catalog_cache[1]
    else:
        raw = _docker_exec_json(container, _CATALOG_CODE)
        if raw is None:
            return []
        _catalog_cache = (now, raw)

    fams, models, files = raw.get("families", {}), raw.get("models", {}), raw.get("files", {})
    out: list[Tier] = []
    for fname, f in fams.items():
        for mname, m in models.items():
            if fname not in m.get("families", ["qwen", "swift"]):
                continue
            first = files.get(f"{fname}/{mname}")
            if not first:
                continue
            # Vision resolves model-first, family-second, and only an explicit False blocks it -
            # the same expression setup uses. A missing key on both means "allowed", so `is False`
            # matters: `m.get("vision") or f.get("vision")` would read None as "no" and wrongly
            # hide images on every GSQ-RCO tier.
            v = m.get("vision", f.get("vision"))
            out.append(Tier(
                family=fname, model=mname,
                family_title=f.get("title", fname), family_by=f.get("by", ""),
                tag=f.get("tag", ""),
                download_gb=m.get("download_gb", 0.0), ram_gb=m.get("ram_gb", 0),
                arena_gb=m.get("arena_gb", 0.0),
                experimental=bool(m.get("experimental")),
                vision=v is not False,
                shards=int(m.get("shards") or f.get("shards") or 2),
                first_shard=first,
                hash_pinned=bool(f.get("hash_pinned")),
                canonical_repo=_repo_from_hf(f.get("hf", "")),
            ))
    out.sort(key=lambda t: (t.family, t.download_gb))
    return out


def _repo_from_hf(url: str) -> str:
    """"owner/name" out of a family's Hub URL, which carries a path and placeholders after it."""
    m = re.search(r"huggingface\.co/([^/]+/[^/{}]+)", url or "")
    return m.group(1).rstrip("/") if m else ""


def parse_shard(name: str) -> tuple[int, int] | None:
    """(index, total) from a `-00001-of-00002.gguf` suffix, or None when it has none."""
    m = _SHARD_RE.search(name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _tier_token_in(name: str, model: str) -> bool:
    """Whether `name` carries exactly this tier's token, bounded so IQ3_S never matches IQ3_XXS.

    Plain substring is wrong in both directions here: "Q2_0" is inside "UD-Q2_K_XL", and
    "IQ3_S" is not inside "IQ3_XXS" but "Q3_S" nearly is. Bounding on the separators these
    names actually use (-, _, ., start, end) is what keeps the tiers apart.
    """
    return re.search(rf"(?:^|[-_.]){re.escape(model)}(?:$|[-_.])", name, re.IGNORECASE) is not None


def classify_repo(repo_files: list[dict], tiers: list[Tier] | None = None,
                  container: str = "strata") -> list[RepoMatch]:
    """Which catalog tiers a Hub repo can be installed as, and the rename each file needs.

    `repo_files` is [{"path": "IQ3_S/Foo-IQ3_S-00001-of-00002.gguf", "size": int, "sha": str}].
    `sha` is optional and only used to spot shards already on disk later.

    A tier matches when the repo holds a COMPLETE shard set for it that looks like the right
    thing: the architecture marker, the quant-format marker, that tier's token, and the shard
    count the family declares. Partial sets are dropped rather than offered, because a missing
    shard surfaces as a pack-builder failure tens of GB into a download.
    """
    tiers = tiers if tiers is not None else catalog(container)
    if not tiers:
        return []

    ggufs = [f for f in repo_files
             if str(f.get("path", "")).lower().endswith(".gguf")
             and "mmproj" not in str(f.get("path", "")).rsplit("/", 1)[-1].lower()]
    mmprojs = [f for f in repo_files
               if "mmproj" in str(f.get("path", "")).rsplit("/", 1)[-1].lower()]

    out: list[RepoMatch] = []
    for tier in tiers:
        if tier.family not in MIRROR_SLOTS:
            # unsloth: hash-pinned, so only the genuine upstream file can ever satisfy it.
            continue
        want = parse_shard(tier.first_shard)
        want_total = want[1] if want else tier.shards

        picked: dict[int, dict] = {}
        for f in ggufs:
            base = str(f["path"]).rsplit("/", 1)[-1]
            low = base.lower()
            if ARCH_MARKER not in low or GSQ_MARKER not in low:
                continue
            if not _tier_token_in(base, tier.model):
                continue
            sh = parse_shard(base)
            if not sh or sh[1] != want_total:
                continue
            picked.setdefault(sh[0], f)

        if len(picked) != want_total:
            continue

        files = []
        for idx in sorted(picked):
            src = picked[idx]
            files.append({
                "src": src["path"],
                "dst": re.sub(r"-\d{5}-of-\d{5}\.gguf$", f"-{idx:05d}-of-{want_total:05d}.gguf",
                              tier.first_shard, flags=re.IGNORECASE),
                "size": int(src.get("size") or 0),
                "sha": src.get("sha") or "",
            })

        warnings: list[str] = []
        # Size sanity: the catalog knows roughly what this tier weighs. A repo whose "IQ3_S" is
        # half the expected size is not that tier however it is named, and finding that out
        # from a pack failure after the download is the outcome this check exists to prevent.
        got_gb = sum(f["size"] for f in files) / 1e9
        if tier.download_gb and got_gb and not (0.75 <= got_gb / tier.download_gb <= 1.25):
            warnings.append(f"this repo's {tier.model} is {got_gb:.1f} GB, but the catalog "
                            f"expects about {tier.download_gb:.1f} GB — check it is really "
                            f"{tier.model} before installing")

        mm = None
        if mmprojs and tier.vision:
            # Smallest mmproj: repos ship BF16/F16/Q8_0 of one encoder and the pack only uses one.
            pick = min(mmprojs, key=lambda f: int(f.get("size") or 0))
            mm = {"src": pick["path"], "dst": _mmproj_name(tier, container),
                  "size": int(pick.get("size") or 0), "sha": pick.get("sha") or ""}

        out.append(RepoMatch(tier=tier, files=files, mmproj=mm, warnings=warnings))
    return out


_MMPROJ_CACHE: dict[str, str] = {}


def _mmproj_name(tier: Tier, container: str) -> str:
    """The vision encoder filename this family's setup looks for."""
    key = tier.family
    if key in _MMPROJ_CACHE:
        return _MMPROJ_CACHE[key]
    raw = _docker_exec_json(
        container,
        "import json, setup; print(json.dumps({'n': setup.FAMILIES[%r].get('mmproj','')}))" % tier.family)
    name = (raw or {}).get("n") or "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
    _MMPROJ_CACHE[key] = name
    return name


def free_slots(installed_tags: set[str], tiers: list[Tier]) -> list[Tier]:
    """Catalog pairs whose pack/config name is not already taken.

    See the module docstring: the ceiling on simultaneously installed mirrors of one tier is
    the number of family slots, and the UI should say which are left BEFORE a download.
    """
    return [t for t in tiers if t.install_tag not in installed_tags]


# ---------------------------------------------------------------- installed state

# Where this app can see Strata's volume. model-loader mounts ./strata as /strata, and the
# engine's data lives under it, so the installed set is read from the FILESYSTEM rather than
# by exec'ing into the container. That difference matters: the engines take turns on the cards,
# so Strata is stopped much of the time, and "what is installed" has to answer while it is.
DATA_DIR = "/strata/data"


# How a path inside the ENGINE container maps to one this app can see. The two containers
# mount the same data under different names, so "does this file exist" is answerable here only
# after translating. /opt/strata lives inside the engine's image and is always present, so it is
# not checked rather than guessed at.
_PATH_VIEWS = (
    ("/gguf-root/", "/models/"),      # the models root, ./models in both compose services
)


def _missing_sources(cfg: dict, root=None) -> list[str]:
    """Paths an install needs that are NOT there any more.

    This exists because an install can rot without anything touching it. `/gguf` used to be a
    bind of ONE model directory; replacing it with `/gguf-root` (the whole models root) left
    every earlier install pointing at a mount the container no longer has. Nothing reported
    that: the config was valid, the pack was intact, and the failure only appeared as a
    container that would not start - the same silent, crash-looping shape as the staged-config
    bug. An install whose files have moved should say so where it is listed, not when it is
    selected.
    """
    out: list[str] = []
    paths = [v for v in (cfg.get("args") or []) if isinstance(v, str) and v.startswith("/")]
    vision = cfg.get("vision")
    if isinstance(vision, dict):
        paths += [v for v in vision.values() if isinstance(v, str) and v.startswith("/")]
    for raw in paths:
        if raw.startswith("/opt/strata"):
            continue                              # inside the engine image; always present
        local = None
        # the engine's /data is wherever the deployment's volume is (./strata/data on 0.1,
        # ./strata/data2 on 0.2), passed in by the caller
        views = _PATH_VIEWS + ((("/data/", str(root).rstrip("/") + "/"),) if root else ())
        for pref, here in views:
            if raw.startswith(pref):
                local = here + raw[len(pref):]
                break
        from pathlib import Path as _P
        if local is None:
            out.append(raw)                       # an old mount name, e.g. the retired /gguf
        elif not _P(local).exists():
            out.append(raw)
    return out


def installed(data_dir: str | None = None) -> dict[str, dict]:
    """{install_tag: {"config": path, "pack": path|None, "bytes": int}} for prepared models.

    An install is a prepared PACK plus an install CONFIG, both named after family tag + tier
    (`strata-swift-iq3_s.json`, `packs/swift-iq3_s`). The config alone is what the entry point
    tests for with `[ -f $cfg ]` to decide whether to run a setup pass, so the config is the
    authority on "installed" and the pack is reported beside it for size.
    """
    from pathlib import Path
    from . import strata as strata_engine
    root = Path(data_dir) if data_dir else strata_engine.live_data_dir()
    out: dict[str, dict] = {}
    cfg_dir = strata_engine.configs_dir(root)
    try:
        entries = sorted(cfg_dir.glob("strata-*.json"))
    except OSError:
        return out
    for c in entries:
        tag = c.stem[len("strata-"):]
        pack = strata_engine.packs_dir(root) / tag
        size = 0
        if pack.is_dir():
            try:
                size = sum(f.stat().st_size for f in pack.rglob("*") if f.is_file())
            except OSError:
                size = 0
        try:
            cfg = json.loads(c.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cfg = {}
        out[tag] = {"config": str(c), "pack": str(pack) if pack.is_dir() else None, "bytes": size,
                    "missing": _missing_sources(cfg, root)}
    return out


# ---------------------------------------------------------------- download planning

def _index_local(models_root: str) -> dict[int, list[str]]:
    """{size: [path, ...]} for GGUFs already on disk - candidates to verify, never to trust."""
    from pathlib import Path
    out: dict[int, list[str]] = {}
    try:
        for p in Path(models_root).rglob("*.gguf"):
            try:
                out.setdefault(p.stat().st_size, []).append(str(p))
            except OSError:
                continue
    except OSError:
        pass
    return out


_sha_cache: dict[tuple[str, int, int], str] = {}


def local_sha256(path: str) -> str:
    """sha256 of a local file, cached on (path, size, mtime). "" when it cannot be read.

    Hashing tens of GB is slow, so this is only ever called on a file whose SIZE already
    matches something we were about to download - i.e. when the alternative is fetching those
    same tens of GB over the network. A minute of disk read beats half an hour of download.
    """
    import hashlib
    from pathlib import Path
    try:
        st = Path(path).stat()
    except OSError:
        return ""
    key = (path, st.st_size, int(st.st_mtime))
    if key in _sha_cache:
        return _sha_cache[key]
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(16 << 20)
                if not b:
                    break
                h.update(b)
    except OSError:
        return ""
    _sha_cache[key] = h.hexdigest()
    return _sha_cache[key]


def plan_install(match: RepoMatch, models_root: str, dest_subdir: str,
                 verify: bool = True) -> dict:
    """What actually has to be downloaded for `match`, after reusing what is already here.

    Returns {"fetch", "reuse", "candidates", "fetch_bytes", "saved_bytes", "dest"}.

    REUSE IS THE POINT OF THIS FUNCTION, not an optimisation. These families publish ONE
    n-gram-table shard and share it across every quant tier and across mirror repos: the
    abliterated IQ3_S shard 2 and ISTA-DASLab's IQ3_S shard 2 are the same 28.8 GB object
    (sha256 316b46f3...). Re-fetching it would double a 55 GB download for nothing.

    REUSE IS BY CONTENT HASH ONLY. An earlier cut of this matched on byte size alone and, in
    its first test, proposed installing a DIFFERENT model's vision encoder
    (Qwen3.8-Flash-Next-UD-Q3_K_XL/mmproj-BF16.gguf) because it happened to weigh the same -
    while the Hub etags prove the abliterated encoder is a different file. Weights and encoders
    of one architecture collide on size routinely, so size is a way to FIND a candidate and
    never a reason to trust one. A size candidate is hashed (`verify=True`) and either confirmed
    or discarded; with `verify=False` it is reported under "candidates" for the UI to offer,
    and is NOT counted as reuse.

    `match.files[*]["sha"]` is the Hub's LFS etag, which is the file's sha256. Fill it in with
    `fetch_etags()` before planning, or every reuse decision degrades to a candidate.
    """
    from pathlib import Path
    by_size = _index_local(models_root)
    dest = Path(models_root) / dest_subdir

    fetch, reuse, candidates = [], [], []
    for f in list(match.files) + ([match.mmproj] if match.mmproj else []):
        want_sha = (f.get("sha") or "").lower()
        target = dest / f["dst"]

        # Already in place and the right size: confirm by hash when we have one to confirm against.
        if target.exists() and target.stat().st_size == f["size"]:
            if not want_sha or local_sha256(str(target)) == want_sha:
                reuse.append({**f, "from": str(target), "how": "already in place"})
                continue

        hits = [h for h in by_size.get(f["size"], []) if h != str(target)]
        if hits and want_sha and verify:
            found = next((h for h in hits if local_sha256(h) == want_sha), None)
            if found:
                reuse.append({**f, "from": found, "how": "identical file already on disk"})
                continue
        elif hits and not want_sha:
            candidates.append({**f, "from": hits[0],
                               "how": "same size, but the Hub gave no hash to confirm it"})
        elif hits and not verify:
            candidates.append({**f, "from": hits[0], "how": "same size, not yet verified"})

        fetch.append(f)

    return {
        "dest": str(dest),
        "fetch": fetch, "reuse": reuse, "candidates": candidates,
        "fetch_bytes": sum(f["size"] for f in fetch),
        "saved_bytes": sum(f["size"] for f in reuse),
    }


async def fetch_etags(repo_id: str, paths: list[str], revision: str = "main") -> dict[str, str]:
    """{path: sha256} from the Hub, read from the `x-linked-etag` header.

    THE HEADER CHOICE IS THE WHOLE FUNCTION. A resolve URL answers with TWO 64-hex values and
    only one of them is the file's content hash:

        x-linked-etag: 316b46f3...   <- sha256 of the bytes; matches `sha256sum` on disk
        etag:          076b2d0a...   <- the CDN/Xet object hash; matches NOTHING local

    Both look like a sha256, so reading the wrong one fails silently and expensively: every
    reuse check then compares a real local hash against a CDN hash, never matches, and the
    planner re-downloads tens of GB it already has. (That is not hypothetical - it happened
    while this was being written, and cost a 28.8 GB shard its correct reuse.)

    `follow_redirects` is OFF for the same reason: huggingface.co answers the HEAD with
    x-linked-etag and then redirects to a CDN whose response carries only `etag`. Following the
    redirect throws the useful header away.
    """
    import httpx
    out: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
        for p in paths:
            url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{p}"
            try:
                r = await client.head(url)
            except httpx.HTTPError:
                continue
            tag = (r.headers.get("x-linked-etag") or "").strip().strip('"')
            if re.fullmatch(r"[0-9a-f]{64}", tag):
                out[p] = tag
    return out


# ---------------------------------------------------------------- carrying out an install

def materialize_reuse(plan: dict) -> list[str]:
    """Put the reused files in place as HARDLINKS, falling back to a copy across filesystems.

    A hardlink because these are 1-29 GB objects and a second copy buys nothing: the sources
    are read-only inputs to the pack builder, which never writes to them. On this box that is
    the difference between an install costing 55 GB and costing 85 GB.

    Returns the list of messages for the UI; never raises, because a failed link is a reason to
    download that file instead, not a reason to fail the whole install.
    """
    import os
    import shutil
    from pathlib import Path
    out: list[str] = []
    dest = Path(plan["dest"])
    dest.mkdir(parents=True, exist_ok=True)
    for f in plan.get("reuse", []):
        src, dst = Path(f["from"]), dest / f["dst"]
        if dst.exists() and dst.stat().st_size == f["size"]:
            continue
        try:
            os.link(src, dst)
            out.append(f"linked {dst.name} ({f['size'] / 1e9:.1f} GB, no extra disk)")
        except OSError:
            try:
                shutil.copy2(src, dst)
                out.append(f"copied {dst.name} ({f['size'] / 1e9:.1f} GB)")
            except OSError as e:
                out.append(f"could not place {dst.name}: {e}")
    return out


# ---------------------------------------------------------------- which model the engine runs

_SEL_KEYS = ("FAMILY", "MODEL", "GGUF_DIR")


def read_selection(env_path: str) -> dict[str, str]:
    """{FAMILY, MODEL, GGUF_DIR} as the pin file currently names them ("" when it does not)."""
    out = {k: "" for k in _SEL_KEYS}
    try:
        text = open(env_path, encoding="utf-8").read()
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        key = key.replace("export", "").strip()
        if key in out:
            out[key] = val.strip().strip('"').strip("'")
    return out


def set_selection(env_path: str, family: str, model: str, gguf_dir: str) -> tuple[bool, str]:
    """Rewrite FAMILY / MODEL / GGUF_DIR in the pin file, leaving every comment and the GPU
    pin untouched - the same single-writer discipline `strata.set_pin` uses on the same file.

    THIS ONLY TAKES EFFECT IF THE ENTRY POINT EXPORTS THESE NAMES. compose bakes `env_file`
    into a container at create time and its own `environment:` block outranks the file, so the
    service's entry point must re-source this file AND export these three the way it already
    does for GPU/GPUS. Without that half, this function writes a value nothing reads.
    """
    from pathlib import Path
    p = Path(env_path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        return False, f"cannot read {p}: {e}"

    want = {"FAMILY": family, "MODEL": model, "GGUF_DIR": gguf_dir}
    seen: set[str] = set()
    out: list[str] = []
    for line in text.splitlines():
        key = line.split("=", 1)[0].replace("export", "").strip()
        if key in want and not line.strip().startswith("#"):
            if key not in seen:
                out.append(f"{key}={want[key]}")
                seen.add(key)
            continue
        out.append(line)
    for k in _SEL_KEYS:
        if k not in seen:
            out.append(f"{k}={want[k]}")
    # Atomic, like set_pin and write_env_setting: the entry point sources this file at start,
    # and a plain write_text during a start can be read half-done.
    from . import strata as strata_engine
    try:
        strata_engine._atomic_replace(p, "\n".join(out) + "\n")
    except OSError as e:
        return False, f"cannot write {p}: {e}"
    return True, f"next start: {family} / {model}" + (f" from {gguf_dir}" if gguf_dir else "")


# ---------------------------------------------------------------- where an install came from

# A prepared pack records the family slot and the tier - `swift-iq3_s` - and nothing about the
# repo the files came from. That is fine for the engine and wrong for a person: once a mirror
# is installed into a borrowed slot, the slot name actively misdescribes it. SC117's abliterated
# IQ3_S sitting in the `swift` slot is not Swift 1.5, and a list that calls it `swift-iq3_s` is
# telling the user something false every time they look at it.
#
# So the app keeps its own small record beside the engine's. It is NOT written into Strata's
# config: setup owns that file and regenerates it, and anything we add there is lost on the next
# pass (the lesson the hand-edited vision block keeps teaching). This lives in the app's own kv
# store, where the app is the only writer and a re-setup cannot touch it.
_PROV_KEY = "strata.install_provenance"


def provenance() -> dict[str, dict]:
    """{install_tag: {"repo": "owner/name", "family": str, "model": str, "at": epoch}}."""
    from . import db
    try:
        return json.loads(db.get_setting(_PROV_KEY, "") or "{}")
    except (ValueError, TypeError):
        return {}


def record_provenance(install_tag: str, repo_id: str, family: str, model: str,
                      canonical_repo: str = "") -> None:
    """Remember which repo an install came from. Overwrites: a slot holds one thing at a time.

    `borrowed` is decided HERE, where the family's own repo is known from the catalog, and
    stored as a fact. Deriving it later by comparing the repo id against the slot name is
    string-guessing that gets `unsloth-ud-iq4_xs` wrong - "unsloth-ud" is not a family name,
    so Unsloth's own file in Unsloth's own slot reads as borrowed.
    """
    from . import db
    cur = provenance()
    cur[install_tag] = {"repo": repo_id, "family": family, "model": model,
                        "at": int(time.time()),
                        "borrowed": bool(canonical_repo) and repo_id.lower() != canonical_repo.lower()}
    db.set_setting(_PROV_KEY, json.dumps(cur))


def describe_install(install_tag: str, prov: dict[str, dict] | None = None) -> dict:
    """How to label an installed pack: its source repo when known, the slot name when not.

    `borrowed` is the case worth flagging in the UI - the pack sits in a family slot whose name
    is not its own, which is normal for a mirror and confusing without a label.
    """
    p = (prov if prov is not None else provenance()).get(install_tag) or {}
    repo = p.get("repo") or ""
    owner, _, name = repo.partition("/")
    return {
        "tag": install_tag,
        "repo": repo,
        "owner": owner,
        "title": name or install_tag,
        "model": p.get("model", ""),
        "family": p.get("family", ""),
        "borrowed": bool(p.get("borrowed")),
    }


def install_label(about: dict) -> str:
    """A person's name for an install: the repo's model name and the tier - Qwen3.8-Flash-Next-GSQ-RCO-
    abliterated-IQ3_S - or "" when the install has no recorded source (then the slot name IS its name)."""
    if not about.get("repo"):
        return ""
    title = about.get("title") or ""
    for suffix in ("-GGUF", "-gguf", "_GGUF"):
        if title.endswith(suffix):
            title = title[: -len(suffix)]
    tier = about.get("model") or ""
    return f"{title}-{tier}" if tier and not title.lower().endswith(tier.lower()) else title


_SERVED_TTL_S = 30.0
_served_cache: tuple[float, dict] | None = None


def served_labels(data_dir: str | None = None) -> dict[str, dict]:
    """{model id Strata serves: {"tag", "repo", "owner", "label"}} for installs with a recorded source.

    Strata names a model after the catalog slot it sits in (setup.py rebuilds model_name as
    <family>-<tier> on every start - it is one of its SETUP_KEYS), and a community model can only
    be installed by borrowing a slot. So SC117's abliterated IQ3_S, installed into the free Swift
    slot, is served as "swift-1.5-iq3_s" - a name for a different model. On 0.1 the hand-edited
    config kept "qwen3.8-flash-next-abliterated-iq3_s"; 0.2 rewrites it at every start. The
    install's own record of where it came from is what tells the two apart, so every place this
    app names a Strata model asks here first. Cached 30 s: the overview asks twice a second.
    """
    global _served_cache
    now = time.time()
    if data_dir is None and _served_cache and now - _served_cache[0] < _SERVED_TTL_S:
        return _served_cache[1]
    from pathlib import Path
    prov = provenance()
    out: dict[str, dict] = {}
    for tag, d in installed(data_dir).items():
        about = describe_install(tag, prov)
        label = install_label(about)
        if not label:
            continue
        try:
            served = json.loads(Path(d["config"]).read_text(encoding="utf-8-sig")).get("model_name") or ""
        except (OSError, ValueError):
            continue
        if served:
            out[served] = {"tag": tag, "repo": about["repo"], "owner": about["owner"], "label": label,
                           "borrowed": about["borrowed"]}
    if data_dir is None:
        _served_cache = (now, out)
    return out


# ---------------------------------------------------------------- whose format is it

def strata_only(filename: str) -> bool:
    """True for a GGUF only the Strata engine can read.

    GSQ-RCO is ISTA-DASLab's quantization for this engine's pack builder; llama.cpp has no
    kernels for it and cannot load the file at all. That matters beyond labelling, because
    autoconfig will otherwise size one of these like any other GGUF and write a complete,
    confident models.ini section for it - context, flash-attn, cache types, mmproj - describing
    how to run something that cannot run. A recipe that detailed is worse than no recipe: it
    reads as a tested configuration.

    Unsloth's UD-* packs are deliberately NOT included. Those are ordinary GGUF quantizations
    that llama.cpp reads perfectly well; Strata merely also knows how to build a pack from
    them, so a section for one is legitimate.
    """
    return GSQ_MARKER in (filename or "").lower()


# ---------------------------------------------------------------- tuning an installed model

# Strata has no models.ini. Its equivalent is the `args` list inside an install config, which
# setup writes and the engine reads at start - so this app owns a SMALL, named set of them the
# same way `ini` owns models.ini: single writer, re-applied, never hand-edited.
#
# Only knobs that cannot stop the engine starting are here. --kv, --max-context, --spec and
# --batch-groups all change what has to fit in VRAM, and this box reports 552 MiB free, so a
# wrong value there is a container that will not start - the failure mode this engine already
# has too many of. Those wait for a fit check that can refuse BEFORE the restart.
TUNABLES = {
    "batch": {
        "label": "Concurrent slots",
        "about": ("How many conversations decode together. Measured here 2026-10-16 on both "
                  "cards, IQ3_S, greedy, 256 tokens, agent-shaped prompts (shared prefix, "
                  "different task per caller): at 4 callers total throughput 66.6 -> 103.2 "
                  "tok/s (+55%) and the slowest wait 12.6s -> 2.7s; at 2 callers 65.6 -> 71.8 "
                  "and 4.6s -> 1.0s. The price is a single request alone: 85.6 -> 74.5 tok/s "
                  "(-13%), because the slots' sessions come out of the expert cache and "
                  "residency fell 100% -> 95%. Worth it for agents that fan out, backwards "
                  "for one chat at a time."),
        "choices": [("1", "off (one at a time)"), ("2", "2 slots"), ("4", "4 slots")],
        "metric": "batch_slots",   # the engine reports what it actually runs, not what we asked
    },
    "conversation-cache-mib": {
        "label": "Conversation cache",
        "about": ("Keeps each client's conversation so an agent resending its history does not "
                  "re-read it. Measured here 2026-10-16 with six conversations alternating: "
                  "follow-up turns 3.05s -> 0.52s to first token with 99% of the prompt "
                  "reused, first turns unchanged, and expert residency 95% either way - it "
                  "costs nothing in decode. Needs enough slots below, or it parks and evicts "
                  "without ever reading back."),
        "choices": [("0", "off"), ("4096", "4 GB"), ("8192", "8 GB"), ("16384", "16 GB")],
        "metric": "conversation_cache_mib",   # the engine reports this, so the UI shows truth
    },
    "conversation-cache-slots": {
        "label": "Parked conversations",
        "about": ("How many conversations the cache may hold, which is a separate limit from "
                  "its size and usually the binding one: a parked conversation of ~5.4k tokens "
                  "snapshots at ~320 MB, so 8 GB has room for roughly 25 but the slot count "
                  "stops it far earlier. With more live conversations than slots the cache "
                  "parks and evicts without ever being read back (the engine log says "
                  "reused_kv_bytes=0). Set it above the number of conversations an agent keeps "
                  "in flight."),
        "choices": [("4", "4"), ("8", "8"), ("16", "16")],
        "metric": "conversation_cache_slots",
    },
    "prefill": {
        "label": "Prompt chunk",
        "about": ("How much of a prompt is read per pass. Upstream measured auto:32768 reading "
                  "prompts 21-35% faster on machines with plenty of RAM (#433 #440 #834); this "
                  "one has 126 GB."),
        "choices": [("auto", "auto"), ("auto:16384", "auto, up to 16K"), ("auto:32768", "auto, up to 32K")],
        "metric": None,                        # not reported; read back from the config
    },
}


def _config_path(install_tag: str, data_dir: str | None = None) -> "Path":
    from pathlib import Path
    from . import strata as strata_engine
    root = Path(data_dir) if data_dir else strata_engine.live_data_dir()
    return strata_engine.configs_dir(root) / f"strata-{install_tag}.json"


# ---------------------------------------------------------------- 0.2: the settings live in the pin file
#
# 0.2's setup rewrites the run config's args on every start (it runs inside `strata launch`), so
# a flag written there lasts until the next restart. The documented surface is the launcher's:
# PARALLEL, and STRATA_SET for engine flags (`strata launch --set prefill=auto:32768`). Both are
# read from the pin file by entrypoint-0.2.sh on every start - the same file, and the same
# single-writer discipline, as the GPU pin and the model selection.

_ENV_OF = {"batch": "PARALLEL"}            # tunables with a launcher variable of their own


def _layout_02(data_dir: str | None) -> bool:
    from pathlib import Path
    from . import strata as strata_engine
    root = Path(data_dir) if data_dir else strata_engine.live_data_dir()
    return strata_engine.layout(root) == "0.2"


def _env_lines(env_path: str) -> list[str]:
    try:
        return open(env_path, encoding="utf-8").read().splitlines()
    except OSError:
        return []


def _env_key(line: str) -> str:
    t = line.strip()
    if t.startswith("#") or "=" not in t:
        return ""
    return t.partition("=")[0].replace("export", "").strip()


def env_settings(env_path: str) -> dict[str, str]:
    """{tunable: value} as the pin file sets it: PARALLEL, and the flags in STRATA_SET."""
    out: dict[str, str] = {}
    for line in _env_lines(env_path):
        key = _env_key(line)
        if not key:
            continue
        val = line.strip().partition("=")[2].strip().strip('"').strip("'")
        if key == "STRATA_SET":
            for tok in val.split():
                k, _, v = tok.partition("=")
                if k and v:
                    out[k] = v
        else:
            for name, env in _ENV_OF.items():
                if key == env and val:
                    out[name] = val
    return out


def write_env_setting(env_path: str, flag: str, value: str) -> tuple[bool, str]:
    """Set one tunable in the pin file, keeping every other line (comments included) as it is."""
    from pathlib import Path
    from . import strata as strata_engine
    lines = _env_lines(env_path)
    if not lines:
        return False, f"cannot read {env_path}"
    out: list[str] = []
    done = False
    if flag in _ENV_OF:
        key = _ENV_OF[flag]
        for line in lines:
            if _env_key(line) == key:
                if not done:
                    out.append(f"{key}={value}")
                    done = True
                continue
            out.append(line)
        if not done:
            out.append(f"{key}={value}")
    else:
        for line in lines:
            if _env_key(line) == "STRATA_SET":
                cur = line.strip().partition("=")[2].strip().strip('"').strip("'").split()
                toks = [x for x in cur if x.partition("=")[0] != flag] + [f"{flag}={value}"]
                if not done:
                    out.append('STRATA_SET="' + " ".join(toks) + '"')
                    done = True
                continue
            out.append(line)
        if not done:
            out.append(f'STRATA_SET="{flag}={value}"')
    try:
        strata_engine._atomic_replace(Path(env_path), "\n".join(out) + "\n")
    except OSError as e:
        return False, f"cannot write {env_path}: {e}"
    return True, f"{TUNABLES[flag]['label']} set to {value}; applies on the next restart"


def tuning_state(install_tag: str, metrics: dict | None = None,
                 data_dir: str | None = None) -> dict[str, str]:
    """{flag: current value} for an install, preferring what the ENGINE reports over what the
    config says. The two differ exactly between a change and the restart that applies it, and
    showing the config there would claim a setting is live when it is not."""
    out: dict[str, str] = {}
    try:
        cfg = json.loads(_config_path(install_tag, data_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    args = cfg.get("args") or []
    from . import strata as strata_engine
    pinned = env_settings(str(strata_engine.ENV_PATH)) if _layout_02(data_dir) else {}
    for flag, spec in TUNABLES.items():
        val = ""
        key = spec.get("metric")
        if key and isinstance(metrics, dict):
            live = (metrics.get("engine") or {}).get(key)
            if live is not None:
                val = str(live)
        if not val:
            val = pinned.get(flag, "")
        if not val:
            tok = f"--{flag}"
            val = args[args.index(tok) + 1] if tok in args and args.index(tok) + 1 < len(args) else ""
        out[flag] = val or (spec["choices"][0][0])
    return out


def set_tunable(install_tag: str, flag: str, value: str, container: str = "strata",
                data_dir: str | None = None) -> tuple[bool, str]:
    """Write one tuning flag into an install's config, to take effect on the next start.

    THE STAGED COPY HAS TO GO, and that is not tidiness. docker-entrypoint.sh copies
    /opt/strata/strata-<tag>.json BACK OVER the config on the volume whenever the two differ -
    that is how changes made during a run are persisted. So a config edited here while a staged
    copy exists is silently REVERTED on the next start, by a mechanism that looks like it is
    protecting data. Removing the staged copy makes the next start re-stage from what we wrote.
    """
    if flag not in TUNABLES:
        return False, f"not a tunable: {flag}"
    if value not in {v for v, _ in TUNABLES[flag]["choices"]}:
        return False, f"not an allowed value for {flag}: {value!r}"
    if _layout_02(data_dir):
        from . import strata as strata_engine
        return write_env_setting(str(strata_engine.ENV_PATH), flag, value)

    p = _config_path(install_tag, data_dir)
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return False, f"cannot read {p.name}: {e}"

    args = list(cfg.get("args") or [])
    tok = f"--{flag}"
    if tok in args:
        i = args.index(tok)
        if i + 1 < len(args):
            args[i + 1] = value
        else:
            args.append(value)
    else:
        args += [tok, value]
    cfg["args"] = args
    try:
        p.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    except OSError as e:
        return False, f"cannot write {p.name}: {e}"

    # Remember what we asked for, so a setup pass that regenerates args can be detected - the
    # same reason the vision block keeps getting lost.
    from . import db
    want = json.loads(db.get_setting("strata.tuning", "") or "{}")
    want.setdefault(install_tag, {})[flag] = value
    db.set_setting("strata.tuning", json.dumps(want))

    _drop_staged(install_tag, container)
    return True, f"{TUNABLES[flag]['label']} set to {value} — applies on the next restart"


def _drop_staged(install_tag: str, container: str) -> None:
    """Remove the engine container's staged copy of this config (see set_tunable)."""
    from .utils import docker_client, timed_shell
    cli = docker_client()
    if cli is None:
        return
    try:
        c = cli.containers.get(container)
        if c.status == "running":
            c.exec_run(timed_shell(f"rm -f /opt/strata/strata-{install_tag}.json"), demux=False)
    except Exception:  # noqa: BLE001 - a stopped engine has no staged copy to drop
        pass
