"""Are the engines behind their upstreams?

The engines on this box move fast - Maya shipped 1.1.17 and 1.1.18 the day after it was
installed, Strata goes through point releases weekly, llama.cpp publishes several builds a day -
and finding out used to mean someone remembering to look. This answers it in one place, per
engine: what runs here, what upstream has, and where the runbook is. It never upgrades anything:
each engine's upgrade has its own checks and its own rollback (see the runbooks), and an app that
swapped engines by itself is how this box would lose a working one.

What "running here" means differs per engine, so it is read the way each one can tell it:

  * Strata and Maya are built locally, one image per release, tagged `rocm7-v<version>`, with
    `rocm7` pointing at the live one. The version is the tag sharing `rocm7`'s image id. When the
    engine is serving, its own report wins (Strata's /metrics engine.version, Maya's /health
    version), because that is what is actually loaded.
  * llama.cpp is a pulled image. The meaningful question is "would a pull change it", so each
    local llama image's digest is compared with the registry's for the same tag, and the build it
    runs (llama's /props build_info, bNNNNN) against the newest release.

Results are cached in the settings db for CHECK_EVERY_S, so the Overview costs nothing and a
restart does not re-ask GitHub. Unauthenticated GitHub allows 60 requests an hour; a full check
is 4.
"""

from __future__ import annotations

import json
import re
import threading
import time

import httpx

from . import db

CHECK_EVERY_S = 6 * 3600.0
_KEY = "updates.state"
_HTTP_TIMEOUT = httpx.Timeout(10.0)

# The engines and where their upstreams and runbooks are. `image` is the local repository the
# release tags live in; `live_tag` the one compose runs.
ENGINES = [
    {"engine": "strata", "label": "Strata", "repo": "Niko1221/Strata", "kind": "tags",
     "image": "ai-lab/strata", "live_tag": "rocm7", "container": "strata",
     "runbook": "~/ai-lab/strata/UPGRADING.md"},
    {"engine": "maya", "label": "Maya", "repo": "mw00/project-maya", "kind": "tags",
     "image": "ai-lab/maya", "live_tag": "rocm7", "container": "maya",
     "runbook": "~/ai-lab/maya/UPGRADING.md"},
    {"engine": "llama", "label": "llama.cpp", "repo": "ggml-org/llama.cpp", "kind": "release",
     "image": "ghcr.io/ggml-org/llama.cpp", "runbook": "docker compose pull <service>, then recreate it"},
]

_VER_RE = re.compile(r"^v?(\d+(?:\.\d+)+)$")
_LOCAL_TAG_RE = re.compile(r"-v(\d+(?:\.\d+)+)$")
_BUILD_RE = re.compile(r"^b(\d+)")

_lock = threading.Lock()
_running = False


def _vkey(v: str) -> tuple:
    return tuple(int(x) for x in v.split("."))


def newer(latest: str, current: str) -> bool | None:
    """Is `latest` a later version than `current`? None when either is not a version."""
    try:
        return _vkey(latest) > _vkey(current)
    except (ValueError, AttributeError):
        return None


# ------------------------------------------------------------------------------- upstream

def latest_tag(client: httpx.Client, repo: str) -> str:
    """The highest version-shaped tag of a GitHub repo ("1.1.18"), "" when there is none."""
    r = client.get(f"https://api.github.com/repos/{repo}/tags", params={"per_page": 100})
    r.raise_for_status()
    vers = [m.group(1) for t in r.json() if (m := _VER_RE.match(str(t.get("name") or "")))]
    return max(vers, key=_vkey) if vers else ""


def latest_release(client: httpx.Client, repo: str) -> dict:
    """GitHub's latest release: {tag, published, url}."""
    r = client.get(f"https://api.github.com/repos/{repo}/releases/latest")
    r.raise_for_status()
    d = r.json()
    return {"tag": str(d.get("tag_name") or ""), "published": str(d.get("published_at") or ""),
            "url": str(d.get("html_url") or "")}


def registry_digest(client: httpx.Client, image: str, tag: str) -> str:
    """The digest a `docker pull <image>:<tag>` would fetch, from ghcr's v2 API. "" if unknown."""
    if not image.startswith("ghcr.io/"):
        return ""
    name = image[len("ghcr.io/"):]
    tok = client.get("https://ghcr.io/token", params={"scope": f"repository:{name}:pull"})
    tok.raise_for_status()
    accept = ", ".join([
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json"])
    r = client.head(f"https://ghcr.io/v2/{name}/manifests/{tag}",
                    headers={"Authorization": f"Bearer {tok.json().get('token', '')}",
                             "Accept": accept})
    r.raise_for_status()
    return r.headers.get("docker-content-digest", "")


# ------------------------------------------------------------------------------- local

def local_version(docker, image: str, live_tag: str) -> str:
    """The release the live tag points at: the `*-v<version>` tag sharing its image id."""
    try:
        live = docker.images.get(f"{image}:{live_tag}")
    except Exception:  # noqa: BLE001 - no image is "unknown", never an error page
        return ""
    vers = [m.group(1) for t in (live.tags or []) if (m := _LOCAL_TAG_RE.search(t))]
    return max(vers, key=_vkey) if vers else ""


def _running_report(engine: str, container: str) -> str:
    """The version the engine itself reports while serving, "" when it is not serving."""
    try:
        with httpx.Client(timeout=httpx.Timeout(3.0)) as c:
            if engine == "strata":
                v = (c.get(f"http://{container}:8080/metrics").json().get("engine") or {}).get("version")
            else:
                v = c.get(f"http://{container}:8080/health").json().get("version")
        return str(v or "")
    except (httpx.HTTPError, ValueError, AttributeError):
        return ""


def llama_images(docker) -> list[dict]:
    """Each llama.cpp image a container on this box uses: {tag, digest, containers, build}."""
    by_tag: dict[str, dict] = {}
    try:
        containers = docker.containers.list(all=True)
    except Exception:  # noqa: BLE001
        return []
    for c in containers:
        ref = str((c.attrs.get("Config") or {}).get("Image") or "")
        if not ref.startswith("ghcr.io/ggml-org/llama.cpp:"):
            continue
        tag = ref.split(":", 1)[1]
        row = by_tag.setdefault(tag, {"tag": tag, "digest": "", "containers": [], "build": ""})
        row["containers"].append(c.name)
        if not row["digest"]:
            try:
                digests = docker.images.get(ref).attrs.get("RepoDigests") or []
                row["digest"] = next((d.split("@", 1)[1] for d in digests if "@" in d), "")
            except Exception:  # noqa: BLE001
                pass
        if c.status == "running" and not row["build"]:
            row["build"] = _llama_build(c)
    return sorted(by_tag.values(), key=lambda r: r["tag"])


def _llama_build(c) -> str:
    """bNNNNN-<commit> from a running llama server's /props, "" when it does not say."""
    from .services import _extract_ports
    _, port = _extract_ports(c.attrs or {})
    if port is None:
        return ""
    try:
        with httpx.Client(timeout=httpx.Timeout(3.0)) as h:
            return str(h.get(f"http://{c.name}:{port}/props").json().get("build_info") or "")
    except (httpx.HTTPError, ValueError, AttributeError):
        return ""


# ------------------------------------------------------------------------------- the check

def check(docker, http: httpx.Client | None = None) -> dict:
    """One full check: {checked_at, engines: [row]}. Network failures land in a row's `error`."""
    own = http is None
    http = http or httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=True,
                                headers={"Accept": "application/vnd.github+json",
                                         "User-Agent": "model-loader-update-check"})
    rows = []
    try:
        for e in ENGINES:
            row = {"engine": e["engine"], "label": e["label"], "repo": e["repo"],
                   "runbook": e["runbook"], "current": "", "latest": "", "status": "unknown",
                   "url": "", "error": "", "detail": []}
            try:
                if e["kind"] == "tags":
                    row["current"] = (_running_report(e["engine"], e["container"])
                                      or local_version(docker, e["image"], e["live_tag"]))
                    row["latest"] = latest_tag(http, e["repo"])
                    row["url"] = f"https://github.com/{e['repo']}/releases"
                    if row["current"] and row["latest"]:
                        row["url"] = (f"https://github.com/{e['repo']}/compare/"
                                      f"v{row['current']}...v{row['latest']}")
                    is_new = newer(row["latest"], row["current"])
                    row["status"] = ("update" if is_new else "current") if is_new is not None else "unknown"
                else:
                    rel = latest_release(http, e["repo"])
                    row["latest"], row["url"] = rel["tag"], rel["url"]
                    images = llama_images(docker)
                    stale = []
                    for img in images:
                        remote = registry_digest(http, e["image"], img["tag"])
                        img["newer"] = bool(remote and img["digest"] and remote != img["digest"])
                        if img["newer"]:
                            stale.append(img["tag"])
                        row["detail"].append(img)
                    builds = [i["build"] for i in images if i["build"]]
                    row["current"] = builds[0].split("-")[0] if builds else ""
                    row["status"] = "update" if stale else ("current" if images else "unknown")
                    lb = _BUILD_RE.match(row["latest"])
                    cb = _BUILD_RE.match(row["current"])
                    if lb and cb:
                        row["behind"] = int(lb.group(1)) - int(cb.group(1))
            except (httpx.HTTPError, ValueError, KeyError) as err:
                row["error"] = f"{type(err).__name__}: {err}"[:200]
            rows.append(row)
    finally:
        if own:
            http.close()
    return {"checked_at": time.time(), "engines": rows}


def cached() -> dict:
    """The last check, {} if none has run."""
    try:
        return json.loads(db.get_setting(_KEY, "") or "{}")
    except ValueError:
        return {}


def refresh(docker, force: bool = False) -> bool:
    """Run a check in the background if the last one is stale (or `force`). True if started.

    Never on the request path: GitHub and ghcr each take a round trip or two, and the Overview
    must not wait on either.
    """
    global _running
    last = cached()
    if not force and last and time.time() - float(last.get("checked_at") or 0) < CHECK_EVERY_S:
        return False
    with _lock:
        if _running:
            return False
        _running = True

    def _go() -> None:
        global _running
        try:
            db.set_setting(_KEY, json.dumps(check(docker)))
        except Exception:  # noqa: BLE001 - a failed check leaves the last good one in place
            pass
        finally:
            with _lock:
                _running = False

    threading.Thread(target=_go, daemon=True, name="engine-update-check").start()
    return True


def in_progress() -> bool:
    with _lock:
        return _running
