from __future__ import annotations

import configparser
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


from .. import ini
from ..config import settings
from ..utils import human_bytes, shard_key

from .backends import _effective_container_names


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
    from .. import gguf_meta
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
    from .. import hw
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
