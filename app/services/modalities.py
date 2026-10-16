from __future__ import annotations

import asyncio
import configparser
import re
import shutil
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import docker
import httpx
from docker.errors import APIError, DockerException, NotFound

from .. import diagnose
from .. import ini
from .. import strata as strata_engine
from ..config import settings
from ..utils import docker_client, human_bytes, shard_key

from .. import gguf_meta


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
        from .. import gguf_meta
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
