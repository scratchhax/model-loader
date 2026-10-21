"""strata.env: the file Strata's entry point sources at every start. One reader, one writer.

Three things live in it, each set from a different page: the GPU pin (GPU / GPUS, Containers), the
model selection (FAMILY / MODEL / GGUF_DIR, Strata tab) and the launcher's tunables (PARALLEL and
STRATA_SET, Strata tab). Each used to parse and rewrite the file its own way. They all go through
here now, so the rules are stated once:

  * it is read the way sh reads it: the last assignment of a key wins, `export` is allowed, a
    quoted value keeps its spaces, and an unquoted value ends at whitespace or a `#` comment;
  * a write changes only the keys it names. A key's first assignment is rewritten in place, any
    later assignment of it is dropped (sh would take the last one, which would undo the write),
    a key not there yet is appended, and every other line, comments included, is kept as it is.
    The comments are the file's record of why it says what it says;
  * the write is atomic (fileio.atomic_replace), because the entry point may be sourcing it.

An EMPTY value is written, not deleted: sourcing an empty assignment is what clears a variable
compose baked into the container, and deleting the line clears nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import fileio

_ASSIGN_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def _parse_value(raw: str) -> str:
    raw = raw.strip()
    if raw[:1] in ("'", '"'):
        end = raw.find(raw[0], 1)
        return raw[1:end] if end > 0 else raw[1:]
    return re.split(r"\s+#|\s", raw, maxsplit=1)[0] if raw else ""


def key_of(line: str) -> str:
    """The variable a line assigns, or "" for a comment, a blank or anything else."""
    if line.lstrip().startswith("#"):
        return ""
    m = _ASSIGN_RE.match(line)
    return m.group(1) if m else ""


def read(path: str | Path) -> dict[str, str]:
    """{key: value} as sourcing the file would leave them. {} when it cannot be read."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return {}
    out: dict[str, str] = {}
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = _ASSIGN_RE.match(line)
        if m:
            out[m.group(1)] = _parse_value(m.group(2))
    return out


_PLAIN_RE = re.compile(r"^[A-Za-z0-9_./:,=+@%-]*$")


def _format(key: str, value: str) -> str:
    """KEY=value; "value" when only spaces need quoting (the file's own style); otherwise
    single-quoted. The file is SOURCED by sh, so a value like $(...) or `...` would run unquoted
    or in double quotes; inside single quotes sh expands nothing."""
    if _PLAIN_RE.match(value):
        return f"{key}={value}"
    if _PLAIN_RE.match(value.replace(" ", "")):
        return f'{key}="{value}"'
    return f"{key}='" + value.replace("'", "'\\''") + "'"


def update(path: str | Path, values: dict[str, str]) -> tuple[bool, str]:
    """Set each of `values` in the file, keeping everything else. (ok, "" or the reason)."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False, f"cannot read {p}: it is missing (is ./strata mounted into this container?)"
    except OSError as e:
        return False, f"cannot read {p}: {e}"
    out: list[str] = []
    written: set[str] = set()
    for line in text.splitlines():
        key = key_of(line)
        if key in values:
            if key not in written:
                out.append(_format(key, values[key]))
                written.add(key)
            continue
        out.append(line)
    out += [_format(k, v) for k, v in values.items() if k not in written]
    try:
        fileio.atomic_replace(p, "\n".join(out) + "\n")
    except OSError as e:
        return False, f"cannot write {p}: {e}"
    return True, ""
