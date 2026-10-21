"""strata.env has one reader and one writer, app/strata_env.py (review item 11). The GPU pin, the
model selection and the tunables all go through it."""
from __future__ import annotations

from app import strata as strata_engine
from app import strata_env
from app import strata_models as sm

FILE = """# why this file is not HIP_VISIBLE_DEVICES
GPU=1
GPUS=
export FAMILY=swift
MODEL=IQ3_S
GGUF_DIR=
PARALLEL=4
STRATA_SET="prefill=auto:32768 ple-io=direct"
"""


def _env(tmp_path, text=FILE):
    p = tmp_path / "strata.env"
    p.write_text(text, encoding="utf-8")
    return p


def test_read_is_how_sh_sources_it(tmp_path):
    p = _env(tmp_path, FILE + "MODEL=IQ4_XS   # changed my mind\n")
    env = strata_env.read(p)
    assert env["GPU"] == "1" and env["GPUS"] == "" and env["FAMILY"] == "swift"
    assert env["MODEL"] == "IQ4_XS"                       # the last assignment wins
    assert env["STRATA_SET"] == "prefill=auto:32768 ple-io=direct"
    assert strata_env.read(tmp_path / "missing") == {}


def test_update_rewrites_in_place_drops_repeats_and_appends(tmp_path):
    p = _env(tmp_path, FILE + "MODEL=stale\n")
    ok, _ = strata_env.update(p, {"MODEL": "UD-IQ4_XS", "NEW": "x", "STRATA_SET": "a=1 b=2"})
    assert ok
    lines = p.read_text().splitlines()
    assert lines[0] == "# why this file is not HIP_VISIBLE_DEVICES"
    assert lines[4] == "MODEL=UD-IQ4_XS" and "MODEL=stale" not in lines
    assert 'STRATA_SET="a=1 b=2"' in lines and lines[-1] == "NEW=x"


def test_update_missing_file_says_so(tmp_path):
    ok, why = strata_env.update(tmp_path / "nope", {"A": "1"})
    assert not ok and "cannot read" in why and "missing" in why


def test_pin_one_card_clears_the_split(tmp_path):
    p = _env(tmp_path, "# keep\nGPUS=0,1\n")
    ok, msg = strata_engine.set_pin("1", p)
    assert ok and "card 1" in msg
    assert strata_env.read(p) == {"GPUS": "", "GPU": "1"} and "# keep" in p.read_text()
    assert strata_engine.read_pin(p) == "1"


def test_pin_a_split_clears_the_single(tmp_path):
    p = _env(tmp_path)
    ok, _ = strata_engine.set_pin("1,0", p)
    assert ok and strata_engine.read_pin(p) == "1,0"
    assert strata_env.read(p)["GPU"] == ""
    assert sm.read_selection(str(p)) == {"FAMILY": "swift", "MODEL": "IQ3_S", "GGUF_DIR": ""}


def test_pin_refusals(tmp_path):
    p = _env(tmp_path)
    assert strata_engine.set_pin("x", p)[0] is False
    assert strata_engine.set_pin("1,1", p)[0] is False
    assert strata_engine.read_pin(tmp_path / "missing") == ""


def test_tunables_and_selection_share_the_file(tmp_path):
    p = _env(tmp_path)
    assert sm.env_settings(str(p)) == {"prefill": "auto:32768", "ple-io": "direct", "batch": "4"}
    ok, _ = sm.write_env_setting(str(p), "conversation-cache-slots", "16")
    assert ok and sm.env_settings(str(p))["conversation-cache-slots"] == "16"
    ok, _ = sm.write_env_setting(str(p), "batch", "2")
    assert ok and strata_env.read(p)["PARALLEL"] == "2"
    assert strata_engine.read_pin(p) == "1" and "# why this file" in p.read_text()
