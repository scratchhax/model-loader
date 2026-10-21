"""A model downloaded from HF search can be chosen, and its first start uses its files (review
item 7). 0.2 lists an install only once setup has written its run config, which happens at a
start, so a fresh download was listed nowhere; and a selection without a folder makes setup
download the family's own model instead."""
from __future__ import annotations

import json

import pytest

from app import db, strata_env
from app import strata_models as sm
from tests.test_routes_smoke import web  # noqa: F401 - the fixture: real app, empty box

REPO = "SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF"
FILES = ["Swift-Q-IQ3_S-00001-of-00002.gguf", "Swift-Q-IQ3_S-00002-of-00002.gguf"]


@pytest.fixture
def box(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "d.db")
    db.init()
    folder = tmp_path / "models" / "strata-abliterated-IQ3_S"
    folder.mkdir(parents=True)
    sm.record_provenance("swift-iq3_s", REPO, "swift", "IQ3_S", "unsloth/whatever",
                         subdir="strata-abliterated-IQ3_S", files=FILES)
    return tmp_path / "models", folder


def test_listed_with_its_folder_once_every_file_is_there(box):
    models, folder = box
    got = sm.downloaded_installs(set(), models)
    assert got["swift-iq3_s"]["complete"] is False and got["swift-iq3_s"]["present"] == 0
    for f in FILES:
        (folder / f).write_bytes(b"x")
    (folder / "extra.gguf.download").write_bytes(b"")
    assert sm.downloaded_installs(set(), models)["swift-iq3_s"]["complete"] is False   # still fetching
    (folder / "extra.gguf.download").unlink()
    d = sm.downloaded_installs(set(), models)["swift-iq3_s"]
    assert d["complete"] and d["gguf_dir"] == "/gguf-root/strata-abliterated-IQ3_S"
    assert (d["family"], d["model"]) == ("swift", "IQ3_S") and d["about"]["repo"] == REPO


def test_gone_once_strata_has_set_it_up(box):
    models, _ = box
    assert sm.downloaded_installs({"swift-iq3_s"}, models) == {}


def test_selection_writes_the_folder_and_refuses_anything_shell_shaped(tmp_path):
    env = tmp_path / "strata.env"
    env.write_text("GPU=1\n")
    ok, _ = sm.set_selection(str(env), "swift", "IQ3_S", "/gguf-root/strata-abliterated-IQ3_S")
    assert ok and strata_env.read(env)["GGUF_DIR"] == "/gguf-root/strata-abliterated-IQ3_S"
    for fam, mdl, d in (("swift", "$(reboot)", ""), ("a;b", "M", ""),
                        ("swift", "M", "/etc/passwd"), ("swift", "M", "/gguf-root/../x")):
        assert sm.set_selection(str(env), fam, mdl, d)[0] is False, (fam, mdl, d)


def test_the_writer_quotes_what_sh_would_expand(tmp_path):
    env = tmp_path / "strata.env"
    env.write_text("")
    strata_env.update(env, {"A": "$(touch /tmp/x)", "B": "plain-1.0", "C": "it's"})
    text = env.read_text()
    assert "A='$(touch /tmp/x)'" in text and "B=plain-1.0" in text
    assert strata_env.read(env)["A"] == "$(touch /tmp/x)"


def test_the_strata_tab_offers_it(web, box, monkeypatch):
    models, folder = box
    for f in FILES:
        (folder / f).write_bytes(b"x")
    from app.config import settings
    monkeypatch.setattr(settings, "models_dir", models)
    r = web.get("/strata/models")
    assert r.status_code == 200 and "Not set up yet" in r.text
    assert 'name="gguf_dir" value="/gguf-root/strata-abliterated-IQ3_S"' in r.text


# ------------------------------------------------------------------ stranded: run config deleted by setup

def _tier(family, model, first, shards):
    return sm.Tier(family=family, model=model, family_title="", family_by="",
                   tag="" if family == "qwen" else f"{family}-", download_gb=0, ram_gb=0,
                   arena_gb=0, experimental=False, vision=True, shards=shards,
                   first_shard=first, hash_pinned=False)


def test_a_pack_whose_run_config_setup_deleted_is_offered_with_its_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "d.db")
    db.init()
    data = tmp_path / "data2"
    f = data / "Strata-data" / "models" / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S"
    f.mkdir(parents=True)
    for i in (1, 2):
        (f / f"Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{i}-of-00002.gguf").write_bytes(b"x")
    models = tmp_path / "models"
    models.mkdir()
    tiers = [_tier("qwen", "IQ3_S", "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf", 2),
             _tier("qwen", "IQ3_XXS", "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf", 2)]
    got = sm.stranded_installs(tiers, set(), data, models)
    assert list(got) == ["iq3_s"]
    assert got["iq3_s"]["gguf_dir"] == "/data/Strata-data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S"
    assert got["iq3_s"]["stranded"] and got["iq3_s"]["complete"]
    assert sm.stranded_installs(tiers, {"iq3_s"}, data, models) == {}       # set up: not stranded
    (f / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf").unlink()
    assert sm.stranded_installs(tiers, set(), data, models) == {}           # a shard missing: not offered
