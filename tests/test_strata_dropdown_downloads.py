"""A finished download Strata has not set up yet can be chosen from the engine row's dropdown.

The Strata tab's own row button could start one, but the dropdown above it (and on the Router
tab) listed installed packs only, so a model the app had just downloaded looked unselectable.
Its Set handler also split the tag to guess family and tier, with no folder: Gyro came out wrong,
and a download chosen that way would have been fetched again by Strata's setup.
"""
from __future__ import annotations

import pytest

from app import db, engine_controls
from app import strata_models as sm
from tests.test_routes_smoke import web  # noqa: F401 - the fixture: real app, empty box
from tests.test_strata_catalog_install import NAMES, unsloth

SUB = "strata-Qwen3.8-Flash-Next-GGUF-UD-Q4_K_XL"


def gyro():
    return sm.Tier(family="gyro", model="Gyro-S", family_title="Qwen3.8-Flash-Next (Gyro)",
                   family_by="Gyro", tag="gyro-", download_gb=58.5, ram_gb=20, arena_gb=25.8,
                   experimental=False, vision=True, shards=2, first_shard="g-00001-of-00002.gguf",
                   hash_pinned=True, canonical_repo="agentionai/x")


@pytest.fixture
def box(web, tmp_path, monkeypatch):  # noqa: F811
    from app.config import settings
    from app.routers import strata as rs
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "d.db")
    db.init()
    models = tmp_path / "models"
    (models / SUB).mkdir(parents=True)
    for n in NAMES:
        (models / SUB / n).write_bytes(b"x")
    monkeypatch.setattr(settings, "models_dir", models)
    monkeypatch.setattr(settings, "router_readonly", False, raising=False)
    env = tmp_path / "strata.env"
    env.write_text("GPU=0,1\nFAMILY=swift\nMODEL=IQ3_S\n", encoding="utf-8")
    monkeypatch.setattr(rs, "STRATA_ENV_FILE", str(env))
    monkeypatch.setattr(sm, "catalog", lambda *a, **k: [unsloth(), gyro()])
    monkeypatch.setattr(sm, "installed", lambda *a, **k: {})
    monkeypatch.setattr(sm, "stranded_installs", lambda *a, **k: {})
    marked = []
    monkeypatch.setattr(sm, "mark_in_background", lambda folder, pins: marked.append(folder))
    sm.record_provenance("unsloth-ud-q4_k_xl", "unsloth/Qwen3.8-Flash-Next-GGUF", "unsloth", "UD-Q4_K_XL",
                         canonical_repo="unsloth/Qwen3.8-Flash-Next-GGUF", subdir=SUB, files=NAMES)
    return web, env, models, marked


def test_the_dropdown_offers_a_finished_download(box):
    choices = dict(engine_controls.strata_choices())
    assert "unsloth-ud-q4_k_xl" in choices
    assert "not set up yet" in choices["unsloth-ud-q4_k_xl"]
    assert "Qwen3.8-Flash-Next-UD-Q4_K_XL" in choices["unsloth-ud-q4_k_xl"]


def test_a_running_download_is_not_offered(box):
    _, _, models, _ = box
    (models / SUB / (NAMES[3] + ".download")).write_bytes(b"")
    assert "unsloth-ud-q4_k_xl" not in dict(engine_controls.strata_choices())


def test_setting_it_writes_its_folder_and_leaves_finish_marks(box):
    web, env, models, marked = box
    r = web.post("/router/default", data={"engine": "strata", "model": "unsloth-ud-q4_k_xl"},
                 headers={"origin": "http://testserver"})
    assert "restart Strata" in r.text, r.text
    sel = sm.read_selection(str(env))
    assert sel == {"FAMILY": "unsloth", "MODEL": "UD-Q4_K_XL", "GGUF_DIR": f"/gguf-root/{SUB}"}
    assert marked == [models / SUB]
    assert "GPU=0,1" in env.read_text()


def test_setting_a_running_download_is_refused(box):
    web, env, models, _ = box
    (models / SUB / (NAMES[3] + ".download")).write_bytes(b"")
    r = web.post("/router/default", data={"engine": "strata", "model": "unsloth-ud-q4_k_xl"},
                 headers={"origin": "http://testserver"})
    assert "still downloading" in r.text
    assert sm.read_selection(str(env))["MODEL"] == "IQ3_S"


def test_an_installed_pack_resolves_through_the_catalog_not_its_tag(box):
    from app.routers import strata as rs
    assert rs.selection_for_tag("gyro-gyro-s") == ("gyro", "Gyro-S", "")
    assert rs.selection_for_tag("unsloth-ud-q4_k_xl") == ("unsloth", "UD-Q4_K_XL", f"/gguf-root/{SUB}")
