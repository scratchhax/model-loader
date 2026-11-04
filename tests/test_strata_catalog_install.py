"""Every tier in Strata's catalog installs from Model Loader, hash-pinned families included.

Unsloth's and Gyro's files are pinned by SHA-256 in Strata's setup, so they could not be renamed
in from a mirror, and the app used to offer no way in at all: the Unsloth repo showed as refused
and the catalog said "upstream only". They now install from the family's own repo, at the commit
setup pins, under the names it looks for, with the Hub's hashes checked against the pins first.
"""
from __future__ import annotations

import asyncio
import hashlib

import pytest

from app import db, downloader, hf
from app import strata_models as sm
from tests.test_routes_smoke import web  # noqa: F401 - the fixture: real app, empty box

REPO = "unsloth/Qwen3.8-Flash-Next-GGUF"
REV = "38bb39ee97821de2c9009abb7e93950eec396e66"
NAMES = [f"Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{i}-of-00004.gguf" for i in range(1, 5)]
PINS = {n: [100 + i, f"{i:064x}"] for i, n in enumerate(NAMES)}
MM_URL = "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f920/"


def unsloth(vision=True, pins=PINS):
    return sm.Tier(family="unsloth", model="UD-Q4_K_XL", family_title="Qwen3.8-Flash-Next (Unsloth)",
                   family_by="Unsloth", tag="unsloth-", download_gb=111.3, ram_gb=48, arena_gb=77.0,
                   experimental=False, vision=vision, shards=4, first_shard=NAMES[0], hash_pinned=True,
                   canonical_repo=REPO, shard_files=list(NAMES), pins=dict(pins),
                   hf_url=f"https://huggingface.co/{REPO}/resolve/{REV}/{{q}}/",
                   mmproj_url=MM_URL, mmproj_file="mmproj-Qwen3.8-Flash-Next-BF16.gguf")


# ---------------------------------------------------------------- the engine's recipe

def test_hf_location_reads_repo_commit_and_tier_folder():
    assert sm.hf_location(f"https://huggingface.co/{REPO}/resolve/{REV}/{{q}}/", "UD-Q4_K_XL") == \
        (REPO, REV, "UD-Q4_K_XL/")
    assert sm.hf_location("https://huggingface.co/a/b/resolve/abc/", "X") == ("a/b", "abc", "")
    assert sm.hf_location("https://example.com/a/b") is None
    assert sm.hf_location("") is None


def test_catalog_match_is_setups_own_files_at_its_commit():
    m = sm.catalog_match(unsloth())
    assert [f["dst"] for f in m.files] == NAMES                       # setup's names, no renaming
    assert {f["src"] for f in m.files} == {f"UD-Q4_K_XL/{n}" for n in NAMES}
    assert {(f["repo"], f["rev"]) for f in m.files} == {(REPO, REV)}  # the pinned commit, not main
    assert m.files[0]["size"] == 100 and m.files[0]["sha"] == f"{0:064x}" and m.files[0]["pinned"]
    # the vision encoder comes from ISTA-DASLab's repo for every family
    assert (m.mmproj["repo"], m.mmproj["rev"], m.mmproj["src"]) == \
        ("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "ed59f920", "mmproj-Qwen3.8-Flash-Next-BF16.gguf")
    assert sm.catalog_match(unsloth(vision=False)).mmproj is None


def test_no_recipe_no_match():
    t = unsloth()
    t.hf_url = ""
    assert sm.catalog_match(t) is None


def test_catalog_reads_shard_names_and_pins_from_the_engine(monkeypatch):
    raw = {"families": {"unsloth": {"title": "U", "tag": "unsloth-", "shards": 4, "hash_pinned": True,
                                    "hf": f"https://huggingface.co/{REPO}/resolve/{REV}/{{q}}/",
                                    "mmproj_hf": MM_URL, "mmproj": "mm.gguf"}},
           "models": {"UD-Q4_K_XL": {"download_gb": 111.3, "shards": 4, "families": ["unsloth"]}},
           "files": {"unsloth/UD-Q4_K_XL": NAMES[0]},
           "shard_files": {"unsloth/UD-Q4_K_XL": NAMES},
           "pins": {"unsloth/UD-Q4_K_XL": PINS}}
    monkeypatch.setattr(sm, "_docker_exec_json", lambda *a, **k: raw)
    (t,) = sm.catalog(force=True)
    assert t.shard_files == NAMES and t.pins == PINS and t.canonical_repo == REPO
    assert t.mmproj_file == "mm.gguf" and t.hash_pinned


# ---------------------------------------------------------------- search rows

def _repo_files(names=NAMES, folder="UD-Q4_K_XL/"):
    return [{"path": folder + n, "size": 7} for n in names] + [{"path": "mmproj-BF16.gguf", "size": 3}]


def test_the_publishers_own_repo_is_installable():
    (m,) = sm.classify_repo(_repo_files(), [unsloth()], repo_id=REPO)
    assert m.tier.family == "unsloth" and [f["dst"] for f in m.files] == NAMES
    assert m.files[0]["size"] == 100                                  # the pin's size wins


def test_a_pinned_family_never_matches_someone_elses_repo():
    assert sm.classify_repo(_repo_files(), [unsloth()], repo_id="someone/Qwen3.8-Flash-Next-GGUF") == []
    assert sm.classify_repo(_repo_files(), [unsloth()]) == []


def test_a_missing_shard_is_not_offered():
    assert sm.classify_repo(_repo_files(NAMES[:3]), [unsloth()], repo_id=REPO) == []


# ---------------------------------------------------------------- finish marks

def test_marks_are_written_only_for_the_pinned_hash(tmp_path):
    good, bad = b"right bytes", b"wrong bytes"
    pins = {"a.gguf": [len(good), hashlib.sha256(good).hexdigest()],
            "b.gguf": [len(bad), hashlib.sha256(b"something else!").hexdigest()],
            "c.gguf": [5, "0" * 64]}
    (tmp_path / "a.gguf").write_bytes(good)
    (tmp_path / "b.gguf").write_bytes(bad)
    (tmp_path / "c.gguf").write_bytes(b"size is off")
    assert sm.write_done_marks(tmp_path, pins) == ["a.gguf"]
    assert (tmp_path / "a.gguf.done").read_text() == f"sha256 {pins['a.gguf'][1]}"   # what verify_sha256 reads
    assert not (tmp_path / "b.gguf.done").exists() and not (tmp_path / "c.gguf.done").exists()
    assert sm.write_done_marks(tmp_path, pins) == []                                # already marked


# ---------------------------------------------------------------- downloads at a commit

def test_enqueue_fetches_the_commit_it_is_given(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "models_dir", tmp_path)

    async def noop(self, job):
        pass
    monkeypatch.setattr(downloader.DownloadManager, "_run", noop)

    async def scenario():
        mgr = downloader.DownloadManager()
        j = mgr.enqueue(repo_id=REPO, hf_path="UD-Q4_K_XL/x.gguf", filename="d/x.gguf",
                        total_bytes=1, revision=REV)
        assert j.url == f"{hf.HF_RESOLVE}/{REPO}/resolve/{REV}/UD-Q4_K_XL/x.gguf"
        j._task.cancel()
    asyncio.run(scenario())


# ---------------------------------------------------------------- the install route

@pytest.fixture
def route(web, tmp_path, monkeypatch):  # noqa: F811
    from app.config import settings
    from app.routers import strata as rs
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "d.db")
    db.init()
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    t = unsloth()
    monkeypatch.setattr(rs, "_strata_state", lambda: {"tiers": [t], "installed": {}})
    queued = []
    monkeypatch.setattr(rs.manager, "enqueue", lambda **k: queued.append(k))
    monkeypatch.setattr(rs.manager, "snapshot", lambda: [])
    etags = {f"UD-Q4_K_XL/{n}": PINS[n][1] for n in NAMES}

    async def fake_etags(repo, paths, revision="main"):
        return {p: etags[p] for p in paths if p in etags}

    async def fake_detail(repo, revision="main"):
        return hf.HfRepoDetail(id=repo, files=[hf.HfFile(path="mmproj-Qwen3.8-Flash-Next-BF16.gguf", size=900,
                                                         quant=None, shard_base="", shard_index=None,
                                                         shard_total=None)], readme_snippet=None)
    monkeypatch.setattr(sm, "fetch_etags", fake_etags)
    monkeypatch.setattr(hf, "repo_detail", fake_detail)
    return web, queued, etags


def _post(web):
    return web.post("/strata/install", data={"repo_id": REPO, "family": "unsloth", "model": "UD-Q4_K_XL"},
                    headers={"origin": "http://testserver"})


def test_install_queues_every_file_at_the_pinned_commit(route):
    web, queued, _ = route
    r = _post(web)
    assert r.status_code == 200 and "Queued" in r.text, r.text
    shards = [q for q in queued if q["repo_id"] == REPO]
    assert [q["filename"].rsplit("/", 1)[-1] for q in shards] == NAMES
    assert {q["revision"] for q in shards} == {REV}
    (mm,) = [q for q in queued if q["repo_id"] != REPO]
    assert mm["revision"] == "ed59f920" and mm["total_bytes"] == 900
    prov = sm.provenance()["unsloth-ud-q4_k_xl"]
    assert prov["repo"] == REPO and not prov["borrowed"] and prov["files"][:4] == NAMES


def test_a_hub_hash_that_disagrees_with_the_pin_is_refused(route):
    web, queued, etags = route
    etags[f"UD-Q4_K_XL/{NAMES[2]}"] = "f" * 64
    r = _post(web)
    assert "does not match the SHA-256 Strata pins" in r.text and queued == []


def test_a_pinned_family_refuses_a_mirror(route):
    web, queued, _ = route
    r = web.post("/strata/install", data={"repo_id": "someone/else", "family": "unsloth",
                                          "model": "UD-Q4_K_XL"}, headers={"origin": "http://testserver"})
    assert "hash-pinned" in r.text and queued == []


def test_not_enough_disk_is_refused_before_anything_is_queued(route, monkeypatch):
    web, queued, _ = route
    from app.routers import strata as rs
    monkeypatch.setattr(rs.shutil, "disk_usage", lambda p: type("U", (), {"free": 10})())
    r = _post(web)
    assert "free" in r.text and queued == []


def test_the_catalog_offers_install_for_every_tier_not_installed(web, monkeypatch):  # noqa: F811
    from app.routers import strata as rs
    t = unsloth()
    real = rs._strata_state

    def state():
        try:
            st = real()
        except Exception:  # noqa: BLE001 - no engine in tests
            st = {}
        st.update(tiers=[t], installed={}, downloaded={}, engine_up=True,
                  tier_by_tag={t.install_tag: t}, free={t.install_tag})
        return st
    monkeypatch.setattr(rs, "_strata_state", state)
    html = web.get("/strata/models").text
    assert f'name="repo_id" value="{REPO}"' in html and "upstream only" not in html
