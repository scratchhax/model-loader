from __future__ import annotations

import pytest
from docker.errors import NotFound
from starlette.testclient import TestClient

from app import db, hf, hw, ini
from app import utils as app_utils
from app.config import settings
from app.main import app


class FakeContainers:
    def list(self, all=False, filters=None, **kw):
        return []

    def get(self, name):
        raise NotFound("no such container")


class FakeClient:
    def __init__(self):
        self.containers = FakeContainers()

    def ping(self):
        return True


@pytest.fixture
def web(tmp_path, monkeypatch):
    """Every GET route against an empty box: no containers, no models, fresh db.

    Docker is stubbed at from_env (the one seam every module shares), the hw sampler thread is
    switched off, and the three HF network calls the search pages make are stubbed to canned
    answers. Startup still runs for real: db.init, router tables, prompt seeds.
    """
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(settings, "models_dir", models)
    monkeypatch.setattr(settings, "models_ini_path", models / "models.ini")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(db, "_DB_PATH", None)
    fake = FakeClient()
    monkeypatch.setattr(app_utils.docker, "from_env", lambda *a, **k: fake)
    monkeypatch.setattr(app_utils, "_DOCKER_CLIENT", None)
    monkeypatch.setattr(hw, "start_sampler", lambda: None)
    # the startup reconcile thread outlives the fixture; left running it reaches db after the
    # tmp data_dir is torn down and the restored /data default raises in a daemon thread
    monkeypatch.setattr("app.main._reconcile_strata_config", lambda: None)

    async def one_result(query="", limit=30, sort="downloads"):
        return [hf.HfModel(id="o/m", downloads=1, likes=0, last_modified="",
                           pipeline_tag=None, tags=["gguf"], gguf_count=1)]

    async def no_avatars(owners):
        return {}

    async def one_detail(repo_id, revision="main"):
        return hf.HfRepoDetail(id=repo_id, files=[hf.HfFile(
            path="m-Q4_K_M.gguf", size=1000, quant="Q4_K_M",
            shard_base="m", shard_index=None, shard_total=None)], readme_snippet=None)

    monkeypatch.setattr(hf, "search_models", one_result)
    monkeypatch.setattr(hf, "owner_avatars", no_avatars)
    monkeypatch.setattr(hf, "repo_detail", one_detail)
    with TestClient(app) as c:
        yield c


PAGES = [
    ("/", "Model Loader"),
    ("/models", "Models directory"),
    ("/settings", "Settings"),
    ("/search", "Search Hugging Face"),
    ("/downloads", "Downloads"),
    ("/config", "models.ini"),
    ("/containers", "llama.cpp containers"),
    ("/prompts", "Prompt library"),
    ("/benchmark", "Model Loader"),
]


@pytest.mark.parametrize("url,marker", PAGES)
def test_pages_render_on_an_empty_box(web, url, marker):
    r = web.get(url)
    assert r.status_code == 200
    assert marker in r.text


@pytest.mark.parametrize("url", ["/hero", "/gpu-strip", "/power", "/hoststats"])
def test_polling_partials_render(web, url):
    assert web.get(url).status_code == 200


def test_v2_redirects_to_dashboard_permanently(web):
    r = web.get("/v2", allow_redirects=False)
    assert r.status_code == 308 and r.headers["location"] == "/"


def test_palette_json(web):
    r = web.get("/palette.json")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_classic_dashboard_is_gone(web):
    assert web.get("/classic").status_code == 404


def test_model_detail_unknown_file_is_404(web):
    assert web.get("/model/zzz-not-here.gguf").status_code == 404


def test_search_results_use_stubbed_hub(web):
    r = web.get("/search/results", params={"q": "deep"})
    assert r.status_code == 200 and "o/m" in r.text


def test_search_repo_page_groups_stubbed_files(web):
    r = web.get("/search/repo/o/r")
    assert r.status_code == 200 and "m-Q4_K_M.gguf" in r.text


def test_config_section_edit_redirects_without_hx(web):
    r = web.get("/config/section/voice/edit", allow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/config?edit=voice")


def test_config_section_edit_unknown_is_404(web):
    r = web.get("/config/section/ghost/edit", headers={"HX-Request": "true"})
    assert r.status_code == 404


def test_config_section_edit_form_for_seeded_section(web):
    ini.upsert_section("voice", {"model": "/models/voice/x.gguf", "alias": "voice"}, "")
    r = web.get("/config/section/voice/edit", headers={"HX-Request": "true"})
    assert r.status_code == 200 and "<form" in r.text


def test_autoconfig_panel_without_gguf(web):
    ini.upsert_section("voice", {"model": "/models/voice/x.gguf"}, "")
    r = web.get("/config/section/voice/autoconfig")
    assert r.status_code == 200 and "No GGUF found" in r.text


def test_prompts_body_roundtrip_and_404(web):
    pid = db.add_prompt("smoke", "say hi")
    r = web.get(f"/prompts/{pid}/body")
    assert r.status_code == 200 and r.text == "say hi"
    assert web.get("/prompts/999999/body").status_code == 404


def test_container_subroutes_degrade_without_docker(web):
    assert "unknown backend" in web.get("/containers/nope/dashboard").text
    assert "container not found" in web.get("/containers/nope/logs").text
    assert web.get("/containers/nope/diagnose").status_code == 200


def test_engines_switch_reports_no_switch(web):
    r = web.get("/engines/switch")
    assert r.status_code == 200 and r.json() == {}


def test_downloads_rows_empty(web):
    assert web.get("/downloads/rows").status_code == 200


def test_clear_finished_post_works_with_no_jobs(web):
    assert web.post("/downloads/clear").status_code == 200


def test_readonly_guard_blocks_posts_but_not_gets(web, monkeypatch):
    monkeypatch.setattr(settings, "router_readonly", True)
    r = web.post("/downloads/clear")
    assert r.status_code == 403 and "read-only" in r.json()["detail"]
    assert web.get("/").status_code == 200
