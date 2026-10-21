"""A Strata model in a borrowed catalog slot is served under the slot's name; this app says what it is.

SC117's abliterated IQ3_S was installed into Strata's free Swift slot. 0.2's setup rebuilds model_name
as <family>-<tier> on every start, so Strata serves it as "swift-1.5-iq3_s" - which reads as a different
model. The install record (where it came from) is what names it.
"""
from __future__ import annotations

import asyncio
import json

from app import router_api as ra
from app import strata_models

ABOUT = {"repo": "SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF", "owner": "SC117",
         "title": "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF", "model": "IQ3_S", "borrowed": True}


def test_label_is_the_repo_model_and_tier():
    assert strata_models.install_label(ABOUT) == "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S"
    assert strata_models.install_label({"repo": "", "title": "swift-iq3_s"}) == ""
    # a repo name that already ends in its tier is not doubled
    assert strata_models.install_label({"repo": "a/M-IQ3_S", "title": "M-IQ3_S", "model": "IQ3_S"}) == "M-IQ3_S"


def test_served_name_maps_back_to_the_install(tmp_path, monkeypatch):
    root = tmp_path / "data2"
    (root / "install").mkdir(parents=True)
    (root / "Strata-data" / "packs").mkdir(parents=True)
    (root / "install" / "strata-swift-iq3_s.json").write_text(json.dumps({"model_name": "swift-1.5-iq3_s", "args": []}))
    (root / "install" / "strata-unsloth-ud-iq4_xs.json").write_text(
        json.dumps({"model_name": "qwen3.8-flash-next-unsloth-ud-iq4_xs", "args": []}))
    monkeypatch.setattr(strata_models, "provenance", lambda: {
        "swift-iq3_s": {"repo": ABOUT["repo"], "family": "swift", "model": "IQ3_S", "borrowed": True}})
    got = strata_models.served_labels(str(root))
    assert got == {"swift-1.5-iq3_s": {"tag": "swift-iq3_s", "repo": ABOUT["repo"], "owner": "SC117",
                                        "label": "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S", "borrowed": True}}


def test_router_offers_and_routes_the_real_name(monkeypatch):
    monkeypatch.setattr(ra, "_backends", lambda include_stopped=False: [
        {"name": "strata", "engine": "strata", "running": True, "url": "http://strata:8080",
         "models": [], "headers": {}}])

    async def models_of(client, b):
        return ["swift-1.5-iq3_s"]

    async def ready(client, b):
        return True
    monkeypatch.setattr(ra, "_models_of", models_of)
    monkeypatch.setattr(ra, "_ready", ready)
    monkeypatch.setattr(strata_models, "served_labels",
                        lambda data_dir=None: {"swift-1.5-iq3_s": {"label": "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S"}})
    bs, by_model = asyncio.run(ra._live_map())
    assert by_model["Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S"]["name"] == "strata"
    ids = [d["id"] for d in asyncio.run(ra.list_models())["data"]]
    assert "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S" in ids and "swift-1.5-iq3_s" in ids
