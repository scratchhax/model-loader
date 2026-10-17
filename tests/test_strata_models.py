"""Characterization tests for app/strata_models.py: catalog classification,
install planning (reuse by content hash), the pin file, and provenance.

No docker, no network: the docker seam (_docker_exec_json) and the Hub
(fetch_etags) are stubbed; everything else runs against tmp filesystems.
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import time

import httpx
import pytest

from app import db
from app import strata_models as sm


@pytest.fixture
def no_mmproj_cache():
    sm._MMPROJ_CACHE.clear()
    yield
    sm._MMPROJ_CACHE.clear()


@pytest.fixture
def rdb(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "db.sqlite")
    db.init()
    yield


def tier(family="qwen", model="IQ3_S", tag="", download_gb=55.0, vision=True,
         shards=2, first="Qwen3.8-Flash-Next-IQ3_S-00001-of-00002.gguf"):
    return sm.Tier(family=family, model=model, family_title="Qwen", family_by="ISTA",
                   tag=tag, download_gb=download_gb, ram_gb=64, arena_gb=40.0,
                   experimental=False, vision=vision, shards=shards, first_shard=first,
                   hash_pinned=False, canonical_repo="ISTA-DASLab/Qwen")


def f(path, size=27_500_000_000, sha=""):
    return {"path": path, "size": size, "sha": sha}


GOOD = [f("IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"),
        f("IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf")]


# ---------------------------------------------------------------- name plumbing

class TestNames:
    def test_repo_from_hf_strips_path_and_placeholders(self):
        assert sm._repo_from_hf("https://huggingface.co/A/B/tree/main/{file}") == "A/B"
        assert sm._repo_from_hf("https://huggingface.co/A/B/") == "A/B"
        assert sm._repo_from_hf("") == ""
        assert sm._repo_from_hf("https://example.com/A/B") == ""

    def test_parse_shard(self):
        assert sm.parse_shard("X-IQ3_S-00001-of-00002.gguf") == (1, 2)
        assert sm.parse_shard("x-00002-OF-00002.GGUF") == (2, 2)
        assert sm.parse_shard("X-IQ3_S.gguf") is None

    def test_tier_token_is_bounded(self):
        assert sm._tier_token_in("Foo-IQ3_S.gguf", "IQ3_S")
        assert sm._tier_token_in("Foo-Q2_0.gguf", "Q2_0")
        assert not sm._tier_token_in("Foo-IQ3_XXS.gguf", "IQ3_S")
        assert not sm._tier_token_in("Foo-UD-Q2_K_XL.gguf", "Q2_0")

    def test_strata_only_is_the_gsq_marker(self):
        assert sm.strata_only("Foo-GSQ-RCO-IQ3_S.gguf")
        assert not sm.strata_only("Foo-UD-Q4_K_XL.gguf")   # llama reads these fine
        assert not sm.strata_only("")


# ---------------------------------------------------------------- classify_repo

class TestClassifyRepo:
    def test_complete_set_matches_with_normalized_names(self, monkeypatch, no_mmproj_cache):
        monkeypatch.setattr(sm, "_docker_exec_json", lambda *a, **k: {"n": "mmproj-X.gguf"})
        out = sm.classify_repo(GOOD, [tier()])
        assert len(out) == 1
        m = out[0]
        assert [x["dst"] for x in m.files] == [
            "Qwen3.8-Flash-Next-IQ3_S-00001-of-00002.gguf",
            "Qwen3.8-Flash-Next-IQ3_S-00002-of-00002.gguf"]
        assert m.total_bytes == 2 * 27_500_000_000
        assert m.warnings == []

    def test_partial_shard_set_is_dropped_not_offered(self):
        assert sm.classify_repo(GOOD[:1], [tier()]) == []

    def test_wrong_shard_total_is_dropped(self):
        wrong = [f("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00003.gguf"),
                 f("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00003.gguf")]
        assert sm.classify_repo(wrong, [tier()]) == []

    def test_missing_gsq_or_arch_marker_is_not_our_tier(self):
        no_gsq = [x.replace("GSQ-RCO-", "") for x in
                  ["Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
                   "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"]]
        assert sm.classify_repo([f(p) for p in no_gsq], [tier()]) == []
        no_arch = [f(p.replace("Qwen3.8-Flash-Next", "SomeOtherModel")) for p in
                   ["Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
                    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"]]
        assert sm.classify_repo(no_arch, [tier()]) == []

    def test_hash_pinned_slots_are_not_mirror_targets(self):
        assert sm.classify_repo(GOOD, [tier(family="unsloth", tag="unsloth-")]) == []

    def test_empty_catalog_answers_empty(self):
        assert sm.classify_repo(GOOD, []) == []

    def test_size_sanity_warning(self):
        half = [f(p, size=14_000_000_000) for p in
                ["Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
                 "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"]]
        out = sm.classify_repo(half, [tier(download_gb=55.0)])
        assert len(out) == 1 and len(out[0].warnings) == 1

    def test_mmproj_smallest_for_vision_none_otherwise(self, monkeypatch, no_mmproj_cache):
        monkeypatch.setattr(sm, "_docker_exec_json", lambda *a, **k: {"n": "mmproj-X.gguf"})
        files = GOOD + [f("mmproj-BF16.gguf", size=1_200_000_000),
                        f("mmproj-Q8_0.gguf", size=600_000_000)]
        vis = sm.classify_repo(files, [tier(vision=True)])[0]
        assert vis.mmproj["src"] == "mmproj-Q8_0.gguf"
        assert vis.mmproj["dst"] == "mmproj-X.gguf"
        plain = sm.classify_repo(files, [tier(vision=False)])[0]
        assert plain.mmproj is None


# ---------------------------------------------------------------- slots & state

class TestFreeSlots:
    def test_occupied_tags_are_excluded(self):
        tiers = [tier(), tier(model="IQ4_XS", tag="swift-", first="s-00001-of-00002.gguf")]
        assert sm.free_slots({"swift-iq4_xs"}, tiers) == [tiers[0]]


class TestMissingSources:
    def test_image_paths_are_never_checked(self):
        assert sm._missing_sources({"args": ["/opt/strata/lib/setup.sh"]}) == []

    def test_retired_mount_name_is_missing(self):
        assert sm._missing_sources({"args": ["/gguf/models/x.gguf"]}) == ["/gguf/models/x.gguf"]

    def test_viewed_paths_check_the_local_side(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sm, "_PATH_VIEWS", (("/gguf-root/", str(tmp_path) + "/"),))
        ok = tmp_path / "x.gguf"
        ok.write_text("x")
        cfg = {"args": ["/gguf-root/x.gguf", "/gguf-root/gone.gguf"],
               "vision": {"mmproj": "/gguf-root/gone2.gguf"}}
        assert sm._missing_sources(cfg) == ["/gguf-root/gone.gguf", "/gguf-root/gone2.gguf"]


class TestInstalled:
    def test_config_is_the_authority_pack_is_reported_beside_it(self, tmp_path):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "strata-swift-iq3_s.json").write_text('{"args": []}')
        pack = tmp_path / "packs" / "swift-iq3_s"
        pack.mkdir(parents=True)
        (pack / "a.bin").write_bytes(b"12345")
        out = sm.installed(str(tmp_path))
        assert list(out) == ["swift-iq3_s"]
        assert out["swift-iq3_s"]["bytes"] == 5
        assert out["swift-iq3_s"]["pack"] == str(pack)
        assert out["swift-iq3_s"]["missing"] == []

    def test_config_without_pack_and_broken_json_still_list(self, tmp_path):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "strata-qwen-iq3_s.json").write_text("{not json")
        out = sm.installed(str(tmp_path))
        assert out["qwen-iq3_s"]["pack"] is None and out["qwen-iq3_s"]["bytes"] == 0

    def test_missing_data_dir_is_empty(self, tmp_path):
        assert sm.installed(str(tmp_path / "nope")) == {}


# ---------------------------------------------------------------- install planning

def _match(files, mmproj=None):
    t = tier()
    return sm.RepoMatch(tier=t, files=files, mmproj=mmproj)


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


class TestPlanInstall:
    def test_already_in_place_and_hash_matches_is_reuse(self, tmp_path):
        root = tmp_path / "models"
        sha = _write(root / "dest" / "a.gguf", b"payload")
        plan = sm.plan_install(_match([{"src": "r/a.gguf", "dst": "a.gguf",
                                        "size": 7, "sha": sha}]), str(root), "dest")
        assert plan["reuse"][0]["how"] == "already in place"
        assert plan["fetch"] == [] and plan["saved_bytes"] == 7

    def test_identical_elsewhere_is_reuse_by_hash_not_size(self, tmp_path):
        root = tmp_path / "models"
        sha = _write(root / "other" / "a.gguf", b"payload")
        plan = sm.plan_install(_match([{"src": "r/a.gguf", "dst": "a.gguf",
                                        "size": 7, "sha": sha}]), str(root), "dest")
        assert plan["reuse"][0]["how"] == "identical file already on disk"
        assert plan["fetch"] == []

    def test_size_collision_with_different_content_must_download(self, tmp_path):
        root = tmp_path / "models"
        _write(root / "other" / "a.gguf", b"XXXXXXX")   # same size, different bytes
        plan = sm.plan_install(_match([{"src": "r/a.gguf", "dst": "a.gguf",
                                        "size": 7, "sha": "a" * 64}]), str(root), "dest")
        assert plan["reuse"] == [] and len(plan["fetch"]) == 1

    def test_no_hash_offers_a_candidate_but_still_fetches(self, tmp_path):
        root = tmp_path / "models"
        _write(root / "other" / "a.gguf", b"payload")
        plan = sm.plan_install(_match([{"src": "r/a.gguf", "dst": "a.gguf", "size": 7}]),
                               str(root), "dest")
        assert plan["candidates"][0]["how"].startswith("same size, but the Hub gave no hash")
        assert len(plan["fetch"]) == 1          # a candidate is an offer, not a reuse

    def test_verify_off_reports_unverified_candidates(self, tmp_path):
        root = tmp_path / "models"
        _write(root / "other" / "a.gguf", b"payload")
        plan = sm.plan_install(_match([{"src": "r/a.gguf", "dst": "a.gguf",
                                        "size": 7, "sha": "a" * 64}]),
                               str(root), "dest", verify=False)
        assert plan["candidates"][0]["how"] == "same size, not yet verified"
        assert plan["reuse"] == [] and len(plan["fetch"]) == 1

    def test_mmproj_planned_too(self, tmp_path):
        root = tmp_path / "models"
        plan = sm.plan_install(_match([], mmproj={"src": "m", "dst": "mm.gguf", "size": 3}),
                               str(root), "dest")
        assert plan["fetch_bytes"] == 3 and plan["dest"] == str(root / "dest")


class TestMaterializeReuse:
    def test_hardlink_no_extra_disk(self, tmp_path):
        src = tmp_path / "src.gguf"
        _write(src, b"payload")
        plan = {"dest": str(tmp_path / "dest"),
                "reuse": [{"src": "s", "dst": "a.gguf", "size": 7, "from": str(src)}]}
        msgs = sm.materialize_reuse(plan)
        dst = tmp_path / "dest" / "a.gguf"
        assert dst.exists() and dst.stat().st_ino == src.stat().st_ino
        assert "no extra disk" in msgs[0]

    def test_existing_correct_file_is_skipped(self, tmp_path):
        src = tmp_path / "src.gguf"
        _write(src, b"payload")
        _write(tmp_path / "dest" / "a.gguf", b"payload")
        msgs = sm.materialize_reuse({"dest": str(tmp_path / "dest"), "reuse": [
            {"src": "s", "dst": "a.gguf", "size": 7, "from": str(src)}]})
        assert msgs == []

    def test_cross_filesystem_falls_back_to_copy(self, tmp_path, monkeypatch):
        src = tmp_path / "src.gguf"
        _write(src, b"payload")

        def no_link(a, b):
            raise OSError("different filesystems")
        monkeypatch.setattr("os.link", no_link)
        msgs = sm.materialize_reuse({"dest": str(tmp_path / "dest"), "reuse": [
            {"src": "s", "dst": "a.gguf", "size": 7, "from": str(src)}]})
        assert "copied" in msgs[0]
        assert (tmp_path / "dest" / "a.gguf").read_bytes() == b"payload"

    def test_failed_placement_is_a_message_not_an_exception(self, tmp_path, monkeypatch):
        def no_link(a, b):
            raise OSError("no link")
        monkeypatch.setattr("os.link", no_link)
        monkeypatch.setattr("shutil.copy2", lambda a, b: (_ for _ in ()).throw(OSError("ro")))
        msgs = sm.materialize_reuse({"dest": str(tmp_path / "dest"), "reuse": [
            {"src": "s", "dst": "a.gguf", "size": 7, "from": str(tmp_path / "gone.gguf")}]})
        assert "could not place" in msgs[0]


# ---------------------------------------------------------------- the pin file

ENV = """# strata pin file
GPU=1
export FAMILY=qwen
MODEL="Qwen3.8-Flash-Next-IQ3_S"
# keep this comment
# MODEL=commented-out
GGUF_DIR=IQ3_S
"""


class TestSelection:
    def test_read_strips_quotes_and_ignores_comments(self, tmp_path):
        p = tmp_path / "strata.env"
        p.write_text(ENV)
        sel = sm.read_selection(str(p))
        assert sel == {"FAMILY": "qwen", "MODEL": "Qwen3.8-Flash-Next-IQ3_S",
                       "GGUF_DIR": "IQ3_S"}

    def test_inline_comments_are_not_stripped_latent_quirk(self, tmp_path):
        # Pinned as-is: a shell would drop the trailing comment, this reader keeps it
        # inside the value - and its quote-strip then eats only the leading quote.
        # Harmless for files this app writes (set_selection writes bare KEY=value),
        # surprising for hand-edited ones.
        p = tmp_path / "strata.env"
        p.write_text('MODEL="M"   # note\n')
        assert sm.read_selection(str(p))["MODEL"] == 'M"   # note'

    def test_read_missing_file_is_all_empty(self, tmp_path):
        assert sm.read_selection(str(tmp_path / "nope")) == {
            "FAMILY": "", "MODEL": "", "GGUF_DIR": ""}

    def test_set_rewrites_only_the_three_keys(self, tmp_path):
        p = tmp_path / "strata.env"
        p.write_text(ENV)
        ok, msg = sm.set_selection(str(p), "swift", "M2", "IQ4")
        assert ok and "next start: swift / M2 from IQ4" == msg
        text = p.read_text()
        assert "GPU=1" in text
        assert "# keep this comment" in text
        assert "# MODEL=commented-out" in text
        assert sm.read_selection(str(p)) == {"FAMILY": "swift", "MODEL": "M2",
                                             "GGUF_DIR": "IQ4"}

    def test_set_appends_keys_that_were_absent(self, tmp_path):
        p = tmp_path / "strata.env"
        p.write_text("GPU=1\n")
        sm.set_selection(str(p), "qwen", "M", "")
        assert sm.read_selection(str(p)) == {"FAMILY": "qwen", "MODEL": "M", "GGUF_DIR": ""}

    def test_set_unreadable_file_fails_loudly(self, tmp_path):
        ok, msg = sm.set_selection(str(tmp_path / "nope"), "f", "m", "")
        assert ok is False and "cannot read" in msg


# ---------------------------------------------------------------- provenance

class TestProvenance:
    def test_borrowed_is_decided_at_record_time(self, rdb):
        sm.record_provenance("swift-iq3_s", "SC117/Mirror", "swift", "IQ3_S",
                             canonical_repo="ISTA-DASLab/Qwen")
        p = sm.provenance()
        assert p["swift-iq3_s"]["borrowed"] is True
        d = sm.describe_install("swift-iq3_s")
        assert d["owner"] == "SC117" and d["borrowed"] is True

    def test_own_repo_is_not_borrowed_case_insensitively(self, rdb):
        sm.record_provenance("unsloth-ud-iq4_xs", "unsloth/UD", "unsloth", "UD",
                             canonical_repo="Unsloth/UD")
        assert sm.provenance()["unsloth-ud-iq4_xs"]["borrowed"] is False

    def test_unknown_install_falls_back_to_the_slot_name(self, rdb):
        d = sm.describe_install("qwen-iq3_s")
        assert d["title"] == "qwen-iq3_s" and d["repo"] == "" and d["borrowed"] is False

    def test_corrupt_record_is_tolerated(self, rdb):
        db.set_setting("strata.install_provenance", "{broken")
        assert sm.provenance() == {}


# ---------------------------------------------------------------- hub etags

class TestFetchEtags:
    def test_linked_etag_is_the_content_hash_not_the_cdn_etag(self, monkeypatch):
        def h(req):
            return httpx.Response(200, headers={
                "x-linked-etag": "316b46f3" + "0" * 56, "etag": "076b2d0a" + "0" * 56})
        monkeypatch.setattr(httpx, "AsyncClient",
                            functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(h)))
        out = asyncio.run(sm.fetch_etags("A/B", ["x.gguf"]))
        assert out == {"x.gguf": "316b46f3" + "0" * 56}

    def test_cdn_only_or_malformed_headers_are_skipped(self, monkeypatch):
        def h(req):
            if "good" in req.url.path:
                return httpx.Response(200, headers={"x-linked-etag": '"' + "a" * 64 + '"'})
            if "cdn" in req.url.path:
                return httpx.Response(200, headers={"etag": "b" * 64})
            return httpx.Response(200, headers={"x-linked-etag": "nothex"})
        monkeypatch.setattr(httpx, "AsyncClient",
                            functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(h)))
        out = asyncio.run(sm.fetch_etags("A/B", ["good.gguf", "cdn.gguf", "bad.gguf"]))
        assert out == {"good.gguf": "a" * 64}

    def test_network_error_skips_the_path(self, monkeypatch):
        def h(req):
            raise httpx.ConnectError("down")
        monkeypatch.setattr(httpx, "AsyncClient",
                            functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(h)))
        assert asyncio.run(sm.fetch_etags("A/B", ["x"])) == {}
