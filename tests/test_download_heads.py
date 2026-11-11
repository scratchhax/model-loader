"""Which draft head POST /download queues beside a main GGUF.

A head is a small companion (60 MB to ~1.4 GB). Some repos use `-MTP` in the names of full quants
that carry embedded MTP layers; queuing one of those as a "head" pulled a 101 GB file into another
quant's folder.
"""
from __future__ import annotations

import asyncio

import pytest

from app import hf
from app.routers import downloads


def _file(path: str, size: int) -> hf.HfFile:
    return hf.HfFile(path=path, size=size, quant=hf.infer_quant(path),
                     shard_base=path[:-5], shard_index=None, shard_total=None)


def _queue(monkeypatch, repo_id: str, files: list[hf.HfFile], main: str, size: int = 0) -> list[str]:
    """Run download_single against a stubbed repo; return the hf_paths it enqueued, in order."""
    queued: list[str] = []

    async def detail(rid, revision="main"):
        return hf.HfRepoDetail(id=rid, files=files, readme_snippet=None)

    async def not_foreign(rid, path):
        return ""

    monkeypatch.setattr(hf, "repo_detail", detail)
    monkeypatch.setattr(downloads, "_foreign_repo", not_foreign)
    monkeypatch.setattr(downloads.manager, "enqueue",
                        lambda repo_id, hf_path, filename, total_bytes: queued.append(hf_path))
    asyncio.run(downloads.download_single(repo_id=repo_id, path=main, size=size))
    return queued


SYNDICATE_REPO = "SyndicateLabs/GLM-5.3-Flash-CYBERSECURITY-GGUF"
SYNDICATE = [
    _file("GLM-5.3-Flash-CYBER-IQ1_M-MTP.gguf", 101208277792),
    _file("GLM-5.3-Flash-CYBER-IQ2_M-MTP.gguf", 121865093920),
    _file("GLM-5.3-Flash-CYBER-IQ2_M.gguf", 115537726208),
    _file("GLM-5.3-Flash-CYBER-IQ3_M-MTP.gguf", 154853950240),
    _file("GLM-5.3-Flash-CYBER-IQ3_M.gguf", 151263365888),
    _file("GLM-5.3-Flash-CYBER-IQ4_XS-MTP.gguf", 175981445920),
    _file("GLM-5.3-Flash-CYBER-Q4_K_M-MTP.gguf", 195512877856),
]


@pytest.mark.parametrize("main", [f.path for f in SYNDICATE])
@pytest.mark.parametrize("pass_size", [True, False])
def test_mtp_quants_of_the_same_model_are_not_heads(monkeypatch, main, pass_size):
    size = next(f.size for f in SYNDICATE if f.path == main) if pass_size else 0
    assert _queue(monkeypatch, SYNDICATE_REPO, SYNDICATE, main, size) == [main]


def test_the_2026_11_11_case(monkeypatch):
    main = "GLM-5.3-Flash-CYBER-IQ3_M.gguf"
    queued = _queue(monkeypatch, SYNDICATE_REPO, SYNDICATE, main, 151263365888)
    assert "GLM-5.3-Flash-CYBER-IQ1_M-MTP.gguf" not in queued
    assert queued == [main]


# The real heads on wheatley, with their target's size.
@pytest.mark.parametrize("main,main_size,head,head_size", [
    ("Qwen3.8-27B-UD-Q4_K_M.gguf", 16464440224, "mtp-Qwen3.8-27B-Q4_0.gguf", 1369590656),
    ("gemma-4-31B-it-Q6_K.gguf", 25201486784, "mtp-gemma-4-31B-it.gguf", 514687104),
    ("gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf", 4215695776, "mtp-gemma-4-E4B-it.gguf", 59678016),
])
def test_real_small_heads_are_still_queued(monkeypatch, main, main_size, head, head_size):
    files = [_file(main, main_size), _file(head, head_size)]
    assert _queue(monkeypatch, "o/m", files, main, main_size) == [main, head]


def test_only_the_smallest_head_is_taken(monkeypatch):
    main = "Qwen3.8-27B-Q4_K_M.gguf"
    files = [_file(main, 16_000_000_000),
             _file("mtp-Qwen3.8-27B-BF16.gguf", 170_000_000),
             _file("mtp-Qwen3.8-27B-Q8_0.gguf", 90_000_000),
             _file("mtp-Qwen3.8-27B-Q4_0.gguf", 60_000_000)]
    assert _queue(monkeypatch, "o/m", files, main) == [main, "mtp-Qwen3.8-27B-Q4_0.gguf"]


def test_mmproj_still_rides_along(monkeypatch):
    main = "gemma-4-31B-it-Q6_K.gguf"
    files = [_file(main, 25201486784), _file("mmproj-F16.gguf", 1198957024),
             _file("mtp-gemma-4-31B-it.gguf", 514687104)]
    assert _queue(monkeypatch, "o/m", files, main) == [main, "mmproj-F16.gguf",
                                                       "mtp-gemma-4-31B-it.gguf"]


class TestDraftHead:
    def test_over_the_absolute_cap(self):
        files = [_file("mtp-Big-Q8_0.gguf", 3 * 1024 ** 3)]
        assert downloads._draft_head(files, "Big-Q4_K_M.gguf", 400 * 1024 ** 3) is None

    def test_too_large_a_share_of_a_small_model(self):
        files = [_file("mtp-Tiny.gguf", 900_000_000)]
        assert downloads._draft_head(files, "Tiny-Q4_K_M.gguf", 2_000_000_000) is None

    def test_unknown_size_is_not_a_head(self):
        files = [_file("mtp-Model.gguf", 0)]
        assert downloads._draft_head(files, "Model-Q4_K_M.gguf", 10_000_000_000) is None

    def test_sibling_quant_is_rejected_on_name_alone(self):
        # Tiny enough to pass both size bounds; the name still says it is the model itself.
        files = [_file("Model-IQ1_M-MTP.gguf", 100_000_000)]
        assert downloads._draft_head(files, "Model-Q4_K_M.gguf", 0) is None

    def test_sibling_quant_names(self):
        assert downloads._is_sibling_quant("GLM-5.3-Flash-CYBER-IQ1_M-MTP.gguf",
                                           "GLM-5.3-Flash-CYBER-IQ3_M.gguf")
        assert downloads._is_sibling_quant("GLM-5.3-Flash-CYBER-IQ3_M-MTP.gguf",
                                           "GLM-5.3-Flash-CYBER-IQ3_M.gguf")
        assert not downloads._is_sibling_quant("mtp-Qwen3.8-27B-Q4_0.gguf",
                                               "Qwen3.8-27B-Q4_K_M.gguf")
        assert not downloads._is_sibling_quant("Other-Model-IQ1_M-MTP.gguf",
                                               "GLM-5.3-Flash-CYBER-IQ3_M.gguf")
