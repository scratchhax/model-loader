from __future__ import annotations

import asyncio
import functools

import httpx
import pytest

from app import hf


class TestPureHelpers:
    def test_infer_quant_uppercases_and_misses(self):
        assert hf.infer_quant("Model-Q4_K_M.gguf") == "Q4_K_M"
        assert hf.infer_quant("model-f16.gguf") == "F16"
        assert hf.infer_quant("README.md") is None

    def test_support_files(self):
        assert hf._is_support_file("vision.mmproj")
        assert hf._is_support_file("vision-mmproj.gguf")
        assert hf._is_support_file("chat_template.jinja")
        assert hf._is_support_file("tokenizer.model")
        assert not hf._is_support_file("model-Q4_K_M.gguf")

    def test_match_repo_path_exact_then_basename_then_unique(self):
        meta = {"Qwen/Qwen-Q4.gguf": {}, "top.gguf": {},
                "a/x.gguf": {}, "b/x.gguf": {}}
        assert hf._match_repo_path("Qwen/Qwen-Q4.gguf", meta) == "Qwen/Qwen-Q4.gguf"
        assert hf._match_repo_path("mydir/top.gguf", meta) == "top.gguf"
        assert hf._match_repo_path("mine/x.gguf", meta) == ""  # ambiguous
        assert hf._match_repo_path("mine/zzz.gguf", meta) == ""


def client_with(handler, monkeypatch):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx, "AsyncClient",
                        functools.partial(httpx.AsyncClient, transport=transport))
    monkeypatch.setattr(hf, "get_token", lambda: "")


class TestSearchModels:
    def test_params_and_response_mapping(self, monkeypatch):
        seen = {}

        def handler(request):
            seen.update(dict(request.url.params))
            return httpx.Response(200, json=[
                {"modelId": "o/m1", "downloads": "100", "likes": 5,
                 "lastModified": "2026-01-01", "tags": ["gguf"],
                 "siblings": [{"rfilename": "a.gguf"}, {"rfilename": "b.txt"}]},
                {"id": "o/m2"},
            ])

        client_with(handler, monkeypatch)
        out = asyncio.run(hf.search_models("qwen", limit=7))
        assert seen["filter"] == "gguf" and seen["search"] == "qwen"
        assert seen["limit"] == "7" and seen["direction"] == "-1"
        assert out[0].id == "o/m1"  # modelId fallback
        assert out[0].downloads == 100 and out[0].gguf_count == 1
        assert out[1].id == "o/m2" and out[1].gguf_count is None
        assert out[1].downloads == 0

    def test_empty_query_omits_search_param(self, monkeypatch):
        seen = {}

        def handler(request):
            seen.update(dict(request.url.params))
            return httpx.Response(200, json=[])

        client_with(handler, monkeypatch)
        asyncio.run(hf.search_models(""))
        assert "search" not in seen


class TestRepoDetail:
    def test_pagination_filtering_and_lfs_size(self, monkeypatch):
        calls = []

        def handler(request):
            calls.append(str(request.url))
            if request.url.params.get("cursor"):
                return httpx.Response(200, json=[
                    {"type": "file", "path": "README.md", "size": 10},
                    {"type": "file", "path": "m.chat_template.jinja", "size": 5},
                    {"type": "file", "path": "m-Q4_K_M-00001-of-00002.gguf", "size": 7},
                ])
            return httpx.Response(
                200,
                json=[{"type": "file", "path": "m-Q4.gguf", "lfs": {"size": 1000}},
                      {"type": "dir", "path": "q8_0"}],
                headers={"link": ('<https://huggingface.co/api/models/o/r/tree/main'
                                  '?cursor=abc>; rel="next"')})

        client_with(handler, monkeypatch)
        detail = asyncio.run(hf.repo_detail("o/r"))
        assert len(calls) == 2  # followed the cursor to the end
        paths = [f.path for f in detail.files]
        assert paths == ["m-Q4.gguf", "m-Q4_K_M-00001-of-00002.gguf",
                         "m.chat_template.jinja"]  # ggufs first, then support
        assert detail.files[0].size == 1000  # lfs.size preferred
        assert detail.files[0].quant == "Q4"
        q8 = detail.files[1]
        assert q8.quant == "Q4_K_M" and q8.shard_index == 1 and q8.shard_total == 2
