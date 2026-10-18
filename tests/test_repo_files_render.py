"""Render tests for the search repo page's fit chips (_repo_files.html)."""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).resolve().parents[1] / "app" / "templates"


def render(chips):
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)))
    groups = [{
        "shard_base": "IQ3_S/model", "shard_total": 2, "total_size_h": "78.1 GiB",
        "total_size": 0, "quant": "IQ3_S", "fit_chips": chips,
        "files": [{"path": "IQ3_S/model-00001-of-00002.gguf", "size": 1,
                   "size_h": "1 B", "quant": "IQ3_S"}],
    }]
    return env.get_template("_repo_files.html").render(
        repo_id="r/r", groups=groups, error=None, gated="", foreign="")


def test_streams_chip_renders_blue_with_the_combined_pool():
    out = render([{"name": "strata", "vram_gb": 63.7, "host_ram_gb": 120.0,
                   "verdict": "streams", "ratio_pct": 42}])
    assert "⇄" in out
    assert "strata (63.7G+120.0G RAM)" in out
    assert "bg-sky-100" in out
    assert "streams the routed experts from host RAM" in out


def test_oom_chip_is_unchanged():
    out = render([{"name": "llama-9700x2", "vram_gb": 63.7,
                   "verdict": "oom", "ratio_pct": 123}])
    assert "✗" in out
    assert "needs CPU offload" in out
    assert "llama-9700x2 (63.7G)" in out
