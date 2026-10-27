"""NVIDIA GPU tenants: host-sensors' host-side nvidia-smi table, read back by the app.

On a box that pins one container per card, no container sees the whole picture: the llama
container pinned to GPU0 cannot see ComfyUI on GPU1, or that GPU1 exists. These pin the path
that closes that gap - the publisher's parsing, the tenant table built from it, the card list,
and the overview strip widening to the host's cards with the backend's own numbering re-keyed.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from app import gpu_procs as gp
from app import hw
from app.routers import overview

GLADOS = {
    "gpus": [
        {"index": 0, "bus": "00000000:01:00.0", "name": "NVIDIA GeForce RTX 5070",
         "mem_used_mib": 11716, "mem_total_mib": 12227, "util_pct": 37, "power_w": 120,
         "power_cap_w": 175, "temp_c": 57, "fan_pct": 30},
        {"index": 1, "bus": "00000000:08:00.0", "name": "NVIDIA GeForce RTX 5070",
         "mem_used_mib": 7828, "mem_total_mib": 12227, "util_pct": 0, "power_w": 1,
         "power_cap_w": 175, "temp_c": 52, "fan_pct": 0},
    ],
    "procs": [
        {"pid": 435428, "gpu": 0, "mem_mib": 11704, "sm_pct": 21},
        {"pid": 2886019, "gpu": 1, "mem_mib": 7796, "sm_pct": 0},
    ],
}
OWNERS = {435428: ("llama-gpu0", "llama-server"), 2886019: ("comfy", "python")}


def _load_publisher():
    path = Path(__file__).resolve().parents[1] / "contrib" / "host-sensors.py"
    spec = importlib.util.spec_from_file_location("host_sensors_contrib", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPublisher:
    def test_parses_cards_processes_and_pmon(self, monkeypatch):
        hs = _load_publisher()
        out = {
            "--query-gpu": ["0, 00000000:01:00.0, NVIDIA GeForce RTX 5070, 11716, 12227, 37, "
                            "120.5, 175.00, 57, [N/A]",
                            "1, 00000000:08:00.0, NVIDIA GeForce RTX 5070, 7828, 12227, 0, "
                            "1.04, 175.00, 52, 0"],
            "--query-compute-apps": ["435428, 11704, 00000000:01:00.0",
                                     "2886019, 7796, 00000000:08:00.0",
                                     "999, 10, 00000000:ff:00.0"],
            "pmon": ["# gpu         pid   type     sm    mem  command",
                     "# Idx           #    C/G      %      %  name",
                     "    0     435428     C     21      9  llama-server",
                     "    1    2886019     C      -      -  python"],
        }
        monkeypatch.setattr(hs, "_smi", lambda args: next(
            v for k, v in out.items() if args[0].startswith(k)))
        nv = hs.read_nvidia()
        assert [g["bus"] for g in nv["gpus"]] == ["00000000:01:00.0", "00000000:08:00.0"]
        assert nv["gpus"][0]["fan_pct"] == 0.0           # [N/A] is "not reported", not an error
        procs = {p["pid"]: p for p in nv["procs"]}
        assert set(procs) == {435428, 2886019}           # a bus no card has is dropped
        assert procs[435428] == {"pid": 435428, "gpu": 0, "mem_mib": 11704.0, "sm_pct": 21.0}
        assert procs[2886019]["gpu"] == 1 and procs[2886019]["sm_pct"] == 0.0

    def test_no_driver_is_none(self, monkeypatch):
        hs = _load_publisher()
        monkeypatch.setattr(hs, "_smi", lambda args: None)
        assert hs.read_nvidia() is None


class TestTenants:
    def _wire(self, monkeypatch, nvidia):
        monkeypatch.setattr(gp, "card_index_by_gpu_id", lambda *_a, **_k: {})
        monkeypatch.setattr(gp, "host_sensors", lambda: {"nvidia": nvidia} if nvidia else {})
        monkeypatch.setattr(gp, "_owner_map", lambda pids: {p: OWNERS[p] for p in pids if p in OWNERS})

    def test_comfy_is_a_foreign_tenant_on_the_hosts_second_card(self, monkeypatch):
        self._wire(monkeypatch, GLADOS)
        ts = {t.container: t for t in gp.tenants({"llama-gpu0", "llama-gpu1"})}
        comfy, llama = ts["comfy"], ts["llama-gpu0"]
        assert comfy.foreign and not llama.foreign
        assert set(comfy.per_card_gb) == {1} and round(comfy.per_card_gb[1], 2) == 7.61
        assert not comfy.working                          # resident, running nothing
        assert llama.working and llama.cu_pct == 21 and llama.util_src == "sm"
        assert gp.foreign_by_card(list(ts.values())) == {1: [("comfy", comfy.per_card_gb[1])]}

    def test_without_the_publisher_there_are_none(self, monkeypatch):
        self._wire(monkeypatch, None)
        assert gp.tenants({"llama-gpu0"}) == []


class TestHostCards:
    def test_cards_from_host_sensors(self, monkeypatch):
        monkeypatch.setattr(hw, "host_sensors", lambda: {"nvidia": GLADOS})
        cards = hw.host_nvidia_cards()
        assert [(c.index, c.bus) for c in cards] == [(0, "00000000:01:00.0"), (1, "00000000:08:00.0")]
        assert cards[1].vram_used_gb == 7.64 and cards[0].power_cap_w == 175

    def test_stale_or_amd_is_empty(self, monkeypatch):
        monkeypatch.setattr(hw, "host_sensors", lambda: {"stale": True})
        assert hw.host_nvidia_cards() == []


class TestIndexMap:
    def test_pinned_container_maps_to_its_host_card(self):
        host = [SimpleNamespace(index=0, bus="A"), SimpleNamespace(index=1, bus="B")]
        # llama-gpu1's own view: one card, numbered 0, which is the host's card 1.
        assert overview._host_index_map([SimpleNamespace(index=0, bus="B")], host) == {0: 1}

    def test_no_bus_no_map(self):
        host = [SimpleNamespace(index=0, bus="A")]
        assert overview._host_index_map([SimpleNamespace(index=0, bus="")], host) == {}


class TestStripWidening:
    def test_strip_shows_both_cards_and_comfy(self, monkeypatch):
        backend_card = hw.GpuCard(index=0, name="NVIDIA GeForce RTX 5070", util_pct=37,
                                  vram_used_gb=11.44, vram_total_gb=11.94, temp_c=57,
                                  power_w=120, bus="00000000:01:00.0")
        st = SimpleNamespace(ok=True, gpu=SimpleNamespace(cards=[backend_card]))
        monkeypatch.setattr(overview.services, "_effective_container_names", lambda: ["llama-gpu0"])
        monkeypatch.setattr(overview.hw, "stats_for", lambda n: st)
        monkeypatch.setattr(overview.hw, "history_for", lambda n: [])
        monkeypatch.setattr(overview.hw, "host_sensors", lambda: {"nvidia": GLADOS})
        monkeypatch.setattr(overview, "_solo_backend", lambda: "")
        monkeypatch.setattr(gp, "card_index_by_gpu_id", lambda *_a, **_k: {})
        monkeypatch.setattr(gp, "host_sensors", lambda: {"nvidia": GLADOS})
        monkeypatch.setattr(gp, "_owner_map", lambda pids: {p: OWNERS[p] for p in pids if p in OWNERS})
        monkeypatch.setattr(gp, "gpu_capable_containers", lambda *_a, **_k: [])
        monkeypatch.setattr(overview.services, "engine_for", lambda n: "llama")
        monkeypatch.setattr(overview.services, "gpu_engines", lambda: [], raising=False)
        ctx = overview._gpu_strip_context()
        assert [c.index for c in ctx["cards"]] == [0, 1]
        seg = ctx["card_breakdowns"][1]["segments"][0]
        assert seg["key"] == "foreign" and seg["label"] == "comfy"
        rows = {r["label"]: r for r in ctx["tenants"]}
        assert rows["comfy"]["per_card"][0] is None and rows["comfy"]["per_card"][1] > 7
