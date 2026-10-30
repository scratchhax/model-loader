"""Characterization tests for app/gpu_procs.py: kernel-side GPU tenant accounting.

The parsing (node properties, topology), the pure tenant math (foreign-by-card,
last-seen, eviction totals) and the CU-per-card discovery run against tmp sysfs
trees; docker and the sampler thread stay out of it.
"""
from __future__ import annotations

import pytest

from app import gpu_procs as gp


PROPS = """
cpu_id 0
simd_count 128
simd_per_cu 2
location_id 3
"""


class TestNodeProp:
    def test_key_found(self):
        assert gp._node_prop(PROPS, "simd_count") == 128
        assert gp._node_prop(PROPS, "location_id") == 3

    def test_missing_key_is_zero(self):
        assert gp._node_prop(PROPS, "nope") == 0

    def test_non_numeric_value_is_zero(self):
        assert gp._node_prop("simd_count abc", "simd_count") == 0


class TestParseTopo:
    def test_ordered_by_location_id(self):
        # gpu 1 sits at location 6, gpu 2 at location 3: card order follows the bus.
        assert gp._parse_topo("1 6\n2 3") == {"2": 0, "1": 1}

    def test_noise_lines_are_skipped(self):
        assert gp._parse_topo("x y\n1 0\n\n1\n") == {"1": 0}

    def test_empty_is_empty(self):
        assert gp._parse_topo("") == {}


class TestCuPerCard:
    @pytest.fixture(autouse=True)
    def no_cache(self):
        # cu_per_card caches a real answer for the life of the process; these tests
        # need to see the discovery, not a sibling test's cached value.
        gp._CU_TOTAL = None
        yield
        gp._CU_TOTAL = None

    def test_local_sysfs_math(self, tmp_path, monkeypatch):
        node = tmp_path / "node2"
        node.mkdir()
        (node / "properties").write_text(PROPS)
        monkeypatch.setattr(gp, "_KFD_NODES", tmp_path)
        assert gp.cu_per_card(exec_candidates=()) == 64      # simd_count / simd_per_cu

    def test_unreadable_sysfs_is_zero_not_a_crash(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gp, "_KFD_NODES", tmp_path / "gone")
        assert gp.cu_per_card(exec_candidates=()) == 0


def _tenant(pid=1, comm="x", container="", backend="", per_card=None,
            cu=None, evicted=0):
    # Tenant.foreign is derived: no backend name means the app does not manage it.
    return gp.Tenant(pid=pid, comm=comm, container=container, backend=backend,
                     per_card_gb=per_card or {}, cu_by_card=cu or {},
                     cu_total=0, evicted_ms=evicted)


class TestTenantMath:
    def test_totals_and_peak(self):
        t = _tenant(per_card={0: 3.5, 1: 2.0}, cu={0: 12, 1: 40})
        assert t.total_gb == 5.5 and t.cu_peak == 40

    def test_holding_vram_is_not_working(self):
        # Chatterbox holds ~6 GB around the clock while doing nothing.
        assert _tenant(per_card={0: 6.0}).working is False
        assert _tenant(per_card={0: 6.0}, cu={0: 1}).working is True


class TestForeignByCard:
    def test_only_foreign_named_and_floor_applied(self):
        ours = _tenant(backend="llama", per_card={0: 20.0})
        ghost = _tenant(container="chatterbox", per_card={0: 5.88, 1: 0.004})
        out = gp.foreign_by_card([ours, ghost])
        assert out == {0: [("chatterbox", 5.88)]}   # 0.004 is below the 0.01 floor

    def test_sorted_by_card(self):
        t = _tenant(container="x", per_card={1: 2.0, 0: 1.0})
        out = gp.foreign_by_card([t])
        assert list(out) == [0, 1]


class TestNoteSeen:
    def test_stopped_tenants_keep_their_last_number(self):
        gp._LAST_SEEN.clear()
        try:
            gp.note_seen([_tenant(container="chatterbox", per_card={0: 5.88}),
                          _tenant(container="llama", backend="llama", per_card={0: 20.0}),
                          _tenant(container="empty", per_card={})])
            assert gp._LAST_SEEN == {"chatterbox": 5.88}
        finally:
            gp._LAST_SEEN.clear()


class TestEvictedMs:
    def test_sums_across_cards(self, tmp_path, monkeypatch):
        pid = tmp_path / "4242"
        for card, val in (("stats_0", 100), ("stats_1", 50)):
            d = pid / card
            d.mkdir(parents=True)
            (d / "evicted_ms").write_text(str(val))
        monkeypatch.setattr(gp, "_KFD_PROC", tmp_path)
        assert gp._evicted_ms(4242) == 150

    def test_unknown_pid_is_zero(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gp, "_KFD_PROC", tmp_path)
        assert gp._evicted_ms(99) == 0


class TestSlowStop:
    """docker-py stops reading a stop call at ~70 s while the daemon keeps stopping. A container
    granted 120 s (Strata, strata-ds, glados' comfy) made a switch report failure, start nothing,
    and then exit anyway (2026-10-30)."""

    class _C:
        def __init__(self, polls_until_stopped, err="UnixHTTPConnectionPool: Read timed out. (read timeout=70)"):
            self.status, self.attrs = "running", {"Config": {"StopTimeout": 120}}
            self._left, self._err = polls_until_stopped, err

        def stop(self):
            raise OSError(self._err)

        def reload(self):
            self._left -= 1
            if self._left <= 0:
                self.status = "exited"

    def _wire(self, monkeypatch, c):
        from types import SimpleNamespace
        from app import gpu_procs
        monkeypatch.setattr(gpu_procs, "docker_client",
                            lambda: SimpleNamespace(containers=SimpleNamespace(get=lambda n: c)))
        monkeypatch.setattr(gpu_procs.time, "sleep", lambda s: None)
        return gpu_procs

    def test_a_timed_out_stop_that_completes_is_a_stop(self, monkeypatch):
        gp = self._wire(monkeypatch, self._C(polls_until_stopped=3))
        ok, msg = gp.set_container_running("comfy", start=False)
        assert ok and "stopped" in msg

    def test_one_that_never_stops_fails_after_its_grace(self, monkeypatch):
        gp = self._wire(monkeypatch, self._C(polls_until_stopped=10**9))
        clock = iter(range(0, 10**6, 10))
        monkeypatch.setattr(gp.time, "time", lambda: float(next(clock)))
        ok, msg = gp.set_container_running("comfy", start=False)
        assert not ok and "150 s after the stop request" in msg

    def test_any_other_error_is_still_an_error(self, monkeypatch):
        gp = self._wire(monkeypatch, self._C(polls_until_stopped=1, err="permission denied"))
        ok, msg = gp.set_container_running("comfy", start=False)
        assert not ok and "permission denied" in msg
