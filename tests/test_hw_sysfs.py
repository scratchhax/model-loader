"""AMD telemetry from sysfs: the source that needs no GPU container shipping rocm-smi.

Since Strata moved to upstream's 0.2 image, no running container had rocm-smi while Strata held the
cards, and the GPU strip - including the engine switch's progress - went blank for the whole time.
"""
from __future__ import annotations

import os

from app import hw


def _card(root, n, pci, vendor="0x1002", device="0x7551", vram_total=34208743424, vram_used=0,
          busy=0, temp_m=0, power_uw=0, cap_uw=0, pwm=0, rpm=0, hwmon=True):
    """A /sys/class/drm/cardN whose device symlinks to a PCI address, as the kernel lays it out."""
    pdev = root / "devices" / pci
    pdev.mkdir(parents=True)
    (pdev / "vendor").write_text(vendor + "\n")
    (pdev / "device").write_text(device + "\n")
    if vram_total is not None:
        (pdev / "mem_info_vram_total").write_text(f"{vram_total}\n")
        (pdev / "mem_info_vram_used").write_text(f"{vram_used}\n")
    (pdev / "gpu_busy_percent").write_text(f"{busy}\n")
    if hwmon:
        h = pdev / "hwmon" / "hwmon3"
        h.mkdir(parents=True)
        for name, val in (("temp1_input", temp_m), ("power1_average", power_uw), ("power1_cap", cap_uw),
                          ("pwm1", pwm), ("fan1_input", rpm)):
            (h / name).write_text(f"{val}\n")
    drm = root / "drm" / f"card{n}"
    drm.mkdir(parents=True)
    os.symlink(pdev, drm / "device")


def _wheatley(tmp_path):
    # As measured: drm card0 is 0000:08:00.0 (the chipset link), card1 the Intel iGPU, card2 is
    # 0000:03:00.0 (the CPU's lanes). The engines number them by PCI address: 03:00.0 first.
    _card(tmp_path, 0, "0000:08:00.0", vram_used=10054754304, busy=4, temp_m=47000,
          power_uw=12000000, cap_uw=210000000, pwm=51, rpm=894)
    _card(tmp_path, 1, "0000:00:02.0", vendor="0x8086", device="0xa780", vram_total=None, hwmon=False)
    _card(tmp_path, 2, "0000:03:00.0", vram_used=8733859840, busy=6, temp_m=49000,
          power_uw=19000000, cap_uw=210000000, pwm=51, rpm=1066)
    (tmp_path / "drm" / "card0-DP-5").mkdir()                 # a connector, not a device
    return str(tmp_path / "drm")


def test_cards_in_pci_order_not_drm_order(tmp_path):
    g = hw._read_amd_sysfs(_wheatley(tmp_path))
    assert g.gpu_count == 2 and g.name == "2 × AMD Radeon AI PRO R9700"
    c0, c1 = g.cards
    assert (c0.index, c0.fan_rpm, round(c0.vram_used_gb, 1)) == (0, 1066.0, 8.1)    # 03:00.0
    assert (c1.index, c1.fan_rpm, round(c1.vram_used_gb, 1)) == (1, 894.0, 9.4)     # 08:00.0


def test_units(tmp_path):
    c = hw._read_amd_sysfs(_wheatley(tmp_path)).cards[0]
    assert c.vram_total_gb == 31.86 and c.util_pct == 6.0 and c.temp_c == 49.0
    assert (c.power_w, c.power_cap_w) == (19.0, 210.0)
    assert c.fan_pct == 20.0                                   # pwm 51 of 255


def test_no_amd_cards_is_none(tmp_path):
    _card(tmp_path, 0, "0000:00:02.0", vendor="0x8086", vram_total=None, hwmon=False)
    assert hw._read_amd_sysfs(str(tmp_path / "drm")) is None
    assert hw._read_amd_sysfs(str(tmp_path / "nowhere")) is None


def test_unknown_device_named_by_id_and_missing_hwmon_is_zero(tmp_path):
    _card(tmp_path, 0, "0000:05:00.0", device="0x744c", hwmon=False)
    c = hw._read_amd_sysfs(str(tmp_path / "drm")).cards[0]
    assert c.name == "AMD Radeon (0x744c)" and c.temp_c == 0.0 and c.power_w == 0.0


def test_strip_says_a_switch_is_running_even_without_readings(monkeypatch):
    from app.web import templates
    html = templates.get_template("_gpu_strip.html").render(
        request=None, cards=[], engines=[], engine_switch={"engine": "strata", "phase": "loading",
                                                            "msg": "strata up 42s, model not loaded yet"})
    assert "Switching to strata: loading the model" in html and "up 42s" in html
