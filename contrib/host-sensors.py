#!/usr/bin/env python3
"""Publish the host sensors a container cannot read, into a file Model Loader already mounts.

Two things are invisible from inside the app container:

  * CPU package power. RAPL's energy counter is root-only (-r-------- on energy_uj, a
    side-channel hardening measure) and /sys/devices/virtual/powercap is not exposed in a
    container's sysfs namespace at all - mounting /sys does not help, which was worth proving
    before writing this.
  * CPU and NVMe temperatures are readable in-container, but are gathered here anyway so the
    app has one consistent source with one timestamp.

GPU power and temperature are NOT here: the llama container reads those through rocm-smi, and
duplicating them would invite two numbers that disagree.

Writes /home/jason/ai-lab/model_loader_data/host_sensors.json (mounted as /data) atomically,
every few seconds. A stale file is worse than none, so the reader checks the timestamp.
"""
import json
import os
import pathlib
import time

OUT = pathlib.Path("/home/jason/ai-lab/model_loader_data/host_sensors.json")
INTERVAL_S = 2.0
RAPL = pathlib.Path("/sys/class/powercap")


def rapl_domains():
    """Package-level RAPL domains only: summing packages and their subdomains double-counts."""
    out = []
    for d in sorted(RAPL.glob("intel-rapl:*")):
        if ":" in d.name.split("intel-rapl:")[1]:
            continue            # skip intel-rapl:0:0 style subdomains (core, uncore, dram)
        try:
            name = (d / "name").read_text().strip()
            energy = d / "energy_uj"
            wrap = int((d / "max_energy_range_uj").read_text().strip())
            energy.read_text()  # prove it is readable before relying on it
            out.append((name, energy, wrap))
        except OSError:
            continue
    return out


def read_temps():
    """(cpu package C, hottest NVMe C). Labels vary, so match on the hwmon device name."""
    cpu = None
    nvme = None
    for h in sorted(pathlib.Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            name = (h / "name").read_text().strip()
        except OSError:
            continue
        temps = []
        for t in sorted(h.glob("temp*_input")):
            try:
                temps.append(int(t.read_text().strip()) / 1000.0)
            except (OSError, ValueError):
                pass
        if not temps:
            continue
        if name == "coretemp":
            # temp1 is "Package id 0" on Intel; fall back to the hottest core.
            cpu = max(cpu or 0.0, temps[0] if temps else 0.0, max(temps))
        elif name == "nvme":
            nvme = max(nvme or 0.0, max(temps))
    return cpu, nvme


def main() -> None:
    domains = rapl_domains()
    prev = {name: (int(path.read_text().strip()), time.monotonic()) for name, path, _w in domains}
    while True:
        time.sleep(INTERVAL_S)
        watts = {}
        for name, path, wrap in domains:
            try:
                uj = int(path.read_text().strip())
            except (OSError, ValueError):
                continue
            now = time.monotonic()
            last_uj, last_t = prev.get(name, (uj, now))
            dt = now - last_t
            duj = uj - last_uj
            if duj < 0:
                duj += wrap          # the counter wraps; a negative delta is a wrap, not idle
            if dt > 0:
                watts[name] = round(duj / dt / 1_000_000.0, 1)
            prev[name] = (uj, now)
        cpu_t, nvme_t = read_temps()
        payload = {
            "ts": time.time(),
            "cpu_package_w": round(sum(watts.values()), 1) if watts else None,
            "cpu_domains_w": watts or None,
            "cpu_temp_c": round(cpu_t, 1) if cpu_t else None,
            "nvme_temp_max_c": round(nvme_t, 1) if nvme_t else None,
        }
        tmp = OUT.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, OUT)     # atomic: the reader never sees a half-written file
        except OSError:
            pass


if __name__ == "__main__":
    main()
