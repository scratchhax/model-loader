#!/usr/bin/env python3
"""Publish the host sensors a container cannot read, into a file Model Loader already mounts.

Two things are invisible from inside the app container:

  * CPU package power. RAPL's energy counter is root-only (-r-------- on energy_uj, a
    side-channel hardening measure) and /sys/devices/virtual/powercap is not exposed in a
    container's sysfs namespace at all - mounting /sys does not help, which was worth proving
    before writing this.
  * CPU and NVMe temperatures are readable in-container, but are gathered here anyway so the
    app has one consistent source with one timestamp.

On an AMD box GPU power and temperature are NOT here: the llama container reads those through
rocm-smi, and the kernel's KFD tables already say which process holds which card's VRAM.

On an NVIDIA box one more thing is invisible, and it is the GPU's: WHO is on each card. nvidia-smi
inside a container lists only that container's own processes, under its own pids, and only for
the cards it was given - so a llama container pinned to GPU0 cannot see ComfyUI on GPU1, or that
GPU1 exists at all. Run here, as root on the host, it sees every card and every process under
the host pids that `docker top` maps to a container. So when nvidia-smi is on the PATH this also
publishes every card and every compute process (with its SM share from `nvidia-smi pmon`).

Writes host_sensors.json into the directory Model Loader mounts as /data, atomically, every few
seconds. A stale file is worse than none, so the reader checks the timestamp.
"""
import json
import os
import pathlib
import shutil
import subprocess
import time

# Where Model Loader's data directory is on THIS host. The env var lets one copy of the script
# serve boxes whose data directories differ, without hand-editing it on each.
OUT = pathlib.Path(os.environ.get("HOST_SENSORS_OUT")
                   or "/home/jason/ai-lab/model_loader_data/host_sensors.json")
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


def _smi(args):
    """One nvidia-smi call's stdout lines, or None when it fails. Bounded: a wedged driver must
    stall one sample, never the whole publisher."""
    try:
        r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def _num(text):
    """nvidia-smi prints '[N/A]' or '-' for a field a card does not report; that is 0, not an error."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def read_nvidia():
    """{"gpus": [...], "procs": [...]} for every NVIDIA card on the host, or None without one.

    Card `index` is nvidia-smi's, which is PCI bus order by default; `bus` is carried so the
    reader can match a container's own (renumbered from 0) view of a card to this one.
    """
    rows = _smi(["--query-gpu=index,pci.bus_id,name,memory.used,memory.total,utilization.gpu,"
                 "power.draw,power.limit,temperature.gpu,fan.speed",
                 "--format=csv,noheader,nounits"])
    if not rows:
        return None
    gpus = []
    for ln in rows:
        p = [x.strip() for x in ln.split(",")]
        if len(p) < 10:
            continue
        gpus.append({"index": int(_num(p[0])), "bus": p[1], "name": p[2],
                     "mem_used_mib": _num(p[3]), "mem_total_mib": _num(p[4]),
                     "util_pct": _num(p[5]), "power_w": _num(p[6]), "power_cap_w": _num(p[7]),
                     "temp_c": _num(p[8]), "fan_pct": _num(p[9])})
    by_bus = {g["bus"]: g["index"] for g in gpus}
    procs = {}
    for ln in _smi(["--query-compute-apps=pid,used_memory,gpu_bus_id",
                    "--format=csv,noheader,nounits"]) or []:
        p = [x.strip() for x in ln.split(",")]
        if len(p) < 3 or p[2] not in by_bus or not p[0].isdigit():
            continue
        procs[(int(p[0]), by_bus[p[2]])] = {"pid": int(p[0]), "gpu": by_bus[p[2]],
                                            "mem_mib": _num(p[1]), "sm_pct": 0.0}
    # Per-process SM share. pmon's columns are "gpu pid type sm mem ...", with '-' for a process
    # that ran nothing in the sample window. One sample is ~0.1 s on a 2-card box.
    for ln in _smi(["pmon", "-c", "1", "-s", "u"]) or []:
        p = ln.split()
        if ln.lstrip().startswith("#") or len(p) < 4 or not p[0].isdigit() or not p[1].isdigit():
            continue
        row = procs.get((int(p[1]), int(p[0])))
        if row is not None:
            row["sm_pct"] = _num(p[3])
    return {"gpus": gpus, "procs": list(procs.values())}


def main() -> None:
    has_nvidia = shutil.which("nvidia-smi") is not None
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
        if has_nvidia:
            payload["nvidia"] = read_nvidia()
        tmp = OUT.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, OUT)     # atomic: the reader never sees a half-written file
        except OSError:
            pass


if __name__ == "__main__":
    main()
