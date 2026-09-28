# Host sensors (optional)

The overview reports **power at the wall**. Two of the three inputs are measured:

| Input | Source | Where from |
|---|---|---|
| GPU socket power, per card | `rocm-smi` / `nvidia-smi` | inside the llama container, already |
| CPU package power | RAPL | **this service** |
| Board, RAM, fans, drives | — | nothing meters them; you set an allowance on the page |

CPU power needs a host-side publisher for two reasons, both worth knowing before assuming it is
overengineering:

* `/sys/class/powercap/intel-rapl:0/energy_uj` is mode `-r--------`, root only. This is
  deliberate: the counter leaks enough timing information to be a side channel.
* The path lives under `/sys/devices/virtual/powercap`, which is **not** exposed in a
  container's sysfs namespace. Mounting `/sys` into the container does not help — verified.

So a small root service samples it and writes JSON into the directory Model Loader already
mounts. If it is not running, the panel says CPU power is unavailable instead of quietly
reporting a total that is 50-150 W light.

## Install

Copy `contrib/host-sensors.py` to `/usr/local/bin/`, then:

```ini
# /etc/systemd/system/host-sensors.service
[Unit]
Description=Publish host CPU power and temperatures for Model Loader
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/local/bin/host-sensors.py
Restart=always
RestartSec=5
Nice=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now host-sensors.service
```

Edit the output path at the top of the script if your `model_loader_data` lives elsewhere. It
writes every 2 s, atomically, and Model Loader ignores the file if it is more than 15 s old.

## What it publishes

```json
{"ts": 1790567427.9, "cpu_package_w": 52.3, "cpu_domains_w": {"package-0": 52.3},
 "cpu_temp_c": 70.0, "nvme_temp_max_c": 71.8}
```

GPU power and temperature are deliberately **not** in here: the llama container already reports
them through its own vendor tool, and two sources for one number is how they end up disagreeing.

## AMD notes

`amd-smi set --power-cap` is per card and applies to the **last** `-g` flag only, so cap each
card in its own invocation. It also reports success while the kernel keeps the old value — read
`/sys/bus/pci/devices/<bdf>/hwmon/hwmon*/power1_cap` back to confirm. On gfx1201 neither
`--clk-limit` nor `--perf-determinism` is supported; the power cap is the only lever, and its
floor is 210 W.
