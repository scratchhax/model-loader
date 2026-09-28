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

**Read GPU power in its own `rocm-smi` call.** `Average Graphics Package Power` is averaged over
the interval since the previous read, so it depends on when in the invocation each card is
sampled. Bundled behind the other queries, the gap before each card differs enough to skew them
in opposite directions. Measured on two evenly loaded cards capped at 210 W, 14 consecutive reads
of the bundled command:

| command | result |
|---|---|
| `--showid --showproductname --showuse --showmemuse --showmeminfo vram --showtemp --showpower --showfan` | split worse than 60 W in **12 of 14**: 63/302, 263/49, 65/270 |
| `--showpower` alone | 152/153, 152/150, 152/154 — matches hwmon |

The pair total stayed correct in both, which is the signature of a timing artefact rather than a
bad sensor. If per-card wattage ever looks mirrored again — one card absurdly high, the other as
absurdly low, total about right — this is the first thing to check.

**Neither power sensor is exact.** `power1_average` under hwmon, which is what nvtop displays,
overshot the enforced cap in 29 of 72 steady-load readings on this box; rocm-smi's SMU figure did
so in 2 of 72. nvtop is not doing anything smarter, it just prints the counter, which is why it
too will occasionally show a wattage the cap forbids. Model Loader takes a median of the last
three samples per card, which measured sd 21.5 W down to 4.7 W with no bias.


`amd-smi set --power-cap` is per card and applies to the **last** `-g` flag only, so cap each
card in its own invocation. It also reports success while the kernel keeps the old value — read
`/sys/bus/pci/devices/<bdf>/hwmon/hwmon*/power1_cap` back to confirm. On gfx1201 neither
`--clk-limit` nor `--perf-determinism` is supported; the power cap is the only lever, and its
floor is 210 W.
