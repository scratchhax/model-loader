"""Every host disk on the strip, not just the models one.

The case that motivated it: on wheatley Docker's data-root (/HDDPool, md1) filled to 100% and
broke the nightly backup, while the strip showed only the models disk (md0p2) at 62%. The
container never mounts md1 by name; Docker's own /etc/hosts, /etc/hostname and /etc/resolv.conf
bind mounts are the only trace of it.
"""

from __future__ import annotations

from collections import namedtuple

from app import hw
from app.routers import power
from tests.test_routes_smoke import web  # noqa: F401  (the fixture)

# wheatley's model-loader, trimmed: the overlay root and kernel filesystems must be ignored,
# the four md0p2 binds must collapse into one disk, the three Docker files into another.
WHEATLEY = """\
1201 1083 0:310 / / rw,relatime master:571 - overlay overlay rw,lowerdir=/x,upperdir=/y,workdir=/z
1202 1201 0:313 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw
1203 1201 0:314 / /dev rw,nosuid - tmpfs tmpfs rw,size=65536k,mode=755
1208 1201 0:28 / /sys/fs/cgroup ro,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw
1209 1201 9:2 /home/jason/ai-lab/model_loader_data /data rw,relatime - ext4 /dev/md0p2 rw
1210 1201 9:2 /home/jason/ai-lab/models /models rw,relatime - ext4 /dev/md0p2 rw
1211 1201 9:2 /home/jason/ai-lab/strata /strata rw,relatime - ext4 /dev/md0p2 rw
1212 1201 9:2 /home/jason/ai-lab/maya/data/conf /maya/conf rw,relatime - ext4 /dev/md0p2 rw
1213 1201 9:1 /docker-data/containers/abc/resolv.conf /etc/resolv.conf rw,relatime - ext4 /dev/md1 rw
1214 1201 9:1 /docker-data/containers/abc/hostname /etc/hostname rw,relatime - ext4 /dev/md1 rw
1215 1201 9:1 /docker-data/containers/abc/hosts /etc/hosts rw,relatime - ext4 /dev/md1 rw
1216 1203 0:312 / /dev/shm rw,nosuid,nodev,noexec,relatime - tmpfs shm rw,size=65536k
"""

U = namedtuple("U", "total used free")
GB = 1024 ** 3


def _usage(table: dict[str, tuple[float, float]]):
    """A disk_usage stand-in: path -> (free GB, total GB)."""
    def usage(path):
        free, total = table[path]
        return U(total * GB, (total - free) * GB, free * GB)
    return usage


def _mounts(tmp_path, text=WHEATLEY):
    f = tmp_path / "mountinfo"
    f.write_text(text)
    return hw._read_mounts(str(f))


def test_only_real_filesystems_are_read(tmp_path):
    mounts = _mounts(tmp_path)
    assert {m for _d, m in mounts} == {"/data", "/models", "/strata", "/maya/conf",
                                       "/etc/resolv.conf", "/etc/hostname", "/etc/hosts"}


def test_wheatley_shows_the_models_disk_and_dockers_disk(tmp_path):
    usage = _usage({"/models": (1011, 2760), "/etc/resolv.conf": (0, 5630)})
    disks = hw.host_disks("/models", _mounts(tmp_path), usage)
    assert [d["label"] for d in disks] == ["models", "docker"]
    assert disks[0]["paths"] == ["/data", "/models", "/strata", "/maya/conf"]
    assert (disks[1]["free"], disks[1]["total"]) == (0, 5630)


def test_one_disk_holding_both_is_named_for_both(tmp_path):
    text = WHEATLEY.replace(" 9:1 ", " 9:2 ")
    disks = hw.host_disks("/models", _mounts(tmp_path, text), _usage({"/models": (500, 2000)}))
    assert [d["label"] for d in disks] == ["models · docker"]


def test_another_disk_is_named_for_its_mount_point(tmp_path):
    text = WHEATLEY + "1217 1201 8:17 /scratch /scratch rw - xfs /dev/sdb1 rw\n"
    usage = _usage({"/models": (1011, 2760), "/etc/resolv.conf": (900, 5630), "/scratch": (10, 100)})
    disks = hw.host_disks("/models", _mounts(tmp_path, text), usage)
    assert [d["label"] for d in disks] == ["models", "docker", "scratch"]


# glados: ZFS datasets each get a device number, but /models and /data draw on one pool (tank)
# and Docker on another (fast).
GLADOS = """\
951 942 0:47 /models /models rw,noatime - zfs tank rw,xattr
952 942 0:54 /llama/ml-data /data rw,noatime - zfs tank/services rw,xattr
953 942 0:48 /containers/x/resolv.conf /etc/resolv.conf rw,noatime - zfs fast/docker rw,xattr
954 942 0:48 /containers/x/hostname /etc/hostname rw,noatime - zfs fast/docker rw,xattr
955 942 0:48 /containers/x/hosts /etc/hosts rw,noatime - zfs fast/docker rw,xattr
"""


def test_zfs_datasets_of_one_pool_are_one_disk(tmp_path):
    usage = _usage({"/models": (10745, 10955), "/etc/resolv.conf": (2698, 2698)})
    disks = hw.host_disks("/models", _mounts(tmp_path, GLADOS), usage)
    assert [(d["label"], d["paths"]) for d in disks] == [
        ("models", ["/models", "/data"]),
        ("docker", ["/etc/resolv.conf", "/etc/hostname", "/etc/hosts"]),
    ]


def test_btrfs_subvolumes_of_one_device_are_one_disk(tmp_path):
    text = ("1 0 0:41 /@models /models rw - btrfs /dev/nvme0n1p2 rw\n"
            "2 0 0:42 /@data /data rw - btrfs /dev/nvme0n1p2 rw\n")
    disks = hw.host_disks("/models", _mounts(tmp_path, text), _usage({"/models": (100, 1000)}))
    assert [d["paths"] for d in disks] == [["/models", "/data"]]


def test_escaped_mount_points_are_decoded(tmp_path):
    text = "1 0 8:1 / /mnt/big\\040disk rw - ext4 /dev/sda1 rw\n"
    assert _mounts(tmp_path, text) == [("8:1", "/mnt/big disk")]


def test_no_mount_table_falls_back_to_the_models_dir():
    disks = hw.host_disks("/models", [], _usage({"/models": (300, 1000)}))
    assert [(d["label"], d["paths"]) for d in disks] == [("models", ["/models"])]


def test_an_unreadable_disk_is_left_out(tmp_path):
    def usage(path):
        if path == "/etc/resolv.conf":
            raise OSError("gone")
        return U(2760 * GB, 0, 1011 * GB)
    disks = hw.host_disks("/models", _mounts(tmp_path), usage)
    assert [d["label"] for d in disks] == ["models"]


class TestAlerts:
    def test_a_full_disk_is_critical(self):
        # The bug this replaced: `0 < free < 60` read a full disk as unreadable and stayed calm.
        d = power._disk_entry({"label": "docker", "paths": [], "free": 0.0, "total": 5630.0})
        assert d["crit"] and not d["warn"] and d["used_pct"] == 100.0

    def test_big_disk_uses_the_fixed_limits(self):
        e = lambda free: power._disk_entry({"label": "m", "paths": [], "free": free, "total": 2760.0})
        assert e(1011)["crit"] is False and e(1011)["warn"] is False
        assert e(100)["warn"] and not e(100)["crit"]
        assert e(50)["crit"]

    def test_small_disk_scales_its_limits(self):
        e = lambda free: power._disk_entry({"label": "s", "paths": [], "free": free, "total": 120.0})
        assert not e(80)["warn"]          # 150 GB free is impossible here; 15% (18 GB) applies
        assert e(10)["warn"] and not e(10)["crit"]
        assert e(5)["crit"]                # under 5% (6 GB)


def test_the_strip_draws_a_cell_per_disk(monkeypatch, web):  # noqa: F811
    monkeypatch.setattr(hw, "host_disks", lambda _p: [
        {"label": "models", "paths": ["/models"], "free": 1011.0, "total": 2760.0},
        {"label": "docker", "paths": ["/etc/hosts"], "free": 0.0, "total": 5630.0},
    ])
    html = web.get("/boxbar").text
    assert 'id="bar-disk-0"' in html and 'id="bar-disk-1"' in html
    assert "models 63%" in html and "docker 100%" in html
    # the full one is tinted red, the healthy one is not
    cell1 = html.split('id="bar-disk-1"')[1].split(">")[0]
    cell0 = html.split('id="bar-disk-0"')[1].split(">")[0]
    assert "bg-red-500/10" in cell1 and "bg-red-500/10" not in cell0
