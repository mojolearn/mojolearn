"""tools/hotaisle_leg.sh's segment lease against a local stand-in for the Hot Aisle API, with no cloud.

The stand-in serves what the leg calls (teams, balance, available offerings,
ssh keys, the VM list, create, get, state, PATCH, DELETE). The "VM" is a
directory on this machine: an ssh stand-in runs the leg's remote commands
with `sh -c` after mapping /root, /var/lib/mojolearn-hotaisle, /dev/kfd and
/dev/dri into it, and a PATH of stand-in sudo, setsid, timeout, docker,
rocm-smi, rocminfo and pixi (docker runs the container's command directly).
The leg runs from a throwaway git repository (a clean tree is one of its
guards). Nothing here rents anything.

    PYTHONPATH=python .pixi/envs/test/bin/python -m pytest tools/tests/test_hotaisle_leg_shim.py -q
"""
import calendar
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

REPO = Path(__file__).resolve().parents[2]
KEY = "hotaisle-test-key-3e8b1d7c9a"
TEAM = "andrews-team"
PRICE = 598          # cents an hour, the 2x MI300X offering (probe 2026-09-11)
LEASE = 1800         # minutes: a 30-hour segment

sys.argv = [sys.argv[0]]
_SPEC = importlib.util.spec_from_file_location("lm_run_driver", REPO / "tools" / "lm_run_driver.py")
drv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(drv)


# ---------------------------------------------------------------- the API stand-in

class Shim:
    def __init__(self):
        self.vms = {}              # deployment_id -> dict
        self.deleted = []
        self.creates = []
        self.balance = 50000
        self.quantity = 1
        self.fingerprint = ""
        self.log = []

    def offering(self):
        return [{"Quantity": self.quantity, "OnDemandPrice": PRICE, "MinimumReservationMinutes": 60,
                 "Specs": {"cpu_cores": 26, "ram_capacity": 448 * 2 ** 30, "disk_capacity": 2 ** 40,
                           "gpus": [{"count": 2, "model": "MI300X"}]}},
                {"Quantity": 3, "OnDemandPrice": 199, "MinimumReservationMinutes": 1,
                 "Specs": {"cpu_cores": 13, "ram_capacity": 224 * 2 ** 30, "disk_capacity": 2 ** 40,
                           "gpus": [{"count": 1, "model": "MI300X"}]}}]

    def handler(self):
        shim = self
        base = "/teams/%s/virtual_machines" % TEAM

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj=None):
                body = json.dumps(obj).encode() if obj is not None else b""
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _auth(self):
                if self.headers.get("Authorization") != "Token " + KEY:
                    self._send(401, {"error": "unauthorized"})
                    return False
                return True

            def _path(self):
                return self.path.split("?")[0].rstrip("/")

            def _vm(self, p):
                ref = p[len(base) + 1:].split("/")[0]
                for i, v in shim.vms.items():
                    if ref in (i, v["name"]):
                        return v
                return None

            def do_GET(self):
                p = self._path()
                shim.log.append(("GET", p))
                if not self._auth():
                    return
                if p == "/teams":
                    return self._send(200, [{"handle": TEAM, "effective_roles": ["operator"], "maximum_virtual_machines": 2}])
                if p == "/teams/%s/balance" % TEAM:
                    return self._send(200, {"available_balance": shim.balance, "hourly_rate": 0,
                                            "active_stripe_products": [], "virtual_machine_count": len(shim.vms),
                                            "bare_metal_server_count": 0})
                if p == "/user/ssh_keys":
                    return self._send(200, [{"fingerprint": shim.fingerprint}])
                if p == base + "/available":
                    return self._send(200, shim.offering())
                if p == base:
                    return self._send(200, [dict(name=v["name"], deployment_id=i, ip_address=v["ip_address"],
                                                 description=v["description"], status="ready") for i, v in shim.vms.items()])
                v = self._vm(p)
                if v is None:
                    return self._send(404, {"error": "not found"})
                if p.endswith("/state"):
                    return self._send(200, {"state": "running"})
                return self._send(200, v)

            def do_POST(self):
                p = self._path()
                shim.log.append(("POST", p))
                if not self._auth():
                    return
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if p == base:
                    shim.creates.append(data)
                    i = "5d0c1e2a-7b3f-4c8d-9e6a-%012d" % len(shim.creates)
                    v = dict(name="enc1-gpuvm%03d" % len(shim.creates), deployment_id=i, ip_address="127.0.0.1",
                             ssh_access={"ip_address": "127.0.0.1", "port": 22}, description="", status="ready",
                             cpu_cores=data.get("cpu_cores"), gpus=data.get("gpus"))
                    shim.vms[i] = v
                    return self._send(200, v)
                return self._send(404, {"error": "no such path"})

            def do_PATCH(self):
                p = self._path()
                shim.log.append(("PATCH", p))
                if not self._auth():
                    return
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                v = self._vm(p)
                if v is None:
                    return self._send(404, {"error": "not found"})
                v["description"] = data.get("description", "")
                return self._send(204)

            def do_DELETE(self):
                p = self._path()
                shim.log.append(("DELETE", self.path))
                if not self._auth():
                    return
                v = self._vm(p)
                if v is None:
                    return self._send(404, {"error": "not found"})
                del shim.vms[v["deployment_id"]]
                shim.deleted.append(v["deployment_id"])
                return self._send(204)
        return H


@pytest.fixture
def shim():
    s = Shim()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), s.handler())
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    s.url = "http://127.0.0.1:%d" % srv.server_address[1]
    yield s
    srv.shutdown()


# ---------------------------------------------------------------- the VM stand-ins

SSH_SHIM = r'''#!/usr/bin/env python3
import os, re, subprocess, sys
FAKE = @FAKE@
BOXBIN = @BOXBIN@
PAT = re.compile(r"/root|/var/lib/mojolearn-hotaisle|/dev/kfd|/dev/dri|/etc/os-release")
def rw(s):
    return PAT.sub(lambda m: FAKE + m.group(0), s)
args = sys.argv[1:]
i = 0
no_stdin = False
while i < len(args):
    a = args[i]
    if a == "-n":
        no_stdin = True; i += 1
    elif a in ("-o", "-i", "-p", "-l", "-F"):
        i += 2
    elif a.startswith("-"):
        i += 1
    else:
        break
cmd = rw(" ".join(args[i + 1:]))
data = b"" if no_stdin else sys.stdin.buffer.read()
try:
    data = rw(data.decode()).encode()
except UnicodeDecodeError:
    pass
env = dict(os.environ, HOME=FAKE + "/root", PATH=BOXBIN + ":" + os.environ["PATH"])
p = subprocess.run(["sh", "-c", cmd], input=data, env=env, cwd=FAKE + "/root")
sys.exit(p.returncode)
'''

DOCKER = r'''#!/usr/bin/env python3
import os, sys
a = sys.argv[1:]
if not a or a[0] in ("info", "pull", "rm", "--version"):
    print("Docker version 29.5.3 (stand-in)" if a and a[0] == "--version" else "")
    sys.exit(0)
if a[0] != "run":
    sys.exit(0)
WITH_VALUE = {"--name", "--device", "-e", "-v", "-w", "--security-opt", "--network"}
i = 1
devices = []
while i < len(a):
    if a[i] in WITH_VALUE:
        if a[i] == "--device":
            devices.append(a[i + 1])
        i += 2
    elif a[i].startswith("-"):
        i += 1
    else:
        break
with open(@FAKE@ + "/docker_runs.txt", "a") as f:
    f.write("devices=%s image=%s cmd=%s\n" % (",".join(devices), a[i], " ".join(a[i + 1:])))
os.chdir(@FAKE@ + "/root/mojolearn")
os.execvp(a[i + 1], a[i + 1:])
'''

BOX_TOOLS = {
    "sudo": "#!/bin/sh\nwhile [ $# -gt 0 ]; do case \"$1\" in -n|-H|-E) shift ;; *) break ;; esac; done\nexec \"$@\"\n",
    "setsid": "#!/bin/sh\nexec \"$@\"\n",
    "timeout": "#!/bin/sh\nwhile [ $# -gt 0 ]; do case \"$1\" in -k) shift 2 ;; -*) shift ;; *) break ;; esac; done\nshift\nexec \"$@\"\n",
    "rocm-smi": "#!/bin/sh\necho 'GPU[0]\t\t: Card Series: \t\tAMD Instinct MI300X VF'\n"
                "echo 'GPU[1]\t\t: Card Series: \t\tAMD Instinct MI300X VF'\n"
                "echo 'GPU[0]\t\t: PCI Bus: 0000:C1:00.0'\necho 'GPU[1]\t\t: PCI Bus: 0000:C2:00.0'\n",
    "rocminfo": "#!/bin/sh\necho '  Name:                    AMD EPYC'\nfor i in 0 1; do echo '  Name:                    gfx942'; done\n",
    "pixi": "#!/bin/sh\necho \"pixi $*\"\nexit 0\n",
}

BODY = """#!/bin/sh
mkdir -p /root/gemm_leg_out/lm-segment-B-1
echo "arm=amd route=B segment=1 steps=1000" > /root/gemm_leg_out/lm-segment-B-1/status.txt
echo BODY_RAN
echo "archs=$MOJOLEARN_GPU_ARCHS column=$MOJOLEARN_TARGET_COLUMN"
sleep 3
echo "segment verdict PASS" >> /root/gemm_leg_out/lm-segment-B-1/status.txt
"""


@pytest.fixture
def world(tmp_path, shim):
    """A clean throwaway repository holding the leg, a fake VM, the stand-ins and the env."""
    w = type("World", (), {})()
    w.tmp, w.shim = tmp_path, shim
    w.fake = tmp_path / "box"
    for d in ("root", "var/lib", "dev/dri", "etc"):
        (w.fake / d).mkdir(parents=True, exist_ok=True)
    (w.fake / "dev" / "kfd").write_text("")
    (w.fake / "etc" / "os-release").write_text('PRETTY_NAME="Ubuntu 24.04.4 LTS (stand-in)"\n')
    bin_, boxbin = tmp_path / "bin", tmp_path / "boxbin"
    bin_.mkdir()
    boxbin.mkdir()
    f = bin_ / "ssh"
    f.write_text(SSH_SHIM.replace("@FAKE@", repr(str(w.fake))).replace("@BOXBIN@", repr(str(boxbin))))
    f.chmod(0o755)
    for name, text in dict(BOX_TOOLS, docker=DOCKER.replace("@FAKE@", repr(str(w.fake)))).items():
        f = boxbin / name
        f.write_text(text)
        f.chmod(0o755)
    # the ssh key the account "has"
    w.sshkey = tmp_path / "id_ed25519"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(w.sshkey)], check=True)
    fp = subprocess.run(["ssh-keygen", "-lf", str(w.sshkey) + ".pub"], capture_output=True, text=True, check=True).stdout.split()[1]
    shim.fingerprint = fp
    # the repository the leg runs from, committed and clean
    w.repo = tmp_path / "repo"
    (w.repo / "tools").mkdir(parents=True)
    shutil.copy(REPO / "tools" / "hotaisle_leg.sh", w.repo / "tools" / "hotaisle_leg.sh")
    shutil.copy(REPO / "tools" / "hotaisle_vm_lib.sh", w.repo / "tools" / "hotaisle_vm_lib.sh")
    shutil.copy(REPO / "tools" / "stage_from_r2.sh", w.repo / "tools" / "stage_from_r2.sh")
    (w.repo / "pixi.toml").write_text("[workspace]\nname = 'shim'\n")
    (w.repo / "pixi.lock").write_text("version: 6\n")
    (w.repo / "kernel.mojo").write_text("fn main():\n    pass\n")
    git = ["git", "-C", str(w.repo), "-c", "user.name=shim", "-c", "user.email=shim@example.invalid", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q", str(w.repo)], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "shim"], check=True)
    w.body = tmp_path / "body.sh"
    w.body.write_text(BODY)
    w.key = tmp_path / "hotaisle_key"
    w.key.write_text(KEY + "\n")
    w.key.chmod(0o600)
    w.out = tmp_path / "out"
    w.slots = tmp_path / "slot"
    (tmp_path / "t").mkdir()
    w.env = dict(os.environ,
                 TMPDIR=str(tmp_path / "t"),
                 MOJOLEARN_HOTAISLE_API=shim.url,
                 MOJOLEARN_HOTAISLE_KEY_FILE=str(w.key),
                 MOJOLEARN_HOTAISLE_SSH_BIN=str(bin_ / "ssh"),
                 MOJOLEARN_HOTAISLE_SSH_KEY=str(w.sshkey),
                 MOJOLEARN_HOTAISLE_SSH_KEY_FP=fp,
                 MOJOLEARN_HOTAISLE_SLOT_PREFIX=str(w.slots),
                 MOJOLEARN_HOTAISLE_CREATE_LOCK=str(tmp_path / "create.lock"),
                 MOJOLEARN_HOTAISLE_POLL_SECONDS="1",
                 MOJOLEARN_HOTAISLE_STATUS_EVERY="1",
                 MOJOLEARN_HOTAISLE_STOCK_WAIT_MINUTES="0",
                 MOJOLEARN_HOTAISLE_SLOT_WAIT_MINUTES="0",
                 MOJOLEARN_HOTAISLE_SPEC="2gpu",
                 MOJOLEARN_HOTAISLE_GPU_ONLY="1",
                 MOJOLEARN_GPU_ARCHS="gfx942",
                 MOJOLEARN_GEMM_LEG_EXTRA=str(w.body),
                 MOJOLEARN_GEMM_LEG_OUT=str(w.out),
                 MOJOLEARN_STAGE_KEYS="",
                 FETCH_RESERVE="60")
    for k in ("MOJOLEARN_HOTAISLE_FROZEN", "MOJOLEARN_HOTAISLE_REPO", "MOJOLEARN_BINCACHE"):
        w.env.pop(k, None)
    yield w
    # the on-box watchdog sleeps to its (30-hour) deadline on this machine; the
    # VM is gone only in the stand-in
    subprocess.run(["pkill", "-f", str(tmp_path)], check=False)


def run_leg(w, *args, timeout=300):
    p = subprocess.run(["bash", str(w.repo / "tools" / "hotaisle_leg.sh"), "amd", "--rent", "--skip-gates", *args],
                       env=w.env, cwd=w.repo, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def segment(w, lease=LEASE, cap="200", *extra):
    return run_leg(w, "--one-body", "--segment-lease", str(lease), "--dollar-cap", cap, *extra)


def kv(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out.setdefault(k, v)
    return out


def no_key_anywhere(root):
    for p in Path(root).rglob("*"):
        if p.is_file():
            assert KEY.encode() not in p.read_bytes(), "the key reached %s" % p


# ---------------------------------------------------------------- argument refusals (no API call)

@pytest.mark.parametrize("args, phrase", [
    (["--minutes", "120"], "60 is the maximum lease"),
    (["--one-body", "--segment-lease", "1800"], "--segment-lease and --dollar-cap go together"),
    (["--one-body", "--segment-lease", "60", "--dollar-cap", "10"], "ABOVE one hour"),
    (["--one-body", "--segment-lease", "3000", "--dollar-cap", "900"], "at most 2880 minutes"),
    (["--one-body", "--segment-lease", "120", "--dollar-cap", "ten"], "a dollar figure"),
    (["--one-body", "--minutes", "60", "--segment-lease", "120", "--dollar-cap", "10"], "two ways to say one thing"),
    (["--segment-lease", "120", "--dollar-cap", "20"], "needs --one-body"),
])
def test_argument_refusals(world, args, phrase):
    rc, text = run_leg(world, *args)
    assert rc == 2, text
    assert phrase in text, text
    assert not world.shim.log, "no API call before the arguments are sound"


def test_one_body_needs_the_2gpu_spec(world):
    world.env["MOJOLEARN_HOTAISLE_SPEC"] = "13core"
    rc, text = segment(world)
    assert rc == 2 and "it needs --spec 2gpu" in text, text
    assert not world.shim.log


def test_one_body_needs_gpu_only(world):
    world.env.pop("MOJOLEARN_HOTAISLE_GPU_ONLY")
    rc, text = segment(world)
    assert rc == 2 and "MOJOLEARN_HOTAISLE_GPU_ONLY=1" in text, text
    assert not world.shim.creates


# ---------------------------------------------------------------- refusals the run driver walks past

def test_over_the_dollar_cap_is_refused_before_the_create(world):
    rc, text = segment(world, LEASE, "100")
    assert rc == 2, text
    assert "above the --dollar-cap of $100" in text and "$179.40" in text, text
    assert not world.shim.creates
    leg = kv(world.out / "leg.txt")
    assert leg["max_cost_cents"] == "17940" and leg["cap_cents"] == "10000"
    assert leg["segment_lease_verdict"] == "REFUSED_OVER_CAP"
    assert not any(world.tmp.glob("slot.*")), "the slot is released on a refusal"
    assert not drv._amd_busy(text), "over the cap is not busy: the driver skips hotaisle by its own rule"


def test_a_balance_short_of_the_whole_lease_is_refused(world):
    world.shim.balance = 17000        # above the $5 floor, below $179.40 + $5.00
    rc, text = segment(world)
    assert rc == 3, text
    assert "REFUSED: balance $170.00 is below $184.40" in text and "the whole lease" in text, text
    assert not world.shim.creates
    assert kv(world.out / "leg.txt")["balance_required_cents"] == "18440"
    assert drv._amd_busy(text)


def test_no_stock_is_refused_by_name(world):
    world.shim.quantity = 0
    rc, text = segment(world)
    assert rc == 3, text
    assert "showed no stock for 0 minutes" in text, text
    assert not world.shim.creates
    assert drv._amd_busy(text)


def test_all_slots_held_is_refused_by_name(world):
    for n in (1, 2):
        d = Path(str(world.slots) + ".%d" % n)
        d.mkdir()
        (d / "owner").write_text("lane=other\npid=%d\nnonce=someone-else\n" % os.getpid())
    rc, text = segment(world)
    assert rc == 3, text
    assert "no slot freed in 0 minutes" in text, text
    assert not world.shim.creates
    assert all(Path(str(world.slots) + ".%d" % n).exists() for n in (1, 2)), "another leg's slots are left alone"
    assert drv._amd_busy(text)


# ---------------------------------------------------------------- the whole segment lease

def _epoch(utc):
    return calendar.timegm(time.strptime(utc, "%Y-%m-%dT%H:%M:%SZ"))


def test_segment_lease_happy_path(world):
    t0 = time.time()
    rc, text = segment(world)
    assert rc == 0, text
    s = world.shim
    # one create, of the 2x MI300X offering
    assert len(s.creates) == 1, text
    assert s.creates[0]["cpu_cores"] == 26 and s.creates[0]["gpus"] == [{"count": 2, "model": "MI300X"}]
    (vm_id,) = s.deleted
    assert not s.vms
    leg = kv(world.out / "leg.txt")
    for k, v in dict(provider="hotaisle", spec="2gpu", size="mi300x-2gpu-vm", body_gpus="all", segment_lease=str(LEASE),
                     dollar_cap="200", max_cost="$179.40", max_cost_cents="17940", segment_lease_verdict="UNDER_CAP",
                     lease_cents="17940", balance_required_cents="18440", vm_id=vm_id, vm_ref=vm_id,
                     gpu_agents="2", runtime="docker", gpu_archs="gfx942").items():
        assert leg.get(k) == v, (k, leg.get(k), v)
    assert "price_cents_per_hour=598 min_reservation_minutes=60" in (world.out / "leg.txt").read_text()
    # gpu.txt from rocm-smi on the host, and the body's own
    gpu = (world.out / "gpu.txt").read_text()
    assert "GPU[0]" in gpu and "GPU[1]" in gpu and "MI300X" in gpu
    assert "MI300X" in (world.out / "remote" / "gpu.txt").read_text()
    # the container saw both GPUs: /dev/kfd and the whole /dev/dri, no visible-devices pin
    runs = (world.fake / "docker_runs.txt").read_text()
    assert "devices=%s/dev/kfd,%s/dev/dri " % (world.fake, world.fake) in runs, runs
    assert "VISIBLE_DEVICES" not in runs
    # the long deadline reached the on-box watchdog and the Mac dead-man
    dm = kv(world.out / "deadman.txt")
    assert abs(int(dm["watchdog_seconds"]) - LEASE * 60) < 600, dm
    assert abs(_epoch(dm["mac_deadman_fires_at"]) - (t0 + LEASE * 60)) < 600, dm
    assert abs(_epoch(dm["watchdog_fires_at"]) - (t0 + LEASE * 60)) < 600, dm
    wd = (world.out / "watchdog.sh").read_text()
    assert "fires_in=%s" % dm["watchdog_seconds"] in wd and "/virtual_machines/%s/?force=true" % vm_id in wd
    for w in ("watchdog_WATCHDOG_ALIVE", "watchdog_REF_BAKED_IN=1", "watchdog_TOKEN_GET_HTTP=200", "watchdog_DESC_MATCH",
              "watchdog_WATCHDOG_STILL_ALIVE_SECOND_SESSION"):
        assert any(line.startswith(w) for line in (world.out / "deadman.txt").read_text().splitlines()), w
    assert "mac_deadman=cancelled" in (world.out / "deadman.txt").read_text()
    # the body's timeout(1) is the long lease less the fetch reserve
    assert int(leg["work_seconds"]) > (LEASE - 10) * 60
    # the body ran and came home, and its status was copied here while it ran
    assert "BODY_RAN" in (world.out / "remote" / "extra.log").read_text()
    assert "archs=gfx942 column=amd" in (world.out / "remote" / "extra.log").read_text()
    live = (world.out / "status_live.txt").read_text()
    assert live.startswith("copied_utc=") and "lm-segment-B-1/status.txt" in live and "segment verdict PASS" in live
    remote = kv(world.out / "remote" / "leg.txt")
    assert remote["provider"] == "hotaisle" and remote["size"] == "mi300x-2gpu-vm" and remote["extra_exit"] == "0"
    assert leg["source_sha256_match"] == "yes" and leg["local_key_in_ps"] == "not_visible" and leg["box_key_in_ps"] == "not_visible"
    # the verified delete
    td = (world.out / "teardown.txt").read_text()
    assert "delete ref=%s attempt 1 -> HTTP 204" % vm_id in td
    assert "verified_gone ref=%s yes get=404" % vm_id in td and "destroy_confirmed=1" in td and "exit=0" in td
    assert ("DELETE", "/teams/%s/virtual_machines/%s/?force=true" % (TEAM, vm_id)) in s.log
    assert not any(world.tmp.glob("slot.*")), "the slot is released after the verified delete"
    no_key_anywhere(world.out)


def test_one_body_refuses_a_vm_that_shows_one_gpu(world):
    (world.tmp / "boxbin" / "rocminfo").write_text("#!/bin/sh\necho '  Name:                    gfx942'\n")
    rc, text = segment(world)
    assert rc == 6, text
    assert "shows 1 GPU agents, not 2" in text, text
    assert len(world.shim.creates) == 1 and not world.shim.vms, "the VM is deleted unused"
    assert "destroy_confirmed=1" in (world.out / "teardown.txt").read_text()


# ---------------------------------------------------------------- the driver's reading of the leg's words

def test_driver_busy_phrases():
    busy = [
        "REFUSED: the 2gpu 2x MI300X spec showed no stock for 5 minutes (last HTTP 200, none, quantity 0). Nothing was created.",
        "REFUSED: no slot freed in 5 minutes. Nothing was created.",
        "REFUSED: balance $44.65 is below $184.40, the whole lease (1800 min at 598 cents/h = $179.40) plus the $5.00 floor. Nothing was created.",
        "REFUSED: balance $4.00 is below the $5.00 floor (500 cents). Nothing was created.",
        "create REFUSED by the API (HTTP 404) and no new VM appears. Nothing is billing.",
    ]
    for t in busy:
        assert drv._amd_busy(t), t
    for t in ("segment lease REFUSED: 1800 minutes of the 2x MI300X VM at $5.98/h is up to $179.40, above the --dollar-cap of $100; nothing was created",
              "--one-body on the 2x MI300X VM: rocminfo on the host shows 1 GPU agents, not 2. Deleting.",
              "THE ON-BOX WATCHDOG COULD NOT BE VERIFIED"):
        assert not drv._amd_busy(t), t
