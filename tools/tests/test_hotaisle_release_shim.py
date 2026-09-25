"""The release's Hot Aisle routes against a local stand-in for the Hot Aisle API, with no cloud.

Two runners use tools/hotaisle_vm_lib.sh:
  * the AMD COLUMN, tools/release_wheel_smoke.sh --vendor hip --provider hotaisle
    (and auto, which walks runpod, hotaisle, do);
  * the AMD BUILD LEG, tools/hotaisle_release_leg.sh.
The stand-in serves what they call (teams, balance, ssh keys, offerings, the VM
list, create, get, state, PATCH, DELETE) and a RunPod that has no MI300X. The
"VM" is a directory on this machine: an ssh stand-in runs each remote command
with bash after `sudo -n -H` (a stand-in that just execs), records it, and
stands in for the three things a Mac cannot run: the smoke's box.sh (a column
that ran to the end), the build leg's host preparation, and the pinned Ubuntu
22.04 container build (which writes the build tree release061_remote_build.sh
writes, over the unpacked archive's real native inventory). Nothing here rents.

    PYTHONPATH=python .pixi/envs/test/bin/python -m pytest tools/tests/test_hotaisle_release_shim.py -q
"""
import hashlib
import json
import re
import os
from pathlib import Path
import subprocess
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

REPO = Path(__file__).resolve().parents[2]
KEY = "hotaisle-test-key-7c1e0b9d44"
TEAM = "andrews-team"
FP = "SHA256:shimshimshimshimshimshimshimshimshimshimshi"
COMMIT_W = "c" * 40


# ---------------------------------------------------------------- the API stand-in

class Cloud:
    def __init__(self):
        self.vms = {}
        self.creates, self.deleted, self.log = [], [], []
        self.balance = 4882
        self.q1, self.q2 = 1, 1
        self.delete_fails = False
        self.max_vms = 2

    def offers(self):
        out = [{"Quantity": self.q2, "OnDemandPrice": 598, "MinimumReservationMinutes": 60,
                "Specs": {"cpu_cores": 26, "ram_capacity": 448 * 2 ** 30, "gpus": [{"count": 2, "model": "MI300X"}]}}]
        if self.q1 is not None:
            out.append({"Quantity": self.q1, "OnDemandPrice": 299, "MinimumReservationMinutes": 1,
                        "Specs": {"cpu_cores": 13, "ram_capacity": 224 * 2 ** 30, "gpus": [{"count": 1, "model": "MI300X"}]}})
        return out

    def handler(self):
        c = self
        base = "/ha/teams/%s/virtual_machines" % TEAM

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

            def _body(self):
                return json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")

            def _vm(self, p):
                ref = p[len(base) + 1:].split("/")[0]
                return next((v for i, v in c.vms.items() if ref in (i, v["name"])), None)

            def _auth(self):
                if self.path.startswith("/ha/") and self.headers.get("Authorization") != "Token " + KEY:
                    self._send(401, {"error": "unauthorized"})
                    return False
                return True

            def do_GET(self):
                p = self.path.split("?")[0].rstrip("/")
                c.log.append(("GET", self.path))
                if not self._auth():
                    return
                if p == "/rp/pods":
                    return self._send(200, [])
                if p == "/ha/teams":
                    return self._send(200, [{"handle": TEAM, "effective_roles": ["operator"], "maximum_virtual_machines": c.max_vms}])
                if p == "/ha/teams/%s/balance" % TEAM:
                    return self._send(200, {"available_balance": c.balance})
                if p == "/ha/user/ssh_keys":
                    return self._send(200, [{"fingerprint": FP}])
                if p == base + "/available":
                    return self._send(200, c.offers())
                if p == base:
                    return self._send(200, [dict(name=v["name"], deployment_id=i, ip_address="127.0.0.1",
                                                 description=v["description"]) for i, v in c.vms.items()])
                v = self._vm(p)
                if v is None:
                    return self._send(404, {"error": "not found"})
                if p.endswith("/state"):
                    return self._send(200, {"state": "running"})
                return self._send(200, v)

            def do_POST(self):
                p = self.path.split("?")[0].rstrip("/")
                data = self._body()
                c.log.append(("POST", self.path, data))
                if not self._auth():
                    return
                if p == "/rp/pods":
                    return self._send(200, {"error": "create pod: There are no instances currently available", "status": 500})
                if p == base:
                    c.creates.append(data)
                    i = "9a1d7c2e-0b3f-4c8d-9e6a-%012d" % len(c.creates)
                    v = dict(name="enc1-gpuvm%03d" % len(c.creates), deployment_id=i, description="",
                             ssh_access={"ip_address": "127.0.0.1", "port": 2222}, specs=data)
                    c.vms[i] = v
                    return self._send(200, v)
                return self._send(404, {})

            def do_PATCH(self):
                p = self.path.split("?")[0].rstrip("/")
                data = self._body()
                c.log.append(("PATCH", self.path, data))
                if not self._auth():
                    return
                v = self._vm(p)
                if v is None:
                    return self._send(404, {})
                v["description"] = data.get("description", "")
                return self._send(204)

            def do_DELETE(self):
                p = self.path.split("?")[0].rstrip("/")
                c.log.append(("DELETE", self.path))
                if not self._auth():
                    return
                if c.delete_fails:
                    return self._send(500, {"error": "try later"})
                v = self._vm(p)
                if v is None:
                    return self._send(404, {})
                del c.vms[v["deployment_id"]]
                c.deleted.append(v["deployment_id"])
                return self._send(204)
        return H

    def posts(self, prefix):
        return [x for x in self.log if x[0] == "POST" and x[1].startswith(prefix)]


@pytest.fixture
def cloud():
    c = Cloud()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), c.handler())
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    c.url = "http://127.0.0.1:%d" % srv.server_address[1]
    yield c
    srv.shutdown()
    srv.server_close()


# ---------------------------------------------------------------- the VM stand-ins

SSH = r'''#!/usr/bin/env python3
# every call passes the remote command as its last argument; log it, then run it
import json, os, re, subprocess, sys
cmd = sys.argv[-1]
with open(os.environ["SHIM_LOG"], "a") as f:
    f.write(json.dumps(cmd) + "\n")
cmd = cmd.replace("/dev/kfd", os.environ["FAKE_KFD"])
inner = cmd
m = re.match(r"^sudo -n -H bash -c '(.*)'$", cmd, re.S)
if m:
    inner = m.group(1).replace("'\\''", "'")
# 1. the smoke's box.sh: a column that ran to the end, bytes from FAKE_COLUMN
if "nohup bash " in inner and "/box.sh " in inner:
    d = os.environ["FAKE_BOX_DIR"]
    open(d + "/column.json", "w").write(open(os.environ["FAKE_COLUMN"]).read())
    open(d + "/column.exit", "w").write("0\n")
    open(d + "/column.txt", "w").write("install_exit=0\nversion 0.0.0 vendor hip\nselftest_exit=0\nenv=" + inner.split("nohup")[0] + "\n")
    open(d + "/box.txt", "w").write("GPU[0] : Card Series: AMD Instinct MI300X VF (stand-in)\n")
    open(d + "/box.done", "w").write("done\n")
    print("STARTED"); sys.exit(0)
# 2. the build leg's host preparation (apt, pixi, patchelf are the box's business)
if re.match(r"^bash \S*/hotaisle_host_prep\.sh$", inner):
    print("APT_EXIT=0 need= patchelf python3-venv python3-pip"); print("PIXI_INSTALL_EXIT=0")
    print("patchelf 0.17.2"); print("PIXI_ENV_OK"); sys.exit(0)
# 3. the pinned container: prepare pulls, run builds
if re.match(r"^bash \S*/release_ubuntu22_build\.sh prepare$", inner):
    print("pulled (stand-in)"); sys.exit(0)
if "release_ubuntu22_build.sh run hip gfx942" in inner:
    subprocess.run([sys.executable, os.environ["FAKE_BUILD"]], check=True)
    print("STARTED"); sys.exit(0)
env = dict(os.environ, PATH=os.environ["BOXBIN"] + ":" + os.environ["PATH"])
sys.exit(subprocess.run(["bash", "-c", cmd], env=env).returncode)
'''

# The build release061_remote_build.sh writes, over the box's unpacked archive.
FAKE_BUILD = r'''
import hashlib, json, os, sys
from pathlib import Path
root = Path(os.environ["FAKE_BOX_ROOT"])
src = root / "mojolearn"
sys.path.insert(0, str(src / "tools"))
from check_linux_release_qualification import native_inventory
from verify_linux_surface_qualification import MODES, expected_bindings
commit = (src / "commit.txt").read_text().strip()
out = root / "rel061-build"
out.mkdir()
inv = native_inventory(src)
(out / "preflight.json").write_text(json.dumps(dict(vendor="hip", device_architecture="gfx942",
    source_inventory=inv, source_commit=commit)))
ext = {}
sets = out / "build" / "sets" / "hip" / "gfx942"
for mode in MODES:
    for b in sorted(expected_bindings(mode, True)):
        p = sets / mode / (b + ".so")
        p.parent.mkdir(parents=True, exist_ok=True)
        data = ("%s/%s/%s" % (os.environ.get("FAKE_DRAW", "draw"), mode, b)).encode()
        p.write_bytes(data)
        ext["mojolearn/hip/gfx942/%s/%s.so" % (mode, b)] = hashlib.sha256(data).hexdigest()
(sets / "host").mkdir()
(sets / "host" / "_mojolearn_core_host.so").write_bytes(b"core-host")
(sets / "readback.txt").write_text("host _mojolearn_core_host cpu\n")
(sets / "arch_readback.txt").write_text("host _mojolearn_core_host NONE-BY-DESIGN\n")
(out / "build" / "build-provenance.json").write_text(json.dumps(dict(
    complete=True, build_exit=0, source_commit=commit, source_inventory=inv, extensions=ext)))
(out / "exit_code").write_text("0\n")
(root / "rel061-build.log").write_text("core_host_probe=skipped (compared at pack time)\nbuilt (stand-in)\n")
(root / "rel061.exit").write_text(os.environ.get("FAKE_BUILD_EXIT", "0") + "\n")
'''

BOXBIN = {
    "sudo": "#!/bin/bash\nwhile [ $# -gt 0 ]; do case \"$1\" in -n|-H|-E) shift ;; *) break ;; esac; done\nexec \"$@\"\n",
    "rocm-smi": "#!/bin/sh\necho 'GPU[0]\t\t: Card Series: \t\tAMD Instinct MI300X VF'\n",
    "rocminfo": "#!/bin/sh\necho '  Name:                    AMD EPYC'\necho '  Name:                    gfx942'\n",
    "sha256sum": "#!/bin/sh\nexec shasum -a 256 \"$@\"\n",
}


def make_wheel(directory, commit=COMMIT_W):
    path = Path(directory) / "mojolearn-0.0.0-py3-none-manylinux_2_35_x86_64.whl"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mojolearn/identity_columns/COMMIT", commit + "\n")
        z.writestr("mojolearn/verify_reference/models/models.json", '{"models": []}')
    return path


@pytest.fixture
def world(tmp_path, cloud):
    w = type("World", (), {})()
    w.tmp, w.cloud = tmp_path, cloud
    w.bin = tmp_path / "bin"
    w.bin.mkdir()
    (w.bin / "ssh").write_text(SSH)
    (w.bin / "fake_build.py").write_text(FAKE_BUILD)
    for name, text in BOXBIN.items():
        (w.bin / name).write_text(text)
    for f in w.bin.iterdir():
        f.chmod(0o755)
    (tmp_path / "kfd").write_text("")
    w.key = tmp_path / "hotaisle.key"
    w.key.write_text(KEY + "\n")
    w.key.chmod(0o600)
    (tmp_path / "t").mkdir()
    w.shim_log = tmp_path / "ssh_commands.jsonl"
    w.env = dict(os.environ,
                 PATH=str(w.bin) + ":" + os.environ["PATH"],
                 TMPDIR=str(tmp_path / "t"),
                 BOXBIN=str(w.bin), SHIM_LOG=str(w.shim_log), FAKE_KFD=str(tmp_path / "kfd"),
                 FAKE_BUILD=str(w.bin / "fake_build.py"),
                 MOJOLEARN_HOTAISLE_API=cloud.url + "/ha",
                 MOJOLEARN_HOTAISLE_KEY_FILE=str(w.key),
                 MOJOLEARN_HOTAISLE_SSH_KEY=str(tmp_path / "id_ed25519"),
                 MOJOLEARN_HOTAISLE_SSH_KEY_FP=FP,
                 MOJOLEARN_HOTAISLE_SLOT_PREFIX=str(tmp_path / "slot"),
                 MOJOLEARN_HOTAISLE_CREATE_LOCK=str(tmp_path / "create.lock"),
                 MOJOLEARN_HOTAISLE_POLL_SECONDS="1",
                 MOJOLEARN_HOTAISLE_VERIFY_SECONDS="3",
                 MOJOLEARN_HOTAISLE_BUILD_POLL_SECONDS="1",
                 # never the real RunPod or DigitalOcean from a test
                 RP=cloud.url + "/rp", RP_V2=cloud.url + "/rpv2",
                 MOJOLEARN_SMOKE_DO_API=cloud.url + "/do",
                 MOJOLEARN_RUNPOD_KEY_FILE=str(tmp_path / "no-runpod-key"),
                 MOJOLEARN_DO_TOKEN_FILE=str(tmp_path / "no-do-token"),
                 MOJOLEARN_DO_GPU_LOCK=str(tmp_path / "do-gpu.lock"))
    for k in ("RUNPOD_API_KEY", "MOJOLEARN_HOTAISLE_RELEASE_SPEC", "MOJOLEARN_HOTAISLE_STOCK_WAIT_MINUTES",
              "MOJOLEARN_HOTAISLE_SLOT_WAIT_MINUTES", "MOJOLEARN_EXPECT_CORE_HOST_SHA256", "MOJOLEARN_EXPECT_CORE_HOST_FROM"):
        w.env.pop(k, None)
    yield w
    # the on-box watchdog and the Mac dead-man sleep to their deadlines on this machine
    subprocess.run(["pkill", "-f", str(tmp_path)], check=False)


def commands(w):
    if not w.shim_log.exists():
        return []
    return [json.loads(line) for line in w.shim_log.read_text().splitlines()]


def kv(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out.setdefault(k.strip(), v)
    return out


def no_key_anywhere(root):
    for p in Path(root).rglob("*"):
        if p.is_file():
            assert KEY.encode() not in p.read_bytes(), "the key reached %s" % p


def assert_verified_delete(w, record):
    (vm,) = w.cloud.deleted
    assert not w.cloud.vms
    assert ("DELETE", "/ha/teams/%s/virtual_machines/%s/?force=true" % (TEAM, vm)) in w.cloud.log
    text = Path(record).read_text()
    assert "delete ref=%s attempt 1 -> HTTP 204" % vm in text
    assert "verified_gone ref=%s yes get=404" % vm in text
    assert "destroy_confirmed=1" in text
    assert "mac_deadman=cancelled" in text
    assert not list(w.tmp.glob("slot.*")), "the slot is released after the verified delete"
    return vm


# ================================================================ the AMD column

SMOKE = REPO / "tools" / "release_wheel_smoke.sh"


def smoke(w, provider="hotaisle", *extra, spec=None):
    wheel = make_wheel(w.tmp)
    sel = w.tmp / "selection-hip.json"
    sel.write_text(json.dumps({"backend": "hip", "lanes": ["hf-checkpoint", "kmeans"]}))
    col = w.tmp / "col.json"
    col.write_text(json.dumps({"lanes": {}}))
    box = w.tmp / "box" / "wheel-smoke"
    env = dict(w.env, MOJOLEARN_SMOKE_REMOTE_DIR=str(box), FAKE_BOX_DIR=str(box), FAKE_COLUMN=str(col))
    if spec:
        env["MOJOLEARN_HOTAISLE_RELEASE_SPEC"] = spec
    if provider == "auto":
        (w.tmp / "runpod.key").write_text("fake-runpod-key\n")
        (w.tmp / "runpod.key").chmod(0o600)
        env["MOJOLEARN_RUNPOD_KEY_FILE"] = str(w.tmp / "runpod.key")
    w.out = w.tmp / "out"
    p = subprocess.run(["bash", str(SMOKE), str(wheel), "--expected-source-commit", COMMIT_W, "--vendor", "hip",
                        "--provider", provider, "--column", str(sel), "--rent", "--lease", "10",
                        "--smoke-seconds", "60", "--out", str(w.out), *extra],
                       env=env, cwd=REPO, capture_output=True, text=True, timeout=300)
    return p.returncode, p.stdout + p.stderr


def test_column_on_a_1x_vm(world):
    rc, text = smoke(world)
    assert rc == 0, text
    c = world.cloud
    assert len(c.creates) == 1 and c.creates[0]["gpus"] == [{"count": 1, "model": "MI300X"}], c.creates
    assert c.creates[0]["cpu_cores"] == 13
    vm = assert_verified_delete(world, world.out / "teardown.txt")
    desc = [x[2]["description"] for x in c.log if x[0] == "PATCH"][0]
    assert desc.startswith("mojolearn:release-smoke-0.0.0:")
    prov = (world.out / "provider.txt").read_text()
    assert "provider=hotaisle vm=%s name=enc1-gpuvm001 spec=1gpu ip=127.0.0.1 port=2222 price_cents_per_hour=299 gfx=gfx942" % vm in prov
    assert "pin=" not in prov
    assert "horizon_minutes=30" in prov and "max_cost_cents=150" in prov and "cap_cents=1000" in prov   # 30 min at $2.99/h
    # the watchdog: verified from two sessions, its ref and deadline baked in
    wd = (world.out / "hotaisle_watchdog_check.txt").read_text()
    for s in ("WATCHDOG_ALIVE pid=", "REF_BAKED_IN=1", "TOKEN_GET_HTTP=200", "DESC_MATCH", "WATCHDOG_STILL_ALIVE_SECOND_SESSION"):
        assert s in wd, (s, wd)
    assert "virtual_machines/%s/?force=true" % vm in (world.out / "hotaisle_watchdog.sh").read_text()
    # the watchdog fires at the lease (10 min from the create), and the record says so
    # (a with_timeout call once overwrote the recorded number with 60, 2026-09-25)
    secs = int(kv(world.out / "provider.txt")["watchdog_seconds"].split()[0])
    assert 540 <= secs <= 600, secs
    assert "fires_in=%ds" % secs in (world.out / "hotaisle_watchdog.sh").read_text()
    # every box command ran as root through sudo, in this order, and the column came home
    cmds = commands(world)
    box = str(world.tmp / "box" / "wheel-smoke")
    assert all(x == "true" or x.startswith("sudo -n -H bash -c '") for x in cmds), cmds
    flat = "\n".join(cmds)
    order = ["rm -rf %s && mkdir -p %s" % (box, box), "cat > %s/mojolearn-0.0.0-py3-none-manylinux_2_35_x86_64.whl" % box,
             "cat > %s/qualify_verifier_wheel.py" % box, "cat > %s/box.sh" % box, "sha256sum mojolearn-0.0.0",
             "nohup bash %s/box.sh" % box, "cat %s/box.done" % box, "tar czf - box.txt"]
    pos = [flat.index(o) for o in order]
    assert pos == sorted(pos), order
    assert "ROCR_VISIBLE_DEVICES" not in flat
    assert (world.out / "column-hip.json").is_file()
    assert "rented=hotaisle target=-p 2222" in (world.out / "smoke.txt").read_text()
    assert "box_sudo=1" in (world.out / "smoke.txt").read_text()
    assert "Hot Aisle dead-man cancelled" in text
    assert not [x for x in c.log if x[1].startswith("/do")]
    no_key_anywhere(world.out)
    assert KEY not in text


def test_auto_walks_runpod_then_hotaisle(world):
    rc, text = smoke(world, "auto")
    assert rc == 0, text
    c = world.cloud
    assert len(c.posts("/rp/pods")) == 1                      # RunPod tried ONCE
    assert "RunPod created nothing; FALLING BACK to the next provider" in text
    assert len(c.posts("/ha/")) == 1
    seq = [x[1].split("?")[0] for x in c.log if x[0] in ("POST", "DELETE")]
    assert seq[0] == "/rp/pods" and seq[1].startswith("/ha/teams/%s/virtual_machines" % TEAM) and seq[-1].startswith("/ha/"), seq
    assert "runpod=no_stock" in (world.out / "provider.txt").read_text()
    assert not [x for x in c.log if x[1].startswith("/do")], "Hot Aisle created a VM: DigitalOcean is never asked"
    assert_verified_delete(world, world.out / "teardown.txt")


def test_2gpu_vm_when_no_1x_stock_pins_gpu_0(world):
    world.cloud.q1 = 0
    rc, text = smoke(world)
    assert rc == 0, text
    assert world.cloud.creates[0]["gpus"] == [{"count": 2, "model": "MI300X"}]
    prov = (world.out / "provider.txt").read_text()
    assert "spec=2gpu" in prov and "pin=ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0" in prov
    assert "billed_minutes_at_most=60 max_cost_cents=598" in prov          # the 60-minute minimum at $5.98/h
    start = [x for x in commands(world) if "/box.sh >" in x][0]
    assert "env ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 $S nohup bash" in start
    assert_verified_delete(world, world.out / "teardown.txt")


@pytest.mark.parametrize("setup, phrase", [
    (lambda c: (setattr(c, "q1", 0), setattr(c, "q2", 0)), "no stock: no MI300X VM of spec auto is available"),
])
def test_refusals_create_nothing(world, setup, phrase):
    setup(world.cloud)
    rc, text = smoke(world)
    assert rc != 0, text
    assert "Hot Aisle REFUSED: " + phrase in text, text
    assert "Hot Aisle: " + phrase in text
    assert not world.cloud.creates and not commands(world)
    assert not list(world.tmp.glob("slot.*")), "the slot is released on a refusal"
    assert "hotaisle_refused=" in (world.out / "provider.txt").read_text()


def test_a_low_balance_still_creates(world):
    """The team balance tops up automatically (2026-09-25): a balance under
    $5 is recorded, never a refusal."""
    world.cloud.balance = 400
    rc, text = smoke(world)
    assert rc == 0, text
    assert "REFUSED" not in text and world.cloud.creates
    assert_verified_delete(world, world.out / "teardown.txt")


def test_cap_refuses_the_2gpu_vm(world):
    world.cloud.q1 = 0
    rc, text = smoke(world, "hotaisle", "--hotaisle-cap", "5")
    assert rc != 0
    assert "is up to $5.98, above the cap of $5.00; nothing was created" in text, text
    assert not world.cloud.creates


def test_held_slots_are_left_alone(world):
    for n in (1, 2):
        d = world.tmp / ("slot.%d" % n)
        d.mkdir()
        (d / "owner").write_text("lane=another-leg\nnonce=someone-else\n")
    rc, text = smoke(world)
    assert rc != 0
    assert "no Hot Aisle slot free in 0 minutes" in text, text
    assert not world.cloud.creates
    assert all((world.tmp / ("slot.%d" % n) / "owner").read_text().startswith("lane=another-leg") for n in (1, 2))


def test_auto_goes_on_to_digitalocean_when_hotaisle_refuses(world):
    world.cloud.q1 = world.cloud.q2 = 0
    rc, text = smoke(world, "auto")
    assert rc != 0                     # no DigitalOcean token in the test: rent_do refuses by name
    assert "Hot Aisle created nothing (no stock" in text and "FALLING BACK to DigitalOcean" in text, text
    assert "no usable DigitalOcean token" in text
    assert not world.cloud.creates


def test_an_unverified_delete_leaves_the_dead_man_armed(world):
    world.cloud.delete_fails = True
    rc, text = smoke(world)
    assert rc != 0
    td = (world.out / "teardown.txt").read_text()
    assert "NOT_VERIFIED ref=" in td and "destroy_confirmed=0" in td
    assert "MAY STILL BE BILLING" in text and "tools/hotaisle_leg.sh reap" in text
    assert list(world.tmp.glob("slot.*")), "the slot stays held until the VM is gone"
    assert "mac_deadman=cancelled" not in td


# ================================================================ the AMD build leg

LEG = REPO / "tools" / "hotaisle_release_leg.sh"


def head():
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def leg(w, *extra, spec=None, build_exit="0"):
    br = w.tmp / "box" / "root"
    br.mkdir(parents=True, exist_ok=True)
    env = dict(w.env, MOJOLEARN_HOTAISLE_BOX_ROOT=str(br), FAKE_BOX_ROOT=str(br), FAKE_BUILD_EXIT=build_exit,
               MOJOLEARN_HOTAISLE_REMOTE_PY="/bin/sh", MOJOLEARN_RELEASE_RESULTS_ROOT=str(w.tmp / "legs"))
    if spec:
        env["MOJOLEARN_HOTAISLE_RELEASE_SPEC"] = spec
    w.out = w.tmp / "legs" / "hip-gfx942"
    w.br = br
    p = subprocess.run(["bash", str(LEG), head(), *extra], env=env, cwd=REPO, capture_output=True, text=True, timeout=600)
    return p.returncode, p.stdout + p.stderr


def test_build_leg_dry_run_rents_nothing(world):
    rc, text = leg(world)
    assert rc == 0, text
    assert "DRY RUN -- nothing rented" in text and "compose (sh -n, bash -n)" in text
    assert "1x MI300X: found quantity 1 299 cents/h" in text
    assert not world.cloud.creates and not commands(world)
    assert not world.out.exists()


def test_build_leg_builds_fetches_and_deletes(world):
    rc, text = leg(world, "--rent")
    assert rc == 0, text
    c = world.cloud
    assert len(c.creates) == 1 and c.creates[0]["gpus"] == [{"count": 1, "model": "MI300X"}]
    vm = assert_verified_delete(world, world.out / "hotaisle.txt")
    state = kv(world.out / "leg.txt")
    for k, v in dict(vendor="amd", arch="hip/gfx942", provider="hotaisle", commit=head(), vm=vm, spec="1gpu",
                     gfx="gfx942", build_exit="0", expect_core_host_sha256="skip source=none",
                     build_environment="ROCm 6.4.1 Ubuntu 22.04 pinned container").items():
        assert state.get(k) == v, (k, state.get(k), v)
    # the extension count follows the wheel's manifest (the FAST tier for the
    # classical models added bindings on 2026-09-25), so it is read from the
    # proof the leg fetched, never from a literal
    m = re.match(r"BUILT_NOT_INSTALLED hip/gfx942 (\d+) extensions, fetched bytes match proof", state["admission"])
    assert m, state
    proofs = list(world.out.rglob("build-provenance.json"))
    assert proofs, "no build-provenance.json came home"
    assert int(m.group(1)) == len(json.loads(proofs[0].read_text())["extensions"]), state["admission"]
    assert state["destroyed"].split()[1].startswith("verified_gone")
    assert state["container_helper_sha256"] == hashlib.sha256((REPO / "tools/release_ubuntu22_build.sh").read_bytes()).hexdigest()
    assert int(state["work_seconds"]) <= 2400
    # the same tree tools/release.py reads, with the proof of this commit
    rb = world.out / "release-build"
    proof = json.loads((rb / "build" / "build-provenance.json").read_text())
    assert proof["complete"] is True and proof["source_commit"] == head()
    for f in ("exit_code", "preflight.json", "build/sets/hip/gfx942/readback.txt", "build/sets/hip/gfx942/arch_readback.txt",
              "build/sets/hip/gfx942/host/_mojolearn_core_host.so", "build/sets/hip/gfx942/identical/_mojolearn_byte_lm.so"):
        assert (rb / f).is_file(), f
    for f in ("source_inventory_local.json", "release_ubuntu22_build.sh", "host_prep.sh", "prep-console.log",
              "container-prepare.log", "rel061-build.log", "rel061.exit", "device.txt"):
        assert (world.out / f).is_file(), f
    # the exact remote commands, as root, in order
    br = str(world.br)
    flat = "\n".join(commands(world))
    order = ["cat > %s/src.tgz" % br, "sha256sum %s/src.tgz" % br, "test ! -e %s/mojolearn && mkdir %s/mojolearn" % (br, br),
             "cat > %s/hotaisle_host_prep.sh" % br, "bash %s/hotaisle_host_prep.sh" % br,
             "cat > %s/release_ubuntu22_build.sh" % br, "sha256sum %s/release_ubuntu22_build.sh" % br,
             "bash %s/release_ubuntu22_build.sh prepare" % br,
             "bash %s/release_ubuntu22_build.sh run hip gfx942 %s/rel061-build > %s/rel061-build.log" % (br, br, br),
             "cat %s/rel061.exit" % br, "cd %s/rel061-build && tar czf - ." % br]
    pos = [flat.index(o) for o in order]
    assert pos == sorted(pos), order
    start = [x for x in commands(world) if "release_ubuntu22_build.sh run" in x][0]
    for s in ("MOJOLEARN_COMMIT=" + head(), "MOJOLEARN_EXPECT_CORE_HOST_SHA256=skip", "MOJOLEARN_BUILD_JOBS=4",
              "MOJOLEARN_PYTHON=/bin/sh", "export HOME=" + br):
        assert s in start, s
    prep = (world.out / "host_prep.sh").read_text()
    assert "tools/amd_serial_guard.py" in prep and "pixi install --locked --environment default" in prep
    assert "patchelf==0.17.2.4" in prep and "@BR@" not in prep and "@PREP_SECONDS@" not in prep
    no_key_anywhere(world.out)


def test_build_leg_failed_build_exits_10_and_still_deletes(world):
    rc, text = leg(world, "--rent", build_exit="2")
    assert rc == 10, text
    assert kv(world.out / "leg.txt")["build_exit"] == "2"
    assert_verified_delete(world, world.out / "hotaisle.txt")


def test_build_leg_refuses_the_2gpu_vm_before_anything(world):
    rc, text = leg(world, "--rent", spec="2gpu")
    assert rc == 2, text
    assert "needs the 1x MI300X VM" in text and "amd_serial_guard.py" in text
    assert not world.cloud.log


def test_build_leg_no_1x_stock_creates_nothing(world):
    world.cloud.q1 = 0                     # the 2x VM is in stock, and never taken by the build leg
    rc, text = leg(world, "--rent")
    assert rc == 2, text
    assert "Hot Aisle created nothing: no stock: no MI300X VM of spec 1gpu" in text
    assert not world.cloud.creates


def test_build_leg_cap_and_lease_refusals(world):
    rc, text = leg(world, "--rent", "--cap", "3")
    assert rc == 2 and "above the cap of $3.00" in text, text        # 80 min at $2.99/h = $3.99
    assert not world.cloud.creates
    rc, text = leg(world, "--rent", "--lease", "90")
    assert rc == 2 and "--lease must be 30..60 minutes" in text


def overlay_tarball(w, members):
    """A route overlay (tools/release_tooling.py write_overlay's shape)."""
    import io
    import tarfile
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as t:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755
            t.addfile(info, io.BytesIO(data))
    p = w.tmp / ("overlay-%d.tgz" % len(list(w.tmp.glob("overlay-*.tgz"))))
    p.write_bytes(raw.getvalue())
    return p, hashlib.sha256(raw.getvalue()).hexdigest()


def test_build_leg_applies_the_route_overlay_after_the_source(world):
    """The release tooling's copy of a box-side tool replaces the frozen
    source's after the unpack, before the build, with both digests recorded."""
    new = b"# the tooling checkout's guard (route overlay)\n"
    tgz, digest = overlay_tarball(world, {"tools/amd_serial_guard.py": new})
    world.env.update(MOJOLEARN_ROUTE_OVERLAY=str(tgz), MOJOLEARN_ROUTE_OVERLAY_SHA256=digest,
                     MOJOLEARN_SOURCE_CHECKOUT=str(REPO))
    rc, text = leg(world, "--rent")
    assert rc == 0, text
    assert (world.br / "mojolearn" / "tools" / "amd_serial_guard.py").read_bytes() == new
    ro = (world.out / "route-overlay.txt").read_text()
    old = hashlib.sha256((REPO / "tools" / "amd_serial_guard.py").read_bytes()).hexdigest()
    assert "before tools/amd_serial_guard.py " + old in ro, ro
    assert "after tools/amd_serial_guard.py " + hashlib.sha256(new).hexdigest() in ro, ro
    assert "overlay_sha256=" + digest in ro
    assert kv(world.out / "leg.txt")["route_overlay_sha256"] == digest
    flat = "\n".join(commands(world))
    br = str(world.br)
    order = ["test ! -e %s/mojolearn && mkdir %s/mojolearn" % (br, br), "cd %s/mojolearn || exit 1" % br,
             "bash %s/hotaisle_host_prep.sh" % br]
    pos = [flat.index(o) for o in order]
    assert pos == sorted(pos), order
    assert_verified_delete(world, world.out / "hotaisle.txt")


def test_build_leg_refuses_an_overlay_of_source_before_renting(world):
    tgz, digest = overlay_tarball(world, {"bindings/build.sh": b"echo not tooling\n"})
    world.env.update(MOJOLEARN_ROUTE_OVERLAY=str(tgz), MOJOLEARN_ROUTE_OVERLAY_SHA256=digest)
    rc, text = leg(world, "--rent")
    assert rc == 2, text
    assert "REFUSING bindings/build.sh, it is in the build's source inventory" in text, text
    assert not world.cloud.creates and not commands(world)
    tgz, _ = overlay_tarball(world, {"tools/amd_serial_guard.py": b"x\n"})
    world.env.update(MOJOLEARN_ROUTE_OVERLAY=str(tgz), MOJOLEARN_ROUTE_OVERLAY_SHA256="0" * 64)
    rc, text = leg(world, "--rent")
    assert rc == 2 and "is not the recorded" in text, text
    assert not world.cloud.creates
