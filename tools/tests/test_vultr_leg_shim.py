"""tools/vultr_leg.sh against a local shim of the Vultr API, with no cloud.

The shim serves the endpoints the leg calls (bare-metals create, list, get,
delete; plans-metal; region availability; ssh-keys; os; account) and the
amdgpu-install package. The "box" is a directory on this machine: an ssh
and scp stand-in run the leg's remote commands with `sh -c` after mapping
/root, /etc/, /proc/, /sys/module, /opt/rocm and /dev/kfd into it, and a
PATH of stand-in rocm-smi, rocminfo, apt-get, amdgpu-install, dpkg-query,
systemctl, reboot and pixi. The leg runs from a throwaway git repository
(a clean tree is one of its guards). Nothing here rents anything.

    PYTHONPATH=python .pixi/envs/test/bin/python -m pytest tools/tests/test_vultr_leg_shim.py -q
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

REPO = Path(__file__).resolve().parents[2]
TOKEN = "vultr-test-token-7f3a9c1e5b"
PLAN = "vbm-256c-2048gb-8-mi300x-gpu"
NAME = "mojolearn-extra-amd-vultr"
TAG = "mojolearn-extra"
PUBKEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIShimTestKeyShimTestKeyShimTestKey shim@test"

sys.argv = [sys.argv[0]]
_SPEC = importlib.util.spec_from_file_location("lm_run_driver", REPO / "tools" / "lm_run_driver.py")
drv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(drv)


# ---------------------------------------------------------------- the API shim

class Shim:
    def __init__(self):
        self.bare_metals = {}      # id -> dict
        self.deleted = set()
        self.gets = {}             # id -> GET count (pending, then active)
        self.creates = []
        self.stock = {"ewr": [], "ord": [PLAN]}
        self.plan = {"id": PLAN, "cpu_count": 128, "monthly_cost": 21450.24, "hourly_cost": 31.92,
                     "type": "SSD", "locations": ["ewr", "ord"]}
        self.create_error = None   # (code, message)
        self.ssh_keys = []
        self.log = []

    def handler(self):
        shim = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj=None, raw=None):
                body = raw if raw is not None else (json.dumps(obj).encode() if obj is not None else b"")
                self.send_response(code)
                self.send_header("Content-Type", "application/octet-stream" if raw is not None else "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _auth(self):
                if self.headers.get("Authorization") != "Bearer " + TOKEN:
                    self._send(401, {"error": "Invalid API token.", "status": 401})
                    return False
                return True

            def _path(self):
                return self.path.split("?")[0].rstrip("/")

            def do_GET(self):
                p = self._path()
                shim.log.append(("GET", p))
                if p in ("", "/uplink"):
                    return self._send(200, {"ok": True})
                if p == "/amdgpu-install.deb":
                    return self._send(200, raw=b"!<arch>\nshim amdgpu-install package\n")
                if not self._auth():
                    return
                if p == "/v2/bare-metals":
                    return self._send(200, {"bare_metals": list(shim.bare_metals.values()), "meta": {"total": len(shim.bare_metals), "links": {"next": "", "prev": ""}}})
                if p.startswith("/v2/bare-metals/"):
                    i = p.rsplit("/", 1)[1]
                    if i not in shim.bare_metals:
                        return self._send(404, {"error": "Bare Metal not found.", "status": 404})
                    n = shim.gets[i] = shim.gets.get(i, 0) + 1
                    bm = shim.bare_metals[i]
                    if n >= 3 and bm["status"] == "pending":
                        bm.update(status="active", main_ip="127.0.0.1")
                    return self._send(200, {"bare_metal": bm})
                if p == "/v2/plans-metal":
                    return self._send(200, {"plans_metal": [shim.plan], "meta": {"total": 1, "links": {"next": "", "prev": ""}}})
                if p.startswith("/v2/regions/") and p.endswith("/availability"):
                    r = p.split("/")[3]
                    return self._send(200, {"available_plans": shim.stock.get(r, [])})
                if p == "/v2/ssh-keys":
                    return self._send(200, {"ssh_keys": shim.ssh_keys, "meta": {"total": len(shim.ssh_keys)}})
                if p == "/v2/os":
                    return self._send(200, {"os": [{"id": 1743, "name": "Ubuntu 22.04 LTS x64", "arch": "x64", "family": "ubuntu"},
                                                   {"id": 2284, "name": "Ubuntu 24.04 LTS x64", "arch": "x64", "family": "ubuntu"}]})
                if p == "/v2/account":
                    return self._send(200, {"account": {"balance": -500, "pending_charges": 0, "email": "x"}})
                return self._send(404, {"error": "no such path " + p})

            def do_POST(self):
                p = self._path()
                shim.log.append(("POST", p))
                if not self._auth():
                    return
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if p == "/v2/ssh-keys":
                    k = dict(id="key-%d" % (len(shim.ssh_keys) + 1), name=data.get("name"), ssh_key=data.get("ssh_key"), date_created="now")
                    shim.ssh_keys.append(k)
                    return self._send(201, {"ssh_key": k})
                if p == "/v2/bare-metals":
                    shim.creates.append(data)
                    if shim.create_error:
                        code, msg = shim.create_error
                        return self._send(code, {"error": msg, "status": code})
                    i = "cb676a46-66fd-4dfb-b839-%012d" % len(shim.creates)
                    bm = dict(id=i, os="Ubuntu 24.04 LTS x64", main_ip="0.0.0.0", status="pending", plan=data.get("plan"),
                              region=data.get("region"), label=data.get("label"), tags=data.get("tags") or [],
                              default_password="shim-root-password-SECRET", date_created="now")
                    shim.bare_metals[i] = bm
                    return self._send(202, {"bare_metal": bm})
                return self._send(404, {"error": "no such path"})

            def do_DELETE(self):
                p = self._path()
                shim.log.append(("DELETE", p))
                if not self._auth():
                    return
                i = p.rsplit("/", 1)[1]
                if i in shim.bare_metals:
                    del shim.bare_metals[i]
                    shim.deleted.add(i)
                    return self._send(204)
                return self._send(404, {"error": "Bare Metal not found."})
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


# ---------------------------------------------------------------- the box stand-ins

SSH_SHIM = r'''#!/usr/bin/env python3
import os, re, subprocess, sys
FAKE = @FAKE@
BOXBIN = @BOXBIN@
PAT = re.compile(r"/root|/etc/|/proc/|/sys/module|/opt/rocm|/dev/kfd")
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

SCP_SHIM = r'''#!/usr/bin/env python3
import shutil, sys
FAKE = @FAKE@
args = [a for a in sys.argv[1:]]
src, dst = args[-2], args[-1]
shutil.copyfile(src, FAKE + dst.split(":", 1)[1])
'''

BOX_TOOLS = {
    "apt-get": "#!/bin/sh\necho \"apt-get $*\"\nexit 0\n",
    "amdgpu-install": "#!/bin/sh\necho \"amdgpu-install $*\"\nmkdir -p @FAKE@/opt/rocm/.info\necho 6.4.0-47 > @FAKE@/opt/rocm/.info/version\n",
    "dpkg-query": "#!/bin/sh\nprintf '1:6.12.12.60400-2158079.24.04'\n",
    "systemctl": "#!/bin/sh\nexit 0\n",
    "reboot": "#!/bin/sh\necho boot-2 > @FAKE@/proc/sys/kernel/random/boot_id\n: > @FAKE@/dev/kfd\nmkdir -p @FAKE@/sys/module/amdgpu\necho 6.12.12 > @FAKE@/sys/module/amdgpu/version\n",
    "rocm-smi": "#!/bin/sh\necho 'GPU[0]\t\t: Card Series: \t\tAMD Instinct MI300X'\necho 'Driver version: 6.12.12'\n",
    "rocminfo": "#!/bin/sh\nfor i in 0 1 2 3 4 5 6 7; do echo '  Name:                    gfx942'; done\n",
    "pixi": "#!/bin/sh\necho \"pixi $*\"\nexit 0\n",
}


@pytest.fixture
def world(tmp_path, shim):
    """A clean throwaway repository holding the leg, a fake box, the stand-ins and the env."""
    w = type("World", (), {})()
    w.tmp = tmp_path
    w.shim = shim
    w.fake = tmp_path / "box"
    for d in ("root", "etc", "proc/sys/kernel/random", "dev", "opt"):
        (w.fake / d).mkdir(parents=True, exist_ok=True)
    (w.fake / "etc" / "os-release").write_text('NAME="Ubuntu"\nVERSION_CODENAME=noble\n')
    (w.fake / "proc/sys/kernel/random/boot_id").write_text("boot-1\n")
    bin_ = tmp_path / "bin"
    boxbin = tmp_path / "boxbin"
    bin_.mkdir()
    boxbin.mkdir()
    for name, text in (("ssh", SSH_SHIM), ("scp", SCP_SHIM)):
        f = bin_ / name
        f.write_text(text.replace("@FAKE@", repr(str(w.fake))).replace("@BOXBIN@", repr(str(boxbin))))
        f.chmod(0o755)
    for name, text in BOX_TOOLS.items():
        f = boxbin / name
        f.write_text(text.replace("@FAKE@", str(w.fake)))
        f.chmod(0o755)
    # the repository the leg runs from, committed and clean
    w.repo = tmp_path / "repo"
    (w.repo / "tools").mkdir(parents=True)
    shutil.copy(REPO / "tools" / "vultr_leg.sh", w.repo / "tools" / "vultr_leg.sh")
    shutil.copy(REPO / "tools" / "stage_from_r2.sh", w.repo / "tools" / "stage_from_r2.sh")
    (w.repo / "pixi.toml").write_text("[workspace]\nname = 'shim'\n")
    (w.repo / "pixi.lock").write_text("version: 6\n")
    (w.repo / "kernel.mojo").write_text("fn main():\n    pass\n")
    git = ["git", "-C", str(w.repo), "-c", "user.name=shim", "-c", "user.email=shim@example.invalid", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q", str(w.repo)], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "shim"], check=True)
    w.body = tmp_path / "body.sh"
    w.body.write_text('echo BODY_RAN\necho "archs=$MOJOLEARN_GPU_ARCHS column=$MOJOLEARN_TARGET_COLUMN"\n')
    w.token = tmp_path / "vultr_token"
    w.token.write_text(TOKEN + "\n")
    w.token.chmod(0o600)
    w.pub = tmp_path / "id_ed25519.pub"
    w.pub.write_text(PUBKEY + "\n")
    w.lock = tmp_path / "vultr-gpu.lock"
    w.out = tmp_path / "out"
    (tmp_path / "t").mkdir()
    w.env = dict(os.environ,
                 TMPDIR=str(tmp_path / "t"),
                 MOJOLEARN_VULTR_API=shim.url + "/v2",
                 MOJOLEARN_VULTR_TOKEN_FILE=str(w.token),
                 MOJOLEARN_VULTR_SSH_BIN=str(bin_ / "ssh"),
                 MOJOLEARN_VULTR_SCP_BIN=str(bin_ / "scp"),
                 MOJOLEARN_VULTR_SSH_PUBKEY=str(w.pub),
                 MOJOLEARN_VULTR_GPU_LOCK=str(w.lock),
                 MOJOLEARN_VULTR_ROCM_DEB_URL=shim.url + "/amdgpu-install.deb",
                 MOJOLEARN_VULTR_UPLINK_HOSTS=shim.url + "/uplink",
                 MOJOLEARN_VULTR_POLL_SECONDS="1",
                 MOJOLEARN_GPU_ARCHS="gfx942",
                 MOJOLEARN_GEMM_LEG_EXTRA=str(w.body),
                 MOJOLEARN_GEMM_LEG_OUT=str(w.out),
                 MOJOLEARN_STAGE_KEYS="",
                 FETCH_RESERVE="60")
    w.env.pop("MOJOLEARN_VULTR_LEG_FROZEN", None)
    yield w
    # the on-box dead-man sleeps to its deadline on this machine (the box is gone
    # only in the shim), and a leg that died early may leave its local one
    subprocess.run(["pkill", "-f", str(tmp_path)], check=False)


def run_leg(w, *args, timeout=240):
    p = subprocess.run(["bash", str(w.repo / "tools" / "vultr_leg.sh"), "amd", "--skip-gates", *args],
                       env=w.env, cwd=w.repo, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def no_token_anywhere(root):
    for p in Path(root).rglob("*"):
        if p.is_file():
            assert TOKEN.encode() not in p.read_bytes(), "the token reached %s" % p


# ---------------------------------------------------------------- refusals

def test_refuses_without_the_token_file(world):
    world.token.unlink()
    rc, text = run_leg(world)
    assert rc == 2, text
    assert "REFUSING to rent" in text and str(world.token) in text and "does not exist" in text
    assert not world.shim.log, "no API call before the token check"


def test_refuses_an_empty_token_file(world):
    world.token.write_text("")
    rc, text = run_leg(world)
    assert rc == 2, text
    assert str(world.token) in text and "EMPTY" in text
    assert not world.shim.log


def test_refuses_a_dirty_tree(world):
    (world.repo / "stray.txt").write_text("x")
    rc, text = run_leg(world)
    assert rc == 3, text
    assert "dirty tree" in text
    assert not world.shim.creates


def test_refuses_over_the_dollar_cap(world):
    rc, text = run_leg(world, "--segment-lease", "120", "--dollar-cap", "10")
    assert rc == 2, text
    assert "above the --dollar-cap of $10" in text and "31.92" in text
    assert not world.shim.creates
    lease = (world.out / "lease.txt").read_text()
    assert "price_hourly=31.92" in lease and "max_cost=63.84" in lease and "verdict=REFUSED_OVER_CAP" in lease
    assert not world.lock.exists(), "the lock is released on a refusal"


def test_refuses_on_a_held_lock(world):
    world.lock.mkdir()
    (world.lock / "owner").write_text("lane=other\nnonce=someone-else\n")
    rc, text = run_leg(world)
    assert rc == 3, text
    assert "REFUSING to create" in text and str(world.lock) in text and "is held" in text
    assert not world.shim.creates
    assert world.lock.exists(), "another leg's lock is left alone"
    assert drv._amd_busy(text)


def test_refuses_when_a_tagged_bare_metal_exists(world):
    world.shim.bare_metals["old-1"] = dict(id="old-1", label="someone", plan="vc2-1c-1gb", tags=[TAG], region="ord",
                                           status="active", main_ip="10.0.0.1", date_created="then")
    rc, text = run_leg(world)
    assert rc == 3, text
    assert "ONE Vultr GPU bare metal at a time" in text and "old-1" in text
    assert not world.shim.creates
    assert drv._amd_busy(text)


def test_refuses_when_a_gpu_bare_metal_exists(world):
    world.shim.bare_metals["gpu-1"] = dict(id="gpu-1", label="by-hand", plan=PLAN, tags=[], region="ord",
                                           status="active", main_ip="10.0.0.2", date_created="then")
    rc, text = run_leg(world)
    assert rc == 3 and "gpu-1" in text, text
    assert not world.shim.creates


def test_no_stock_is_named_and_busy_to_the_driver(world):
    world.shim.stock = {"ewr": [], "ord": ["vbm-other"]}
    rc, text = run_leg(world)
    assert rc == 3, text
    assert "Vultr has NO STOCK of %s" % PLAN in text
    assert not world.shim.creates
    assert drv._amd_busy(text)
    assert "verdict=NO_STOCK" in (world.out / "lease.txt").read_text()


def test_region_override_outside_the_plan_is_no_stock(world):
    rc, text = run_leg(world, "--region", "sjc")
    assert rc == 3 and "NO STOCK" in text, text
    assert drv._amd_busy(text)


def test_account_limit_create_refusal_is_busy_and_swept(world):
    world.shim.create_error = (400, "Server add failed: You have reached the maximum monthly fee limit for this account.")
    rc, text = run_leg(world)
    assert rc == 4, text
    assert "Vultr REFUSED the create on account limits or stock" in text
    assert drv._amd_busy(text)
    td = (world.out / "teardown.txt").read_text()
    assert "sweep_by_name=none" in td and "destroy_confirmed=1" in td
    assert not world.lock.exists()
    no_token_anywhere(world.out)


def test_a_malformed_create_is_not_busy(world):
    world.shim.create_error = (400, "Invalid os_id.")
    rc, text = run_leg(world)
    assert rc == 4, text
    assert "Vultr create FAILED" in text
    assert not drv._amd_busy(text)


# ---------------------------------------------------------------- the happy path

def test_happy_path(world):
    rc, text = run_leg(world, "--minutes", "30")
    assert rc == 0, text
    s = world.shim
    assert len(s.creates) == 1
    req = json.loads((world.out / "create_request.json").read_text())
    assert req == s.creates[0]
    assert req["plan"] == PLAN and req["region"] == "ord" and req["os_id"] == 2284
    assert req["label"] == NAME and req["tags"] == [TAG] and req["sshkey_id"] == ["key-1"]
    assert req["activation_email"] is False
    assert [k["ssh_key"] for k in s.ssh_keys] == [" ".join(PUBKEY.split()[:2])], "the key is registered once, without its comment"
    resp = (world.out / "create_response.json").read_text()
    assert "shim-root-password-SECRET" not in resp and '"default_password": "<redacted>"' in resp
    lease = (world.out / "lease.txt").read_text()
    for want in ("provider=vultr", "plan=" + PLAN, "price_hourly=31.92", "price_source=hourly_cost",
                 "max_cost=15.96", "region=ord", "bare_metal_id=cb676a46", "deadline="):
        assert want in lease, (want, lease)
    # the verified delete
    (bm_id,) = s.deleted
    assert not s.bare_metals
    td = (world.out / "teardown.txt").read_text()
    assert "delete %s attempt 1 -> HTTP 204" % bm_id in td
    assert "verified_gone=%s GET HTTP 404" % bm_id in td and "destroy_confirmed=1" in td and "exit=0" in td
    # provisioning was polled pending -> active
    poll = (world.out / "status_poll.txt").read_text()
    assert "status=pending" in poll and "status=active main_ip=127.0.0.1" in poll
    # ROCm: installed, rebooted into the dkms driver, checked; the logs came home
    leg = (world.out / "leg.txt").read_text()
    assert "rebooted=1 boot_id_before=boot-1 boot_id_after=boot-2" in leg
    assert "rocm_ready=yes" in leg and "rocm_reboot_needed=1" in leg
    rocm = (world.out / "rocm" / "rocm.txt").read_text()
    assert "rocm_version=6.4.0-47" in rocm and "installer_exit=0" in rocm and "os_codename=noble" in rocm
    assert "amdgpu-install -y --usecase=rocm" in (world.out / "rocm" / "install.log").read_text()
    assert "gfx942_agents=8" in (world.out / "rocm" / "check.txt").read_text()
    # the dead-men: on the box armed, and again after the reboot; local cancelled after the 404
    dm = (world.out / "deadman.txt").read_text()
    assert "on_box_armed_ON_BOX_DEADMAN_ARMED" in dm and "on_box_armed_TOKEN_GET_HTTP=200" in dm
    assert "on_box_after_reboot_ON_BOX_DEADMAN_ARMED" in dm and "on_box_REBOOT_REARM=systemd_enabled" in dm
    assert "local_deadman=cancelled" in dm
    curlrc = world.fake / "root" / ".mojolearn-vultr.curlrc"
    assert oct(curlrc.stat().st_mode & 0o777) == "0o600"
    # the body ran and its results came home
    assert "BODY_RAN" in (world.out / "remote" / "extra.log").read_text()
    assert "archs=gfx942 column=amd" in (world.out / "remote" / "extra.log").read_text()
    remote_leg = (world.out / "remote" / "leg.txt").read_text()
    assert "provider=vultr" in remote_leg and "extra_exit=0" in remote_leg and "rocm_version=6.4.0-47" in remote_leg
    assert "source_sha256_match=yes" in leg and "box_key_in_ps=not_visible" in leg and "local_key_in_ps=not_visible" in leg
    assert (world.out / "remote_body.sh").exists() and (world.out / "extra_body.sh").exists()
    assert not world.lock.exists()
    no_token_anywhere(world.out)


def test_dry_run_needs_no_token_and_calls_nothing(world):
    world.token.unlink()
    rc, text = run_leg(world, "--dry-run")
    assert rc == 0, text
    assert "DRY RUN: GREEN" in text
    assert not world.shim.log


# ---------------------------------------------------------------- the driver

def test_driver_busy_mapping():
    busy = [
        "REFUSING to create mojolearn-extra-amd-vultr: Vultr has NO STOCK of %s in any of: ewr ord" % PLAN,
        "Vultr REFUSED the create on account limits or stock (HTTP 400): limit",
        "REFUSING to create x: the Vultr GPU lock /tmp/mojolearn-vultr-gpu.lock is held (5s old) by: lane=a",
        "REFUSING to create x: ONE Vultr GPU bare metal at a time on this account, and these already exist:",
        "REFUSING to create mojolearn-extra-amd: ONE GPU droplet at a time on this account",
    ]
    for t in busy:
        assert drv._amd_busy(t), t
    for t in ("Vultr create FAILED (HTTP 400): Invalid os_id.",
              "segment lease REFUSED: 120 minutes of x at $31.92/h is up to $63.84, above the --dollar-cap of $10",
              "ROCm 6.4.0 IS NOT READY on the box"):
        assert not drv._amd_busy(t), t


def test_driver_vultr_command(tmp_path, monkeypatch):
    calls = []

    class R:
        returncode = 0

    def fake_run(cmd, cwd=None, env=None, stdout=None, stderr=None):
        calls.append((cmd, env))
        return R()

    monkeypatch.setattr(drv.subprocess, "run", fake_run)
    rerendered = []

    def rerender(devices):
        rerendered.append(devices)
        return tmp_path / "bodies" / "A-2-vultr.sh"

    spec = dict(amd_devices="0", lease_minutes=90, dollar_cap=12, vultr_region="ord", vultr_dollar_cap=100)
    e = dict(route="A", segment="2")
    env = dict(MOJOLEARN_GEMM_LEG_EXTRA=str(tmp_path / "bodies" / "A-2.sh"))
    res = tmp_path / "legs"
    res.mkdir()
    got = drv._rent_amd_once(spec, e, res, env, ["--segment-lease", "90", "--dollar-cap", "12"], ["vultr"], tmp_path, 1, rerender=rerender)
    assert got == (res / "leg-vultr-1", 0)
    (cmd, venv), = calls
    assert cmd == ["bash", "tools/vultr_leg.sh", "amd", "--segment-lease", "150", "--dollar-cap", "100",
                   "--plan", "vbm-256c-2048gb-8-mi300x-gpu", "--region", "ord"]
    assert rerendered == ["0,1,2,3,4,5,6,7"]
    assert venv["MOJOLEARN_GEMM_LEG_EXTRA"] == str(tmp_path / "bodies" / "A-2-vultr.sh")
    assert env["MOJOLEARN_GEMM_LEG_EXTRA"].endswith("A-2.sh"), "the other providers keep the spec-wide body"


def test_driver_walks_past_a_busy_vultr(tmp_path, monkeypatch):
    seen = []

    def fake_run(cmd, cwd=None, env=None, stdout=None, stderr=None):
        seen.append(cmd[1])
        if cmd[1] == "tools/vultr_leg.sh":
            stdout.write("REFUSING to create x: Vultr has NO STOCK of y in any of: ewr\n")
            stdout.flush()
            return type("R", (), {"returncode": 3})()
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(drv.subprocess, "run", fake_run)
    res = tmp_path / "legs"
    res.mkdir()
    got = drv._rent_amd_once(dict(amd_devices="0"), dict(route="A", segment="2"), res, {}, ["--minutes", "60"],
                             ["vultr", "do"], tmp_path, 1)
    assert seen == ["tools/vultr_leg.sh", "tools/do_extra_leg.sh"]
    assert got == (res / "leg-do-1", 0)
