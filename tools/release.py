#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ONE COMMAND FOR A RELEASE (2026-09-22): docs/RELEASE_CHECKLIST.md, walked.

    pixi run release 0.8.15 --dry-run          the plan and what is already done
    pixi run release 0.8.15                    everything up to publication
    pixi run release 0.8.15 --publish pypi     ... and publish, finish line, record

Every step is idempotent and resumable. State lives in
~/mojolearn-evidence/release/<version>/state.json (MOJOLEARN_EVIDENCE_ROOT
moves it); everything after the freeze is keyed by the frozen commit, so a new
freeze commit starts those steps over in a fresh <version>/<commit12>/
directory and a rerun at the same commit skips what is done. A step is done
when its OUTPUT checks out (a wheel of the right commit, a PASSED receipt for
that wheel's sha256, a complete release-check record, the file on PyPI), not
merely because the state file says so.

THE STEPS, IN ORDER
  freeze-version     _version.py and pyproject.toml say <version> (edited if not)
  freeze-changelog   CHANGELOG.md has `## <version> (published YYYY-MM-DD)`; the
                     prose is the releaser's, so a missing entry stops here
  freeze-docs-facts  pixi run write-docs-facts (CITATION and marked spans)
  freeze-commit      commit exactly those files if they changed, push, and
                     freeze HEAD (the tree must be clean and pushed)
  rehearsal          pixi run release-rehearsal (docs facts, ext lists, wheel
                     audit, Python suite, leg dry runs), logs kept
  reuse-plan         tools/release_reuse.py: for every binding of every set
                     (cuda sm_90a, cuda sm_89, hip gfx942, the host bindings,
                     the runtime closure, the macOS wheel) REUSE the bytes the
                     last PUBLISHED release shipped when the binding's identity
                     (source closure, toolchain, flags, builder scripts, box
                     image or Apple toolchain) is unchanged, else BUILD; the
                     table is printed, and in --dry-run in full
  linux-builds       a leg per Linux set that has a binding to BUILD (none for a
                     Python-only release: nothing is rented for builds), all
                     launched AT ONCE, detached, with no stagger. The default
                     backend `gpu-legs` builds on the GPU boxes: RunPod NVIDIA
                     (walking NVIDIA_WALK on no stock) and a DigitalOcean MI325X,
                     or a Hot Aisle 1x MI300X when DigitalOcean has a GPU
                     droplet live or no token (--amd-build-provider do|hotaisle
                     or MOJOLEARN_AMD_PROVIDER pins one). `cpu-box` (one RunPod
                     CPU pod per set) is opt-in by name
  macos-build        mac_slot --slots 4 run -- build_release_wheel.sh, byte LM on,
                     while the Linux legs build; its REUSE bindings placed from
                     the published macOS wheel
  macos-smoke        qualify_verifier_wheel.py --scope expanded under the Metal lock
  release-check      pixi run -e test release-check: the Apple (Metal) column of
                     the changed lanes, one fit per cell. The CPU pass runs too
                     only with --cpu-column (opt-in)
  linux-wait         every launched leg finished, its proof complete for this
                     commit, and every host binding byte-identical across legs
  linux-assemble     the set directories the packer packs: each leg's set, or one
                     synthesized from the published wheel, with every REUSE
                     binding's published bytes verified by sha256 (reuse.json)
  linux-pack         pixi run -e pkg pack-linux-wheel, audit.sh, strip; the
                     payload records per binding built or reused, and from which
                     release
  gpu-columns        the NVIDIA and AMD wheel columns AT ONCE, each a detached
                     tools/release_wheel_smoke.sh on the changed lanes from the
                     installed wheel, one fit per cell:
                       NVIDIA  --rent (one RTX 4090 pod, walking SMOKE_WALK on no
                               stock): the expanded smoke AND the NVIDIA column
                       AMD     --vendor hip --rent --provider auto (RunPod MI300X,
                               then Hot Aisle MI300X, then DigitalOcean MI325X;
                               gfx942 every way; --amd-provider pins one)
                     each diffed against the Apple column of this release (and
                     the CPU column with --cpu-column); both are awaited and both
                     reported, and a column with a PASSED record for this wheel
                     is not rerun. Then ONE diff of every column together
                     (Apple, NVIDIA, AMD[, CPU]): any DIVERGENT or MOVED cell
                     stops the release
  publish-linux      tools/release_linux_publish.sh ... --light-smoke   } only with
  publish-macos      tools/release_linux_publish.sh ... --light-smoke   } --publish
  finish-line        pip install mojolearn==<version> in a fresh venv on this Mac
  record             bench/results/release_verification/<date>_pypi_<v>/, committed

Nothing is published without `--publish none|testpypi|pypi`; without it the run
stops after gpu-columns and says so. The Linux wheel publishes only when no
column (Apple, NVIDIA, AMD, and CPU when run) shows a DIVERGENT cell against
any other (bitwise identity across GPUs is the point; only changed lanes run).
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
PUBLISH_CHOICES = ("none", "testpypi", "pypi")
sys.path.insert(0, str(ROOT / "tools"))
import release_reuse  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args, check=True):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                          check=check).stdout.strip()


def wheel_commit(path):
    """The source commit a wheel records (mojolearn/identity_columns/COMMIT), or None."""
    try:
        with zipfile.ZipFile(path) as z:
            return z.read("mojolearn/identity_columns/COMMIT").decode().strip()
    except (OSError, KeyError, zipfile.BadZipFile):
        return None


def smoke_passed(results, wheel):
    """A light-smoke receipt that is PASSED, expanded, and about THIS wheel."""
    try:
        d = json.loads(Path(results).read_text())
    except (OSError, ValueError):
        return False
    return (d.get("status") == "PASSED" and d.get("scope") == "expanded"
            and d.get("wheel_sha256") == sha256(wheel)
            and d.get("source_commit") == wheel_commit(wheel))


# ---------------------------------------------------------------- freeze helpers
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?$")


def set_version(root, version):
    """Write <version> into _version.py and pyproject.toml; return the files changed."""
    changed = []
    for rel, pattern, repl in (
            ("python/mojolearn/_version.py", r'^__version__ = "[^"]*"$', f'__version__ = "{version}"'),
            ("python/pyproject.toml", r'^version = "[^"]*"$', f'version = "{version}"')):
        path = Path(root) / rel
        text = path.read_text()
        new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
        if n != 1:
            raise SystemExit(f"release: no version line in {rel}")
        if new != text:
            path.write_text(new)
            changed.append(rel)
    return changed


def changelog_date(root, version):
    """The `published YYYY-MM-DD` date of <version>'s CHANGELOG heading, or None."""
    text = (Path(root) / "CHANGELOG.md").read_text()
    m = re.search(r"^## " + re.escape(version) + r" \(published (\d{4}-\d{2}-\d{2})\)\s*$", text, re.M)
    return m.group(1) if m else None


def release_files():
    """What a version bump edits by hand or by write-docs-facts' citation sync."""
    return ["python/mojolearn/_version.py", "python/pyproject.toml", "CHANGELOG.md", "CITATION.cff"]


def docs_fact_files():
    """The documents `write-docs-facts` rewrites (tools/docs_facts.py DOCS)."""
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import docs_facts
        return list(docs_facts.DOCS)
    finally:
        sys.path.pop(0)


# ---------------------------------------------------------------- Linux build legs
class Leg:
    """One detached build process and where its release-build tree lands."""

    def __init__(self, name, vendor, arch, command, env, release_build, workdir, out_dir):
        self.name, self.vendor, self.arch = name, vendor, arch
        self.command, self.env = command, env
        self.release_build = Path(release_build)
        self.workdir = Path(workdir)
        #: what a relaunch moves aside (the leg refuses an existing output)
        self.out_dir = Path(out_dir)
        self.log = self.workdir / f"{name}.log"
        self.exit_file = self.workdir / f"{name}.exit"
        self.pid_file = self.workdir / f"{name}.pid"

    def describe(self):
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env.items())
        return f"{env} {' '.join(shlex.quote(c) for c in self.command)}"

    def pid(self):
        try:
            return int(self.pid_file.read_text())
        except (OSError, ValueError):
            return None

    def running(self):
        pid = self.pid()
        if pid is None or self.exit_file.exists():
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def exit_code(self):
        try:
            return int(self.exit_file.read_text().strip())
        except (OSError, ValueError):
            return None

    def proof_ok(self, commit):
        """The leg's build proof: complete, exit 0, this commit, this architecture's set."""
        try:
            proof = json.loads((self.release_build / "build" / "build-provenance.json").read_text())
        except (OSError, ValueError):
            return False
        return (proof.get("complete") is True and proof.get("build_exit") == 0
                and proof.get("source_commit") == commit
                and (self.release_build / "build" / "sets" / self.vendor / self.arch).is_dir())

    def done(self, commit):
        """Finished and its output checks out, so a (re)launch leaves it alone."""
        return self.exit_code() == 0 and self.proof_ok(commit)


class ColumnLeg(Leg):
    """One GPU wheel column (tools/release_wheel_smoke.sh), run detached like a
    build leg: its own session, its exit code written last, a failed attempt
    moved aside before a relaunch. `check` says whether its output (a PASSED
    receipt for this wheel, a column with no DIVERGENT cell) is already there,
    whatever the exit file says."""

    def __init__(self, name, vendor, command, workdir, out_dir, check):
        super().__init__(name, vendor, "", command, {}, out_dir, workdir, out_dir)
        self.check = check

    def done(self, commit):
        return bool(self.check())


def gpu_legs(ctx):
    """THE DEFAULT ROUTE: two RunPod NVIDIA legs and one DigitalOcean MI325X leg,
    exactly docs/RELEASE_CHECKLIST.md section 2's commands, output directories
    pinned under the release directory so this script can find them. The AMD
    leg does not wait for NVIDIA: its core-host probe is skipped and
    pack_wheel.py compares every host binding across the legs (linux-wait
    names a mismatch first)."""
    legs_dir = ctx.rel / "legs"
    runpod = dict(MOJOLEARN_RUNPOD_KEY_FILE=os.path.expanduser(
        os.environ.get("MOJOLEARN_RUNPOD_KEY_FILE", "~/.mojolearn_runpod_key")),
        MOJOLEARN_NVIDIA_CAMPAIGN="7")
    legs = []
    for arch, gpu in ((a, NVIDIA_WALK[a][0]) for a in ("sm_90a", "sm_89")):
        out = legs_dir / f"cuda-{arch}"
        legs.append(Leg(f"cuda-{arch}", "cuda", arch,
                        ["sh", "tools/gemm_remote_leg.sh", "nvidia", "--payload", "mamba",
                         "--source-ref", ctx.commit, "--gpu", gpu, "--allow-concurrent", "--rent",
                         "--segment-lease", "120", "--dollar-cap", "15"],
                        dict(runpod, MOJOLEARN_GPU_ARCHS=arch, MOJOLEARN_GEMM_LEG_OUT=str(out)),
                        out / "remote" / "release-build", legs_dir, out))
    route, why = amd_build_route(ctx)
    command, env = amd_leg_command(ctx, route, legs_dir)
    leg = Leg("hip-gfx942", "hip", "gfx942", command, env,
              legs_dir / "hip-gfx942" / "release-build", legs_dir, legs_dir / "hip-gfx942")
    leg.provider, leg.provider_reason = route, why
    legs.append(leg)
    return legs


# ---------------------------------------------------------------- the AMD build route
#: Where the AMD (hip gfx942) build leg rents. do: tools/do_release061_leg.sh on
#: a DigitalOcean MI325X, the route every release through 0.8.18 used.
#: hotaisle: tools/hotaisle_release_leg.sh on a Hot Aisle 1x MI300X, the same
#: remote build and the same output tree (2026-09-25). auto: do, unless
#: DigitalOcean has a GPU droplet live (the DigitalOcean leg refuses a rental
#: then, one GPU droplet at a time on the account) or no usable token.
AMD_BUILD_PROVIDERS = ("auto", "do", "hotaisle")
DO_LIVE_REFUSAL = "GPU droplet(s) already live"


def amd_build_want(args):
    return getattr(args, "amd_build_provider", None) or os.environ.get("MOJOLEARN_AMD_PROVIDER") or "auto"


def do_gpu_busy(token_file=None, api=None):
    """(busy, why) for the DigitalOcean AMD route, from one free GET. Never raises."""
    token_file = os.path.expanduser(token_file or os.environ.get("MOJOLEARN_DO_TOKEN_FILE", "~/.mojolearn_do_token"))
    api = api or os.environ.get("MOJOLEARN_DO_API", "https://api.digitalocean.com/v2")
    try:
        token = Path(token_file).read_text().strip()
    except OSError:
        return True, f"no DigitalOcean token at {token_file}"
    if not token:
        return True, f"the DigitalOcean token file {token_file} is empty"
    req = urllib.request.Request(api + "/droplets?per_page=200", headers={"Authorization": "Bearer " + token})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            droplets = json.load(r).get("droplets") or []
    except Exception as exc:  # the leg would refuse the same way
        return True, f"the DigitalOcean droplet listing failed ({type(exc).__name__})"
    live = [f"{d.get('id')}:{d.get('name')}:{d.get('size_slug', '')}" for d in droplets
            if str(d.get("size_slug") or "").startswith("gpu-")]
    if live:
        return True, "DigitalOcean GPU droplet(s) live, so its leg would refuse: " + " ".join(live)
    return False, "no DigitalOcean GPU droplet live"


def amd_build_route(ctx):
    """(route, why): the provider the AMD build leg rents from, decided once per run."""
    cached = getattr(ctx, "_amd_route", None)
    if cached:
        return cached
    want = amd_build_want(ctx.args)
    if want not in AMD_BUILD_PROVIDERS:
        raise StepFailed(f"the AMD build provider must be one of {', '.join(AMD_BUILD_PROVIDERS)}, not {want!r}")
    if want != "auto":
        route = (want, f"pinned ({want})")
    else:
        probe = getattr(ctx, "amd_do_probe", None) or do_gpu_busy
        busy, why = probe()
        route = ("hotaisle", "auto: " + why) if busy else ("do", "auto: " + why)
    ctx._amd_route = route
    ctx.say(f"  AMD build leg: {route[0]} ({route[1]})")
    return route


def amd_leg_command(ctx, route, legs_dir):
    """The AMD build leg's command and environment on ROUTE; the same output tree either way."""
    if route == "hotaisle":
        cmd = ["bash", "tools/hotaisle_release_leg.sh", ctx.commit, "--rent"]
        env = dict(MOJOLEARN_RELEASE_RESULTS_ROOT=str(legs_dir))
    else:
        token = os.path.expanduser(os.environ.get("MOJOLEARN_DO_TOKEN_FILE", "~/.mojolearn_do_token"))
        cmd = ["bash", "tools/do_release061_leg.sh", ctx.commit, token, "--rent"]
        env = dict(MOJOLEARN_RELEASE_UBUNTU22="1", MOJOLEARN_RELEASE_RESULTS_ROOT=str(legs_dir))
    if ctx.args.amd_expect_from:
        cmd += ["--expect-from", ctx.args.amd_expect_from]
    return cmd, env


def do_refused_live(log):
    try:
        return DO_LIVE_REFUSAL in log.read_text(errors="replace")
    except OSError:
        return False


#: RunPod stock comes and goes by GPU type. A leg or a smoke whose create is
#: answered "no instances currently available" walks to the next type of the
#: same architecture (0.8.16, 2026-09-23: the L40S leg failed twice on stock
#: and the RTX 4090 smoke once, each stopping the release).
NVIDIA_WALK = {
    "sm_90a": ["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H100 PCIe", "NVIDIA H200"],
    "sm_89": ["NVIDIA L40S", "NVIDIA L40", "NVIDIA RTX 6000 Ada Generation", "NVIDIA GeForce RTX 4090"],
}
#: The NVIDIA column needs any NVIDIA GPU (it tests the wheel's set for the
#: GPU it lands on); the H100 and H200 close the walk when no sm_89 box has
#: stock (0.8.19: none did for hours).
SMOKE_WALK = ["NVIDIA GeForce RTX 4090", "NVIDIA L40S", "NVIDIA L40", "NVIDIA RTX 6000 Ada Generation",
              "NVIDIA H100 80GB HBM3", "NVIDIA H200"]
NO_STOCK = "no instances currently available"
#: RunPod create failures that say nothing about our request, so the walk
#: tries the next GPU type (0.8.19: "Something went wrong" ended the sm_89
#: walk one type short of the RTX 4090).
CREATE_TRANSIENT = ("no longer any instances available", "Something went wrong",
                    "Please try again later")


def no_stock(log, *dirs):
    """Whether RunPod answered the create with no stock. The leg's log carries
    only "create returned HTTP 500"; RunPod's words are in the
    create_response.json it keeps in its output directory (0.8.19,
    2026-09-25: both NVIDIA legs failed on stock and never walked)."""
    texts = [log] + [p for d in dirs if d for p in Path(d).glob("**/create_response.json")]
    for t in texts:
        try:
            text = Path(t).read_text(errors="replace")
            if NO_STOCK in text or any(phrase in text for phrase in CREATE_TRANSIENT):
                return True
        except OSError:
            pass
    return False


#: Build routes by name. A route returns the Leg list for ctx; launch, wait,
#: pack and resume are route-independent. The GPU legs are the default (the
#: release builds on the GPUs it ships for); the CPU pods are opt-in by name.
#: RunPod CPU flavors a release build pod may land on, 32 vCPU each: general
#: purpose (4 GiB per vCPU) then memory optimized (8). 0.8.19's first CPU
#: launch found neither general-purpose flavor in stock; compute optimized
#: (2 GiB per vCPU) is left out, too little memory for 16 compile jobs.
CPU_BOX_FLAVORS = "cpu5g,cpu3g,cpu5m,cpu3m"


def cpu_legs(ctx):
    """OPT-IN ROUTE (--build-backend cpu-box): the three Linux sets compile on
    RunPod CPU pods, one pod per set, all three at once, with no GPU present
    (tools/release_linux_build.sh --archs <one>: Mojo compiles each set ahead
    of time from --target-accelerator alone). CPU pods do not wait on GPU
    stock. Proven at d181d9792 (0.8.14): cuda/sm_90a and cuda/sm_89 132 of 132
    binaries byte-identical to the GPU-box builds; gfx942 codegen varies run to
    run on ANY box, so the binding cache freezes it either way. Identity is
    still read back on the silicon: the NVIDIA and AMD wheel columns run the
    built wheel on real GPUs before anything publishes."""
    legs_dir = ctx.rel / "legs"
    legs = []
    for vendor, arch in (("cuda", "sm_90a"), ("cuda", "sm_89"), ("hip", "gfx942")):
        name = f"{vendor}-{arch}"
        out = legs_dir / name
        legs.append(Leg(name, vendor, arch,
                        ["bash", "tools/release_linux_build.sh", ctx.commit, "--rent", "--archs", arch,
                         "--flavors", CPU_BOX_FLAVORS, "--out", str(out)],
                        {}, out / name / "release-build", legs_dir, out))
    return legs


#: "gpu-legs" is the default (Andrew, 2026-09-25: no CPU anywhere by default);
#: "cpu-box" is opt-in by name.
BUILD_BACKENDS = {"cpu-box": cpu_legs, "gpu-legs": gpu_legs}


def linux_legs(ctx):
    """The legs the reuse plan says to launch: only the sets with a binding
    to BUILD (plus the cheapest leg when only a host binding or the runtime
    needs one). Every leg of the backend when there is no plan."""
    legs = BUILD_BACKENDS[ctx.args.build_backend](ctx)
    plan = ctx.plan()
    if plan is None:
        return legs
    return [l for l in legs if l.name in plan["legs"]]


def launch_detached(ctx, legs):
    """THE ONE LAUNCH POINT for the Linux builds and the GPU columns. Every leg
    is launched at once, with no stagger (Andrew, 2026-09-25: a release is
    parallel only). Each runs detached (its own session, so this script can
    exit or be interrupted without killing a paid rental) and writes its exit
    code last. A leg that is running is left alone, a leg whose output checks
    out is not rerun, and a leg that finished but failed is moved aside (never
    deleted) and relaunched."""
    started = 0
    for leg in legs:
        if leg.running():
            ctx.say(f"  {leg.name}: already running (pid {leg.pid()}), log {leg.log}")
            continue
        if leg.done(ctx.commit):
            ctx.say(f"  {leg.name}: already done")
            continue
        leg.workdir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        for p in (leg.log, leg.exit_file, leg.pid_file):
            if p.exists():
                p.rename(p.with_name(p.name + f".failed-{stamp}"))
        out = leg.out_dir
        if out.exists():
            out.rename(out.with_name(out.name + f".failed-{stamp}"))
            ctx.say(f"  {leg.name}: previous attempt moved aside to {out.name}.failed-{stamp}")
        wrapped = (f"{' '.join(shlex.quote(c) for c in leg.command)} > {shlex.quote(str(leg.log))} 2>&1; "
                   f"echo $? > {shlex.quote(str(leg.exit_file))}")
        proc = ctx.spawn(["sh", "-c", wrapped], dict(os.environ, **leg.env))
        leg.pid_file.write_text(str(proc))
        ctx.say(f"  {leg.name}: launched pid {proc}, log {leg.log}")
        started += 1
    return started


def host_digest_mismatches(legs):
    """{binding: {leg: sha256}} for every host binding whose bytes differ across
    legs, or that a leg lacks. pack_wheel.py refuses the same thing; this names
    the leg before the packer does."""
    table = {}
    for leg in legs:
        host = leg.release_build / "build" / "sets" / leg.vendor / leg.arch / "host"
        for so in sorted(host.glob("*.so")):
            table.setdefault(so.name, {})[leg.name] = sha256(so)
    return {name: row for name, row in table.items()
            if len(row) != len(legs) or len(set(row.values())) != 1}


# ---------------------------------------------------------------- the run
class Release:
    def __init__(self, args, runner=None):
        self.args = args
        self.version = args.version
        ev = Path(os.environ.get("MOJOLEARN_EVIDENCE_ROOT", os.path.expanduser("~/mojolearn-evidence")))
        self.base = Path(args.state_dir) if args.state_dir else ev / "release" / self.version
        self.state_path = self.base / "state.json"
        self.state = self.load()
        self.runner = runner
        self.dry = args.dry_run
        self.commands_shown = 0
        self.evidence = ev
        self._plan = None

    # state
    def load(self):
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {"version": self.version, "commit": None, "steps": {}}

    def save(self):
        if self.dry:
            return
        self.base.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
        tmp.replace(self.state_path)

    @property
    def commit(self):
        return self.state.get("commit") or git("rev-parse", "HEAD")

    @property
    def rel(self):
        return self.base / self.commit[:12]

    def mark(self, step, **data):
        self.state["steps"][step] = dict(done=True, commit=self.state.get("commit"), at=now(), **data)
        self.save()

    def recorded(self, step):
        rec = self.state["steps"].get(step)
        return rec if rec and rec.get("done") and rec.get("commit") == self.state.get("commit") else None

    # execution
    def say(self, msg):
        print(msg, flush=True)

    def sleep(self, seconds):
        if not self.dry:
            time.sleep(seconds)

    def run(self, cmd, env=None, log=None, cwd=None):
        """Run one command to completion; its output teed to `log` when given."""
        shown = " ".join(shlex.quote(str(c)) for c in cmd)
        extra = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in (env or {}).items())
        self.say(f"  $ {extra + ' ' if extra else ''}{shown}")
        self.commands_shown += 1
        if self.dry:
            return 0
        if self.runner:
            return self.runner(cmd, env, log)
        full_env = dict(os.environ, **(env or {}))
        if log:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            with open(log, "w") as fh:
                proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd or ROOT, env=full_env,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    sys.stdout.write("    " + line)
                    fh.write(line)
                return proc.wait()
        return subprocess.run([str(c) for c in cmd], cwd=cwd or ROOT, env=full_env).returncode

    def must(self, cmd, env=None, log=None, what=None):
        rc = self.run(cmd, env, log)
        if rc != 0:
            raise StepFailed(f"{what or cmd[0]} exited {rc}" + (f"; log {log}" if log else ""))

    def spawn(self, cmd, env):
        if self.runner:
            return self.runner(cmd, env, None, detach=True)
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)
        return proc.pid

    # ------------------------------------------------------------ steps
    def step_freeze_version(self):
        files = [f for f in ("python/mojolearn/_version.py", "python/pyproject.toml")]
        current = re.search(r'^__version__ = "([^"]*)"', (ROOT / files[0]).read_text(), re.M).group(1)
        if current == self.version and f'version = "{self.version}"' in (ROOT / files[1]).read_text():
            return "already " + self.version
        if self.dry:
            return f"would set {current} -> {self.version} in {', '.join(files)}"
        return "set in " + ", ".join(set_version(ROOT, self.version))

    def step_freeze_changelog(self):
        date = changelog_date(ROOT, self.version)
        if not date:
            raise StepFailed(f"CHANGELOG.md has no `## {self.version} (published YYYY-MM-DD)` heading. "
                             "Write the entry (publication date in UTC) and rerun.")
        return "published " + date

    def step_freeze_docs_facts(self):
        self.must(["pixi", "run", "write-docs-facts"], what="write-docs-facts")
        return "written"

    def step_freeze_commit(self):
        head = git("rev-parse", "HEAD")
        frozen = self.state.get("commit")
        if frozen and head != frozen and any(self.recorded(s) for s in ("publish-linux", "publish-macos")):
            return (f"kept at {frozen}: a wheel of it is already published, so HEAD ({head[:12]}) is not "
                    "refrozen; a new source state needs a new version")
        allowed = set(release_files()) | set(docs_fact_files())
        # NOT through git(): it strips the output, and the first porcelain line
        # then loses its leading status blank, so ` M CITATION.cff` read as
        # `ITATION.cff` and the release's own citation sync was refused.
        porcelain = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                                   capture_output=True, text=True, check=True).stdout
        dirty = [line[3:] for line in porcelain.splitlines() if line.strip()]
        others = [p for p in dirty if p not in allowed]
        if others:
            raise StepFailed("tracked files changed outside the release bump; commit or revert them first: "
                             + " ".join(others[:10]))
        if dirty:
            self.must(["git", "-C", ROOT, "add", "--", *dirty], what="git add")
            self.must(["git", "-C", ROOT, "commit", "-q", "-m", f"Release {self.version}", "-m",
                       "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"],
                      what="git commit")
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        if branch == "HEAD":
            raise StepFailed("detached HEAD; release from a branch (main) so the commit can be pushed")
        # A rerun after main moved on (a fix landed while the legs built) must
        # not push the frozen commit over the newer main, and needs no push at
        # all when origin already contains it (0.8.16's second run stopped
        # here on a rejected non-fast-forward push).
        self.must(["git", "-C", ROOT, "fetch", "-q", "origin", branch], what="git fetch")
        contained = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", "HEAD", f"origin/{branch}"]).returncode == 0
        if not contained:
            self.must(["git", "-C", ROOT, "push", "origin", f"HEAD:refs/heads/{branch}"], what="git push")
        if self.dry:
            return f"would freeze {'the bump commit' if dirty else head}"
        head = git("rev-parse", "HEAD")
        if self.state.get("commit") != head:
            self.state = {"version": self.version, "commit": head,
                          "steps": {k: v for k, v in self.state["steps"].items() if k.startswith("freeze-")}}
            self.save()
        return "frozen at " + head

    def step_rehearsal(self):
        work = self.rel / "rehearsal"
        self.must(["pixi", "run", "release-rehearsal", "--keep", work], log=self.rel / "rehearsal.log",
                  what="release-rehearsal")
        return "PASS, logs " + str(work)

    # ------------------------------------------------------------ reuse
    @property
    def plan_path(self):
        return self.rel / "reuse" / "plan.json"

    def plan(self):
        """The reuse plan of the frozen commit: read from the release
        directory when this commit's plan is there, else computed (and kept
        in memory only under --dry-run)."""
        if self._plan is not None:
            return self._plan
        try:
            doc = json.loads(self.plan_path.read_text())
            if doc.get("schema") == release_reuse.PLAN_SCHEMA and doc.get("commit") == self.commit:
                self._plan = doc
                return doc
        except (OSError, ValueError):
            pass
        self._plan = release_reuse.make_plan(self.commit, self.evidence / "release" / "identities")
        return self._plan

    def step_reuse_plan(self):
        plan = self.plan()
        self.say(release_reuse.render(plan, full=self.dry))
        prev = plan.get("previous") or {}
        summary = "%d REUSE, %d BUILD, legs: %s" % (
            len(release_reuse.plan_rows(plan, decision="REUSE")), len(release_reuse.plan_rows(plan, decision="BUILD")),
            ", ".join(plan["legs"]) or "none")
        if self.dry:
            return "would write " + str(self.plan_path) + "; " + summary
        if not self.plan_path.is_file() or json.loads(self.plan_path.read_text()).get("commit") != self.commit:
            self.plan_path.parent.mkdir(parents=True, exist_ok=True)
            self.plan_path.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
        return f"against {prev.get('version', 'no published release')}: {summary}; {self.plan_path}"

    def previous_wheel(self, platform):
        """The last published wheel for `platform`, verified against the
        record (a local copy, else PyPI)."""
        prev = (self.plan().get("previous") or {})
        if self.dry:
            info = prev.get(platform) or {}
            return Path("<published %s wheel %s>" % (platform, info.get("wheel", "?")))
        return release_reuse.published_wheel(prev, platform, self.evidence)

    def step_linux_builds(self):
        legs = linux_legs(self)
        prev = (self.plan().get("previous") or {}).get("version")
        if not legs:
            return f"no build leg: every Linux binding is taken from {prev} (nothing rented for builds)"
        for leg in legs:
            self.say(f"  {leg.name}: {leg.describe()}  [{self.plan()['leg_reasons'].get(leg.name, '')}]")
        if self.dry:
            return f"would launch {len(legs)} leg(s) ({self.args.build_backend})"
        launch_detached(self, legs)
        return f"{len(legs)} leg(s) launched or running ({self.args.build_backend})"

    def macos_wheel(self):
        found = sorted((self.rel / "macos").glob("mojolearn-*-macosx_*.whl"))
        return found[-1] if found else None

    def step_macos_build(self):
        w = self.macos_wheel()
        if w and wheel_commit(w) == self.commit:
            return "have " + w.name
        env = dict(MOJOLEARN_PACKAGE_BYTE_LM="1", MOJOLEARN_BUILD_JOBS="4", MOJOLEARN_COMPILE_JOBS="1")
        plan = self.plan()
        reuse = release_reuse.plan_rows(plan, release_reuse.MACOS, "REUSE")
        if reuse:
            store = self.rel / "reuse" / "macos"
            if self.dry:
                self.say(f"  {len(reuse)} macOS binding(s) would be placed from the published "
                         f"{plan['previous']['version']} wheel, verified against its RECORD")
            else:
                release_reuse.assemble_macos(plan, self.previous_wheel("macos"), store, say=self.say)
            env.update(MOJOLEARN_REUSE_PLAN=str(store / "macos-plan.json"), MOJOLEARN_REUSE_DIR=str(store))
        self.must([PY, "tools/mac_slot.py", "--slots", "4", "run", "--", "./packaging/macos/build_release_wheel.sh"],
                  env=env, log=self.rel / "macos-build.log", what="build_release_wheel.sh")
        if self.dry:
            return "would copy python/dist/*.whl to " + str(self.rel / "macos")
        built = sorted((ROOT / "python" / "dist").glob(f"mojolearn-{self.version}-*-macosx_*.whl"))
        if len(built) != 1 or wheel_commit(built[0]) != self.commit:
            raise StepFailed(f"expected one macOS wheel of {self.commit} in python/dist, found {built}")
        (self.rel / "macos").mkdir(parents=True, exist_ok=True)
        dest = self.rel / "macos" / built[0].name
        shutil.copy2(built[0], dest)
        return f"{dest.name} sha256 {sha256(dest)}"

    def smoke_python(self):
        for name in ("python3.12", "python3.13", "python3.11", "python3.14", "python3.10"):
            if shutil.which(name):
                return shutil.which(name)
        raise StepFailed("no python3.10..3.14 on PATH for the macOS smoke venv")

    def step_macos_smoke(self):
        w = self.macos_wheel()
        out = self.rel / "smoke-macos"
        if w and smoke_passed(out / "results.json", w):
            return "PASSED (receipt exists)"
        if not w and not self.dry:
            raise StepFailed("no macOS wheel; run macos-build")
        if out.exists() and not self.dry:
            out.rename(out.with_name(out.name + ".failed-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")))
        self.must([PY, "tools/mac_slot.py", "--wait-timeout", "3600", "metal", "--",
                   PY, "tools/qualify_verifier_wheel.py", w or "<macos wheel>", "--scope", "expanded",
                   "--python", self.smoke_python() if not self.dry else "python3.12",
                   "--expected-source-commit", self.commit, "--output", out],
                  log=self.rel / "macos-smoke.log", what="macOS smoke")
        if not self.dry and not smoke_passed(out / "results.json", w):
            raise StepFailed(f"macOS smoke receipt is not PASSED for this wheel: {out / 'results.json'}")
        return "PASSED"

    def release_check_dir(self):
        base = os.environ.get("MOJOLEARN_RELEASE_CHECK_DIR") or os.path.expanduser("~/mojolearn-evidence/release-check")
        return Path(base) / self.commit[:12]

    def check_backends(self):
        """The release-check passes this release runs: the Apple (Metal) column
        always, the CPU column only with --cpu-column (opt-in, 2026-09-25)."""
        return ("metal", "cpu") if getattr(self.args, "cpu_column", False) else ("metal",)

    def release_check_complete(self):
        for backend in self.check_backends():
            d = self.release_check_dir() / backend
            try:
                s = json.loads((d / "run-summary.json").read_text())
            except (OSError, ValueError):
                return False
            if not (s.get("complete") and not s.get("validation_failures") and (d / "column.json").is_file()):
                return False
        return True

    def step_release_check(self):
        if self.release_check_complete():
            return "complete at " + str(self.release_check_dir())
        if not self.dry and git("rev-parse", "HEAD") != self.commit:
            raise StepFailed("HEAD moved off the frozen commit; release-check verifies HEAD")
        cmd = ["pixi", "run", "-e", "test", "release-check"]
        if "cpu" in self.check_backends():
            cmd.append("--cpu-column")
        self.must(cmd, log=self.rel / "release-check.log", what="release-check")
        if not self.dry and not self.release_check_complete():
            raise StepFailed("release-check exited 0 but its records are not complete")
        return "complete"

    def step_linux_wait(self):
        legs = linux_legs(self)
        if not legs:
            return "no leg to wait for (every Linux binding is reused)"
        if self.dry:
            return "would wait for " + ", ".join(l.name for l in legs)
        if not any(l.pid() or l.exit_code() is not None for l in legs):
            launch_detached(self, legs)
        tried = {l.name: 0 for l in legs}
        # A leg that failed on stock walks to its next GPU type AT ONCE, while
        # the other legs keep running (0.8.19: the sm_89 leg sat an hour behind
        # the AMD build before its walk).
        while True:
            again = []
            for l in legs:
                if l.running() or l.exit_code() is None:
                    continue
                if (l.vendor == "hip" and l.exit_code() != 0 and getattr(l, "provider", "") == "do"
                        and amd_build_want(self.args) == "auto" and do_refused_live(l.log)):
                    # a GPU droplet went live after the route was chosen: walk to Hot Aisle
                    self.say(f"  {l.name}: DigitalOcean refused (a GPU droplet is live); trying Hot Aisle")
                    l.command, l.env = amd_leg_command(self, "hotaisle", l.workdir)
                    l.provider, l.provider_reason = "hotaisle", "walked: DigitalOcean refused a rental"
                    self._amd_route = ("hotaisle", l.provider_reason)
                    again.append(l)
                    continue
                walk = NVIDIA_WALK.get(l.arch) if l.vendor == "cuda" and "--gpu" in l.command else None
                if walk and l.exit_code() != 0 and no_stock(l.log, l.workdir / l.name, l.out_dir) and tried[l.name] + 1 < len(walk):
                    tried[l.name] += 1
                    i = l.command.index("--gpu")
                    self.say(f"  {l.name}: {l.command[i + 1]} has no stock; trying {walk[tried[l.name]]}")
                    l.command[i + 1] = walk[tried[l.name]]
                    again.append(l)
            if again:
                launch_detached(self, again)
                continue
            if not any(l.running() for l in legs):
                break
            self.say("  waiting: " + ", ".join(f"{l.name}={'running' if l.running() else l.exit_code()}"
                                                for l in legs))
            time.sleep(60)
        bad = [l for l in legs if l.exit_code() != 0 or not l.proof_ok(self.commit)]
        if bad:
            raise StepFailed("build leg(s) failed: " + ", ".join(
                f"{l.name} (exit {l.exit_code()}, log {l.log})" for l in bad)
                + ". Rerun `release` to relaunch them.")
        mismatch = host_digest_mismatches(legs)
        if mismatch:
            lines = [f"{name}: " + ", ".join(f"{leg}={d[:12]}" for leg, d in row.items())
                     for name, row in sorted(mismatch.items())]
            raise StepFailed("host bindings differ across legs (pack_wheel.py would refuse):\n  "
                             + "\n  ".join(lines[:12]))
        return "all legs built; host bindings byte-identical across " + ", ".join(l.name for l in legs)

    def linux_final(self):
        found = sorted((self.rel / "linux" / "final").glob("mojolearn-*-manylinux*.whl"))
        return found[-1] if found else None

    def assembled_marker(self):
        return self.rel / "reuse" / "sets" / "assembled.json"

    def assembly_needed(self):
        """True when the plan takes any Linux bytes from the published wheel;
        otherwise the packer reads the legs' sets directly, as before."""
        plan = self.plan()
        return bool(release_reuse.plan_rows(plan, release_reuse.LINUX, "REUSE")) or plan["runtime"]["decision"] == "REUSE"

    def assembled_ok(self):
        try:
            d = json.loads(self.assembled_marker().read_text())
        except (OSError, ValueError):
            return False
        return d.get("commit") == self.commit and d.get("plan_sha256") == sha256(self.plan_path)

    def step_linux_assemble(self):
        if not self.assembly_needed():
            return "not needed: every Linux binding is built by the legs"
        if self.assembled_ok():
            return "have " + str(self.assembled_marker().parent)
        legs = {l.name: l.release_build / "build" / "sets" / l.vendor / l.arch for l in linux_legs(self)}
        plan = self.plan()
        if self.dry:
            n = len(release_reuse.plan_rows(plan, release_reuse.LINUX, "REUSE"))
            return (f"would take {n} binding(s) and the runtime closure from the published "
                    f"{plan['previous']['version']} Linux wheel into {self.rel / 'reuse' / 'sets'}"
                    + (f", over the sets of {', '.join(legs)}" if legs else " (no leg ran)"))
        whl = self.previous_wheel("linux")
        for name, d in legs.items():
            if not d.is_dir():
                raise StepFailed(f"{name} has no set directory at {d}; run linux-wait")
        dest = self.rel / "reuse" / "sets"
        release_reuse.assemble_linux(plan, whl, legs, dest, say=self.say)
        self.assembled_marker().write_text(json.dumps(dict(
            commit=self.commit, plan_sha256=sha256(self.plan_path), wheel=str(whl), wheel_sha256=sha256(whl),
            legs=sorted(legs), at=now()), indent=1) + "\n")
        return f"{dest}: sets assembled from {whl.name}" + (f" and legs {', '.join(sorted(legs))}" if legs else "")

    def pack_inputs(self):
        """(set directories, proofs, manifests) for pack_wheel.py and audit.sh:
        the assembled sets when the release reuses anything, else the legs'."""
        legs = linux_legs(self)
        proofs = [leg.release_build / "build" / "build-provenance.json" for leg in legs]
        if self.assembly_needed():
            base = self.rel / "reuse" / "sets"
            sets = sorted({base / vendor for vendor, _ in release_reuse.LINUX_SETS})
            manifests = [base / vendor / arch / "manifest.json" for vendor, arch in release_reuse.LINUX_SETS]
            return sets, proofs, manifests
        sets = [leg.release_build / "build" / "sets" / leg.vendor for leg in legs]
        manifests = [leg.release_build / "build" / "sets" / leg.vendor / leg.arch / "manifest.json" for leg in legs]
        return sets, proofs, manifests

    def step_linux_pack(self):
        final = self.linux_final()
        if final and wheel_commit(final) == self.commit:
            return "have " + final.name
        if self.assembly_needed() and not self.dry and not self.assembled_ok():
            raise StepFailed("the assembled sets are not there or not this plan's; run linux-assemble")
        sets, proofs, manifests = self.pack_inputs()
        dist = self.rel / "linux"
        if dist.exists() and not self.dry:
            dist.rename(dist.with_name("linux.failed-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")))
        args = ["pixi", "run", "-e", "pkg", "pack-linux-wheel", "--profile", "release-linux3"]
        for s in sets:
            args += ["--set", s]
        for p in proofs:
            args += ["--build-proof", p]
        args += ["--out", dist]
        self.must(args, log=self.rel / "linux-pack.log", what="pack_wheel.py")
        packed = sorted(dist.glob("mojolearn-*-linux_x86_64.whl")) if not self.dry else [dist / "<packed>.whl"]
        if len(packed) != 1:
            raise StepFailed(f"expected one packed wheel in {dist}, found {packed}")
        self.must(["bash", "packaging/linux/audit.sh", packed[0], *manifests], log=self.rel / "linux-audit.log",
                  what="audit.sh")
        repaired = sorted((dist / "audit" / "repaired").glob("mojolearn-*-manylinux*.whl")) if not self.dry \
            else [dist / "audit" / "repaired" / "<repaired>.whl"]
        if len(repaired) != 1:
            raise StepFailed(f"expected one repaired wheel, found {repaired}")
        if not self.dry:
            (dist / "final").mkdir(parents=True, exist_ok=True)
        self.must([PY, "tools/strip_wheel_dir_entries.py", repaired[0], dist / "final" / repaired[0].name,
                   "--receipt", dist / "final" / "dir-entry-strip.json"], what="strip_wheel_dir_entries.py")
        final = self.linux_final()
        return f"{final.name} sha256 {sha256(final)}" if final else "would pack, audit and strip"

    def gpu_selection(self, vendor):
        """The release pass's lanes for this vendor, worked out by the selector
        exactly as the pass would (verify_lanes --gpu-pass --write-selection)."""
        path = self.rel / f"selection-{vendor}.json"
        self.must([PY, "tools/verify_lanes.py", "--gpu-pass", vendor, "--write-selection", path],
                  log=self.rel / f"selection-{vendor}.log", what=f"{vendor} lane selection")
        return path

    def column_refs(self):
        """The columns the NVIDIA and AMD columns are diffed against: the Apple
        (Metal) column of this release, and the CPU column with --cpu-column."""
        d = self.release_check_dir()
        return [d / b / "column.json" for b in self.check_backends()]

    def ref_names(self):
        return " and ".join({"metal": "Apple", "cpu": "CPU"}[b] for b in self.check_backends())

    def gpu_column_ok(self, out, vendor):
        d = out / f"diff-ref-{vendor}.txt"
        return (out / f"column-{vendor}.json").is_file() and d.is_file() and "DIVERGENT" not in d.read_text()

    def nvidia_column_ok(self):
        final, out = self.linux_final(), self.rel / "smoke-linux"
        return bool(final) and smoke_passed(out / "results.json", final) and self.gpu_column_ok(out, "cuda")

    def amd_column_ok(self):
        return bool(self.linux_final()) and self.gpu_column_ok(self.rel / "column-amd", "hip")

    def column_legs(self):
        """The two GPU wheel columns as detached legs, each only when its
        record for this wheel is not already there. The NVIDIA leg carries its
        GPU walk (SMOKE_WALK or --smoke-gpu; the position survives a rerun in
        <columns>/nvidia.gpu); the AMD leg's provider walk (RunPod, Hot Aisle,
        DigitalOcean) is release_wheel_smoke.sh's own --provider auto."""
        final = str(self.linux_final() or "<final linux wheel>")
        work = self.rel / "columns"
        refs = [a for r in self.column_refs() for a in ("--ref-column", str(r))]
        legs = []
        if not self.nvidia_column_ok():
            walk = [g for g in self.args.smoke_gpu.split("|") if g] or SMOKE_WALK
            try:
                at = min(int((work / "nvidia.gpu").read_text()), len(walk) - 1)
            except (OSError, ValueError):
                at = 0
            out = self.rel / "smoke-linux"
            leg = ColumnLeg("nvidia", "cuda",
                            ["bash", "tools/release_wheel_smoke.sh", final, "--expected-source-commit", self.commit,
                             "--out", str(out), "--rent", "--column", str(self.gpu_selection("cuda")), *refs,
                             "--gpu", walk[at]], work, out, self.nvidia_column_ok)
            leg.walk, leg.at = walk, at
            legs.append(leg)
        if not self.amd_column_ok():
            out = self.rel / "column-amd"
            legs.append(ColumnLeg("amd", "hip",
                                  ["bash", "tools/release_wheel_smoke.sh", final, "--expected-source-commit",
                                   self.commit, "--out", str(out), "--rent", "--vendor", "hip",
                                   "--provider", self.args.amd_provider,
                                   "--column", str(self.gpu_selection("hip")), *refs],
                                  work, out, self.amd_column_ok))
        return legs

    def step_gpu_columns(self):
        """THE NVIDIA AND AMD COLUMNS, CONCURRENTLY (2026-09-25). Both are
        launched at once as detached legs and both are awaited; one failing
        never stops the other, and the step reports each. Then every column of
        this release is diffed together, so NVIDIA against AMD is checked as
        well as each against Apple."""
        final = self.linux_final()
        if not final and not self.dry:
            raise StepFailed("no final Linux wheel; run linux-pack")
        missing = [str(r) for r in self.column_refs() if not r.is_file()]
        if missing and not self.dry:
            raise StepFailed("no reference column " + ", ".join(missing) + "; run release-check")
        legs = self.column_legs()
        for name in ("nvidia", "amd"):
            if not any(l.name == name for l in legs):
                self.say(f"  {name}: PASSED for this wheel (record exists), not rerun")
        for leg in legs:
            self.say(f"  {leg.name}: {leg.describe().strip()}")
        if self.dry:
            if legs:
                self.say(f"  would launch {', '.join(l.name for l in legs)} at once and wait for every one")
            return self.joint_diff()
        if legs:
            launch_detached(self, legs)
            self.wait_columns(legs)
        failed = []
        for name, vendor, ok, out in (("NVIDIA", "cuda", self.nvidia_column_ok, self.rel / "smoke-linux"),
                                      ("AMD", "hip", self.amd_column_ok, self.rel / "column-amd")):
            if ok():
                self.say(f"  {name} column: PASSED, no DIVERGENT cell against {self.ref_names()}")
                continue
            leg = next((l for l in legs if l.vendor == vendor), None)
            failed.append(f"{name} column missing, not PASSED or DIVERGENT: {out}"
                          + (f" (exit {leg.exit_code()}, log {leg.log})" if leg else ""))
            self.say(f"  {failed[-1]}")
        if failed:
            raise StepFailed("; ".join(failed) + ". Rerun `release` to relaunch the failed column(s).")
        return self.joint_diff()

    def wait_columns(self, legs):
        """Wait for every column leg; walk the NVIDIA leg to its next GPU type
        when RunPod had no stock, and wait again."""
        work = legs[0].workdir
        # the walk runs AT ONCE for a column out of stock, while the others run
        while True:
            again = []
            for l in legs:
                if l.running() or l.exit_code() is None:
                    continue
                walk = getattr(l, "walk", None)
                if (walk and not l.done(self.commit) and l.exit_code() != 0
                        and no_stock(l.log, l.out_dir) and l.at + 1 < len(walk)):
                    l.at += 1
                    self.say(f"  {l.name}: {walk[l.at - 1]} has no stock; trying {walk[l.at]}")
                    l.command[l.command.index("--gpu") + 1] = walk[l.at]
                    work.mkdir(parents=True, exist_ok=True)
                    (work / "nvidia.gpu").write_text(str(l.at))
                    again.append(l)
            if again:
                launch_detached(self, again)
                continue
            if not any(l.running() for l in legs):
                return
            self.say("  waiting: " + ", ".join(f"{l.name}={'running' if l.running() else l.exit_code()}"
                                                for l in legs))
            self.sleep(60)

    def joint_diff(self):
        """Every column of this release in ONE tools/identity_break.py --diff:
        Apple, NVIDIA, AMD (and CPU with --cpu-column). Any DIVERGENT or MOVED
        cell stops the release; the diff is kept at <release>/diff-columns.txt."""
        cols = self.column_refs() + [self.rel / "smoke-linux" / "column-cuda.json",
                                     self.rel / "column-amd" / "column-hip.json"]
        cmd = [PY, str(ROOT / "tools" / "identity_break.py"), "--diff", *[str(c) for c in cols]]
        path = self.rel / "diff-columns.txt"
        self.say("  $ " + " ".join(shlex.quote(c) for c in cmd) + " > " + str(path))
        self.commands_shown += 1
        if self.dry:
            return "pending: would run the command(s) above"
        got = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        text = got.stdout + got.stderr
        path.write_text(text)
        return judge_joint_diff(text, path, say=self.say)

    def on_pypi(self, wheel):
        """True when PyPI already serves this exact file (name and sha256)."""
        url = f"https://pypi.org/pypi/mojolearn/{self.version}/json"
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                files = json.load(r).get("urls", [])
        except Exception:
            return False
        return any(f.get("filename") == wheel.name and f.get("digests", {}).get("sha256") == sha256(wheel)
                   for f in files)

    def publish(self, platform, wheel, smoke):
        target = self.args.publish
        if target is None:
            raise StepFailed("publication needs an explicit --publish none|testpypi|pypi; stopping here")
        if not self.dry and wheel and self.on_pypi(wheel):
            return "already on PyPI"
        if not self.dry and git("rev-parse", "HEAD") != self.commit:
            raise StepFailed("run the publisher from the frozen checkout: HEAD is not " + self.commit)
        day = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
        tag = f"alpha-api-{self.version}-{platform}-{day}"
        work = self.rel / f"publish-{platform}"
        self.must(["bash", "tools/release_linux_publish.sh", wheel or f"<{platform} wheel>", tag, target, work,
                   "--light-smoke", smoke], env=dict(MOJOLEARN_ARTIFACT_SOURCE_COMMIT=self.commit),
                  log=self.rel / f"publish-{platform}.log", what=f"publish {platform}")
        return f"{target} via {tag}"

    def step_publish_linux(self):
        return self.publish("linux", self.linux_final(), self.rel / "smoke-linux" / "results.json")

    def step_publish_macos(self):
        return self.publish("macos", self.macos_wheel(), self.rel / "smoke-macos" / "results.json")

    def step_finish_line(self):
        if self.args.publish != "pypi":
            return "skipped (not published to PyPI)"
        venv = self.rel / "finish-venv"
        if venv.exists() and not self.dry:
            shutil.rmtree(venv)
        self.must([self.smoke_python() if not self.dry else "python3.12", "-m", "venv", venv], what="venv")
        pip = venv / "bin" / "pip"
        for attempt in range(1, 7):
            rc = self.run([pip, "install", "--no-cache-dir", f"mojolearn=={self.version}"],
                          log=self.rel / f"finish-pip-{attempt}.log")
            if rc == 0:
                break
            self.say(f"  pip install failed (attempt {attempt}/6; the index may still be propagating)")
            self.sleep(30)
        else:
            raise StepFailed("pip install mojolearn==" + self.version + " failed six times")
        self.must([venv / "bin" / "python", "-c",
                   f"import mojolearn; assert mojolearn.__version__ == {self.version!r}; "
                   "print(mojolearn.__version__, mojolearn.vendor())"], what="import")
        return f"pip install mojolearn=={self.version} on this Mac: OK"

    def step_record(self):
        if self.args.publish != "pypi":
            return "skipped (not published to PyPI)"
        date = changelog_date(ROOT, self.version) or dt.date.today().isoformat()
        rec = ROOT / "bench" / "results" / "release_verification" / f"{date}_pypi_{self.version.replace('.', '')}"
        if self.dry:
            return "would write " + str(rec.relative_to(ROOT))
        rec.mkdir(parents=True, exist_ok=True)
        for platform in ("linux", "macos"):
            art = self.rel / f"publish-{platform}"
            for src, name in ((art / "artifact" / "alpha-manifest.json", f"alpha-manifest-{platform}.json"),
                              (art / "file-admission.json", f"file-admission-{platform}.json"),
                              (self.rel / f"smoke-{platform}" / "results.json", f"light-smoke-{platform}.json")):
                if src.is_file():
                    shutil.copy2(src, rec / name)
        (rec / "README.md").write_text(self.readme())
        # The identity of every shipped binding and what it shipped as, so the
        # next release decides REUSE or BUILD from this record alone (the
        # Apple toolchain that built the macOS wheel included).
        release_reuse.record_identities(self.plan(), self.linux_final(), self.macos_wheel(),
                                        rec / "binding-identities.json")
        self.must(["git", "-C", ROOT, "add", "--", rec], what="git add record")
        self.must(["git", "-C", ROOT, "commit", "-q", "-m",
                   f"Record the published {self.version} wheels and their release verification", "-m",
                   "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"], what="git commit record")
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        # main has usually moved on during the builds; the record rides on top
        # of it (0.8.16's record push was rejected as non-fast-forward)
        self.must(["git", "-C", ROOT, "fetch", "-q", "origin", branch], what="git fetch")
        self.must(["git", "-C", ROOT, "rebase", "-q", f"origin/{branch}"], what="git rebase record onto origin")
        self.must(["git", "-C", ROOT, "push", "origin", f"HEAD:refs/heads/{branch}"], what="git push record")
        return str(rec.relative_to(ROOT)) + " committed on " + branch

    def readme(self):
        check = self.release_check_dir()
        rows, sums = [], []
        cols = [(check / b / "column.json", {"metal": "Apple Metal", "cpu": "CPU"}[b], b + "/column.json")
                for b in self.check_backends()]
        cols += [(self.rel / "smoke-linux" / "column-cuda.json", "NVIDIA (installed wheel)", "column-cuda.json"),
                 (self.rel / "column-amd" / "column-hip.json", "AMD (installed wheel)", "column-hip.json")]
        for col, label, name in cols:
            try:
                cells = json.loads(col.read_text()).get("cells", {})
            except (OSError, ValueError):
                cells = {}
            lanes = {k.split("/")[0] for k in cells}
            rows.append(f"| {label} | {len(lanes)} | {len(cells)} | {'yes' if cells else 'no'} |")
            if col.is_file():
                sums.append(f"- {name} sha256 {sha256(col)}")
        try:
            diff = (self.rel / "diff-columns.txt").read_text()
        except OSError:
            diff = ""
        summary = [ln for ln in diff.splitlines() if ln.startswith("summary")]
        wheels = []
        for platform, wheel, smoke in (("linux", self.linux_final(), "smoke-linux"),
                                       ("macos", self.macos_wheel(), "smoke-macos")):
            if not wheel:
                continue
            try:
                r = json.loads((self.rel / smoke / "results.json").read_text())
            except (OSError, ValueError):
                r = {}
            tag = self.recorded(f"publish-{platform}") or {}
            plat = wheel.name.split("-", 4)[-1][:-4]
            wheels.append(f"| {plat} | {sha256(wheel)} | {r.get('status', '?')}, {len(r.get('jobs', []))} jobs "
                          f"| {tag.get('result', '?')} |")
        date = changelog_date(ROOT, self.version)
        return "\n".join([
            f"# mojolearn {self.version}", "",
            f"Source commit {self.commit}. Published {date}. See CHANGELOG.md.", "",
            "## Release verification (the changed lanes, one fit per cell)", "",
            "| column | lanes | cells | complete |", "|---|---|---|---|", *rows, "",
            "Every column together (`tools/identity_break.py --diff`, diff-columns.txt):", "",
            *[f"    {s}" for s in summary], "", *sums, "",
            "## Wheels (light route)", "",
            "| platform | sha256 | smoke | published |", "|---|---|---|---|", *wheels, "",
            f"`pip install mojolearn=={self.version}` on this Mac: "
            f"{(self.recorded('finish-line') or {}).get('result', 'not run')}.", ""])

    STEPS = ["freeze-version", "freeze-changelog", "freeze-docs-facts", "freeze-commit", "rehearsal", "reuse-plan",
             "linux-builds", "macos-build", "macos-smoke", "release-check", "linux-wait", "linux-assemble",
             "linux-pack", "gpu-columns", "publish-linux", "publish-macos", "finish-line", "record"]
    #: Every other step checks its own OUTPUT each time and returns at once when
    #: it is already there (a wheel of this commit, a PASSED receipt for that
    #: wheel, complete release-check records); these are skipped on their record.
    SKIP_IF_RECORDED = {"rehearsal", "publish-linux", "publish-macos", "finish-line", "record"}

    def go(self):
        steps = self.STEPS
        if self.args.only:
            wanted = self.args.only.split(",")
            unknown = [s for s in wanted if s not in steps]
            if unknown:
                raise SystemExit("release: unknown step(s) " + ", ".join(unknown) + "; steps: " + ", ".join(steps))
            steps = [s for s in steps if s in wanted]
        for redo in (self.args.redo.split(",") if self.args.redo else []):
            self.state["steps"].pop(redo, None)
        self.say(f"== mojolearn release {self.version}{' (DRY RUN: nothing runs, nothing is rented)' if self.dry else ''}")
        self.say(f"   state {self.state_path}; frozen commit {self.state.get('commit') or '(not yet)'}")
        for step in steps:
            rec = self.recorded(step)
            if rec and step in self.SKIP_IF_RECORDED:
                self.say(f"-- {step}: done {rec['at']} ({rec.get('result', '')})")
                continue
            self.say(f"-- {step}")
            shown = self.commands_shown
            try:
                result = getattr(self, "step_" + step.replace("-", "_"))()
                if self.dry and self.commands_shown > shown:
                    result = "pending: would run the command(s) above"
            except StepFailed as exc:
                if self.dry:
                    self.say(f"   A REAL RUN WOULD STOP HERE: {exc}")
                    continue
                self.say(f"   STOPPED at {step}: {exc}")
                self.say(f"   Fix it and rerun `pixi run release {self.version}`; finished steps are skipped.")
                return 1
            self.say(f"   {result}")
            if not self.dry:
                self.mark(step, result=result)
        self.say("== done" if not self.dry else "== end of plan")
        return 0


class StepFailed(Exception):
    pass


def judge_joint_diff(text, path, say=print):
    """The verdict of the all-columns diff: a StepFailed on any DIVERGENT or
    MOVED cell (any part), else a one-line summary. Cells only one column
    hashed (a lane one vendor alone selects or can run) are counted in the
    result, never read as agreement."""
    bad, one = [], 0
    for line in text.splitlines():
        if line.startswith("summary"):
            say("  " + line)
            for key, n in re.findall(r"([A-Z_-]+)=(\d+)", line):
                if int(n) and ("DIVERGENT" in key or "MOVED" in key):
                    bad.append(f"{line.split(':')[0]} {key}={n}")
                if key == "ONE-COLUMN" and line.startswith("summary:"):
                    one += int(n)
    if "DIVERGENT" in text and not any("DIVERGENT" in b for b in bad):
        bad.append("DIVERGENT")
    if bad:
        rows = [l for l in text.splitlines() if "DIVERGENT" in l or "MOVED" in l][:20]
        raise StepFailed("the columns of this release disagree (" + ", ".join(bad) + f"), {path}:\n  "
                         + "\n  ".join(rows))
    if "summary:" not in text:
        raise StepFailed(f"the all-columns diff printed no summary; see {path}")
    return (f"NVIDIA and AMD columns PASSED; no DIVERGENT cell across the columns ({path})"
            + (f"; {one} cell(s) hashed by one column only" if one else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("version")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and each step's status; run nothing")
    ap.add_argument("--publish", choices=PUBLISH_CHOICES,
                    help="publish the two wheels (none = the workflow's checks without uploading)")
    ap.add_argument("--only", default="", help="comma-separated step names to run (others are skipped)")
    ap.add_argument("--redo", default="", help="comma-separated steps whose record is discarded first")
    ap.add_argument("--build-backend", default="gpu-legs", choices=sorted(BUILD_BACKENDS))
    ap.add_argument("--cpu-column", action="store_true",
                    help="also run release-check's CPU pass (opt-in) and diff the GPU columns against it too")
    ap.add_argument("--amd-expect-from", default="",
                    help="an NVIDIA release-build dir: run the AMD core-host probe against its STAGED copy")
    ap.add_argument("--smoke-gpu", default="", help="RunPod GPU(s) for the Linux smoke, |-separated, walked on no stock (default the 4090, L40S, L40, RTX 6000 Ada)")
    _amd_env = os.environ.get("MOJOLEARN_AMD_PROVIDER") or "auto"
    ap.add_argument("--amd-provider", default=_amd_env if _amd_env in ("auto", "runpod", "hotaisle", "do") else "auto",
                    choices=["auto", "runpod", "hotaisle", "do"],
                    help="where the AMD column rents: auto = RunPod MI300X, then Hot Aisle MI300X, then DigitalOcean "
                         "MI325X (default MOJOLEARN_AMD_PROVIDER, else auto)")
    ap.add_argument("--amd-build-provider", default=None, choices=list(AMD_BUILD_PROVIDERS),
                    help="where the AMD build leg rents: auto = DigitalOcean MI325X, Hot Aisle 1x MI300X when "
                         "DigitalOcean has a GPU droplet live (default MOJOLEARN_AMD_PROVIDER, else auto)")
    ap.add_argument("--state-dir", default="", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if not VERSION_RE.match(args.version):
        ap.error("version must look like 0.8.15 (or 0.9.0a1)")
    return Release(args).go()


if __name__ == "__main__":
    raise SystemExit(main())
