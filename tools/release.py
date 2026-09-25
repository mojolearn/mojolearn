#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ONE COMMAND FOR A RELEASE (2026-09-22): docs/RELEASE_CHECKLIST.md, walked.

    pixi run release 0.8.15 --dry-run          the plan and what is already done
    pixi run release 0.8.15 --status           per pipeline, leg and column: done, failed, owed
    pixi run release 0.8.15                    everything up to publication
    pixi run release 0.8.15 --publish pypi     ... and publish, finish line, record

TWO PIPELINES, SEVERABLE (2026-09-25). After the common steps a release is
two pipelines that run at once in one invocation, each publishing as soon as
ITS OWN gates pass; a failure in one never blocks or undoes the other:
  macos   macos-build -> macos-smoke -> release-check (the Apple column) -> publish-macos
  linux   linux-builds -> linux-wait -> linux-assemble -> linux-pack
          -> gpu-columns (NVIDIA and AMD, diffed against the Apple column) -> publish-linux
The Linux columns are diffed against the Apple column, so a failed
release-check holds the Linux publish too (bitwise identity across GPUs is the
point); a failed macOS build or smoke does not. The finish line and the record
cover whichever platforms are published, and run again when another is. The
run ends with both outcomes and exits non-zero if either did not publish. A
pipeline is PIPELINES' named builds, checks and one publish, so a platform can
later split into more (core-linux, cuda, rocm) without a new scheduler.

THE SHIPPED SOURCE AND THE RELEASE TOOLING ARE TWO COMMITS. freeze-commit pins
the source commit in state.json and it never moves because main moved:
publication ships exactly it, and only `--refreeze` moves it to HEAD. This
script and everything it drives (the leg runners, the guards, the wheel smoke,
the publisher) run from THIS checkout, the tooling checkout, normally current
main. Steps that read the source (the macOS build, release-check, the pack,
the rehearsal, the lane selection) run in the source checkout: this one when
HEAD is the source commit, else a detached worktree of it under
<version>/source/<commit12>/ (or --source-checkout). A build leg's box gets
the SOURCE archive plus the route overlay (tools/release_tooling.py,
tools/route_overlay_lib.sh): the tooling copy of each box-side tool that
differs, never a file of the source inventory. Every leg, column and record
names both commits (source_commit, tooling_commit and the overlay digests).

RESULTS ARE KEYED TO WHAT THEY BUILT. A completed build leg of an earlier
freeze is taken under a new one when, and only when, its set identity
(release_reuse.set_identity: every binding identity of the set, the host
bindings and the runtime closure) and its build tooling digest equal this
freeze's; its bytes then enter the pack through reuse.json with their origin
(built from X, admitted for Y), and anything unequal or unreadable rebuilds.
A GPU column is taken only for a byte-identical wheel (sha256) and the same
lane selection, and is diffed again against this release's references. Every
leg and column result is recorded once in the PASS LEDGER in R2
(tools/release_ledger.py); a ledger PASS is admitted only when its evidence is
on this machine and verifies.

State lives in ~/mojolearn-evidence/release/<version>/state.json
(MOJOLEARN_EVIDENCE_ROOT moves it); everything after the freeze is under
<version>/<commit12>/. A step is done when its OUTPUT checks out (a wheel of the
right commit, a PASSED receipt for that wheel's sha256, a complete
release-check record, the file on PyPI), not merely because the state file says
so. A rerun resumes only what is not done.

THE STEPS
  freeze-version     _version.py and pyproject.toml say <version> (edited if not)
  freeze-changelog   CHANGELOG.md has `## <version> (published YYYY-MM-DD)`; the
                     prose is the releaser's, so a missing entry stops here
  freeze-docs-facts  pixi run write-docs-facts (CITATION and marked spans)
  freeze-commit      commit exactly those files if they changed, push, and
                     freeze HEAD (the tree must be clean and pushed). Once
                     frozen, the four freeze steps only check the frozen source
  rehearsal          pixi run release-rehearsal in the source checkout, logs kept
  reuse-plan         tools/release_reuse.py: for every binding of every set
                     (cuda sm_90a, cuda sm_89, hip gfx942, the host bindings,
                     the runtime closure, the macOS wheel) REUSE the bytes the
                     last PUBLISHED release shipped when the binding's identity
                     (source closure, toolchain, flags, builder scripts as they
                     run after the overlay, box image or Apple toolchain) is
                     unchanged, else BUILD
  linux-builds       a leg per Linux set that has a binding to BUILD, all
                     launched AT ONCE, detached, unless the leg is done at this
                     freeze or a completed leg of an earlier freeze has the same
                     set identity and build tooling (taken, not launched). The
                     default backend `gpu-legs` builds on the GPU boxes: RunPod
                     NVIDIA (walking NVIDIA_WALK on no stock) and a DigitalOcean
                     MI325X, or a Hot Aisle 1x MI300X when DigitalOcean has a GPU
                     droplet live or no token (--amd-build-provider do|hotaisle
                     or MOJOLEARN_AMD_PROVIDER pins one). `cpu-box` is opt-in
  macos-build        mac_slot --slots 4 run -- build_release_wheel.sh, byte LM on
  macos-smoke        qualify_verifier_wheel.py --scope expanded under the Metal lock
  release-check      pixi run -e test release-check: the Apple (Metal) column of
                     the changed lanes, one fit per cell (CPU pass: --cpu-column)
  linux-wait         every launched leg finished, its proof complete for this
                     commit, its route overlay verified on the box, and every
                     host binding byte-identical across the legs (taken ones too)
  linux-assemble     the set directories the packer packs: a leg's set, a taken
                     leg's set, or one synthesized from the published wheel
  linux-pack         pixi run -e pkg pack-linux-wheel, audit.sh, strip
  gpu-columns        the NVIDIA and AMD wheel columns AT ONCE (detached
                     tools/release_wheel_smoke.sh on the changed lanes), each
                     diffed against the Apple column, then ONE diff of every
                     column together: any DIVERGENT or MOVED cell stops Linux
  publish-linux      tools/release_linux_publish.sh ... --light-smoke   } only with
  publish-macos      tools/release_linux_publish.sh ... --light-smoke   } --publish
  finish-line        per published platform: pip install on this Mac (macOS);
                     pip download of the Linux file, sha256 compared (Linux)
  record             bench/results/release_verification/<date>_pypi_<v>/, committed

Nothing is published without `--publish none|testpypi|pypi`; without it both
pipelines stop at their publish step and say so.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
PUBLISH_CHOICES = ("none", "testpypi", "pypi")
sys.path.insert(0, str(ROOT / "tools"))
import release_ledger  # noqa: E402
import release_reuse  # noqa: E402
import release_tooling  # noqa: E402


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


def git_at(root, *args):
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


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


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return None


def write_json(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)


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


def changelog_date(root, version, text=None):
    """The `published YYYY-MM-DD` date of <version>'s CHANGELOG heading, or None."""
    text = text if text is not None else (Path(root) / "CHANGELOG.md").read_text()
    m = re.search(r"^## " + re.escape(version) + r" \(published (\d{4}-\d{2}-\d{2})\)\s*$", text, re.M)
    return m.group(1) if m else None


def file_at(commit, rel):
    """REL's text at COMMIT (the frozen source), or None."""
    p = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{rel}"], capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


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
        #: written beside the log at launch: both commits, the overlay, the identities
        self.provenance_file = self.workdir / f"{name}.provenance.json"
        self.provenance = None

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


def verify_leg_tree(release_build, vendor, arch, commit, exit_file=None):
    """(ok, why, proof sha256) for a finished build leg's output: exit 0, a
    complete proof of COMMIT, its set directory, and every binary the proof
    names on disk with the proof's sha256. Read-only."""
    rb = Path(release_build)
    if exit_file is not None:
        try:
            if int(Path(exit_file).read_text().strip()) != 0:
                return False, "it did not exit 0", None
        except (OSError, ValueError):
            return False, "it has no exit code (not finished)", None
    proof_path = rb / "build" / "build-provenance.json"
    proof = read_json(proof_path)
    if not proof:
        return False, f"no build proof at {proof_path}", None
    if proof.get("complete") is not True or proof.get("build_exit") != 0 or proof.get("source_commit") != commit:
        return False, "its build proof is not complete, not exit 0 or not of " + commit[:12], None
    if not (rb / "build" / "sets" / vendor / arch).is_dir():
        return False, "no set directory", None
    ext = proof.get("extensions") or {}
    if not ext:
        return False, "its proof names no binary", None
    for name, digest in ext.items():
        parts = Path(name).parts
        if parts[:3] != ("mojolearn", vendor, arch) or ".." in parts:
            return False, f"its proof names {name}, outside {vendor}/{arch}", None
        p = rb / "build" / "sets" / Path(*parts[1:])
        try:
            if sha256(p) != digest:
                return False, f"{name} on disk is not the proof's bytes", None
        except OSError:
            return False, f"{name} is missing", None
    return True, "", sha256(proof_path)


def overlay_verified(prov, out_dir):
    """(ok, why): every file the leg's route overlay carried is, per the box's
    own after-sha256 in <out>/route-overlay.txt, the tooling's bytes."""
    files = ((prov or {}).get("overlay") or {}).get("files") or {}
    if not files:
        return True, ""
    try:
        text = (Path(out_dir) / "route-overlay.txt").read_text()
    except OSError:
        return False, f"no route-overlay.txt in {out_dir} (the overlay was not applied)"
    for rel, row in files.items():
        if f"after {rel} {row['tooling_sha256']}" not in text.splitlines():
            return False, f"{rel} on the box was not the overlaid bytes ({out_dir}/route-overlay.txt)"
    return True, ""


def leg_layout(legs_dir, backend, name):
    """(release_build, out_dir) of a leg named `vendor-arch` on a backend."""
    legs_dir = Path(legs_dir)
    if backend == "cpu-box":
        out = legs_dir / name
        return out / name / "release-build", out
    if name.startswith("cuda-"):
        out = legs_dir / name
        return out / "remote" / "release-build", out
    return legs_dir / name / "release-build", legs_dir / name


def leg_common_env(ctx):
    """What every Linux build leg is told about the source and the tooling:
    the source checkout its inventory is compared with, and the route overlay
    (Release.leg_env). A bare context carries none."""
    fn = getattr(ctx, "leg_env", None)
    return fn() if callable(fn) else {}


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
    common = leg_common_env(ctx)
    legs = []
    for arch, gpu in ((a, NVIDIA_WALK[a][0]) for a in ("sm_90a", "sm_89")):
        rb, out = leg_layout(legs_dir, "gpu-legs", f"cuda-{arch}")
        legs.append(Leg(f"cuda-{arch}", "cuda", arch,
                        ["sh", "tools/gemm_remote_leg.sh", "nvidia", "--payload", "mamba",
                         "--source-ref", ctx.commit, "--gpu", gpu, "--allow-concurrent", "--rent",
                         "--segment-lease", "120", "--dollar-cap", "15"],
                        dict(runpod, MOJOLEARN_GPU_ARCHS=arch, MOJOLEARN_GEMM_LEG_OUT=str(out), **common),
                        rb, legs_dir, out))
    route, why = amd_build_route(ctx)
    command, env = amd_leg_command(ctx, route, legs_dir)
    rb, out = leg_layout(legs_dir, "gpu-legs", "hip-gfx942")
    leg = Leg("hip-gfx942", "hip", "gfx942", command, env, rb, legs_dir, out)
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
    env.update(leg_common_env(ctx))
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
    built wheel on real GPUs before anything publishes. The route overlays its
    own files from this checkout (its route.txt), so it takes no route overlay."""
    legs_dir = ctx.rel / "legs"
    legs = []
    for vendor, arch in (("cuda", "sm_90a"), ("cuda", "sm_89"), ("hip", "gfx942")):
        name = f"{vendor}-{arch}"
        rb, out = leg_layout(legs_dir, "cpu-box", name)
        legs.append(Leg(name, vendor, arch,
                        ["bash", "tools/release_linux_build.sh", ctx.commit, "--rent", "--archs", arch,
                         "--flavors", CPU_BOX_FLAVORS, "--out", str(out)],
                        {}, rb, legs_dir, out))
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
    deleted) and relaunched. The leg's provenance (both commits, the route
    overlay, the identities it was asked for) is written beside its log."""
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
        for p in (leg.log, leg.exit_file, leg.pid_file, leg.provenance_file):
            if p.exists():
                p.rename(p.with_name(p.name + f".failed-{stamp}"))
        out = leg.out_dir
        if out.exists():
            out.rename(out.with_name(out.name + f".failed-{stamp}"))
            ctx.say(f"  {leg.name}: previous attempt moved aside to {out.name}.failed-{stamp}")
        prov = leg.provenance
        if prov is None and type(leg) is Leg and callable(getattr(ctx, "leg_provenance", None)):
            prov = ctx.leg_provenance(leg)
        if prov is not None:
            write_json(leg.provenance_file, dict(prov, command=[str(c) for c in leg.command], launched_at=now()))
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


def selection_digest(path):
    """The lanes a GPU column runs (verify_lanes --write-selection), as one
    digest: backend, column, fixtures and lanes; never the commit that asked."""
    doc = read_json(path)
    if not doc or not isinstance(doc.get("lanes"), list):
        return None
    keep = {k: doc.get(k) for k in ("backend", "column", "fixtures", "lanes")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- the pipelines
#: (step, pipeline, needs, resource). A step runs once every step it NEEDS is
#: done; a failed or blocked need blocks it. Steps sharing a RESOURCE run one
#: at a time ("mac": the Mac's cores and its one Metal GPU; the order of
#: STEPS decides who goes first). Everything else runs at once.
STEP_TABLE = [
    ("freeze-version", "common", [], None),
    ("freeze-changelog", "common", ["freeze-version"], None),
    ("freeze-docs-facts", "common", ["freeze-changelog"], None),
    ("freeze-commit", "common", ["freeze-docs-facts"], None),
    ("rehearsal", "common", ["freeze-commit"], None),
    ("reuse-plan", "common", ["rehearsal"], None),
    ("linux-builds", "linux", ["reuse-plan"], None),
    ("macos-build", "macos", ["reuse-plan"], "mac"),
    ("macos-smoke", "macos", ["macos-build"], "mac"),
    ("release-check", "macos", ["reuse-plan"], "mac"),
    ("linux-wait", "linux", ["linux-builds"], None),
    ("linux-assemble", "linux", ["linux-wait"], None),
    ("linux-pack", "linux", ["linux-assemble"], "mac"),
    ("gpu-columns", "linux", ["linux-pack", "release-check"], None),
    ("publish-linux", "linux", ["gpu-columns"], None),
    ("publish-macos", "macos", ["macos-smoke", "release-check"], None),
    ("finish-line", "finish", ["freeze-commit"], None),
    ("record", "finish", ["finish-line"], None),
]
#: Steps that wait for others to SETTLE (done, failed or blocked), not succeed.
AFTER = {"finish-line": ["publish-linux", "publish-macos"]}
#: A PIPELINE: named builds, named checks, one publish, and the wheel it ships.
#: Generic on purpose: the Linux pipeline can become per-package pipelines
#: (core-linux, cuda, rocm), each with its own builds, checks and publish.
PIPELINES = {
    "macos": dict(builds=["macos-build"], checks=["macos-smoke", "release-check"], publish="publish-macos",
                  platform="macos"),
    "linux": dict(builds=["linux-builds", "linux-wait", "linux-assemble", "linux-pack"], checks=["gpu-columns"],
                  publish="publish-linux", platform="linux"),
}
PIPELINE_OF = {s: p for s, p, _, _ in STEP_TABLE}
NEEDS = {s: n for s, _, n, _ in STEP_TABLE}
RESOURCE = {s: r for s, _, _, r in STEP_TABLE}


class StepFailed(Exception):
    pass


class StepHeld(StepFailed):
    """Stopped on purpose (publication without --publish), not broken."""


# ---------------------------------------------------------------- the run
class Release:
    STEPS = [s for s, _, _, _ in STEP_TABLE]
    #: Every other step checks its own OUTPUT each time and returns at once when
    #: it is already there (a wheel of this commit, a PASSED receipt for that
    #: wheel, complete release-check records); these are skipped on their record
    #: (finish-line and record: when the platforms they covered are still the
    #: published ones).
    SKIP_IF_RECORDED = {"rehearsal", "publish-linux", "publish-macos", "finish-line", "record"}

    def __init__(self, args, runner=None):
        self.args = args
        self.version = args.version
        ev = Path(os.environ.get("MOJOLEARN_EVIDENCE_ROOT", os.path.expanduser("~/mojolearn-evidence")))
        self.base = Path(args.state_dir) if args.state_dir else ev / "release" / self.version
        self.state_path = self.base / "state.json"
        self.state = self.load()
        self.runner = runner
        self.dry = args.dry_run
        self.readonly = bool(getattr(args, "status", False))
        self.commands_shown = 0
        self.evidence = ev
        self._plan = None
        self._overlay = None
        self._tooling = None
        self._leg_env = None
        self._ledger = None
        self._lock = threading.RLock()
        self._out = threading.Lock()
        self._tl = threading.local()
        self.errors, self.blocked_by = {}, {}

    # state
    def load(self):
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {"version": self.version, "commit": None, "steps": {}}

    def save(self):
        if self.dry or self.readonly:
            return
        with self._lock:
            self.base.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
            tmp.replace(self.state_path)

    def head(self):
        return git("rev-parse", "HEAD")

    def head_of(self, checkout):
        return self.head() if Path(checkout) == ROOT else git_at(checkout, "rev-parse", "HEAD")

    @property
    def commit(self):
        """THE SOURCE COMMIT: pinned at freeze, HEAD only before the first freeze."""
        return self.state.get("commit") or self.head()

    @property
    def tooling_commit(self):
        return self.head()

    @property
    def rel(self):
        return self.base / self.commit[:12]

    def mark(self, step, **data):
        with self._lock:
            self.state["steps"][step] = dict(done=True, commit=self.state.get("commit"), at=now(),
                                             tooling_commit=git_at(ROOT, "rev-parse", "HEAD"), **data)
            self.state.setdefault("failures", {}).pop(step, None)
            self.save()

    def fail(self, step, exc, held=False):
        with self._lock:
            self.errors[step] = str(exc)
            self.state.setdefault("failures", {})[step] = dict(at=now(), commit=self.state.get("commit"),
                                                              error=str(exc), held=held)
            self.save()

    def recorded(self, step):
        rec = self.state["steps"].get(step)
        return rec if rec and rec.get("done") and rec.get("commit") == self.state.get("commit") else None

    def published_platforms(self):
        return [PIPELINES[p]["platform"] for p in PIPELINES if self.recorded(PIPELINES[p]["publish"])]

    def skip_recorded(self, step, rec):
        if step not in self.SKIP_IF_RECORDED:
            return False
        if step in ("finish-line", "record") and "platforms" in rec:
            return sorted(rec["platforms"]) == sorted(self.published_platforms())
        return True

    # execution
    def say(self, msg):
        tag = getattr(self._tl, "tag", "")
        with self._out:
            if tag:
                msg = "\n".join(f"[{tag}] {line}" for line in (str(msg).splitlines() or [""]))
            print(msg, flush=True)

    def sleep(self, seconds):
        if not self.dry:
            time.sleep(seconds)

    def run(self, cmd, env=None, log=None, cwd=None):
        """Run one command to completion; its output teed to `log` when given."""
        shown = " ".join(shlex.quote(str(c)) for c in cmd)
        extra = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in (env or {}).items())
        where = f"(in {cwd}) " if cwd and Path(cwd) != ROOT else ""
        self.say(f"  $ {where}{extra + ' ' if extra else ''}{shown}")
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
                    self.say("    " + line.rstrip("\n"))
                    fh.write(line)
                return proc.wait()
        return subprocess.run([str(c) for c in cmd], cwd=cwd or ROOT, env=full_env).returncode

    def must(self, cmd, env=None, log=None, what=None, cwd=None):
        rc = self.run(cmd, env, log, cwd=cwd)
        if rc != 0:
            raise StepFailed(f"{what or cmd[0]} exited {rc}" + (f"; log {log}" if log else ""))

    def spawn(self, cmd, env):
        if self.runner:
            return self.runner(cmd, env, None, detach=True)
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)
        return proc.pid

    # ------------------------------------------------------------ source and tooling
    def source_checkout(self, create=True):
        """Where the steps that read the SOURCE run: this checkout when its HEAD
        is the source commit, else --source-checkout, else a detached worktree
        of the source commit under <version>/source/<commit12>/ (made once)."""
        c = self.commit
        if getattr(self.args, "source_checkout", ""):
            p = Path(self.args.source_checkout).expanduser().resolve()
            if not self.dry and not self.readonly and git_at(p, "rev-parse", "HEAD") != c:
                raise StepFailed(f"--source-checkout {p} is not at the source commit {c}")
            return p
        if self.head_of(ROOT) == c:
            return ROOT
        p = self.base / "source" / c[:12]
        if (p / ".git").exists():
            if git_at(p, "rev-parse", "HEAD") != c and not self.readonly:
                raise StepFailed(f"the source checkout {p} is not at {c}; remove it (git worktree remove) and rerun")
            return p
        if self.dry or self.readonly or not create:
            return p
        with self._lock:
            if not (p / ".git").exists():
                self.say(f"  source checkout: a detached worktree of {c[:12]} at {p} (the tooling runs from {ROOT})")
                p.parent.mkdir(parents=True, exist_ok=True)
                rc = subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(p), c],
                                    capture_output=True, text=True)
                if rc.returncode != 0:
                    raise StepFailed(f"could not add a worktree of {c} at {p}: {rc.stderr.strip()}")
        return p

    @property
    def src(self):
        return self.source_checkout()

    def overlay(self):
        """The route overlay manifest: the tooling copy of each box-side tool
        that differs from the source commit's (tools/release_tooling.py)."""
        with self._lock:
            if self._overlay is None:
                try:
                    self._overlay = release_tooling.overlay_manifest(ROOT, self.tooling_commit, self.commit)
                except release_tooling.OverlayRefused as exc:
                    raise StepFailed(str(exc))
            return self._overlay

    def builders_override(self):
        """The binding-identity builders the box runs from the tooling checkout
        (the route overlay; tools/release_linux_build.sh overlays the same
        builder from this checkout on the cpu-box route)."""
        return release_tooling.effective_builders(self.overlay())

    def tooling(self):
        """The build tooling that runs: its commit, its digest (None when a
        tooling file is uncommitted, so nothing built with it is reused across
        freezes), the files and the uncommitted ones."""
        with self._lock:
            if self._tooling is None:
                commit = self.tooling_commit
                digest, rows = release_tooling.tooling_digest(ROOT, commit)
                dirty = release_tooling.dirty(ROOT, release_tooling.BUILD_TOOLING)
                self._tooling = dict(commit=commit, digest=None if dirty else digest, files=rows, dirty=dirty)
            return self._tooling

    def leg_env(self):
        """The environment every build leg gets: the source checkout its
        inventory is compared with, and the route overlay when there is one
        (the tarball is written once per overlay digest and never rewritten)."""
        with self._lock:
            if self._leg_env is None:
                env = {"MOJOLEARN_SOURCE_CHECKOUT": str(self.source_checkout(create=not self.dry))}
                m = self.overlay()
                if m["files"] and self.args.build_backend != "cpu-box":
                    if self.dry:
                        env.update(MOJOLEARN_ROUTE_OVERLAY="<route-overlay.tgz of " + ", ".join(sorted(m["files"])) + ">",
                                   MOJOLEARN_ROUTE_OVERLAY_SHA256="<sha256>")
                    else:
                        dest = self.rel / "legs" / "route-overlay" / m["digest"][:16]
                        doc = read_json(dest / "route-overlay.json")
                        tgz = dest / "route-overlay.tgz"
                        if doc and doc.get("digest") == m["digest"] and tgz.is_file() \
                                and sha256(tgz) == doc.get("tarball_sha256"):
                            digest = doc["tarball_sha256"]
                        else:
                            try:
                                tgz, digest = release_tooling.write_overlay(ROOT, m, dest)
                            except release_tooling.OverlayRefused as exc:
                                raise StepFailed(str(exc))
                        env.update(MOJOLEARN_ROUTE_OVERLAY=str(tgz), MOJOLEARN_ROUTE_OVERLAY_SHA256=digest)
                self._leg_env = env
            return dict(self._leg_env)

    def set_identity(self, vendor, arch):
        try:
            plan = self.plan()
            return release_reuse.set_identity(plan, vendor, arch) if plan else None
        except (Exception, SystemExit):
            return None

    def leg_provenance(self, leg):
        """What a build leg is, recorded at launch: both commits, the tooling
        digest, the route overlay, and the set identity it was asked for."""
        try:
            t = self.tooling()
        except (Exception, SystemExit):
            t = dict(commit=None, digest=None, dirty=[])
        try:
            m = self.overlay() if self.args.build_backend != "cpu-box" else None
        except (Exception, SystemExit):
            m = None
        return dict(schema="mojolearn.release-leg-provenance.v1", leg=leg.name, vendor=leg.vendor, arch=leg.arch,
                    source_commit=self.commit, tooling_commit=t["commit"], tooling_digest=t["digest"],
                    tooling_dirty=t["dirty"], overlay=m if m and m.get("files") else None,
                    route_overlay_env={k: v for k, v in leg.env.items() if k.startswith("MOJOLEARN_ROUTE_OVERLAY")},
                    set_identity=self.set_identity(leg.vendor, leg.arch), backend=self.args.build_backend,
                    release_build=str(leg.release_build), out_dir=str(leg.out_dir), log=str(leg.log),
                    exit_file=str(leg.exit_file))

    def ledger(self):
        """The PASS ledger: R2 for a real run (not a dry run, not a test's
        injected runner, not a --state-dir run), always the local mirror."""
        with self._lock:
            if self._ledger is None:
                if self.args.state_dir:
                    local = Path(self.args.state_dir) / "ledger"
                    self._ledger = release_ledger.Ledger(local, None, say=self.say)
                elif self.runner is None and not self.dry:
                    self._ledger = release_ledger.Ledger.open(self.evidence, say=self.say)
                else:
                    self._ledger = release_ledger.Ledger(self.evidence / "release" / "ledger", None, say=self.say)
                if self.readonly:
                    self._ledger.cache = False
            return self._ledger

    def ledger_put(self, key, entry):
        if self.dry or self.readonly:
            return None
        try:
            return self.ledger().put(key, dict(entry, recorded_at=now()))
        except Exception as exc:  # the ledger is an index; its failure costs a rerun, never a release
            self.say(f"  ledger: could not record {key} ({type(exc).__name__}: {exc})")
            return None

    # ------------------------------------------------------------ steps
    def frozen(self):
        return bool(self.state.get("commit")) and not getattr(self.args, "refreeze", False)

    def step_freeze_version(self):
        files = ["python/mojolearn/_version.py", "python/pyproject.toml"]
        if self.frozen():
            text = file_at(self.commit, files[0]) or ""
            m = re.search(r'^__version__ = "([^"]*)"', text, re.M)
            if not m or m.group(1) != self.version:
                raise StepFailed(f"the frozen source {self.commit[:12]} says {m.group(1) if m else '?'}, not "
                                 f"{self.version}; --refreeze to freeze HEAD")
            return f"frozen source {self.commit[:12]} says {self.version}"
        current = re.search(r'^__version__ = "([^"]*)"', (ROOT / files[0]).read_text(), re.M).group(1)
        if current == self.version and f'version = "{self.version}"' in (ROOT / files[1]).read_text():
            return "already " + self.version
        if self.dry:
            return f"would set {current} -> {self.version} in {', '.join(files)}"
        return "set in " + ", ".join(set_version(ROOT, self.version))

    def step_freeze_changelog(self):
        if self.frozen():
            date = changelog_date(ROOT, self.version, text=file_at(self.commit, "CHANGELOG.md") or "")
            if not date:
                raise StepFailed(f"the frozen source {self.commit[:12]} has no `## {self.version} (published "
                                 "YYYY-MM-DD)` heading; --refreeze after writing it")
            return "published " + date + " (frozen source)"
        date = changelog_date(ROOT, self.version)
        if not date:
            raise StepFailed(f"CHANGELOG.md has no `## {self.version} (published YYYY-MM-DD)` heading. "
                             "Write the entry (publication date in UTC) and rerun.")
        return "published " + date

    def step_freeze_docs_facts(self):
        if self.frozen():
            return f"part of the frozen source {self.commit[:12]}"
        self.must(["pixi", "run", "write-docs-facts"], what="write-docs-facts")
        return "written"

    def step_freeze_commit(self):
        head = git("rev-parse", "HEAD")
        frozen = self.state.get("commit")
        refreeze = getattr(self.args, "refreeze", False)
        if frozen and not refreeze:
            # THE SOURCE NEVER MOVES BECAUSE MAIN MOVED (2026-09-25): a tooling
            # fix lands on main and this checkout runs it; the legs, columns and
            # receipts of the frozen source stay.
            return (f"source pinned at {frozen}" + (f"; tooling runs from HEAD {head[:12]}" if head != frozen else "")
                    + ("" if head == frozen else " (--refreeze moves the source to HEAD)"))
        if frozen and head != frozen and any(self.recorded(s) for s in ("publish-linux", "publish-macos")):
            raise StepFailed(f"--refreeze refused: a wheel of {frozen[:12]} is already published; "
                             "a new source state needs a new version")
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
            # A new source commit (--refreeze). Nothing is discarded: every
            # later step checks its output, and a completed leg or column of the
            # earlier freeze is taken when what it built or tested is unchanged.
            earlier = [c for c in self.state.get("earlier_commits", []) + [self.state.get("commit")] if c]
            self.state = {"version": self.version, "commit": head, "earlier_commits": earlier,
                          "frozen_at": now(), "steps": {k: v for k, v in self.state["steps"].items()
                                                        if k.startswith("freeze-")}}
            self._plan = self._overlay = self._leg_env = self._tooling = None
            self.__dict__.pop("_admitted", None)
            self.save()
        return "frozen at " + head

    def step_rehearsal(self):
        work = self.rel / "rehearsal"
        self.must(["pixi", "run", "release-rehearsal", "--keep", work], log=self.rel / "rehearsal.log",
                  what="release-rehearsal", cwd=self.src)
        return "PASS, logs " + str(work)

    # ------------------------------------------------------------ reuse
    @property
    def plan_path(self):
        return self.rel / "reuse" / "plan.json"

    def plan(self):
        """The reuse plan of the frozen commit, as the builders run after the
        route overlay: read from the release directory when this commit's
        plan with this overlay is there, else computed (in memory only under
        --dry-run; never under --status)."""
        with self._lock:
            if self._plan is not None:
                return self._plan
            doc = read_json(self.plan_path)
            if doc and doc.get("schema") == release_reuse.PLAN_SCHEMA and doc.get("commit") == self.commit:
                if self.readonly:
                    self._plan = doc
                    return doc
                if (doc.get("builders_override") or {}) == self.builders_override():
                    self._plan = doc
                    return doc
            if self.readonly:
                return None
            override = self.builders_override()
            self._plan = release_reuse.make_plan(self.commit, self.evidence / "release" / "identities",
                                                 builders_override=override or None)
            return self._plan

    def step_reuse_plan(self):
        plan = self.plan()
        m = self.overlay()
        if m["files"]:
            self.say(f"  route overlay from the tooling checkout ({self.tooling_commit[:12]}): "
                     + ", ".join(sorted(m["files"])))
        self.say(release_reuse.render(plan, full=self.dry))
        prev = plan.get("previous") or {}
        summary = "%d REUSE, %d BUILD, legs: %s" % (
            len(release_reuse.plan_rows(plan, decision="REUSE")), len(release_reuse.plan_rows(plan, decision="BUILD")),
            ", ".join(plan["legs"]) or "none")
        if self.dry:
            return "would write " + str(self.plan_path) + "; " + summary
        old = read_json(self.plan_path) or {}
        if old.get("commit") != self.commit or old.get("builders_override") != plan.get("builders_override"):
            write_json(self.plan_path, plan)
        return f"against {prev.get('version', 'no published release')}: {summary}; {self.plan_path}"

    def previous_wheel(self, platform):
        """The last published wheel for `platform`, verified against the
        record (a local copy, else PyPI)."""
        prev = (self.plan().get("previous") or {})
        if self.dry:
            info = prev.get(platform) or {}
            return Path("<published %s wheel %s>" % (platform, info.get("wheel", "?")))
        return release_reuse.published_wheel(prev, platform, self.evidence)

    # ------------------------------------------------------------ leg reuse by identity
    def admission_path(self, name):
        return self.rel / "legs" / f"{name}.reused.json"

    def admitted_leg(self, name):
        """The admission of a completed leg of an earlier freeze for NAME in this
        freeze, still true (its tree verifies), or None. Read-only."""
        doc = read_json(self.admission_path(name))
        if not doc or doc.get("admitted_for") != self.commit:
            return None
        cache = self.__dict__.setdefault("_admitted", {})
        key = (name, json.dumps(doc, sort_keys=True))
        if key in cache:
            return cache[key]
        vendor, arch = name.split("-", 1)
        ok, _, proof_sha = verify_leg_tree(doc["release_build"], vendor, arch, doc["built_from"], doc.get("exit_file"))
        cache[key] = doc if ok and proof_sha == doc.get("proof_sha256") else None
        return cache[key]

    def admitted_legs(self):
        out = {}
        for vendor, arch in release_reuse.LINUX_SETS:
            doc = self.admitted_leg(f"{vendor}-{arch}")
            if doc:
                out[f"{vendor}-{arch}"] = doc
        return out

    def leg_candidates(self, name, vendor, arch, ident):
        out = []
        here = self.rel.resolve()
        for p in sorted(self.base.glob(f"*/legs/{name}.provenance.json")):
            if p.parent.parent.resolve() == here:
                continue
            doc = read_json(p)
            if doc:
                out.append((str(p), doc))
        if ident:
            for e in self.ledger().find(f"build/{vendor}-{arch}/{ident['digest']}/"):
                if e.get("verdict") == "PASS" and e.get("provenance"):
                    out.append(("ledger " + e.get("key", "?"), e["provenance"]))
        return out

    def find_reusable_leg(self, name, vendor, arch, write=True):
        """(admission, why not): a completed leg of an earlier freeze whose set
        identity AND build tooling digest equal this freeze's, whose exit is 0,
        whose proof is complete for the commit it built, whose every binary is
        the proof's bytes and whose route overlay verified on the box. Anything
        else (unequal, unreadable, dirty tooling, missing evidence) is refused
        and the leg is built."""
        ident = self.set_identity(vendor, arch)
        if not ident:
            return None, "the set identity is unreadable (no plan, or a binding without a readable identity)"
        if not self.packer_takes_legs():
            return None, "the frozen source's packer predates leg origins in reuse.json"
        t = self.tooling()
        if not t["digest"]:
            return None, "the build tooling has uncommitted changes (" + ", ".join(t["dirty"]) + ")"
        why = []
        for where, c in self.leg_candidates(name, vendor, arch, ident):
            if c.get("source_commit") == self.commit:
                continue
            if (c.get("set_identity") or {}).get("digest") != ident["digest"]:
                why.append(f"{where}: set identity differs")
                continue
            if c.get("tooling_digest") != t["digest"]:
                why.append(f"{where}: build tooling differs")
                continue
            ok, reason, proof_sha = verify_leg_tree(c["release_build"], vendor, arch, c["source_commit"], c.get("exit_file"))
            if ok:
                ok, reason = overlay_verified(c, c.get("out_dir", ""))
            if not ok:
                why.append(f"{where}: {reason}")
                continue
            doc = dict(schema="mojolearn.release-leg-admission.v1", leg=name, set=f"{vendor}/{arch}",
                       built_from=c["source_commit"], built_with_tooling=c.get("tooling_commit"),
                       admitted_for=self.commit, admitted_with_tooling=t["commit"],
                       because=("the set identity (every binding identity of the set, the host bindings and the "
                                "runtime closure) and the build tooling digest equal this freeze's"),
                       set_identity_digest=ident["digest"], tooling_digest=t["digest"],
                       release_build=c["release_build"], exit_file=c.get("exit_file"),
                       set_dir=str(Path(c["release_build"]) / "build" / "sets" / vendor / arch),
                       proof_sha256=proof_sha, provenance=where, at=now())
            if write and not self.dry and not self.readonly:
                write_json(self.admission_path(name), doc)
            return doc, None
        return None, "; ".join(why) or "no completed leg of an earlier freeze with this set identity"

    def packer_takes_legs(self):
        """The pack runs the SOURCE's pack_wheel.py; one older than leg
        origins would pack a taken leg's files as the published release's."""
        return "LEG_ORIGIN_KEYS" in (file_at(self.commit, "packaging/linux/pack_wheel.py") or "")

    def leg_origin(self, doc):
        """reuse.json's origin of an admitted leg (pack_wheel.LEG_ORIGIN_KEYS)."""
        return dict(source_commit=doc["built_from"], leg=doc["leg"], proof_sha256=doc["proof_sha256"],
                    admitted_for=doc["admitted_for"], set_identity_digest=doc["set_identity_digest"],
                    tooling_digest=doc["tooling_digest"], release_build=doc["release_build"])

    def reused_leg_objects(self):
        out = []
        for name, doc in self.admitted_legs().items():
            vendor, arch = name.split("-", 1)
            out.append(Leg(name, vendor, arch, [], {}, doc["release_build"], self.rel / "legs", doc["release_build"]))
        return out

    def step_linux_builds(self):
        legs = linux_legs(self)
        prev = (self.plan().get("previous") or {}).get("version")
        if not legs:
            return f"no build leg: every Linux binding is taken from {prev} (nothing rented for builds)"
        launch, taken = [], []
        for leg in legs:
            if leg.running():
                self.say(f"  {leg.name}: already running (pid {leg.pid()}), log {leg.log}")
                continue
            if leg.done(self.commit):
                self.say(f"  {leg.name}: already done at this freeze")
                continue
            doc = self.admitted_leg(leg.name)
            why = None
            if not doc:
                doc, why = self.find_reusable_leg(leg.name, leg.vendor, leg.arch)
            if doc:
                taken.append(leg.name)
                self.say(f"  {leg.name}: TAKEN, not launched: built from {doc['built_from'][:12]} "
                         f"({doc['provenance']}), admitted for {doc['admitted_for'][:12]} because {doc['because']}")
                continue
            self.say(f"  {leg.name}: {leg.describe()}  [{self.plan()['leg_reasons'].get(leg.name, '')}]"
                     + (f"  (not reusable: {why})" if why else ""))
            launch.append(leg)
        if self.dry:
            return (f"would launch {len(launch)} leg(s) ({self.args.build_backend})"
                    + (f"; would take {', '.join(taken)} from an earlier freeze" if taken else ""))
        launch_detached(self, launch)
        return (f"{len(launch)} leg(s) launched or running ({self.args.build_backend})"
                + (f"; taken from an earlier freeze: {', '.join(taken)}" if taken else ""))

    # ------------------------------------------------------------ macOS
    def macos_wheel(self):
        found = sorted((self.rel / "macos").glob("mojolearn-*-macosx_*.whl"))
        return found[-1] if found else None

    def step_macos_build(self):
        w = self.macos_wheel()
        if w and wheel_commit(w) == self.commit:
            return "have " + w.name
        env = dict(MOJOLEARN_PACKAGE_BYTE_LM="1", MOJOLEARN_BUILD_JOBS="4", MOJOLEARN_COMPILE_JOBS="1")
        plan = self.plan()
        src = self.src
        reuse = release_reuse.plan_rows(plan, release_reuse.MACOS, "REUSE")
        if reuse:
            store = self.rel / "reuse" / "macos"
            if self.dry:
                self.say(f"  {len(reuse)} macOS binding(s) would be placed from the published "
                         f"{plan['previous']['version']} wheel, verified against its RECORD")
            else:
                release_reuse.assemble_macos(plan, self.previous_wheel("macos"), store, say=self.say)
            env.update(MOJOLEARN_REUSE_PLAN=str(store / "macos-plan.json"), MOJOLEARN_REUSE_DIR=str(store))
        self.must([PY, str(ROOT / "tools" / "mac_slot.py"), "--slots", "4", "run", "--",
                   "./packaging/macos/build_release_wheel.sh"],
                  env=env, log=self.rel / "macos-build.log", what="build_release_wheel.sh", cwd=src)
        if self.dry:
            return "would copy python/dist/*.whl to " + str(self.rel / "macos")
        built = sorted((src / "python" / "dist").glob(f"mojolearn-{self.version}-*-macosx_*.whl"))
        if len(built) != 1 or wheel_commit(built[0]) != self.commit:
            raise StepFailed(f"expected one macOS wheel of {self.commit} in {src}/python/dist, found {built}")
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
        # the smoke is stdlib-only tooling (the Linux column ships the same file)
        self.must([PY, str(ROOT / "tools" / "mac_slot.py"), "--wait-timeout", "3600", "metal", "--",
                   PY, str(ROOT / "tools" / "qualify_verifier_wheel.py"), w or "<macos wheel>", "--scope", "expanded",
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

    def record_apple_column(self):
        col = self.release_check_dir() / "metal" / "column.json"
        if col.is_file():
            self.ledger_put(release_ledger.apple_key(self.commit), dict(
                verdict="PASS", kind="column", vendor="metal", source_commit=self.commit,
                tooling_commit=self.tooling_commit, evidence=dict(dir=str(col.parent), column_sha256=sha256(col))))

    def step_release_check(self):
        if self.release_check_complete():
            self.record_apple_column()
            return "complete at " + str(self.release_check_dir())
        src = self.src
        if not self.dry and self.head_of(src) != self.commit:
            raise StepFailed(f"the source checkout {src} is not at the frozen commit; release-check verifies it")
        cmd = ["pixi", "run", "-e", "test", "release-check"]
        if "cpu" in self.check_backends():
            cmd.append("--cpu-column")
        self.must(cmd, log=self.rel / "release-check.log", what="release-check", cwd=src)
        if not self.dry and not self.release_check_complete():
            raise StepFailed("release-check exited 0 but its records are not complete")
        self.record_apple_column()
        return "complete"

    # ------------------------------------------------------------ Linux
    def step_linux_wait(self):
        legs = [l for l in linux_legs(self) if not self.admitted_leg(l.name)]
        taken = self.reused_leg_objects()
        if not legs:
            if taken:
                return "no leg to wait for: " + ", ".join(l.name for l in taken) + " taken from an earlier freeze"
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
            self.sleep(60)
        bad = []
        for l in legs:
            if l.exit_code() != 0 or not l.proof_ok(self.commit):
                bad.append(f"{l.name} (exit {l.exit_code()}, log {l.log})")
                continue
            ok, why = overlay_verified(read_json(l.provenance_file), l.out_dir)
            if not ok:
                bad.append(f"{l.name} ({why})")
        if bad:
            raise StepFailed("build leg(s) failed: " + ", ".join(bad) + ". Rerun `release` to relaunch them.")
        mismatch = host_digest_mismatches(legs + taken)
        if mismatch:
            lines = [f"{name}: " + ", ".join(f"{leg}={d[:12]}" for leg, d in row.items())
                     for name, row in sorted(mismatch.items())]
            raise StepFailed("host bindings differ across legs (pack_wheel.py would refuse):\n  "
                             + "\n  ".join(lines[:12]))
        for l in legs:
            self.record_leg(l)
        return ("all legs built; host bindings byte-identical across " + ", ".join(l.name for l in legs + taken)
                + (f" ({', '.join(l.name for l in taken)} taken from an earlier freeze)" if taken else ""))

    def record_leg(self, leg):
        """A leg that built, in the PASS ledger, keyed by what it built."""
        prov = read_json(leg.provenance_file) or {}
        ident = prov.get("set_identity") or {}
        if not ident.get("digest") or not prov.get("tooling_digest"):
            return
        ok, _, proof_sha = verify_leg_tree(leg.release_build, leg.vendor, leg.arch, self.commit, leg.exit_file)
        if not ok:
            return
        self.ledger_put(release_ledger.build_key(leg.vendor, leg.arch, ident["digest"], prov["tooling_digest"]), dict(
            verdict="PASS", kind="build", set=f"{leg.vendor}/{leg.arch}", source_commit=self.commit,
            tooling_commit=prov.get("tooling_commit"), provenance=prov,
            evidence=dict(release_build=str(leg.release_build), proof_sha256=proof_sha, log=str(leg.log))))

    def linux_final(self):
        found = sorted((self.rel / "linux" / "final").glob("mojolearn-*-manylinux*.whl"))
        return found[-1] if found else None

    def assembled_marker(self):
        return self.rel / "reuse" / "sets" / "assembled.json"

    def needs_published_linux(self):
        plan = self.plan()
        built = {l.name for l in linux_legs(self)} if not self.dry else set(plan["legs"])
        taken = set(self.admitted_legs())
        return (bool(release_reuse.plan_rows(plan, release_reuse.LINUX, "REUSE"))
                or plan["runtime"]["decision"] == "REUSE"
                or any(f"{v}-{a}" not in built | taken for v, a in release_reuse.LINUX_SETS))

    def assembly_needed(self):
        """True when the plan takes any Linux bytes from the published wheel or
        a leg of an earlier freeze is taken; otherwise the packer reads the
        legs' sets directly, as before."""
        plan = self.plan()
        return (bool(release_reuse.plan_rows(plan, release_reuse.LINUX, "REUSE"))
                or plan["runtime"]["decision"] == "REUSE" or bool(self.admitted_legs()))

    def assembled_ok(self):
        d = read_json(self.assembled_marker())
        if not d:
            return False
        return (d.get("commit") == self.commit and d.get("plan_sha256") == sha256(self.plan_path)
                and sorted(d.get("taken") or []) == sorted(self.admitted_legs()))

    def step_linux_assemble(self):
        if not self.assembly_needed():
            return "not needed: every Linux binding is built by the legs"
        if self.assembled_ok():
            return "have " + str(self.assembled_marker().parent)
        taken = self.admitted_legs()
        built = [l for l in linux_legs(self) if l.name not in taken]
        legs = {l.name: l.release_build / "build" / "sets" / l.vendor / l.arch for l in built}
        leg_reuse = {n: dict(dir=Path(d["set_dir"]), origin=self.leg_origin(d)) for n, d in taken.items()}
        plan = self.plan()
        if self.dry:
            n = len(release_reuse.plan_rows(plan, release_reuse.LINUX, "REUSE"))
            return (f"would take {n} binding(s) and the runtime closure from the published "
                    f"{(plan['previous'] or {}).get('version')} Linux wheel into {self.rel / 'reuse' / 'sets'}"
                    + (f", over the sets of {', '.join(legs)}" if legs else " (no leg ran)")
                    + (f", and the sets of {', '.join(taken)} from an earlier freeze" if taken else ""))
        whl = self.previous_wheel("linux") if self.needs_published_linux() else None
        for name, d in legs.items():
            if not d.is_dir():
                raise StepFailed(f"{name} has no set directory at {d}; run linux-wait")
        dest = self.rel / "reuse" / "sets"
        try:
            release_reuse.assemble_linux(plan, whl, legs, dest, say=self.say, leg_reuse=leg_reuse)
        except SystemExit as exc:
            raise StepFailed(str(exc))
        write_json(self.assembled_marker(), dict(
            commit=self.commit, plan_sha256=sha256(self.plan_path), wheel=str(whl) if whl else None,
            wheel_sha256=sha256(whl) if whl else None, legs=sorted(legs), taken=sorted(taken),
            taken_from={n: d["built_from"] for n, d in taken.items()}, at=now()))
        return (f"{dest}: sets assembled" + (f" from {whl.name}" if whl else "")
                + (f" and legs {', '.join(sorted(legs))}" if legs else "")
                + (f"; taken from an earlier freeze: {', '.join(sorted(taken))}" if taken else ""))

    def pack_inputs(self):
        """(set directories, proofs, manifests) for pack_wheel.py and audit.sh:
        the assembled sets when the release reuses anything, else the legs'.
        A leg taken from an earlier freeze gives no proof (its set is named in
        reuse.json with its origin)."""
        taken = self.admitted_legs()
        legs = [l for l in linux_legs(self) if l.name not in taken]
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
        src = self.src
        dist = self.rel / "linux"
        if dist.exists() and not self.dry:
            dist.rename(dist.with_name("linux.failed-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")))
        args = ["pixi", "run", "-e", "pkg", "pack-linux-wheel", "--profile", "release-linux3"]
        for s in sets:
            args += ["--set", s]
        for p in proofs:
            args += ["--build-proof", p]
        args += ["--out", dist]
        self.must(args, log=self.rel / "linux-pack.log", what="pack_wheel.py", cwd=src)
        packed = sorted(dist.glob("mojolearn-*-linux_x86_64.whl")) if not self.dry else [dist / "<packed>.whl"]
        if len(packed) != 1:
            raise StepFailed(f"expected one packed wheel in {dist}, found {packed}")
        self.must(["bash", "packaging/linux/audit.sh", packed[0], *manifests], log=self.rel / "linux-audit.log",
                  what="audit.sh", cwd=src)
        repaired = sorted((dist / "audit" / "repaired").glob("mojolearn-*-manylinux*.whl")) if not self.dry \
            else [dist / "audit" / "repaired" / "<repaired>.whl"]
        if len(repaired) != 1:
            raise StepFailed(f"expected one repaired wheel, found {repaired}")
        if not self.dry:
            (dist / "final").mkdir(parents=True, exist_ok=True)
        self.must([PY, str(ROOT / "tools" / "strip_wheel_dir_entries.py"), repaired[0],
                   dist / "final" / repaired[0].name, "--receipt", dist / "final" / "dir-entry-strip.json"],
                  what="strip_wheel_dir_entries.py")
        final = self.linux_final()
        return f"{final.name} sha256 {sha256(final)}" if final else "would pack, audit and strip"

    # ------------------------------------------------------------ GPU columns
    def gpu_selection(self, vendor):
        """The release pass's lanes for this vendor, worked out by the selector
        exactly as the pass would (verify_lanes --gpu-pass --write-selection),
        in the source checkout (the lanes the SOURCE changed)."""
        path = self.rel / f"selection-{vendor}.json"
        self.must([PY, "tools/verify_lanes.py", "--gpu-pass", vendor, "--write-selection", path],
                  log=self.rel / f"selection-{vendor}.log", what=f"{vendor} lane selection", cwd=self.src)
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

    def column_specs(self):
        return [("nvidia", "cuda", self.nvidia_column_ok, self.rel / "smoke-linux"),
                ("amd", "hip", self.amd_column_ok, self.rel / "column-amd")]

    def column_candidates(self, vendor, outname, wsha, sel_digest):
        out, here = [], (self.rel / outname).resolve()
        for d in sorted(self.base.glob(f"*/{outname}")):
            if d.resolve() != here and (d / "column-provenance.json").is_file():
                out.append((str(d), d))
        e = self.ledger().get(release_ledger.column_key(vendor, wsha, sel_digest))
        if e and e.get("verdict") == "PASS" and (e.get("evidence") or {}).get("dir"):
            out.append(("ledger " + e.get("key", "?"), Path(e["evidence"]["dir"])))
        return out

    def rediff(self, col, vendor, out):
        cmd = [PY, str(ROOT / "tools" / "identity_break.py"), "--diff", *[str(r) for r in self.column_refs()], str(col)]
        got = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        (out / f"diff-ref-{vendor}.txt").write_text(got.stdout + got.stderr)

    def reuse_column(self, name, vendor, out, sel_digest):
        """Take a PASSED column of an earlier freeze (or another machine's, from
        the ledger) only for a BYTE-IDENTICAL wheel (sha256) and the same lane
        selection, its column file verified against its provenance; it is
        diffed again against this release's reference columns here."""
        final = self.linux_final()
        if not final or not sel_digest:
            return None
        wsha = sha256(final)
        for where, d in self.column_candidates(vendor, out.name, wsha, sel_digest):
            prov = read_json(d / "column-provenance.json") or {}
            col = d / f"column-{vendor}.json"
            if (prov.get("vendor") != vendor or prov.get("wheel_sha256") != wsha
                    or prov.get("selection_digest") != sel_digest or not col.is_file()
                    or sha256(col) != prov.get("column_sha256")):
                continue
            if name == "nvidia" and not smoke_passed(d / "results.json", final):
                continue
            if out.exists():
                out.rename(out.with_name(out.name + ".failed-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")))
            out.mkdir(parents=True)
            for f in (col.name, "results.json"):
                if (d / f).is_file():
                    shutil.copy2(d / f, out / f)
            self.rediff(out / col.name, vendor, out)
            doc = dict(prov, reused_from=str(d), found_by=where, admitted_for=self.commit,
                       admitted_with_tooling=self.tooling_commit,
                       because="the wheel is byte-identical (sha256) and the lane selection is the same")
            write_json(out / "column-provenance.json", doc)
            write_json(out / "reused.json", doc)
            self.say(f"  {name}: TAKEN from {d} ({where}): wheel {wsha[:12]} byte-identical, same lanes; "
                     f"diffed again against {self.ref_names()}")
            return doc
        return None

    def column_legs(self):
        """The two GPU wheel columns as detached legs, each only when its
        record for this wheel is not already there and no PASSED column of a
        byte-identical wheel can be taken. The NVIDIA leg carries its GPU walk
        (SMOKE_WALK or --smoke-gpu; the position survives a rerun in
        <columns>/nvidia.gpu); the AMD leg's provider walk (RunPod, Hot Aisle,
        DigitalOcean) is release_wheel_smoke.sh's own --provider auto."""
        final_path = self.linux_final()
        final = str(final_path or "<final linux wheel>")
        work = self.rel / "columns"
        refs = [a for r in self.column_refs() for a in ("--ref-column", str(r))]
        legs = []
        for name, vendor, ok, out in self.column_specs():
            if ok():
                continue
            sel = self.gpu_selection(vendor)
            sel_digest = selection_digest(sel)
            if not self.dry and self.reuse_column(name, vendor, out, sel_digest) and ok():
                continue
            if name == "nvidia":
                walk = [g for g in self.args.smoke_gpu.split("|") if g] or SMOKE_WALK
                try:
                    at = min(int((work / "nvidia.gpu").read_text()), len(walk) - 1)
                except (OSError, ValueError):
                    at = 0
                leg = ColumnLeg("nvidia", "cuda",
                                ["bash", "tools/release_wheel_smoke.sh", final, "--expected-source-commit", self.commit,
                                 "--out", str(out), "--rent", "--column", str(sel), *refs,
                                 "--gpu", walk[at]], work, out, ok)
                leg.walk, leg.at = walk, at
            else:
                leg = ColumnLeg("amd", "hip",
                                ["bash", "tools/release_wheel_smoke.sh", final, "--expected-source-commit",
                                 self.commit, "--out", str(out), "--rent", "--vendor", "hip",
                                 "--provider", self.args.amd_provider,
                                 "--column", str(sel), *refs],
                                work, out, ok)
            leg.provenance = dict(schema="mojolearn.release-column-provenance.v1", column=name, vendor=vendor,
                                  wheel=final, wheel_sha256=sha256(final_path) if final_path and not self.dry else None,
                                  selection=str(sel), selection_digest=sel_digest, source_commit=self.commit,
                                  tooling_commit=self.tooling_commit, refs=[str(r) for r in self.column_refs()])
            legs.append(leg)
        return legs

    def record_column(self, name, vendor, out):
        """A PASSED column: its provenance beside it, and the ledger entry."""
        prov = read_json(self.rel / "columns" / f"{name}.provenance.json") or read_json(out / "column-provenance.json")
        col = out / f"column-{vendor}.json"
        if not prov or not col.is_file():
            return
        if not (out / "reused.json").is_file():
            prov = dict(prov, column_sha256=sha256(col), verdict="PASS")
            write_json(out / "column-provenance.json", prov)
        if prov.get("wheel_sha256") and prov.get("selection_digest"):
            self.ledger_put(release_ledger.column_key(vendor, prov["wheel_sha256"], prov["selection_digest"]), dict(
                verdict="PASS", kind="column", vendor=vendor, source_commit=self.commit,
                tooling_commit=prov.get("tooling_commit"), provenance=prov,
                evidence=dict(dir=str(out), column_sha256=sha256(col))))

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
        for (name, vendor, ok, out), label in zip(self.column_specs(), ("NVIDIA", "AMD")):
            if ok():
                self.say(f"  {label} column: PASSED, no DIVERGENT cell against {self.ref_names()}")
                self.record_column(name, vendor, out)
                continue
            leg = next((l for l in legs if l.vendor == vendor), None)
            failed.append(f"{label} column missing, not PASSED or DIVERGENT: {out}"
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

    # ------------------------------------------------------------ publication
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
        """Publish ONE platform's wheel, built from the frozen source. The
        publisher runs from the tooling checkout (its tag names the tooling;
        MOJOLEARN_ARTIFACT_SOURCE_COMMIT pins the wheel's source witness)."""
        target = self.args.publish
        if target is None:
            raise StepHeld("publication needs an explicit --publish none|testpypi|pypi; stopping here")
        if not self.dry and wheel and wheel_commit(wheel) != self.commit:
            raise StepFailed(f"the {platform} wheel {wheel.name} records source {wheel_commit(wheel)}, not the frozen "
                             f"{self.commit}")
        if not self.dry and wheel and self.on_pypi(wheel):
            return "already on PyPI"
        day = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
        tag = f"alpha-api-{self.version}-{platform}-{day}"
        work = self.rel / f"publish-{platform}"
        self.must(["bash", "tools/release_linux_publish.sh", wheel or f"<{platform} wheel>", tag, target, work,
                   "--light-smoke", smoke], env=dict(MOJOLEARN_ARTIFACT_SOURCE_COMMIT=self.commit),
                  log=self.rel / f"publish-{platform}.log", what=f"publish {platform}")
        return f"{target} via {tag} (source {self.commit[:12]}, tooling {self.tooling_commit[:12]})"

    def step_publish_linux(self):
        return self.publish("linux", self.linux_final(), self.rel / "smoke-linux" / "results.json")

    def step_publish_macos(self):
        return self.publish("macos", self.macos_wheel(), self.rel / "smoke-macos" / "results.json")

    def finish_macos(self, venv):
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
        return f"macOS: pip install mojolearn=={self.version} on this Mac: OK"

    def finish_linux(self, venv):
        """The Linux wheel cannot install on this Mac: pip resolves and
        downloads it for its platform, and its bytes must be the published ones."""
        final = self.linux_final()
        plat = final.name[:-4].split("-")[-1] if final else "manylinux_2_35_x86_64"
        dest = self.rel / "finish-linux"
        for attempt in range(1, 7):
            if dest.exists() and not self.dry:
                shutil.rmtree(dest)
            rc = self.run([venv / "bin" / "pip", "download", "--no-cache-dir", "--no-deps", "--only-binary=:all:",
                           "--platform", plat, "--python-version", "3.12", "-d", dest,
                           f"mojolearn=={self.version}"], log=self.rel / f"finish-linux-{attempt}.log")
            got = sorted(dest.glob("mojolearn-*.whl")) if not self.dry else []
            if self.dry or (rc == 0 and final and len(got) == 1 and sha256(got[0]) == sha256(final)):
                return f"Linux: pip resolves mojolearn=={self.version} for {plat}; bytes are the published wheel's"
            self.say(f"  pip download failed or differs (attempt {attempt}/6)")
            self.sleep(30)
        raise StepFailed(f"pip download mojolearn=={self.version} for {plat} failed or differs six times")

    def step_finish_line(self):
        if self.args.publish != "pypi":
            return "skipped (not published to PyPI)"
        published = self.published_platforms()
        if not published:
            return "nothing published yet", dict(platforms=[])
        venv = self.rel / "finish-venv"
        if venv.exists() and not self.dry:
            shutil.rmtree(venv)
        self.must([self.smoke_python() if not self.dry else "python3.12", "-m", "venv", venv], what="venv")
        done = [self.finish_macos(venv) if p == "macos" else self.finish_linux(venv) for p in published]
        pending = [PIPELINES[p]["platform"] for p in PIPELINES if PIPELINES[p]["platform"] not in published]
        return "; ".join(done) + (f"; not yet published: {', '.join(pending)}" if pending else ""), \
            dict(platforms=published)

    def release_provenance(self):
        legs = {}
        for v, a in release_reuse.LINUX_SETS:
            name = f"{v}-{a}"
            doc = self.admitted_leg(name)
            prov = read_json(self.rel / "legs" / f"{name}.provenance.json")
            if doc:
                legs[name] = dict(origin="taken", built_from=doc["built_from"], admitted_for=doc["admitted_for"],
                                  because=doc["because"], set_identity_digest=doc["set_identity_digest"])
            elif prov:
                legs[name] = dict(origin="built", source_commit=prov.get("source_commit"),
                                  tooling_commit=prov.get("tooling_commit"), tooling_digest=prov.get("tooling_digest"),
                                  overlay=prov.get("overlay"),
                                  set_identity_digest=(prov.get("set_identity") or {}).get("digest"))
        cols = {name: read_json(out / "column-provenance.json") for name, _, _, out in self.column_specs()}
        return dict(schema="mojolearn.release-provenance.v1", version=self.version, source_commit=self.commit,
                    tooling_commit=self.tooling_commit, earlier_commits=self.state.get("earlier_commits", []),
                    published=self.published_platforms(), legs=legs, columns=cols)

    def step_record(self):
        if self.args.publish != "pypi":
            return "skipped (not published to PyPI)"
        published = self.published_platforms()
        if not published:
            return "nothing published yet", dict(platforms=[])
        date = changelog_date(ROOT, self.version, text=file_at(self.commit, "CHANGELOG.md") or "") \
            or dt.date.today().isoformat()
        rec = ROOT / "bench" / "results" / "release_verification" / f"{date}_pypi_{self.version.replace('.', '')}"
        if self.dry:
            return "would write " + str(rec.relative_to(ROOT)), dict(platforms=published)
        rec.mkdir(parents=True, exist_ok=True)
        for platform in published:
            art = self.rel / f"publish-{platform}"
            for src, name in ((art / "artifact" / "alpha-manifest.json", f"alpha-manifest-{platform}.json"),
                              (art / "file-admission.json", f"file-admission-{platform}.json"),
                              (self.rel / f"smoke-{platform}" / "results.json", f"light-smoke-{platform}.json")):
                if src.is_file():
                    shutil.copy2(src, rec / name)
        (rec / "README.md").write_text(self.readme())
        write_json(rec / "release-provenance.json", self.release_provenance())
        # The identity of every shipped binding and what it shipped as, so the
        # next release decides REUSE or BUILD from this record alone (the
        # Apple toolchain that built the macOS wheel included).
        release_reuse.record_identities(self.plan(), self.linux_final() if "linux" in published else None,
                                        self.macos_wheel() if "macos" in published else None,
                                        rec / "binding-identities.json")
        self.must(["git", "-C", ROOT, "add", "--", rec], what="git add record")
        self.must(["git", "-C", ROOT, "commit", "-q", "-m",
                   f"Record the published {self.version} wheels ({', '.join(published)}) and their release verification",
                   "-m", "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"],
                  what="git commit record")
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        # main has usually moved on during the builds; the record rides on top
        # of it (0.8.16's record push was rejected as non-fast-forward)
        self.must(["git", "-C", ROOT, "fetch", "-q", "origin", branch], what="git fetch")
        self.must(["git", "-C", ROOT, "rebase", "-q", f"origin/{branch}"], what="git rebase record onto origin")
        self.must(["git", "-C", ROOT, "push", "origin", f"HEAD:refs/heads/{branch}"], what="git push record")
        return str(rec.relative_to(ROOT)) + f" ({', '.join(published)}) committed on " + branch, \
            dict(platforms=published)

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
        published = self.published_platforms()
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
                          f"| {tag.get('result', '?') if platform in published else 'not yet published'} |")
        date = changelog_date(ROOT, self.version, text=file_at(self.commit, "CHANGELOG.md") or "")
        return "\n".join([
            f"# mojolearn {self.version}", "",
            f"Source commit {self.commit}; release tooling at {self.tooling_commit}. Published {date}. "
            "See CHANGELOG.md.", "",
            "## Release verification (the changed lanes, one fit per cell)", "",
            "| column | lanes | cells | complete |", "|---|---|---|---|", *rows, "",
            "Every column together (`tools/identity_break.py --diff`, diff-columns.txt):", "",
            *[f"    {s}" for s in summary], "", *sums, "",
            "## Wheels (light route)", "",
            "| platform | sha256 | smoke | published |", "|---|---|---|---|", *wheels, "",
            f"Finish line: {(self.recorded('finish-line') or {}).get('result', 'not run')}.", "",
            "Provenance of every leg and column (built, or taken and why): release-provenance.json.", ""])

    # ------------------------------------------------------------ the run
    def run_step(self, step):
        """Run one step; 'done', 'failed' or 'held'. Never raises in a real run:
        a crash is a failed step, reported, so the other pipeline goes on."""
        rec = self.recorded(step)
        if rec and self.skip_recorded(step, rec):
            self.say(f"-- {step}: done {rec['at']} ({rec.get('result', '')})")
            return "done"
        self.say(f"-- {step}")
        shown = self.commands_shown
        try:
            out = getattr(self, "step_" + step.replace("-", "_"))()
            result, data = out if isinstance(out, tuple) else (out, {})
            if self.dry and self.commands_shown > shown:
                result = "pending: would run the command(s) above"
        except StepHeld as exc:
            if self.dry:
                self.say(f"   A REAL RUN WOULD STOP HERE: {exc}")
                return "done"
            self.say(f"   HELD at {step}: {exc}")
            self.fail(step, exc, held=True)
            return "held"
        except StepFailed as exc:
            if self.dry:
                self.say(f"   A REAL RUN WOULD STOP HERE: {exc}")
                return "done"
            self.say(f"   STOPPED at {step}: {exc}")
            self.fail(step, exc)
            return "failed"
        except (Exception, SystemExit) as exc:
            if self.dry:
                raise
            self.say(f"   CRASHED at {step}: {exc!r}\n" + traceback.format_exc(limit=6))
            self.fail(step, f"{type(exc).__name__}: {exc}")
            return "failed"
        self.say(f"   {result}")
        if not self.dry:
            self.mark(step, result=result, **data)
        return "done"

    def schedule(self, selected):
        """THE PIPELINES, AT ONCE. A step starts when every step it needs is
        done (and its resource is free); a failed, held or blocked need blocks
        it, and nothing else. Returns {step: done|failed|held|blocked|skipped}."""
        status = {s: ("pending" if s in selected else "skipped") for s in self.STEPS}
        busy = set()
        events = queue.Queue()

        def worker(step):
            self._tl.tag = PIPELINE_OF[step] if PIPELINE_OF[step] in PIPELINES else ""
            try:
                events.put((step, self.run_step(step)))
            except BaseException as exc:  # run_step never raises; this is a last resort
                events.put((step, "failed"))
                self.errors[step] = repr(exc)

        while True:
            changed = True
            while changed:
                changed = False
                for s in self.STEPS:
                    if status[s] != "pending":
                        continue
                    bad = [d for d in NEEDS[s] if status[d] in ("failed", "held", "blocked")]
                    if bad:
                        status[s] = "blocked"
                        self.blocked_by[s] = bad[0]
                        changed = True
            for s in self.STEPS:
                if status[s] != "pending":
                    continue
                if any(status[d] not in ("done", "skipped") for d in NEEDS[s]):
                    continue
                if any(status[d] in ("pending", "running") for d in AFTER.get(s, ())):
                    continue
                res = RESOURCE[s]
                if res and res in busy:
                    continue
                status[s] = "running"
                if res:
                    busy.add(res)
                threading.Thread(target=worker, args=(s,), name="release-" + s, daemon=True).start()
            if not any(v == "running" for v in status.values()):
                if any(v == "pending" for v in status.values()):
                    raise RuntimeError("release: a step can never start: " + ", ".join(
                        s for s, v in status.items() if v == "pending"))
                return status
            step, outcome = events.get()
            status[step] = outcome
            busy.discard(RESOURCE[step])

    def report(self, status):
        """Both outcomes, one line each, and what a rerun does."""
        self.say("== pipelines")
        for s in self.STEPS:
            if PIPELINE_OF[s] == "common" and status[s] in ("failed", "held"):
                self.say(f"   common: {status[s].upper()} at {s}: {self.errors.get(s, '')}")
        lines = []
        for name, p in PIPELINES.items():
            steps = [s for s in self.STEPS if PIPELINE_OF[s] == name] + \
                [d for s in self.STEPS if PIPELINE_OF[s] == name for d in NEEDS[s] if PIPELINE_OF[d] not in (name, "common")]
            pub = p["publish"]
            if status[pub] == "done" and self.recorded(pub):
                line = f"{name}: PUBLISHED ({self.recorded(pub).get('result', '')})"
            elif status[pub] == "done":
                line = f"{name}: done through {pub}"
            else:
                order = [s for s in self.STEPS if s in steps]
                first = next((s for s in order if status[s] in ("failed", "held")), None)
                blocked = next((s for s in order if status[s] == "blocked"), None)
                if first:
                    line = f"{name}: {status[first].upper()} at {first}: {self.errors.get(first, '')}"
                elif blocked:
                    by = self.blocked_by.get(blocked)
                    line = f"{name}: BLOCKED at {blocked}, which needs {by} ({status.get(by)})"
                else:
                    line = f"{name}: not published ({pub} {status[pub]})"
            lines.append(line)
            self.say("   " + line)
        for s in ("finish-line", "record"):
            if status[s] in ("done",) and self.recorded(s):
                self.say(f"   {s}: {self.recorded(s).get('result', '')}")
            elif status[s] not in ("skipped",):
                self.say(f"   {s}: {status[s]}" + (f": {self.errors[s]}" if s in self.errors else ""))
        ok = all(v in ("done", "skipped") for v in status.values())
        if not ok:
            self.say(f"   Rerun `pixi run release {self.version}"
                     + (f" --publish {self.args.publish}" if self.args.publish else "")
                     + "`: it resumes only what is not done (`--status` shows what that is).")
        return ok

    def banner(self):
        c = self.state.get("commit")
        head = git_at(ROOT, "rev-parse", "HEAD") or "?"
        self.say(f"   state {self.state_path}; source commit {c or '(not frozen yet: HEAD ' + head[:12] + ')'}")
        if c and head != c:
            self.say(f"   tooling: this checkout {ROOT} at {head[:12]}; the source is read from "
                     f"{self.source_checkout(create=False)}")

    def go(self):
        if getattr(self.args, "status", False):
            return self.status()
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
        self.banner()
        if self.dry:
            for step in steps:
                self.run_step(step)
            self.say("== end of plan (the macos and linux pipelines run at once in a real run; see --status)")
            return 0
        status = self.schedule(steps)
        ok = self.report(status)
        self.say("== done" if ok else "== stopped")
        return 0 if ok else 1

    # ------------------------------------------------------------ --status
    def step_state(self, step):
        rec = self.recorded(step)
        if rec:
            return "done", f"{rec['at']}: {rec.get('result', '')}"
        fail = (self.state.get("failures") or {}).get(step)
        if fail and fail.get("commit") == self.state.get("commit"):
            return ("held" if fail.get("held") else "failed"), f"{fail['at']}: {fail['error']}"
        return "owed", ""

    def rerun_action(self, step, st, states):
        if st == "done":
            if step in ("finish-line", "record"):
                rec = self.recorded(step) or {}
                if sorted(rec.get("platforms", [])) != sorted(self.published_platforms()):
                    return "run it again for the newly published platform(s)"
            return "skip"
        missing = [d for d in NEEDS[step] if states.get(d) != "done"]
        if st == "held":
            return "run it again (with --publish)"
        if st == "failed":
            return "run it again" + (f" after {', '.join(missing)}" if missing else "")
        return "run it" + (f" after {', '.join(missing)}" if missing else "")

    def leg_status(self, name):
        vendor, arch = name.split("-", 1)
        rb, out = leg_layout(self.rel / "legs", self.args.build_backend, name)
        leg = Leg(name, vendor, arch, [], {}, rb, self.rel / "legs", out)
        if leg.running():
            return "running", f"pid {leg.pid()}, log {leg.log}", "wait for it"
        if leg.done(self.commit):
            ok, why = overlay_verified(read_json(leg.provenance_file), leg.out_dir)
            if ok:
                return "done", f"exit 0, proof of {self.commit[:12]}", "skip"
            return "failed", why, "relaunch it"
        doc = self.admitted_leg(name)
        if doc:
            return ("taken", f"built from {doc['built_from'][:12]} ({doc['provenance']}), admitted for "
                    f"{doc['admitted_for'][:12]}: {doc['because']}", "skip")
        if leg.exit_code() is not None:
            return "failed", f"exit {leg.exit_code()}, log {leg.log}", "relaunch it (the failed attempt is moved aside)"
        doc, why = self.find_reusable_leg(name, vendor, arch, write=False)
        if doc:
            return "owed", "", f"take the completed leg built from {doc['built_from'][:12]} ({doc['provenance']})"
        return "owed", f"no reusable leg: {why}", "launch it"

    def column_status(self, name, vendor, ok, out):
        work = self.rel / "columns"
        leg = ColumnLeg(name, vendor, [], work, out, ok)
        final = self.linux_final()
        if ok():
            return "done", f"PASSED for wheel {sha256(final)[:12]}, no DIVERGENT cell against {self.ref_names()}" \
                + (" (taken: " + (read_json(out / "reused.json") or {}).get("reused_from", "") + ")"
                   if (out / "reused.json").is_file() else ""), "skip"
        if leg.running():
            return "running", f"pid {leg.pid()}, log {leg.log}", "wait for it"
        if leg.exit_code() is not None:
            return "failed", f"exit {leg.exit_code()}, log {leg.log}", "relaunch it (the failed attempt is moved aside)"
        sel = selection_digest(self.rel / f"selection-{vendor}.json")
        if final and sel:
            e = self.ledger().get(release_ledger.column_key(vendor, sha256(final), sel))
            if e and e.get("verdict") == "PASS":
                return "owed", f"ledger PASS for this wheel at {(e.get('evidence') or {}).get('dir')}", \
                    "take it if its evidence verifies here, else launch it"
        return "owed", "" if final else "no final Linux wheel yet", "launch it" if final else "launch it after linux-pack"

    def status(self):
        """--status: per pipeline, per leg and column, done / failed (with its
        log) / owed, and what a rerun would do. Runs nothing and writes nothing."""
        say = self.say
        c = self.state.get("commit")
        head = git_at(ROOT, "rev-parse", "HEAD") or "?"
        say(f"== mojolearn release {self.version}: status (read-only)")
        say(f"   state {self.state_path}" + ("" if self.state_path.is_file() else " (no state yet)"))
        if c:
            say(f"   source commit {c} (pinned at freeze; only --refreeze moves it)")
            say(f"   tooling: {ROOT} at {head[:12]}" + (" (the source commit)" if head == c else ""))
            if head != c:
                say(f"   source checkout: {self.source_checkout(create=False)}")
        else:
            say(f"   not frozen yet: a run freezes HEAD {head[:12]}")
        try:
            t = self.tooling()
            if t["dirty"]:
                say("   build tooling has uncommitted changes (no leg is reused across freezes): " + ", ".join(t["dirty"]))
        except Exception:
            pass
        states = {s: self.step_state(s)[0] for s in self.STEPS}
        actions = []

        def row(step, st, detail, action, indent="   "):
            say(f"{indent}{step:<18} {st:<8} {detail}")
            if action not in ("skip",):
                say(f"{indent}{'':<18} {'':<8} rerun: {action}")
                actions.append(f"{step}: {action}")

        for group, title in (("common", "common"), ("macos", "macos pipeline"), ("linux", "linux pipeline"),
                             ("finish", "finish")):
            say(f"-- {title}")
            for step in self.STEPS:
                if PIPELINE_OF[step] != group:
                    continue
                st, detail = self.step_state(step)
                row(step, st, detail, self.rerun_action(step, st, states))
                if step == "linux-builds" and c:
                    plan = read_json(self.plan_path)
                    if not plan or plan.get("commit") != c:
                        say("      (the legs are decided by the reuse plan, not made for this commit yet)")
                    elif not plan.get("legs"):
                        say("      no build leg: every Linux binding is taken from the published wheel")
                    else:
                        for name in plan["legs"]:
                            row(name, *self.leg_status(name), indent="      ")
                if step == "gpu-columns" and c:
                    for name, vendor, ok, out in self.column_specs():
                        row(name, *self.column_status(name, vendor, ok, out), indent="      ")
        summary = []
        for name, p in PIPELINES.items():
            steps = [s for s in self.STEPS if PIPELINE_OF[s] == name or any(
                s in NEEDS[t] and PIPELINE_OF[t] == name for t in self.STEPS)]
            bad = next((s for s in steps if states[s] in ("failed", "held")), None)
            if states[p["publish"]] == "done":
                summary.append(f"{name}: published")
            elif bad:
                summary.append(f"{name}: {states[bad]} at {bad}")
            else:
                summary.append(f"{name}: owed")
        say("== " + ", ".join(summary))
        say(f"   ledger: {self.ledger().where}")
        if actions:
            say("   a rerun would: " + "; ".join(actions[:12]) + (" ..." if len(actions) > 12 else ""))
        else:
            say("   a rerun would do nothing: everything is done")
        return 0


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
    ap.add_argument("--status", action="store_true",
                    help="per pipeline, leg and column: done / failed (with its log) / owed, and what a rerun "
                         "would do; runs nothing, writes nothing")
    ap.add_argument("--publish", choices=PUBLISH_CHOICES,
                    help="publish the two wheels (none = the workflow's checks without uploading)")
    ap.add_argument("--only", default="", help="comma-separated step names to run (others are skipped)")
    ap.add_argument("--redo", default="", help="comma-separated steps whose record is discarded first")
    ap.add_argument("--refreeze", action="store_true",
                    help="move the frozen source commit to HEAD (the only way it moves); completed legs and "
                         "columns are kept where what they built or tested is unchanged")
    ap.add_argument("--source-checkout", default="",
                    help="an existing checkout at the frozen source commit (default: this checkout when its HEAD "
                         "is the source commit, else a worktree under the release directory)")
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
    if args.status and (args.dry_run or args.only or args.redo or args.refreeze):
        ap.error("--status runs nothing; it takes no --dry-run, --only, --redo or --refreeze")
    return Release(args).go()


if __name__ == "__main__":
    raise SystemExit(main())
