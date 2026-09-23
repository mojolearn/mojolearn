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
  linux-builds       the three Linux sets, launched in parallel and detached
                     (launch_linux_builds: ONE function, so the CPU build box
                     route of lane/release-cpu-build-box plugs in as a backend)
  macos-build        mac_slot --slots 4 run -- build_release_wheel.sh, byte LM on
  macos-smoke        qualify_verifier_wheel.py --scope expanded under the Metal lock
  release-check      pixi run -e test release-check (CPU + Apple GPU, the guarantee)
  linux-wait         every leg finished, its proof complete for this commit, and
                     every host binding byte-identical across legs (named if not)
  linux-pack         pixi run -e pkg pack-linux-wheel, audit.sh, strip
  linux-smoke        tools/release_wheel_smoke.sh --rent (one RTX 4090 pod): the
                     expanded smoke AND the NVIDIA release column (the lanes the
                     release changed, from the installed wheel), diffed against
                     the CPU column of this commit
  amd-column         tools/release_wheel_smoke.sh --vendor hip --rent (one MI300X
                     pod, gfx942): the AMD release column, diffed the same way
  publish-linux      tools/release_linux_publish.sh ... --light-smoke   } only with
  publish-macos      tools/release_linux_publish.sh ... --light-smoke   } --publish
  finish-line        pip install mojolearn==<version> in a fresh venv on this Mac
  record             bench/results/release_verification/<date>_pypi_<v>/, committed

Nothing is published without `--publish none|testpypi|pypi`; without it the run
stops after amd-column and says so. The Linux wheel publishes only when the
NVIDIA and AMD columns show no DIVERGENT cell against the CPU column (policy
2026-09-22: bitwise identity across GPUs is the point; only changed lanes run).
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


def gpu_legs(ctx):
    """THE CURRENT ROUTE: two RunPod NVIDIA legs and one DigitalOcean MI325X leg,
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
    for arch, gpu in (("sm_90a", "NVIDIA H100 80GB HBM3"), ("sm_89", "NVIDIA L40S")):
        out = legs_dir / f"cuda-{arch}"
        legs.append(Leg(f"cuda-{arch}", "cuda", arch,
                        ["sh", "tools/gemm_remote_leg.sh", "nvidia", "--payload", "mamba",
                         "--source-ref", ctx.commit, "--gpu", gpu, "--allow-concurrent", "--rent",
                         "--minutes", "60"],
                        dict(runpod, MOJOLEARN_GPU_ARCHS=arch, MOJOLEARN_GEMM_LEG_OUT=str(out)),
                        out / "remote" / "release-build", legs_dir, out))
    token = os.path.expanduser(os.environ.get("MOJOLEARN_DO_TOKEN_FILE", "~/.mojolearn_do_token"))
    amd = ["bash", "tools/do_release061_leg.sh", ctx.commit, token, "--rent"]
    if ctx.args.amd_expect_from:
        amd += ["--expect-from", ctx.args.amd_expect_from]
    legs.append(Leg("hip-gfx942", "hip", "gfx942", amd,
                    dict(MOJOLEARN_RELEASE_UBUNTU22="1", MOJOLEARN_RELEASE_RESULTS_ROOT=str(legs_dir)),
                    legs_dir / "hip-gfx942" / "release-build", legs_dir, legs_dir / "hip-gfx942"))
    return legs


#: Build routes by name. A route returns the Leg list for ctx; launch, wait,
#: pack and resume are route-independent. The GPU legs are the default and
#: stay the default (docs/RELEASE_CHECKLIST.md section 2c, policy 2026-09-22):
#: the release builds on the GPUs it ships for. A CPU build box route may be
#: added here as an opt-in diagnostic, never as the default.
BUILD_BACKENDS = {"gpu-legs": gpu_legs}


def linux_legs(ctx):
    return BUILD_BACKENDS[ctx.args.build_backend](ctx)


def launch_linux_builds(ctx, legs, stagger=90):
    """THE ONE LAUNCH POINT for the Linux builds. Each leg runs detached (its own
    session, so this script can exit or be interrupted without killing a paid
    rental) and writes its exit code last. A leg that is running is left alone;
    a leg that finished but failed is moved aside (never deleted) and relaunched."""
    started = 0
    for leg in legs:
        if leg.running():
            ctx.say(f"  {leg.name}: already running (pid {leg.pid()}), log {leg.log}")
            continue
        if leg.exit_code() == 0 and leg.proof_ok(ctx.commit):
            ctx.say(f"  {leg.name}: already built")
            continue
        leg.workdir.mkdir(parents=True, exist_ok=True)
        if leg.exit_file.exists() or leg.log.exists():
            stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            for p in (leg.log, leg.exit_file, leg.pid_file):
                if p.exists():
                    p.rename(p.with_name(p.name + f".failed-{stamp}"))
            out = leg.out_dir
            if out.exists():
                out.rename(out.with_name(out.name + f".failed-{stamp}"))
                ctx.say(f"  {leg.name}: previous attempt moved aside to {out.name}.failed-{stamp}")
        if started and stagger:
            ctx.say(f"  waiting {stagger} s before the next launch (the checklist's stagger)")
            ctx.sleep(stagger)
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
        dirty = [line[3:] for line in git("status", "--porcelain", "--untracked-files=no").splitlines()]
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

    def step_linux_builds(self):
        legs = linux_legs(self)
        for leg in legs:
            self.say(f"  {leg.name}: {leg.describe()}")
        if self.dry:
            return f"would launch {len(legs)} legs ({self.args.build_backend})"
        launch_linux_builds(self, legs)
        return f"{len(legs)} legs launched or running ({self.args.build_backend})"

    def macos_wheel(self):
        found = sorted((self.rel / "macos").glob("mojolearn-*-macosx_*.whl"))
        return found[-1] if found else None

    def step_macos_build(self):
        w = self.macos_wheel()
        if w and wheel_commit(w) == self.commit:
            return "have " + w.name
        env = dict(MOJOLEARN_PACKAGE_BYTE_LM="1", MOJOLEARN_BUILD_JOBS="4", MOJOLEARN_COMPILE_JOBS="1")
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

    def release_check_complete(self):
        for backend in ("cpu", "metal"):
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
        self.must(["pixi", "run", "-e", "test", "release-check"], log=self.rel / "release-check.log",
                  what="release-check")
        if not self.dry and not self.release_check_complete():
            raise StepFailed("release-check exited 0 but its records are not complete")
        return "complete"

    def step_linux_wait(self):
        legs = linux_legs(self)
        if self.dry:
            return "would wait for " + ", ".join(l.name for l in legs)
        if not any(l.pid() or l.exit_code() is not None for l in legs):
            launch_linux_builds(self, legs)
        while any(l.running() for l in legs):
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

    def step_linux_pack(self):
        final = self.linux_final()
        if final and wheel_commit(final) == self.commit:
            return "have " + final.name
        legs = linux_legs(self)
        dist = self.rel / "linux"
        if dist.exists() and not self.dry:
            dist.rename(dist.with_name("linux.failed-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")))
        args = ["pixi", "run", "-e", "pkg", "pack-linux-wheel", "--profile", "release-linux3"]
        for leg in legs:
            args += ["--set", leg.release_build / "build" / "sets" / leg.vendor]
        for leg in legs:
            args += ["--build-proof", leg.release_build / "build" / "build-provenance.json"]
        args += ["--out", dist]
        self.must(args, log=self.rel / "linux-pack.log", what="pack_wheel.py")
        packed = sorted(dist.glob("mojolearn-*-linux_x86_64.whl")) if not self.dry else [dist / "<packed>.whl"]
        if len(packed) != 1:
            raise StepFailed(f"expected one packed wheel in {dist}, found {packed}")
        manifests = [leg.release_build / "build" / "sets" / leg.vendor / leg.arch / "manifest.json" for leg in legs]
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

    def gpu_column_ok(self, out, vendor):
        d = out / f"diff-cpu-{vendor}.txt"
        return (out / f"column-{vendor}.json").is_file() and d.is_file() and "DIVERGENT" not in d.read_text()

    def step_linux_smoke(self):
        final = self.linux_final()
        out = self.rel / "smoke-linux"
        if final and smoke_passed(out / "results.json", final) and self.gpu_column_ok(out, "cuda"):
            return "PASSED (receipt and NVIDIA column exist)"
        if not final and not self.dry:
            raise StepFailed("no final Linux wheel; run linux-pack")
        if out.exists() and not self.dry:
            out.rename(out.with_name(out.name + ".failed-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")))
        cmd = ["bash", "tools/release_wheel_smoke.sh", final or "<final linux wheel>",
               "--expected-source-commit", self.commit, "--out", out, "--rent",
               "--column", self.gpu_selection("cuda"),
               "--cpu-column", self.release_check_dir() / "cpu" / "column.json"]
        if self.args.smoke_gpu:
            cmd += ["--gpu", self.args.smoke_gpu]
        self.must(cmd, log=self.rel / "linux-smoke.log", what="release_wheel_smoke.sh")
        if not self.dry and not smoke_passed(out / "results.json", final):
            raise StepFailed(f"Linux smoke receipt is not PASSED for this wheel: {out / 'results.json'}")
        if not self.dry and not self.gpu_column_ok(out, "cuda"):
            raise StepFailed(f"the NVIDIA column is missing or DIVERGENT: {out}")
        return "PASSED, NVIDIA column identical to CPU"

    def step_amd_column(self):
        final = self.linux_final()
        out = self.rel / "column-amd"
        if final and self.gpu_column_ok(out, "hip"):
            return "identical to CPU (record exists)"
        if not final and not self.dry:
            raise StepFailed("no final Linux wheel; run linux-pack")
        if out.exists() and not self.dry:
            out.rename(out.with_name(out.name + ".failed-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")))
        self.must(["bash", "tools/release_wheel_smoke.sh", final or "<final linux wheel>",
                   "--expected-source-commit", self.commit, "--out", out, "--rent", "--vendor", "hip",
                   "--column", self.gpu_selection("hip"),
                   "--cpu-column", self.release_check_dir() / "cpu" / "column.json"],
                  log=self.rel / "amd-column.log", what="release_wheel_smoke.sh --vendor hip")
        if not self.dry and not self.gpu_column_ok(out, "hip"):
            raise StepFailed(f"the AMD column is missing or DIVERGENT: {out}")
        return "AMD column identical to CPU"

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
        self.must(["git", "-C", ROOT, "add", "--", rec], what="git add record")
        self.must(["git", "-C", ROOT, "commit", "-q", "-m",
                   f"Record the published {self.version} wheels and their release verification", "-m",
                   "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"], what="git commit record")
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        self.must(["git", "-C", ROOT, "push", "origin", f"HEAD:refs/heads/{branch}"], what="git push record")
        return str(rec.relative_to(ROOT)) + " committed on " + branch

    def readme(self):
        check = self.release_check_dir()
        rows, sums = [], []
        for backend, label in (("cpu", "CPU"), ("metal", "Apple Metal")):
            col = check / backend / "column.json"
            try:
                cells = json.loads(col.read_text()).get("cells", {})
            except (OSError, ValueError):
                cells = {}
            lanes = {k.split("/")[0] for k in cells}
            rows.append(f"| {label} | {len(lanes)} | {len(cells)} | {'yes' if cells else 'no'} |")
            if col.is_file():
                sums.append(f"- {backend}/column.json sha256 {sha256(col)}")
        diff = subprocess.run([PY, str(ROOT / "tools" / "identity_break.py"), "--diff",
                               str(check / "cpu" / "column.json"), str(check / "metal" / "column.json")],
                              capture_output=True, text=True, cwd=ROOT).stdout
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
            "## Release verification (CPU and Apple GPU, `pixi run -e test release-check`)", "",
            "| column | lanes | cells | complete |", "|---|---|---|---|", *rows, "",
            "CPU against Metal (`tools/identity_break.py --diff`):", "",
            *[f"    {s}" for s in summary], "", *sums, "",
            "## Wheels (light route)", "",
            "| platform | sha256 | smoke | published |", "|---|---|---|---|", *wheels, "",
            f"`pip install mojolearn=={self.version}` on this Mac: "
            f"{(self.recorded('finish-line') or {}).get('result', 'not run')}.", ""])

    STEPS = ["freeze-version", "freeze-changelog", "freeze-docs-facts", "freeze-commit", "rehearsal",
             "linux-builds", "macos-build", "macos-smoke", "release-check", "linux-wait", "linux-pack",
             "linux-smoke", "amd-column", "publish-linux", "publish-macos", "finish-line", "record"]
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
    ap.add_argument("--amd-expect-from", default="",
                    help="an NVIDIA release-build dir: run the AMD core-host probe against its STAGED copy")
    ap.add_argument("--smoke-gpu", default="", help="RunPod GPU for the Linux smoke (default RTX 4090)")
    ap.add_argument("--state-dir", default="", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if not VERSION_RE.match(args.version):
        ap.error("version must look like 0.8.15 (or 0.9.0a1)")
    return Release(args).go()


if __name__ == "__main__":
    raise SystemExit(main())
