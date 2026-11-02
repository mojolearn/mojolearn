#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""tools/ops/release_gate.py -- the NVIDIA/AMD release gate as ONE persisted box job,
through the supported job wrappers (plan item 4 of docs/plans/POST_0863_RELEASE_TOOLING.md).

    python3 tools/ops/release_gate.py nvidia|amd VERSION --source-commit SHA --source-tree DIR \\
        --out DIR --state FILE --candidate-wheel W [--candidate-wheel W ...] \\
        [--autograd-prior DIR] [--no-adapters] [--profile release-light-v1] [--dry-run]

tools/release.py run_gate calls it once per vendor and judges --out/gate-smoke.json exactly as before.
Nothing here rents, releases or extends a box: the orchestrator brings the box up first
(tools/nvidia_central.sh up / tools/do_amd_steward.sh up). In order:

1. NVIDIA only: `tools/nvidia_central.sh sync release-gate <source tree>` puts the lane tree
   /root/mojolearn-release-gate at the candidate commit (the commit must be on origin/main, so the
   sync's merge base is the commit itself and its patch is empty).
2. The inputs go to <box>/root/release-gate/<version>/<vendor>/<stem>/in/ through the wrapper's own
   channel (`nvidia_central.sh sh release-gate 'cat > ...'`, `do_amd_steward.sh ssh 'cat > ...'`),
   sha256 checked there: the candidate wheels, the committed job script tools/ops/release_gate_job.sh
   (verbatim; refused when it is not committed and clean), gate.env (the exact build and gate argv
   below) and, when given, a retained autograd artifact set to reuse (--autograd-prior).
3. The job runs persisted to R2 from the box: `nvidia_central.sh submit release-gate --cap 180
   --results <res> <in>/release_gate_job.sh` (queued; this tool polls the R2 manifest) or
   `do_amd_steward.sh run <tag> <res> bash <in>/release_gate_job.sh` (in the foreground).
   On the box the job checks the GPU target, makes the source tree's locked pixi env, BUILDS the
   eight autograd executables (tools/gate_autograd_artifacts.py build, reusing proven-unchanged
   prior binaries), then runs tools/gate_smoke_box.py ... all (release-light-v1: eleven installed
   correctness records, 24 autograd entries, three adapter phases, 30-minute ceiling), all from the
   candidate commit's tree. Reports and the built autograd set land in <res>.
4. `tools/box_persist.py fetch --tag <tag> --extract` brings the final results home; <res>/out is
   copied into --out (gate-smoke.json and the reports the old lq transport brought home).

Exit: 0 when the gate-smoke.json that came home says PASS for this vendor and version, the job's rc
(or 1) otherwise, 3 when no final result landed within --max-wait-hours (a rerun with the same
--state resumes the same job instead of submitting another).
"""
import argparse
import contextlib
import datetime as dt
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

JOB_SCRIPT = "tools/ops/release_gate_job.sh"
#: the one nvidia_central lane the gate's jobs run in (its tree: /root/mojolearn-release-gate)
LANE = "release-gate"
IN_BASE = "/root/release-gate"
#: gate_autograd_artifacts.py: outputs, logs and receipts live under a retained /root/results* directory
RESULTS_BASE = "/root/results-release-gate"
REPO_URL = os.environ.get("MOJOLEARN_DEVPOD_REPO_URL", "https://github.com/mojolearn/mojolearn.git")
#: per vendor: the wrapper, the GPU target the shipped wheels carry and the box holds (L40S/4090/L4 are
#: sm_89; MI300X/MI325X gfx942), and the supported autograd compiler worker policy
#: (tools/RELEASE_CANDIDATE_GATE.md: NVIDIA -j2; AMD -j8 after the reproduced one-worker compiler crash)
VENDORS = {
    "nvidia": dict(wrapper="tools/nvidia_central.sh", target="sm_89", compile_jobs=2),
    "amd": dict(wrapper="tools/do_amd_steward.sh", target="gfx942", compile_jobs=8),
}
#: nvidia_central.sh submit --cap (minutes; the wrapper's maximum is 240): pixi env + eight compiles
#: (bounded by BUILD_MINUTES) + the gate's own 30-minute ceiling
NV_CAP_MIN = 180
BUILD_MINUTES = 120
#: what the job copies from the gate's out root into <res>/out (reports only; the venv and caches stay
#: on the box). The same list the lq transport fetched, plus the whole built autograd set.
GATE_FETCH = ("gate-smoke.json", "correctness.log", "correctness-*", "autograd-smoke.json", "autograd-smoke", "runtime-*",
              "smoke.log", "measure-*.log", "measure-manifest-*.json",
              "setup/wheel-manifest.json", "setup/logs", "autograd-bin",
              "smoke-*/smoke/smoke.json", "smoke-*/smoke/smoke.txt", "measure-*/campaign.json",
              "measure-*/campaign-retention.json", "measure-*/board",
              # the installed-wheel adapter smoke (tools/adapter_gate.py): receipt, logs, per-phase results
              "adapter-smoke/adapter-smoke.json", "adapter-smoke/*.log", "adapter-smoke/*.exit",
              "adapter-smoke/*/*.json",
              # the gate runner overlay the job applied (GATE_OVERLAY): its box-side record and its manifest
              "route-overlay.txt", "route-overlay.json")
#: THE GATE RUNNER OVERLAY: the tooling checkout's copy of each of these files whose bytes differ from the
#: source commit's ships with the job (route-overlay.tgz, tools/release_tooling.py write_overlay) and is
#: applied over the source tree AFTER the autograd build (the build and its reuse proof read the clean tree)
#: and BEFORE the gate runs, so a runner fix never needs a refreeze. tools/release.py gate_ok accepts the
#: report only when the box's route-overlay.txt shows the overlaid bytes (release.overlay_verified).
GATE_OVERLAY = ("tools/gate_smoke_box.py", "tools/autograd_smoke.py", "tools/release_gate_profiles.py")
TERMINAL_NV = ("done", "failed", "cancelled", "timeout", "interrupted")


def say(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')} release-gate] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gate_argv(vendor, version, out_root, candidate_wheels=(), source_commit=None, autograd_manifest=None,
              adapters=True, profile="release-light-v1"):
    """The box argv (run from the source tree) of tools/gate_smoke_box.py VENDOR VERSION <out_root> all.
    ADAPTERS False (MOJOLEARN_RELEASE_ADAPTERS=0: no adapters ship) passes --no-adapters."""
    if vendor not in VENDORS or not re.fullmatch(r"[0-9A-Za-z.+-]+", version):
        raise SystemExit(f"release-gate: takes nvidia|amd and a version, not {vendor!r} {version!r}")
    from release_gate_profiles import PROFILES
    if profile not in PROFILES:
        raise ValueError("unknown gate profile")
    argv = ["tools/gate_smoke_box.py", vendor, version, out_root, "all", "--profile", profile]
    if not adapters:
        argv.append("--no-adapters")
    if candidate_wheels:
        if not source_commit or not autograd_manifest:
            raise SystemExit("release-gate: candidate wheels need the source commit and the autograd manifest")
        for wheel in candidate_wheels:
            argv += ["--candidate-wheel", wheel]
        argv += ["--source-commit", source_commit, "--autograd-manifest", autograd_manifest]
    elif source_commit or autograd_manifest:
        raise SystemExit("release-gate: candidate source/autograd inputs require candidate wheels")
    return argv


def build_argv(vendor, source_commit, tree, out, reuse_from=None):
    """The box argv (run under `pixi run -e default` in the source tree) of the autograd artifact build."""
    v = VENDORS[vendor]
    argv = ["tools/gate_autograd_artifacts.py", "build", "--vendor", vendor, "--source-commit", source_commit,
            "--target", v["target"], "--mojo", f"{tree}/.pixi/envs/default/bin/mojo",
            "--compile-jobs", str(v["compile_jobs"]), "--out", out]
    if reuse_from:
        argv += ["--reuse-from", reuse_from]
    return argv


#: The installed-runtime qualifier (tools/release_runtime_installed.py run: the public runtime registry against
#: the installed candidate wheels, runtime-<vendor> in the ledger) runs on the box after the smoke, bounded by
#: this many minutes. Its outcome is recorded under out/runtime-<vendor>/ and fetched; it never changes the
#: gate verdict (the smoke's), and the ledger admits runtime-<vendor> only from a passed run. 0 skips it.
RUNTIME_MINUTES = int(os.environ.get("MOJOLEARN_GATE_RUNTIME_MINUTES", "30"))


def bash_array(name, items):
    return f"{name}=(" + " ".join(shlex.quote(str(i)) for i in items) + ")"


class Plan:
    """Every path, file and wrapper command of one gate job; pure (reads no file, runs nothing)."""

    def __init__(self, vendor, version, source_commit, source_tree, out, wheels, tag, adapters=True,
                 profile="release-light-v1", prior=None, prior_files=(), staged=None, overlay=None):
        if vendor not in VENDORS:
            raise SystemExit(f"release-gate: vendor must be nvidia or amd, not {vendor!r}")
        if not re.fullmatch(r"[0-9a-f]{40}", source_commit or ""):
            raise SystemExit("release-gate: --source-commit must be a full sha")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
            raise SystemExit("release-gate: the tag must be [A-Za-z0-9_.-]")
        self.vendor, self.version, self.commit, self.tag = vendor, version, source_commit, tag
        self.v = VENDORS[vendor]
        self.wrapper = self.v["wrapper"]
        self.source_tree, self.out = str(source_tree), Path(out)
        stem = self.out.name
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", stem):
            raise SystemExit(f"release-gate: the out directory's name {stem!r} must be [A-Za-z0-9_.-]")
        self.indir = f"{IN_BASE}/{version}/{vendor}/{stem}/in"
        self.res = f"{RESULTS_BASE}/{version}/{vendor}/{stem}/{tag}/results"
        self.work = f"{RESULTS_BASE}/{version}/{vendor}/{stem}/{tag}"
        # NVIDIA: the lane tree the sync puts at the commit; AMD: a tree the job fetches at the commit
        self.tree = f"/root/mojolearn-{LANE}" if vendor == "nvidia" else f"{IN_BASE}/src/{source_commit[:12]}"
        self.fetch_url = "" if vendor == "nvidia" else REPO_URL
        self.adapters, self.profile = adapters, profile
        self.wheels = [Path(w) for w in wheels]
        self.prior = Path(prior) if prior else None
        self.prior_files = list(prior_files)
        #: the local directory that holds the generated gate.env
        self.staged = Path(staged) if staged else self.out.parent / f"inputs-{tag}"
        #: the gate runner overlay manifest (release_tooling.overlay_manifest); an input only when bytes differ
        self.overlay = overlay if overlay and overlay.get("files") else None
        #: the overlay tarball's sha256, set once Runner writes it (gate.env carries it)
        self.overlay_sha256 = None
        gate_out = f"{self.work}/out"
        manifest = f"{gate_out}/autograd-bin/build-manifest.json"
        box_wheels = [f"{self.indir}/{w.name}" for w in self.wheels]
        self.gate = gate_argv(vendor, version, gate_out, box_wheels, source_commit, manifest, adapters, profile)
        reuse = f"{self.indir}/autograd-prior" if self.prior else None
        self.build = build_argv(vendor, source_commit, self.tree, f"{gate_out}/autograd-bin", reuse)
        self.build_fresh = build_argv(vendor, source_commit, self.tree, f"{gate_out}/autograd-bin")

    # ---------------------------------------------------------------- inputs
    def env_text(self):
        """gate.env: what the job script sources (every value validated there again)."""
        lines = ["# tools/ops/release_gate.py: one release gate job (sourced by tools/ops/release_gate_job.sh)",
                 f"GATE_VENDOR={self.vendor}", f"GATE_VERSION={self.version}",
                 f"GATE_SOURCE_COMMIT={self.commit}", f"GATE_TARGET={self.v['target']}",
                 f"GATE_TAG={self.tag}", f"GATE_TREE={self.tree}", f"GATE_FETCH_URL={self.fetch_url}",
                 f"GATE_WORK={self.work}", f"GATE_BUILD_MINUTES={BUILD_MINUTES}",
                 f"GATE_ADAPTERS={1 if self.adapters else 0}", f"GATE_PROFILE={self.profile}",
                 f"GATE_REUSE={1 if self.prior else 0}", f"GATE_RUNTIME_MINUTES={RUNTIME_MINUTES}",
                 bash_array("GATE_WHEELS", [f"{self.indir}/{w.name}" for w in self.wheels]),
                 f"GATE_OVERLAY_SHA256={self.overlay_sha256 or ''}", bash_array("GATE_OVERLAY", GATE_OVERLAY),
                 bash_array("BUILD_ARGV", self.build), bash_array("BUILD_FRESH_ARGV", self.build_fresh),
                 bash_array("GATE_ARGV", self.gate), bash_array("GATE_FETCH", GATE_FETCH)]
        return "\n".join(lines) + "\n"

    def uploads(self):
        """[(local, remote, executable)]: the wheels, the job script, the reused autograd set, gate.env."""
        out = [(w, f"{self.indir}/{w.name}", False) for w in self.wheels]
        out.append((ROOT / JOB_SCRIPT, f"{self.indir}/release_gate_job.sh", True))
        if self.overlay:
            for name in ("route-overlay.tgz", "route-overlay.json"):
                out.append((self.staged / name, f"{self.indir}/{name}", False))
        for name in self.prior_files:
            out.append((self.prior / name, f"{self.indir}/autograd-prior/{name}", name != "build-manifest.json"))
        out.append((self.staged / "gate.env", f"{self.indir}/gate.env", False))
        names = [r for _, r, _ in out]
        if len(set(names)) != len(names):
            raise SystemExit(f"release-gate: two inputs share a box name: {names}")
        return out

    # ---------------------------------------------------------------- wrapper commands
    def box_sh(self, command):
        """argv: one housekeeping command on the box through the vendor's wrapper (stdin passes through)."""
        if self.vendor == "nvidia":
            return ["bash", self.wrapper, "sh", LANE, command]
        return ["bash", self.wrapper, "ssh", command]

    def sync(self):
        return ["bash", self.wrapper, "sync", LANE, self.source_tree] if self.vendor == "nvidia" else None

    def prepare(self):
        dirs = [self.indir] + ([f"{self.indir}/autograd-prior"] if self.prior else [])
        return self.box_sh("set -e; rm -rf " + shlex.quote(f"{self.indir}/autograd-prior") + "; mkdir -p "
                           + " ".join(shlex.quote(d) for d in dirs) + "; echo READY")

    def upload(self, remote):
        return self.box_sh("cat > " + shlex.quote(remote))

    def verify(self, remotes, executables):
        cmd = "sha256sum " + " ".join(shlex.quote(r) for r in remotes)
        if executables:
            cmd = "chmod +x " + " ".join(shlex.quote(r) for r in executables) + " && " + cmd
        return self.box_sh(cmd)

    def launch(self):
        """argv: the persisted job (NVIDIA: queued, prints the job id; AMD: runs in the foreground)."""
        script = f"{self.indir}/release_gate_job.sh"
        if self.vendor == "nvidia":
            return ["bash", self.wrapper, "submit", LANE, "--cap", str(NV_CAP_MIN), "--note", self.tag,
                    "--results", self.res, script]
        return ["bash", self.wrapper, "run", self.tag, self.res, "bash", script]

    def fetch(self, tag, dest, listing=False):
        return [sys.executable, "tools/box_persist.py", "fetch", "--tag", tag, "--dest", str(dest)] + (
            ["--list"] if listing else ["--extract"])

    def status(self, job):
        return ["bash", self.wrapper, "status", job]

    def describe(self):
        """The wrapper commands in order, as shell lines (the dry run prints these)."""
        q = lambda argv: " ".join(shlex.quote(str(a)) for a in argv)  # noqa: E731
        lines = []
        if self.sync():
            lines.append(q(self.sync()))
        lines.append(q(self.prepare()))
        ups = self.uploads()
        for local, remote, _ in ups:
            lines.append(q(self.upload(remote)) + " < " + shlex.quote(str(local)))
        lines.append(q(self.verify([r for _, r, _ in ups], [r for _, r, x in ups if x])) + "  # == local sha256")
        lines.append(q(self.launch()))
        lines.append("  on the box, in " + self.tree + " at " + self.commit[:12] + ": pixi run -e default python3 "
                     + q(self.build))
        lines.append("  on the box, then: python3 " + q(self.gate))
        tag = "<the R2 tag the wrapper prints>"
        lines.append(q(self.fetch(tag, self.staged.parent / "fetch-<tag>"))
                     + f"  # then <fetch>/final/out -> {self.out}")
        return lines


def job_script_committed():
    """Refuse a job script whose bytes are not the committed ones (the box runs exactly a committed file)."""
    rel = JOB_SCRIPT
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel], capture_output=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", rel], capture_output=True)
    if tracked.returncode or dirty.returncode:
        raise SystemExit(f"release-gate: {rel} is not committed and clean in {ROOT}; commit it first")


def gate_overlay(source_commit):
    """The gate runner overlay manifest: each GATE_OVERLAY file whose bytes at the tooling commit (this
    checkout's HEAD) differ from the source commit's (an empty `files` when none differ)."""
    import release_tooling
    tooling = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
    try:
        return release_tooling.overlay_manifest(ROOT, tooling, source_commit, GATE_OVERLAY)
    except release_tooling.OverlayRefused as exc:
        raise SystemExit(f"release-gate: {exc}")


def on_origin_main(commit):
    """The NVIDIA lane sync puts the box tree at the source tree's merge base with origin/main."""
    return subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "origin/main"],
                          capture_output=True).returncode == 0


def prior_files(prior, vendor):
    """The files of a retained autograd set to reuse, or [] when it is not a complete built set for VENDOR.
    The builder proves each entry unchanged before it reuses it; this only skips an obviously unusable set."""
    if not prior:
        return []
    try:
        doc = json.loads((Path(prior) / "build-manifest.json").read_text())
    except (OSError, ValueError):
        return []
    if doc.get("state") != "built" or doc.get("vendor") != vendor:
        return []
    files = [row.get("file") for row in doc.get("files", [])]
    if not files or any(not f or "/" in f or not (Path(prior) / f).is_file() for f in files):
        return []
    return ["build-manifest.json", *files]


def gate_verdict_home(out, vendor, version):
    """(ok, why) of the gate-smoke.json that came home."""
    try:
        doc = json.loads((Path(out) / "gate-smoke.json").read_text())
    except (OSError, ValueError):
        return False, "no gate-smoke.json came home"
    if doc.get("vendor") != vendor or doc.get("version") != version:
        return False, f"gate-smoke.json is for {doc.get('vendor')} {doc.get('version')}"
    if doc.get("verdict") != "PASS":
        return False, f"verdict {doc.get('verdict')}: " + "; ".join(doc.get("verdict_reasons") or [])
    return True, "PASS"


def persist_tag(text):
    """The R2 tag a wrapper minted (tools/box_persist.py prepare prints it on stderr)."""
    m = re.search(r"box_persist: job (\S+) persists to R2", text or "")
    return m.group(1) if m else None


def minted_tag(prefix):
    """The newest R2 tag a wrapper minted for PREFIX (tools/box_persist.py new-tag PREFIX-<utc>-<hex>),
    from the local persist index, or None."""
    home = os.environ.get("MOJOLEARN_PERSIST_HOME") or os.path.expanduser("~/mojolearn-evidence/box-persist")
    found = None
    try:
        with open(os.path.join(home, "index.tsv")) as fh:
            for line in fh:
                fields = line.rstrip("\n").split("\t")
                if len(fields) > 1 and re.fullmatch(re.escape(prefix) + r"-[0-9TZ]+-[0-9a-f]+", fields[1]):
                    found = fields[1]
    except OSError:
        return None
    return found


def manifest_state(text):
    m = re.search(r"^MANIFEST state=(\S+) exit=(\S*)", text or "", re.M)
    return (m.group(1), m.group(2)) if m else (None, None)


class Runner:
    def __init__(self, args, run=subprocess.run, sleep=time.sleep, clock=time.time, popen=subprocess.Popen):
        self.a, self.run, self.sleep, self.clock, self.popen = args, run, sleep, clock, popen

    def call(self, argv, stdin=None, timeout=900, capture=True):
        return self.run([str(c) for c in argv], cwd=str(ROOT), stdin=stdin, timeout=timeout,
                        capture_output=capture, text=stdin is None)

    def stream(self, argv, on_line, timeout):
        """Run ARGV with its output passed through line by line (ON_LINE sees each); its rc."""
        proc = self.popen([str(c) for c in argv], cwd=str(ROOT), stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = self.clock() + timeout
        for line in proc.stdout:
            print("    " + line.rstrip("\n"), flush=True)
            on_line(line)
            if self.clock() > deadline:
                proc.kill()
                break
        return proc.wait()

    def must(self, argv, what, stdin=None, timeout=900):
        got = self.call(argv, stdin=stdin, timeout=timeout)
        if got.returncode != 0:
            tail = ((got.stdout or "") + (got.stderr or ""))[-600:] if stdin is None else ""
            raise SystemExit(f"release-gate: {what} failed (rc {got.returncode}) {tail}")
        return got

    def stage(self, plan):
        if plan.sync():
            say("$ " + " ".join(plan.sync()))
            self.must(plan.sync(), "the lane sync", timeout=3600)
        got = self.must(plan.prepare(), f"preparing {plan.indir}")
        if "READY" not in (got.stdout or ""):
            raise SystemExit(f"release-gate: could not prepare {plan.indir}")
        want, executables = {}, []
        for local, remote, executable in plan.uploads():
            if not Path(local).is_file():
                raise SystemExit(f"release-gate: no file {local}")
            size = Path(local).stat().st_size
            say(f"copying {Path(local).name} ({size} bytes) to {remote}")
            with open(local, "rb") as fh:
                self.must(plan.upload(remote), f"the copy of {local}", stdin=fh, timeout=300 + size // 200000)
            want[remote] = sha256(local)
            if executable:
                executables.append(remote)
        got = self.must(plan.verify(list(want), executables), "the sha256 check on the box")
        have = {}
        for line in (got.stdout or "").splitlines():
            parts = line.split()
            if len(parts) == 2:
                have[parts[1]] = parts[0]
        bad = [r for r, s in want.items() if have.get(r) != s]
        if bad:
            raise SystemExit(f"release-gate: sha256 differs on the box for {bad}")
        say(f"{len(want)} input(s) on the box, sha256 verified")

    def wait_final(self, plan, tag, job=None):
        """(state, exit) of the job's R2 manifest once final, or None at the deadline."""
        deadline = self.clock() + self.a.max_wait_hours * 3600
        dest = Path(self.a.state).parent / f"fetch-{tag}" if self.a.state else self.a.out.parent / f"fetch-{tag}"
        polls, ended = 0, 0
        while self.clock() < deadline:
            got = self.call(plan.fetch(tag, dest, listing=True), timeout=600)
            state, rc = manifest_state(got.stdout)
            if state == "final":
                return state, rc
            if job:
                st = self.call(plan.status(job), timeout=300)
                m = re.search(r"^job \S+: (\S+)", st.stdout or "", re.M)
                if m and m.group(1) in TERMINAL_NV:
                    # an ended job uploads its final results on exit; three polls without them: lost
                    ended += 1
                    if ended >= 3:
                        say(f"job {job} is {m.group(1)} and no final upload of {tag} landed")
                        return "lost", None
            if polls % 10 == 0:
                say(f"waiting for {tag} (R2 manifest state={state})")
            polls += 1
            self.sleep(self.a.poll_seconds)
        return None

    def bring_home(self, plan, tag):
        dest = (Path(self.a.state).parent if self.a.state else self.a.out.parent) / f"fetch-{tag}"
        got = self.call(plan.fetch(tag, dest), timeout=3600)
        if got.returncode != 0:
            say(f"fetch of {tag} exited {got.returncode}: {(got.stdout or '')[-300:]}")
            return False
        return self.copy_home(dest / "final", tag)

    def copy_home(self, final, tag):
        """FINAL/out (and FINAL/job) of one fetched job into --out."""
        if not (final / "out").is_dir():
            say(f"no out/ in the final results of {tag}")
            return False
        self.a.out.mkdir(parents=True, exist_ok=True)
        shutil.copytree(final / "out", self.a.out, dirs_exist_ok=True, symlinks=True)
        if (final / "job").is_dir():
            shutil.copytree(final / "job", self.a.out / "job", dirs_exist_ok=True)
        say(f"{tag}: results home in {self.a.out}")
        return True

    def adopt_landed(self):
        """Reuse a landed result instead of relaunching (Andrew, 2026-10-31: reuse good old runs, automatically).
        Under the state dir (the --state parent, else --out's parent), every fetch-*/final.tgz whose
        manifest.json is final and whose bytes are the manifest's final upload is extracted tolerantly; a
        gate-smoke.json for THIS candidate (vendor, version, source commit, profile and the sha256 of every
        --candidate-wheel) with verdict PASS is brought home exactly as bring_home does. A landed FAIL for this
        candidate is reported, never adopted (the gate relaunches). The adopted tag, or None."""
        a = self.a
        base = Path(a.state).parent if a.state else a.out.parent
        want = sorted(sha256(w) for w in a.candidate_wheel if Path(w).is_file())
        if len(want) != len(a.candidate_wheel):
            return None
        import box_persist
        fetched = sorted(base.glob("fetch-*/final.tgz"), key=lambda p: p.stat().st_mtime, reverse=True)
        for tgz in fetched:
            d = tgz.parent
            try:
                m = json.loads((d / "manifest.json").read_text())
            except (OSError, ValueError):
                continue
            ups = [u for u in m.get("uploads", []) if u.get("name") == "final.tgz"]
            if m.get("state") != "final" or not ups or ups[-1].get("sha256") != sha256(tgz):
                continue
            tag = m.get("tag") or d.name[len("fetch-"):]
            try:
                with contextlib.redirect_stdout(io.StringIO()) as skipped:
                    box_persist.extract_tolerant(str(tgz), str(d / "final"))
            except (OSError, tarfile.TarError) as e:
                say(f"landed {tag}: cannot extract ({e}); not adopted")
                continue
            try:
                doc = json.loads((d / "final" / "out" / "gate-smoke.json").read_text())
            except (OSError, ValueError):
                continue
            cand = doc.get("candidate") or {}
            wheels = cand.get("wheels")
            if (doc.get("vendor") != a.vendor or doc.get("version") != a.version
                    or doc.get("source_commit") != a.source_commit or cand.get("source_commit") != a.source_commit
                    or doc.get("profile", a.profile) != a.profile
                    or not isinstance(wheels, dict) or sorted(wheels.values()) != want):
                continue
            ok, why = gate_verdict_home(d / "final" / "out", a.vendor, a.version)
            if not ok:
                say(f"landed {tag} is this candidate's gate but {why}; not adopted, the gate runs again")
                continue
            n = skipped.getvalue().count("SKIPPED ")
            if not self.copy_home(d / "final", tag):
                continue
            say(f"adopted {tag}: {why} for {a.vendor} {a.version} at {a.source_commit[:12]} "
                f"({len(want)} wheel sha256 equal; {n} link member(s) skipped); no job launched")
            return tag
        return None

    def make_plan(self):
        """(plan, key, state path, state): the launch's plan, its input key and a resumable earlier launch."""
        a = self.a
        prior = prior_files(a.autograd_prior, a.vendor)
        tag = f"rel-{a.version}-gate-{a.vendor}-{dt.datetime.fromtimestamp(self.clock(), dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        state_path = Path(a.state) if a.state else None
        st = {}
        if state_path and state_path.is_file():
            try:
                st = json.loads(state_path.read_text())
            except ValueError:
                st = {}
        overlay = gate_overlay(a.source_commit)
        key_inputs = [a.vendor, a.version, a.source_commit, a.profile, not a.no_adapters,
                      [sha256(w) if Path(w).is_file() else str(w) for w in a.candidate_wheel],
                      sha256(ROOT / JOB_SCRIPT) if (ROOT / JOB_SCRIPT).is_file() else None]
        if overlay["files"]:
            key_inputs.append(overlay["digest"])   # a runner fix is a new job, never a resumed old one
        key = hashlib.sha256(json.dumps(key_inputs).encode()).hexdigest()
        if st.get("key") == key and st.get("tag"):
            tag = st["plan_tag"]
        else:
            st = {}
        plan = Plan(a.vendor, a.version, a.source_commit, a.source_tree, a.out, a.candidate_wheel, tag,
                    adapters=not a.no_adapters, profile=a.profile,
                    prior=a.autograd_prior if prior else None, prior_files=prior,
                    staged=(state_path.parent if state_path else a.out.parent) / f"inputs-{tag}",
                    overlay=overlay)
        return plan, key, state_path, st

    def dry_lines(self):
        return self.make_plan()[0].describe()

    def main(self):
        a = self.a
        plan, key, state_path, st = self.make_plan()
        tag = plan.tag
        if a.dry_run:
            for line in plan.describe():
                print("  would: " + line)
            return 0
        if self.adopt_landed():
            if state_path and state_path.is_file():
                state_path.unlink()   # a landed PASS for this candidate settles any earlier launch
            ok, why = gate_verdict_home(a.out, a.vendor, a.version)
            return 0 if ok else 1
        job_script_committed()
        if a.vendor == "nvidia" and not on_origin_main(a.source_commit):
            raise SystemExit("release-gate: the candidate commit is not on origin/main; the NVIDIA lane sync "
                                 "puts the box tree at its merge base with origin/main (push the commit first)")
        rc = None
        if st and not st.get("tag"):
            # an AMD launch whose connection was lost before its tag reached us: the wrapper minted it
            # locally (tools/box_persist.py index), so the job's results are still found
            st["tag"] = minted_tag(st["plan_tag"])
        if st.get("tag"):
            say(f"resuming {st['tag']} (launched {st.get('launched_at')}); not launched again")
        else:
            plan.staged.mkdir(parents=True, exist_ok=True)
            if plan.overlay:
                import release_tooling
                try:
                    _, plan.overlay_sha256 = release_tooling.write_overlay(ROOT, plan.overlay, plan.staged)
                except release_tooling.OverlayRefused as exc:
                    raise SystemExit(f"release-gate: {exc}")
                say("gate runner overlay: " + ", ".join(sorted(plan.overlay["files"])))
            (plan.staged / "gate.env").write_text(plan.env_text())
            self.stage(plan)
            record = dict(key=key, plan_tag=tag, vendor=a.vendor, launched_at=dt.datetime.now(
                dt.timezone.utc).isoformat(timespec="seconds"), launch=plan.launch())

            def keep(**fields):
                record.update(fields)
                if state_path:
                    state_path.parent.mkdir(parents=True, exist_ok=True)
                    state_path.write_text(json.dumps(record, indent=2) + "\n")

            say("$ " + " ".join(shlex.quote(c) for c in plan.launch()))
            if a.vendor == "nvidia":
                got = self.call(plan.launch(), timeout=900)
                text = (got.stdout or "") + "\n" + (got.stderr or "")
                job = next((l.strip() for l in reversed((got.stdout or "").splitlines())
                            if re.fullmatch(r"nvc[0-9]-[0-9]{4,}", l.strip())), None)
                ptag = persist_tag(text)
                if got.returncode or not job or not ptag:
                    raise SystemExit(f"release-gate: submit refused (rc {got.returncode}): {text[-600:]}")
                keep(tag=ptag, job=job)
                say(f"queued {job}, persisted as {ptag}")
            else:
                # the foreground run: the state is written before it starts and again once the wrapper
                # names the job's R2 tag, so a rerun after a lost connection reads the results from R2
                keep()
                rc = self.stream(plan.launch(), lambda line: persist_tag(line) and keep(tag=persist_tag(line)),
                                 timeout=int(a.max_wait_hours * 3600))
                say(f"the AMD job exited {rc}")
                if not record.get("tag"):
                    record["tag"] = minted_tag(tag)
                if not record.get("tag"):
                    raise SystemExit(f"release-gate: the AMD run (rc {rc}) named no R2 tag; nothing to fetch")
            st = record
        final = self.wait_final(plan, st["tag"], st.get("job"))
        if final is None:
            say(f"no final result for {st['tag']} within {a.max_wait_hours} h; a rerun with this --state resumes it")
            return 3
        if final[0] == "lost":
            if state_path and state_path.is_file():
                state_path.unlink()   # the next run launches a fresh job
            return 1
        try:
            job_rc = int(final[1])
        except (TypeError, ValueError):
            job_rc = rc if rc is not None else 1
        home = self.bring_home(plan, st["tag"])
        if state_path and state_path.is_file():
            state_path.unlink()
        if not home:
            return job_rc or 1
        ok, why = gate_verdict_home(a.out, a.vendor, a.version)
        say(f"gate {a.vendor} {a.version}: {why}")
        return 0 if ok else (job_rc or 1)


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("vendor", choices=sorted(VENDORS))
    ap.add_argument("version")
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--source-tree", required=True, help="a checkout at the source commit (the NVIDIA lane sync)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--state", default="")
    ap.add_argument("--candidate-wheel", action="append", default=[], required=True)
    ap.add_argument("--autograd-prior", default=None,
                    help="a complete retained autograd set (build-manifest.json + binaries) the build may reuse")
    ap.add_argument("--profile", choices=("release-light-v1", "performance-campaign-v1"), default="release-light-v1")
    ap.add_argument("--no-adapters", action="store_true")
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--max-wait-hours", type=float, default=72.0)
    ap.add_argument("--dry-run", action="store_true", help="print the wrapper commands; run nothing")
    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.exit(Runner(parse(sys.argv[1:])).main())
