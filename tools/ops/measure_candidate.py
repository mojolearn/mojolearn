#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/ops/measure_candidate.py -- put a CANDIDATE build's wheels on a measurement box and print the one submit
line of tools/ops/paper_compare_job.sh (lane measure-prep, 2026-11-09; docs/plans/PRE_0867_MEASUREMENT.md).

    python3 tools/ops/measure_candidate.py stage nvidia|amd --version V --source-commit SHA [--wheel-dir DIR]
        [--lane LANE] [--tree DIR] [--b-arm NAME=DIR@SHA ...] [--tag TAG] [--dry-run]

The wheels are the ones tools/release.py packs (its linux-pack step, Release.split_final): for a candidate build
(`tools/release.py candidate`, a local version X.Y.Z.devN+<sha12>[.label]) ~/mojolearn-evidence/release-candidates/
<version>/<commit12>/linux/final/, for a frozen release ~/mojolearn-evidence/release/<version>/<commit12>/linux/final/:
the core and the vendor wheel. They are validated here first with tools/candidate_wheel_install.py build_manifest (wheel RECORD, metadata,
tags, the frozen payload's source commit), then copied through the vendor wrapper's own channel and sha256-checked
on the box by tools/ops/release_gate.py Runner.stage, the same code path the release gate uses for the same wheels
(no second downloader; nothing is fetched from GitHub or R2 here). A B set (--b-arm, an A/B arm of
bench/results/paper_compare/owed-pre0867.json built from another commit the same way) goes beside them and into
the PC_B_ARMS file.

NVIDIA: `nvidia_central.sh sync <lane> <tree>` first puts the lane tree /root/mojolearn-<lane> at --tree (the
worktree of the branch whose tools/ the job runs). AMD: the droplet's /root/mojolearn must already be at that
branch's pushed head (`tools/do_amd_steward.sh update <sha>`); this tool checks it and refuses otherwise.
Nothing here rents, releases or extends a box. It prints the submit line; the orchestrator runs it.
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "ops"))
import candidate_wheel_install as C  # noqa: E402
import release_gate as RG  # noqa: E402

RELEASE_ROOT = Path.home() / "mojolearn-evidence" / "release"
#: tools/release.py candidate's work root: a local version (X.Y.Z.devN+<sha12>[.label]) is a candidate build there
CANDIDATE_ROOT = Path.home() / "mojolearn-evidence" / "release-candidates"


def default_wheel_dir(version, commit):
    root = CANDIDATE_ROOT if "+" in version else RELEASE_ROOT
    return root / version / commit[:12] / "linux" / "final"


def dir_version(wheel_dir, fallback):
    """The version of a wheel dir tools/release.py candidate wrote (its candidate.json), else FALLBACK."""
    try:
        import json
        return json.loads((Path(wheel_dir) / "candidate.json").read_text())["version"]
    except (OSError, ValueError, KeyError):
        return fallback
IN_BASE = "/root/pc-in"
AMD_TREE = "/root/mojolearn"


def pick_wheels(wheel_dir, vendor, version):
    """[core wheel, vendor wheel] of VERSION in WHEEL_DIR (exactly one each; the meta wheel is left out)."""
    d = Path(wheel_dir)
    core = sorted(d.glob(f"mojolearn_core-{version}-*.whl"))
    plug = sorted(d.glob(f"mojolearn_{vendor}-{version}-*.whl"))
    if len(core) != 1 or len(plug) != 1:
        raise SystemExit(f"measure-candidate: {d} must hold one mojolearn_core-{version} and one "
                         f"mojolearn_{vendor}-{version} wheel (found {len(core)} and {len(plug)})")
    return [core[0], plug[0]]


class StagePlan:
    """The duck-typed plan tools/ops/release_gate.py Runner.stage reads: sync, prepare, uploads, upload, verify."""

    def __init__(self, vendor, lane, tree, indir, uploads):
        self.vendor, self.lane, self.tree, self.indir = vendor, lane, tree, indir
        self.wrapper = RG.VENDORS[vendor]["wrapper"]
        self._uploads = uploads
        self.dirs = sorted({os.path.dirname(r) for _, r, _ in uploads})

    def box_sh(self, command):
        if self.vendor == "nvidia":
            return ["bash", self.wrapper, "sh", self.lane, command]
        return ["bash", self.wrapper, "ssh", command]

    def sync(self):
        return ["bash", self.wrapper, "sync", self.lane, str(self.tree)] if self.vendor == "nvidia" else None

    def prepare(self):
        return self.box_sh("set -e; mkdir -p " + " ".join(shlex.quote(d) for d in self.dirs) + "; echo READY")

    def uploads(self):
        return self._uploads

    def upload(self, remote):
        return self.box_sh("cat > " + shlex.quote(remote))

    def verify(self, remotes, executables):
        return self.box_sh("sha256sum " + " ".join(shlex.quote(r) for r in remotes))


def parse_b_arm(text):
    m = re.fullmatch(r"([a-z0-9][a-z0-9.-]*)=(.+)@([0-9a-f]{40})", text)
    if not m:
        raise SystemExit(f"measure-candidate: --b-arm wants NAME=DIR@SHA (40 hex), not {text!r}")
    return m.group(1), Path(m.group(2)).expanduser(), m.group(3)


def submit_line(vendor, lane, tag, env):
    q = " ".join
    res = f"box-out/{tag}"
    if vendor == "nvidia":
        envs = q(f"--env {k}={v}" for k, v in env.items())
        return (f"bash tools/nvidia_central.sh submit {lane} --cap 240 --note {tag} --results {res} {envs} "
                f"tools/ops/paper_compare_job.sh")
    envs = q(f"{k}={v}" for k, v in env.items())
    return f"bash tools/do_amd_steward.sh run {tag} {res} '{envs} bash tools/ops/paper_compare_job.sh'"


def cmd_stage(a):
    commit, version, vendor = a.source_commit, a.version, a.vendor
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit("measure-candidate: --source-commit must be the full frozen sha")
    wdir = Path(a.wheel_dir) if a.wheel_dir else default_wheel_dir(version, commit)
    wheels = pick_wheels(wdir, vendor, version)
    manifest = C.build_manifest(version, vendor, wheels, commit)     # refuses a wheel of another freeze
    print(f"candidate {version} {commit[:12]} {vendor}: " + ", ".join(
        f"{f['file']} {f['sha256'][:16]}" for f in manifest["files"]))
    indir = f"{IN_BASE}/{commit[:12]}"
    uploads = [(w, f"{indir}/{vendor}/{w.name}", False) for w in wheels]
    b_lines = []
    for spec in a.b_arm:
        name, bdir, bsha = parse_b_arm(spec)
        bver = dir_version(bdir, a.b_version or version)     # a B candidate has its own local version
        bw = pick_wheels(bdir, vendor, bver)
        C.build_manifest(bver, vendor, bw, bsha)
        uploads += [(w, f"{indir}/b-{name}/{vendor}/{w.name}", False) for w in bw]
        b_lines.append(f"{name} {indir}/b-{name}/{vendor} {bsha}\n")
    env = {"PC_WHEEL_DIR": f"{indir}/{vendor}", "PC_SOURCE_COMMIT": commit}
    if b_lines:
        local = Path(a.state_dir).expanduser() / f"b-arms-{commit[:12]}-{vendor}.txt"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text("".join(b_lines))
        uploads.append((local, f"{indir}/b-arms-{vendor}.txt", False))
        env["PC_B_ARMS"] = f"{indir}/b-arms-{vendor}.txt"
    tree = Path(a.tree).resolve() if a.tree else ROOT
    plan = StagePlan(vendor, a.lane, tree, indir, uploads)
    tag = a.tag or f"pc-{commit[:12]}-{vendor}"
    line = submit_line(vendor, a.lane, tag, env)
    if a.dry_run:
        q = lambda argv: " ".join(shlex.quote(str(x)) for x in argv)  # noqa: E731
        if plan.sync():
            print(q(plan.sync()))
        print(q(plan.prepare()))
        for local, remote, _ in uploads:
            print(q(plan.upload(remote)) + " < " + shlex.quote(str(local)))
        print(q(plan.verify([r for _, r, _ in uploads], [])) + "  # == local sha256")
        print("SUBMIT " + line)
        return 0
    if vendor == "amd":
        head = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        got = subprocess.run(["bash", plan.wrapper, "ssh", f"git -C {AMD_TREE} rev-parse HEAD"], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=120)
        if got.stdout.strip().splitlines()[-1:] != [head]:
            raise SystemExit(f"measure-candidate: the droplet's {AMD_TREE} is not at {head[:12]}; run "
                             f"`bash tools/do_amd_steward.sh update {head}` first (the branch must be pushed)")
    RG.Runner(a).stage(plan)
    print("SUBMIT " + line)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="measure_candidate", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage", help="validate + copy the candidate wheels to the box; print the submit line")
    s.add_argument("vendor", choices=("nvidia", "amd"))
    s.add_argument("--version", required=True, help="the candidate's version (the freeze, e.g. 0.8.67)")
    s.add_argument("--source-commit", required=True, help="the frozen source commit (40 hex)")
    s.add_argument("--wheel-dir", default=None,
                   help="default ~/mojolearn-evidence/release/<version>/<commit12>/linux/final (release.py linux-pack), "
                        "or release-candidates/<version>/... for a local version (tools/release.py candidate)")
    s.add_argument("--lane", default="measure", help="NVIDIA lane name: the tree /root/mojolearn-<lane> (default measure)")
    s.add_argument("--tree", default=None, help="the worktree whose tools/ the job runs (default this checkout)")
    s.add_argument("--b-arm", action="append", default=[], help="NAME=DIR@SHA: a B wheel set of the owed A/Bs")
    s.add_argument("--b-version", default=None, help="the B sets' version (default: each dir's candidate.json, "
                   "else --version)")
    s.add_argument("--state-dir", default="~/mojolearn-evidence/measure-candidate")
    s.add_argument("--tag", default=None, help="the results tag (default pc-<commit12>-<vendor>)")
    s.add_argument("--dry-run", action="store_true", help="validate locally and print every command; copy nothing")
    a = p.parse_args(argv)
    return cmd_stage(a)


if __name__ == "__main__":
    sys.exit(main())
