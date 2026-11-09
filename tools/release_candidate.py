#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""CANDIDATE WHEELS FOR A MEASUREMENT, NEVER A RELEASE (lane candidate-build, 2026-11-09;
docs/plans/PRE_0867_MEASUREMENT.md).

    python3 tools/release.py candidate --commit <sha40> [--vendors nvidia,amd] [--defines "-D A=1 -D B=2"]
        [--label NAME] [--base-version 0.8.67] [--ref <pushed tooling branch>] [--dry-run]

Builds the Linux wheels of one exact commit with the release's own steps and nothing else: tools/release.py's
reuse-plan, linux-builds (.github/workflows/release-linux-build.yml on GitHub's free runners for that commit,
through tools/release_github_build.py run, with the workflow's `defines` when --defines is given), linux-wait,
linux-assemble and linux-pack (packaging/linux/pack_wheel.py --local-version, audit.sh, the directory-entry strip,
split_audit). The Release runs with `--only` those five steps, so the scheduler counts every other step as skipped:
freeze-version, freeze-changelog, freeze-docs-facts, freeze-commit (the version bump commit and the push to the
release branch), identity-at-freeze, rehearsal, readiness, cross-compile, every gate, column, publish, finish-line
and record never run. The source commit is pinned in the candidate's own state.json, so nothing is frozen.

THE VERSION is local, `<base>.dev0+<commit12>[.<label>]` (PEP 440; PyPI refuses local versions, so these wheels can
never be uploaded). It exists only in the packed wheels under the candidate's work dir: the packer generates the
wheel's _version.py and still proves the committed one at the build commit. No commit, no push, no tag, no PyPI.

THE WORK DIR is <work root>/<version>/ (default ~/mojolearn-evidence/release-candidates): its state.json, its own
local ledger (Release with --state-dir never opens the shared R2 ledger), the legs, and a detached worktree of the
commit at source/<commit12> (the only tree it writes). It refuses a work dir or source tree inside the main
checkout. The wheels land in <work dir>/<commit12>/linux/final/, with candidate.json beside them, the path
tools/ops/measure_candidate.py stage reads for a local --version.

DEFINES (the A/B "B" wheel sets): --defines makes an ordinary candidate of the same commit whose compiles get
MOJOLEARN_BUILD_DEFINES. Only the bindings whose closure names a define compile again: the reuse plan builds the
rows that read one (release_reuse.apply_defines) and the binding cache keys and builds every other one exactly as
the A set (tools/bincache.py scope_build_defines), so those are cache hits on the A set's bytes. Run the A set
first. The label defaults to `d` + 8 hex of the defines, so each define set has its own version and work dir.

--vendors names the vendor wheels validated and staged (tools/candidate_wheel_install.py build_manifest). The
packer packs the core and both plugins from one set of legs, so the legs the plan needs always run.

At the end it asserts that origin/main and the tag lists (origin and local) are what they were at the start.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

CANDIDATE_ROOT = Path(os.environ.get("MOJOLEARN_EVIDENCE_ROOT", os.path.expanduser("~/mojolearn-evidence"))) \
    / "release-candidates"
#: the only Release steps a candidate runs (STEP_TABLE order); every other step is skipped by --only
STEPS = ("reuse-plan", "linux-builds", "linux-wait", "linux-assemble", "linux-pack")
VENDORS = ("nvidia", "amd")
LABEL_RE = re.compile(r"[a-z0-9]+(\.[a-z0-9]+)*")


def sh(*args, cwd=ROOT, check=True):
    p = subprocess.run(list(args), cwd=str(cwd), capture_output=True, text=True)
    if check and p.returncode != 0:
        raise SystemExit(f"candidate: {' '.join(args[:4])} failed: {p.stderr.strip()[:300]}")
    return p.stdout.strip()


def main_checkout():
    """The primary worktree of this repository (the main checkout every lane is told not to edit)."""
    common = Path(sh("git", "rev-parse", "--path-format=absolute", "--git-common-dir"))
    return common.parent.resolve()


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def remote_refs():
    """(origin/main sha, sorted origin tag lines, sorted local tags): what a candidate must never change."""
    main = sh("git", "ls-remote", "origin", "refs/heads/main").split()
    tags = sorted(sh("git", "ls-remote", "--tags", "origin").splitlines())
    local = sorted(sh("git", "tag", "-l").splitlines())
    return dict(origin_main=main[0] if main else None,
                origin_tags_sha256=hashlib.sha256("\n".join(tags).encode()).hexdigest(), origin_tags=len(tags),
                local_tags_sha256=hashlib.sha256("\n".join(local).encode()).hexdigest(), local_tags=len(local))


def next_version(committed):
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", committed or "")
    if not m:
        raise SystemExit(f"candidate: the commit's _version.py says {committed!r}; pass --base-version X.Y.Z")
    return f"{m[1]}.{m[2]}.{int(m[3]) + 1}"


def candidate_version(base, commit, label):
    return f"{base}.dev0+{commit[:12]}" + (f".{label}" if label else "")


def default_label(defines):
    return ("d" + hashlib.sha256(defines.encode()).hexdigest()[:8]) if defines else ""


def source_tree(work, commit, dry):
    """A detached worktree of COMMIT at <work>/source/<commit12> (made once, verified clean after)."""
    tree = work / "source" / commit[:12]
    if (tree / ".git").exists():
        head = sh("git", "-C", str(tree), "rev-parse", "HEAD", check=False)
        dirty = sh("git", "-C", str(tree), "status", "--porcelain", "--untracked-files=no", check=False)
        if head != commit or dirty:
            raise SystemExit(f"candidate: {tree} is not a clean checkout of {commit}; "
                             f"git worktree remove --force {tree} and rerun")
        return tree
    if dry:
        return tree
    tree.parent.mkdir(parents=True, exist_ok=True)
    sh("git", "worktree", "add", "--detach", str(tree), commit)
    return tree


def parse(argv):
    ap = argparse.ArgumentParser(prog="release.py candidate", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--commit", required=True, help="the exact source commit (40 hex), pushed to origin")
    ap.add_argument("--vendors", default="nvidia,amd", help="vendor wheels to validate and stage (default nvidia,amd)")
    ap.add_argument("--defines", default="", help="B set: '-D NAME=V -D NAME2' or NAME=V,NAME2 (MOJOLEARN_BUILD_DEFINES)")
    ap.add_argument("--label", default="", help="the version's label (lower case [a-z0-9.]; default: none for an "
                    "A set, d<8 hex of the defines> for a B set)")
    ap.add_argument("--base-version", default="", help="X.Y.Z of the .dev0 version (default: the commit's "
                    "_version.py plus one patch)")
    ap.add_argument("--ref", default="", help="the pushed tooling branch the workflow runs from (default: this "
                    "checkout's branch; it must hold this checkout's HEAD)")
    ap.add_argument("--work-root", default=str(CANDIDATE_ROOT), help="default ~/mojolearn-evidence/release-candidates")
    ap.add_argument("--dry-run", action="store_true", help="print the Release plan of the five steps; run nothing")
    return ap.parse_args(argv)


def main(argv=None, release=None):
    a = parse(sys.argv[1:] if argv is None else argv)
    if release is None:
        import release  # noqa: F811 - tools/release.py, when this file is run directly
    import release_github_build
    import release_reuse
    import candidate_wheel_install as C
    if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
        raise SystemExit("candidate: --commit must be the full 40-hex sha")
    if sh("git", "cat-file", "-t", a.commit, check=False) != "commit":
        raise SystemExit(f"candidate: {a.commit} is not a commit here; git fetch origin first")
    if not sh("git", "branch", "-r", "--contains", a.commit, check=False):
        raise SystemExit(f"candidate: {a.commit[:12]} is on no origin branch; the runners check out a pushed commit")
    vendors = [v for v in a.vendors.split(",") if v]
    if not vendors or any(v not in VENDORS for v in vendors):
        raise SystemExit("candidate: --vendors is a comma list of nvidia, amd")
    defines = release_github_build.normalize_defines(a.defines)
    label = a.label or default_label(defines)
    if label and not LABEL_RE.fullmatch(label):
        raise SystemExit("candidate: --label is lower case letters and digits, dot separated (a PEP 440 local "
                         "segment; no '-', it splits wheel file names)")
    committed = re.search(r'^__version__ = "([^"]*)"',
                          sh("git", "show", f"{a.commit}:python/mojolearn/_version.py"), re.M)
    base = a.base_version or next_version(committed[1] if committed else None)
    if not re.fullmatch(r"\d+\.\d+\.\d+", base):
        raise SystemExit("candidate: --base-version must be X.Y.Z")
    version = candidate_version(base, a.commit, label)
    work = Path(a.work_root).expanduser().resolve() / version
    main_tree = main_checkout()
    if inside(work, main_tree):
        raise SystemExit(f"candidate: the work dir {work} is inside the main checkout {main_tree}; refused")
    src = source_tree(work, a.commit, a.dry_run)
    if inside(src, main_tree) or src.resolve() == ROOT.resolve():
        raise SystemExit(f"candidate: the source tree {src} is the main checkout or this tooling checkout; refused")

    before = remote_refs()
    state_path = work / "state.json"
    # THE PIN, written for a dry run too (the candidate's own file): Release.commit reads state["commit"], so no
    # freeze step is needed or run
    work.mkdir(parents=True, exist_ok=True)
    state = release.read_json(state_path) or {}
    if state and (state.get("commit") != a.commit or (state.get("candidate") or {}).get("defines") != defines):
        raise SystemExit(f"candidate: {state_path} is another candidate's state; pick another --label")
    if not state:
        state = dict(version=version, commit=a.commit, steps={}, release_modes=["identical"],
                     candidate=dict(version=version, commit=a.commit, defines=defines, label=label,
                                    base_version=base, committed_version=committed[1] if committed else None,
                                    made=release.now()))
        release.write_json(state_path, state)
    argv = [version, "--only", ",".join(STEPS), "--state-dir", str(work), "--source-checkout", str(src),
            "--build-backend", "github", "--identity-on-wheels"] + (["--dry-run"] if a.dry_run else [])
    args = release.release_parser().parse_args(argv)
    args.release_modes = release_reuse.release_modes(default=("identical",))
    args.candidate_version, args.candidate_defines, args.candidate_ref = version, defines, a.ref
    r = release.Release(args)
    if r.commit != a.commit:
        raise SystemExit(f"candidate: the Release resolved {r.commit}, not {a.commit}")
    print(f"candidate {version} of {a.commit[:12]}" + (f" defines {defines}" if defines else "")
          + f"; work dir {work}; steps {', '.join(STEPS)} (every other release step is skipped)")
    rc = r.go()

    after = remote_refs()
    moved = [k for k in ("origin_main", "origin_tags_sha256", "local_tags_sha256") if before[k] != after[k]]
    if moved:
        # this path has no push, tag or publish command; a change here is someone else's, and is reported
        raise SystemExit("candidate: CHANGED during the run: " + ", ".join(f"{k} {before[k]} -> {after[k]}" for k in moved)
                         + " (the candidate path pushes and tags nothing; check who moved it)")
    print(f"candidate: origin/main {str(after['origin_main'])[:12]} and {after['origin_tags']} origin / "
          f"{after['local_tags']} local tags unchanged")
    if rc or a.dry_run:
        return rc
    final = work / a.commit[:12] / "linux" / "final"
    finals = r.split_finals()
    if not finals.get("linux") or release.wheel_commit(finals["linux"]) != a.commit:
        raise SystemExit(f"candidate: no core wheel of {a.commit[:12]} in {final}")
    manifests = {}
    for vendor in vendors:
        wheels = [finals["linux"], finals[vendor]]
        manifests[vendor] = C.build_manifest(version, vendor, wheels, a.commit)   # RECORD, metadata, tags, payload
        for f in manifests[vendor]["files"]:
            payload = f.get("candidate") or {}
            if payload.get("local_version") != version or (payload.get("defines") or "") != defines:
                raise SystemExit(f"candidate: {f['file']} does not record this candidate's version and defines")
    doc = dict(schema="mojolearn.release-candidate.v1", version=version, source_commit=a.commit, defines=defines,
               label=label, wheel_dir=str(final), vendors=vendors, steps=list(STEPS),
               wheels={k: dict(file=w.name, sha256=release.sha256(w)) for k, w in finals.items() if w},
               manifests=manifests, refs_before=before, refs_after=after, made=release.now())
    release.write_json(final / "candidate.json", doc)
    print(f"candidate: {final}/candidate.json")
    for vendor in vendors:
        if defines:
            # a B set rides with its A set's stage line (measure_candidate reads this version from candidate.json)
            print(f"B-ARM {vendor} --b-arm <owed arm name>={final}@{a.commit}")
        else:
            print(f"STAGE python3 tools/ops/measure_candidate.py stage {vendor} --version '{version}' "
                  f"--source-commit {a.commit} --tree <job worktree>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
