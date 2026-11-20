#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The macOS build's STALE-ARTIFACT PRE-FLIGHT (lane release-no-hands, 2026-11-04).

python/pyproject.toml's package-data globs ship every `*.so` (and every merged index, every
Linux-layout file) sitting in python/mojolearn/, so packaging/macos/build_release_wheel.sh stops on
two refusals: an "unselected <tier> artifact" (:204-211) and "python/mojolearn holds extensions this
build does not expect ... SHIP STALE" (the stray check after the build). At 0.8.65 three untracked
files (`_mojolearn_native.so`, `_mojolearn_native_m1.so`, `identical/_mojolearn_scoped_gemm_probe.so`)
each cost a full 75-minute macOS build before the error. tools/release.py runs this pre-flight in the
source checkout before macos-build (and again before macos-smoke): every untracked or ignored file a
shipping glob matches that this build does not produce is MOVED (never deleted) to
<release dir>/<commit12>/stale-artifacts/<timestamp>/<path in the checkout>, each one logged, with a
moved.json receipt (path, sha256, bytes) beside them; only then does the build start.

WHAT THIS BUILD PRODUCES. A merged build (MOJOLEARN_MERGED_BINDINGS=1, the default) produces exactly
`<dir>/<stem>.so` and `<dir>/_mojolearn_merged.json` for each merged directory (the selected tiers,
then host/ when the release profile packs the byte LM and host bindings), as the source's own
`tools/merged_bindings.py plan --dir <dir> --platform darwin` says: the same list the build's stray
check holds the tree to. A per-binding build's list lives only inside the build script, so there every
untracked artifact is moved before the build (the build rebuilds every output it expects: each must
be newer than its start, so a pre-existing one is never used) and, after the build, only the
unselected-tier and Linux-layout files are moved. Directories the build wipes itself (`.dylibs/`,
`_adapters/`) are left alone. Tracked files are never touched.

    python3 tools/release_stale_artifacts.py SOURCE_CHECKOUT --dest DIR [--modes "identical"]
        [--merged 1|0] [--byte-lm 1|0] [--phase build|smoke] [--dry-run]
"""
import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PKG = "python/mojolearn"
#: package-data globs (python/pyproject.toml [tool.setuptools.package-data] mojolearn) that ship BUILT
#: native files: an untracked file one of these matches goes into the wheel.
SHIP_GLOBS = ("*.so", "identical/*.so", "host/*.so",
              "_mojolearn_merged.json", "identical/_mojolearn_merged.json", "host/_mojolearn_merged.json",
              "cuda/*", "cuda/identical/*", "cuda/.libs/*", "cuda/sm_*/*", "cuda/sm_*/identical/*",
              "cuda/sm_*/.libs/*", "hip/*", "hip/identical/*", "hip/.libs/*", "hip/gfx*/*",
              "hip/gfx*/identical/*", "hip/gfx*/.libs/*", ".libs/*")
#: directories build_release_wheel.sh deletes and refills itself (stage_dylibs.py, the adapter build)
REBUILT = (".dylibs/", "_adapters/")
#: the macOS tier directories, relative to the package (fast is the package directory itself)
TIER_DIRS = {"fast": "", "identical": "identical/", "host": "host/"}
RECEIPT = "moved.json"


def say(msg):
    print(f"stale-artifacts: {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def shipped(rel):
    """True when REL (relative to the package) is a file a shipping glob matches. fnmatch's `*` also
    crosses `/`, so a glob matches only names with its own directory depth."""
    return any(fnmatch.fnmatchcase(rel, g) and rel.count("/") == g.count("/") for g in SHIP_GLOBS)


def tracked_files(src):
    """Tracked paths under the package (relative to SRC): never moved."""
    out = subprocess.run(["git", "-C", str(src), "ls-files", "-z", "--", PKG], capture_output=True, check=True)
    return {p for p in out.stdout.decode().split("\0") if p}


def candidates(src, tracked):
    """[package-relative path] of every untracked or ignored file a shipping glob matches."""
    pkg = Path(src) / PKG
    found = []
    if not pkg.is_dir():
        return found
    for path in sorted(pkg.rglob("*")):
        if not (path.is_file() or path.is_symlink()):
            continue
        rel = path.relative_to(pkg).as_posix()
        if rel.startswith(REBUILT) or f"{PKG}/{rel}" in tracked or not shipped(rel):
            continue
        found.append(rel)
    return found


def merged_dirs(modes, byte_lm):
    return [m for m in ("fast", "identical") if m in modes] + (["host"] if byte_lm else [])


def merged_plan(src, directory, byte_lm):
    """The source's own merged plan for one directory: [stem]."""
    out = subprocess.run([sys.executable, "tools/merged_bindings.py", "plan", "--dir", directory, "--platform",
                          "darwin", "--byte-lm", "1" if byte_lm else "0"], cwd=str(src), capture_output=True,
                         text=True, check=True)
    return [line.split()[0] for line in out.stdout.splitlines() if line.strip()]


def expected(src, modes, byte_lm, merged=True, plan=None):
    """{package-relative path} this build produces, or None when only the build script knows (per-binding)."""
    if not merged:
        return None
    plan = plan or merged_plan
    want = set()
    for directory in merged_dirs(modes, byte_lm):
        prefix = TIER_DIRS[directory]
        want.add(prefix + "_mojolearn_merged.json")
        want.update(prefix + stem + ".so" for stem in plan(src, directory, byte_lm))
    return want


def in_selected_tier(rel, modes, byte_lm):
    """REL sits in a directory this build fills (a selected tier, or host/ with the byte LM)."""
    depth_one = rel.split("/", 1)[0] + "/" if "/" in rel else ""
    return depth_one in {TIER_DIRS[d] for d in merged_dirs(modes, byte_lm)} and rel.count("/") <= 1


def stale(found, want, modes, byte_lm, phase):
    """The candidates this build does not produce: WANT known (merged): everything not in it; WANT None
    (per-binding): everything before the build, the unselected-tier and Linux-layout files after it."""
    if want is not None:
        return [rel for rel in found if rel not in want]
    if phase == "build":
        return list(found)
    return [rel for rel in found if not in_selected_tier(rel, modes, byte_lm)]


def move_aside(src, rels, dest, log=say):
    """Move each package-relative REL to DEST/<path in the checkout>; returns the receipt rows."""
    rows = []
    for rel in rels:
        source = Path(src) / PKG / rel
        target = Path(dest) / PKG / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = None if source.is_symlink() else sha256(source)
        size = source.lstat().st_size
        shutil.move(str(source), str(target))
        rows.append(dict(path=f"{PKG}/{rel}", moved_to=str(target), sha256=digest, bytes=size))
        log(f"moved {PKG}/{rel} -> {target} ({size} bytes)")
    return rows


def preflight(src, dest_root, modes, byte_lm=True, merged=True, phase="build", dry=False, log=say,
              plan=None, stamp=None):
    """(stale paths, receipt path or None): find and (unless DRY) move every stale artifact under SRC."""
    if not (Path(src) / PKG).is_dir():
        log(f"{src}/{PKG}: not there yet; nothing to check")
        return [], None
    found = candidates(src, tracked_files(src))
    try:
        want = expected(src, modes, byte_lm, merged, plan=plan) if found else set()
    except (OSError, subprocess.CalledProcessError) as error:
        # the source's own plan is unreadable: judged as a per-binding build (never fewer moves before a build)
        log(f"the merged plan of {src} is unreadable ({error}); judging as a per-binding build")
        want = None
    bad = stale(found, want, modes, byte_lm, phase)
    how = "the merged plan" if want is not None else ("per-binding: every untracked artifact" if phase == "build"
                                                       else "per-binding: unselected tiers and Linux layout")
    if not bad:
        log(f"{src}/{PKG}: no stale artifact ({len(found)} untracked shipping file(s), judged by {how})")
        return [], None
    if dry:
        for rel in bad:
            log(f"would move {PKG}/{rel} (not produced by this build; judged by {how})")
        return bad, None
    dest = Path(dest_root) / (stamp or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    rows = move_aside(src, bad, dest, log=log)
    receipt = dest / RECEIPT
    receipt.write_text(json.dumps(dict(schema="mojolearn.release-stale-artifacts.v1", source=str(src),
                                       phase=phase, modes=list(modes), merged=merged, byte_lm=byte_lm,
                                       judged_by=how, moved=rows), indent=2) + "\n")
    log(f"{len(rows)} stale artifact(s) moved aside before the {phase}; receipt {receipt}")
    return bad, receipt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source")
    ap.add_argument("--dest", required=True, help="stale-artifacts root; a <timestamp>/ directory is made in it")
    ap.add_argument("--modes", default=os.environ.get("MOJOLEARN_RELEASE_MODES", "fast identical"))
    ap.add_argument("--merged", default=os.environ.get("MOJOLEARN_MERGED_BINDINGS", "1"), choices=("0", "1"))
    ap.add_argument("--byte-lm", default=os.environ.get("MOJOLEARN_PACKAGE_BYTE_LM", "1"), choices=("0", "1"))
    ap.add_argument("--phase", default="build", choices=("build", "smoke"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    bad, _ = preflight(a.source, a.dest, tuple(a.modes.split()), a.byte_lm == "1", a.merged == "1", a.phase,
                       a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
