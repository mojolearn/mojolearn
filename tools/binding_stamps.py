#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Which sources each built binding came from, and a refusal when they moved.

WHY THIS EXISTS. On 2026-09-22 a release check started on a tree that carried
a GEMM and Holt-Winters kernel fix while every binding in python/mojolearn was
still the build of the previous commit. Nothing compared the two: the passes
load whatever .so files sit in the package, so the check would have "passed"
the old code and recorded itself as the pass for the new commit, and the next
release would have skipped the lanes the fix touched.

    python3 tools/binding_stamps.py write <build script> <output .so>
        after a build: record the sha256 of the binding's source closure
        (tools/bincache.py source_digest: the Mojo import closure of the file
        the script builds, the shell scripts it execs, pixi.toml, pixi.lock)
    python3 tools/binding_stamps.py check [--quiet]
        before a pass: every mojolearn binding in the package must carry a
        stamp whose digest equals the closure's digest in THIS tree. Prints
        STALE or UNSTAMPED per binding and exits 1 if any is.
    python3 tools/binding_stamps.py --self-test

Stamps live in python/.binding-stamps/ (gitignored, never packaged), one JSON
per binding named after its package path.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import bincache  # noqa: E402

PKG = REPO / "python" / "mojolearn"
STAMPS = REPO / "python" / ".binding-stamps"


def stamp_path(so, stamps=STAMPS, pkg=PKG):
    rel = os.path.relpath(Path(so).resolve(), pkg.resolve())
    return stamps / (rel.replace(os.sep, "__") + ".json")


def digest(script, repo=REPO):
    """The closure's digest record, and the closure itself (`sources`, the
    repository-relative files the build's own import walk reached from the
    script's root): recorded in every stamp since 2026-09-23 so that
    `tools/lane_map_import_graph.py --check-stamps` can hold the derived lane
    map against what a built binding was actually compiled from."""
    info, rels = bincache.source_digest(repo, str(Path(repo) / "bindings" / Path(script).name), [])
    return dict(info, sources=sorted(rels))


def commit(repo=REPO):
    r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


def cmd_write(script, so, stamps=STAMPS, pkg=PKG, repo=REPO):
    info = digest(script, repo)
    stamps.mkdir(parents=True, exist_ok=True)
    rec = dict(schema=1, binding=os.path.relpath(Path(so).resolve(), pkg.resolve()),
               script=Path(script).name, commit=commit(repo), **info)
    stamp_path(so, stamps, pkg).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def bindings(pkg=PKG):
    for p in sorted(pkg.rglob("_mojolearn*.so")):
        if ".dylibs" not in p.parts and ".libs" not in p.parts:
            yield p


def cmd_check(quiet=False, stamps=STAMPS, pkg=PKG, repo=REPO):
    bad, seen, cache = [], 0, {}
    for so in bindings(pkg):
        seen += 1
        sp = stamp_path(so, stamps, pkg)
        rel = os.path.relpath(so, pkg)
        if not sp.is_file():
            bad.append(f"UNSTAMPED {rel}: no record of the sources it was built from")
            continue
        rec = json.loads(sp.read_text())
        if rec.get("script") not in cache:
            cache[rec.get("script")] = digest(rec["script"], repo)["digest"]
        if cache[rec["script"]] != rec.get("digest"):
            bad.append(f"STALE {rel}: built at {rec.get('commit', '?')[:12]} from sources that "
                       f"have changed since ({rec['script']})")
    for line in bad:
        print(line)
    if not seen:
        print(f"NO BINDINGS under {pkg}: nothing a pass could run")
        return 1
    if bad:
        print(f"# {len(bad)} of {seen} bindings are not built from this tree; nothing is verified. "
              "Rebuild them (packaging/macos/build_release_wheel.sh) and rerun.")
        return 1
    if not quiet:
        print(f"# {seen} of {seen} bindings are built from this tree's sources")
    return 0


def self_test():
    """A stamp written for a script's sources must go STALE when one of them changes."""
    import shutil
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        (repo / "bindings").mkdir()
        (repo / "python" / "mojolearn" / "identical").mkdir(parents=True)
        (repo / "k.mojo").write_text("fn f() -> Int:\n    return 1\n")
        (repo / "bindings" / "build_k.sh").write_text("mojo build k.mojo -o python/mojolearn/identical/_mojolearn_k.so\n")
        so = repo / "python" / "mojolearn" / "identical" / "_mojolearn_k.so"
        so.write_bytes(b"\0")
        pkg, stamps = repo / "python" / "mojolearn", repo / "python" / ".binding-stamps"
        subprocess.run(["git", "init", "-q", str(repo)])
        ok = []
        ok.append(cmd_check(True, stamps, pkg, repo) == 1)          # unstamped refuses
        cmd_write("build_k.sh", so, stamps, pkg, repo)
        ok.append(cmd_check(True, stamps, pkg, repo) == 0)          # fresh passes
        (repo / "k.mojo").write_text("fn f() -> Int:\n    return 2\n")
        ok.append(cmd_check(True, stamps, pkg, repo) == 1)          # a source edit is STALE
        shutil.rmtree(stamps)
    print("self-test", "PASS" if all(ok) else f"FAIL {ok}")
    return 0 if all(ok) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", nargs="?", choices=("write", "check"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.cmd == "write":
        if len(a.args) != 2:
            ap.error("write <build script> <output .so>")
        cmd_write(a.args[0], a.args[1])
        return 0
    if a.cmd == "check":
        return cmd_check(a.quiet)
    ap.error("write or check")


if __name__ == "__main__":
    sys.exit(main())
