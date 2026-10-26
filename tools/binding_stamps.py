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
    python3 tools/binding_stamps.py write-merged <output .so> <member,member,...>
        after a merged build (bindings/merged_build.sh): one stamp for the
        merged file, recording every member's own build script digest (the
        closure of the module body it compiles) plus the merged build's own
        inputs (MERGED_INPUTS); check holds every one of them
    python3 tools/binding_stamps.py check [--quiet] [--development-cache DIR]
        before a pass: legacy bindings need current source stamps. Active
        development indexes need successful cache receipts, current source and
        toolchain closures, and matching installed bytes. Set the cache with
        --development-cache or MOJOLEARN_BINCACHE_DEVELOPMENT_DIR (default:
        ~/.cache/mojolearn-development). Inactive development history is ignored
        only after its active index is validated. Any failed check exits 1.
    python3 tools/binding_stamps.py --self-test

Stamps live in python/.binding-stamps/ (gitignored, never packaged), one JSON
per binding named after its package path.
"""
import argparse
import hashlib
import json
import os
import re
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


#: The files a merged build reads besides the members' closures: the build
#: script, the root generator and the manifest the root is generated from.
MERGED_INPUTS = ("bindings/merged_build.sh", "bindings/build_defines.sh",
                 "tools/merged_bindings.py", "bindings/merged_groups.json")


def member_script(name):
    """The per-binding build script of a member (build.sh for _mojolearn)."""
    return "build.sh" if name == "_mojolearn" else "build_" + name[len("_mojolearn_"):] + ".sh"


def merged_inputs_digest(repo=REPO):
    h = hashlib.sha256()
    for rel in MERGED_INPUTS:
        f = Path(repo) / rel
        h.update(rel.encode() + b"\0" + (hashlib.sha256(f.read_bytes()).hexdigest().encode()
                                          if f.is_file() else b"absent") + b"\n")
    return h.hexdigest()


def cmd_write_merged(so, members, stamps=STAMPS, pkg=PKG, repo=REPO):
    names = [n for n in members.split(",") if n]
    if not names:
        raise SystemExit("binding_stamps: write-merged needs at least one member")
    rows = {}
    for n in names:
        info = digest(member_script(n), repo)
        rows[n] = dict(script=member_script(n), scope=info["scope"], digest=info["digest"])
    stamps.mkdir(parents=True, exist_ok=True)
    rec = dict(schema=2, binding=os.path.relpath(Path(so).resolve(), pkg.resolve()), script="merged_build.sh",
               commit=commit(repo), merged_inputs=merged_inputs_digest(repo), members=rows)
    stamp_path(so, stamps, pkg).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def bindings(pkg=PKG):
    for p in sorted(pkg.rglob("_mojolearn*.so")):
        if ".dylibs" not in p.parts and ".libs" not in p.parts:
            yield p


def development_bindings(pkg, repo, cache):
    """Validate active development groups without requiring inactive history.

    Receipts retain the build configuration; the verifier's process environment
    is not a request to rebuild with different compiler flags. Reconstruct every
    group using that configuration and CURRENT source/toolchain/dependency facts,
    then hold both the cached and actually indexed bytes against its receipt.
    This is source/binary admission only, never runtime qualification.
    """
    import development_build as dev

    pkg, repo, cache = Path(pkg).resolve(), Path(repo).resolve(), Path(cache).resolve()
    indexes = {}
    for mode in ("host", "identical"):
        path = pkg / mode / "_mojolearn_merged.json"
        if path.is_file():
            index = json.loads(path.read_text())
            if index.get("layout") == "development":
                indexes[mode] = index
    if not indexes:
        return set(), set()
    if dev.merged.REPO.resolve() != repo or pkg != repo / "python/mojolearn":
        raise ValueError("development validation must use the selected source root's build tools")
    facts = dev.toolchain_facts(repo)
    if not facts.get("available"):
        raise ValueError("development compiler/dependency facts are unavailable")
    active, ignored = set(), set()
    for mode, index in indexes.items():
        provenance = index.get("provenance")
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError(f"{mode}: missing development provenance")
        receipts = {}
        for stem, key in provenance.items():
            if not re.fullmatch(r"_mojolearn_dev_[a-zA-Z0-9_]+", stem) or not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
                raise ValueError(f"{mode}: malformed development group/key")
            receipt = json.loads((cache / "artifacts" / key / "receipt.json").read_text())
            if receipt.get("state") != "built" or receipt.get("exit_code") != 0 or receipt.get("key") != key:
                raise ValueError(f"{mode}:{stem}: no successful build receipt")
            receipts[stem] = receipt
        first = next(iter(receipts.values()))["fields"]
        retained = first["environment"]
        link = first["link_environment"]
        if any(r["fields"]["environment"] != retained or r["fields"]["link_environment"] != link
               for r in receipts.values()):
            raise ValueError(f"{mode}: inconsistent active build configuration")
        # PATH locates the supported C compiler; all recorded compiler/link
        # inputs come from the build receipt, not the test runner's settings.
        env = {"PATH": os.environ.get("PATH", ""), **retained, **link}
        plan = dev.make_plan(repo, cache, [mode], env, facts=facts)
        expected = {"format": dev.merged.INDEX_FORMAT, "layout": "development", "files": {}, "provenance": {}}
        for row in plan["groups"]:
            stem = row["stem"]
            if stem not in receipts:
                raise ValueError(f"{mode}:{stem}: required group missing from active index")
            # make_plan records its controller's sys.executable as the root
            # generator. Verification may use the test Python instead. Retain
            # the producer interpreter identity: the current generator recipe
            # AND exact generated Mojo root bytes are still compared below.
            # No compiler, flag, source, dependency or binary field is relaxed.
            producer = retained.get("MOJOLEARN_MERGED_PYTHON")
            if not isinstance(producer, str) or not Path(producer).is_absolute():
                raise ValueError(f"{mode}:{stem}: missing producer interpreter identity")
            row["fields"]["environment"]["MOJOLEARN_MERGED_PYTHON"] = producer
            row["key"] = dev.digest(row["fields"])
            receipt = receipts[stem]
            if receipt["fields"] != row["fields"] or receipt["key"] != row["key"]:
                raise ValueError(dev.input_change_reason(f"{mode}:{stem}: stale receipt", receipt["fields"], row["fields"]))
            artifact = dev.cached_artifact(cache, row)
            if artifact is None:
                raise ValueError(f"{mode}:{stem}: cached binary/receipt mismatch")
            relative = f".development/{row['key']}/{stem}.so"
            installed = pkg / mode / relative
            if installed.is_symlink() or installed.resolve() != installed or not installed.is_file() or bincache.sha256_file(installed) != receipt["sha256"]:
                raise ValueError(f"{mode}:{stem}: active binary/receipt mismatch")
            for member in row["members"]:
                if (pkg / mode / (member + ".so")).exists():
                    raise ValueError(f"{mode}:{member}: direct binding shadows active development index")
            expected["files"][relative] = row["members"]
            expected["provenance"][stem] = row["key"]
            active.add(installed)
        if index != expected:
            raise ValueError(f"{mode}: active index membership/paths/provenance do not match current plan")
        for row in plan["support"]:
            artifact = dev.cached_artifact(cache, row)
            receipt_path = cache / "artifacts" / row["key"] / "receipt.json"
            receipt = json.loads(receipt_path.read_text())
            installed = pkg / row["install"]
            if (artifact is None or receipt.get("state") != "built" or not installed.is_file()
                    or bincache.sha256_file(installed) != receipt.get("sha256")):
                raise ValueError(f"support:{row['name']}: current dependency receipt/installed bytes mismatch")
        # Immutable historical groups are unreachable through the exact index
        # just validated. Other direct bindings still follow legacy stamp checks.
        ignored.update((pkg / mode / ".development").rglob("_mojolearn*.so"))
        logical_members = {member for row in plan["groups"] for member in row["members"]}
        for stem, _, members in dev.merged.plan(mode, plan["platform"], True, "release"):
            # A leftover release aggregate cannot own a logical import after
            # the development index selects every member. Never ignore a
            # direct logical binding (including an optional disabled member).
            if stem not in logical_members and stem not in members:
                ignored.add(pkg / mode / (stem + ".so"))
    return active, ignored


def cmd_check(quiet=False, stamps=STAMPS, pkg=PKG, repo=REPO, development_cache=None):
    bad, seen, cache = [], 0, {}
    development_cache = development_cache or os.environ.get("MOJOLEARN_BINCACHE_DEVELOPMENT_DIR") or Path.home() / ".cache/mojolearn-development"
    try:
        active, ignored = development_bindings(pkg, repo, development_cache)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"STALE DEVELOPMENT: {exc}; nothing is verified")
        return 1
    seen += len(active)
    for so in bindings(pkg):
        if so in ignored:
            continue
        seen += 1
        sp = stamp_path(so, stamps, pkg)
        rel = os.path.relpath(so, pkg)
        if not sp.is_file():
            bad.append(f"UNSTAMPED {rel}: no record of the sources it was built from")
            continue
        rec = json.loads(sp.read_text())
        if "members" in rec:
            # a merged file: every member's closure and the merged build's inputs
            if rec.get("merged_inputs") != merged_inputs_digest(repo):
                bad.append(f"STALE {rel}: built at {rec.get('commit', '?')[:12]} by a merged build whose "
                           f"inputs ({', '.join(MERGED_INPUTS)}) have changed since")
                continue
            moved = []
            for n, row in sorted(rec["members"].items()):
                if row["script"] not in cache:
                    cache[row["script"]] = digest(row["script"], repo)["digest"]
                if cache[row["script"]] != row.get("digest"):
                    moved.append(n)
            if moved:
                bad.append(f"STALE {rel}: built at {rec.get('commit', '?')[:12]}; the sources of "
                           f"{', '.join(moved)} have changed since")
            continue
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
              "Repair the affected build inputs or receipts before running verification.")
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
        # a merged file: stale when any member's sources or the merged inputs move
        so.unlink()
        (repo / "j.mojo").write_text("fn g() -> Int:\n    return 1\n")
        (repo / "bindings" / "build_j.sh").write_text("mojo build j.mojo -o python/mojolearn/identical/_mojolearn_j.so\n")
        merged = repo / "python" / "mojolearn" / "identical" / "_mojolearn_native.so"
        merged.write_bytes(b"\0")
        ok.append(cmd_check(True, stamps, pkg, repo) == 1)          # unstamped merged refuses
        cmd_write_merged(merged, "_mojolearn_k,_mojolearn_j", stamps, pkg, repo)
        ok.append(cmd_check(True, stamps, pkg, repo) == 0)          # fresh merged passes
        (repo / "j.mojo").write_text("fn g() -> Int:\n    return 2\n")
        ok.append(cmd_check(True, stamps, pkg, repo) == 1)          # a member's source edit is STALE
        cmd_write_merged(merged, "_mojolearn_k,_mojolearn_j", stamps, pkg, repo)
        (repo / "bindings" / "merged_groups.json").write_text("{}\n")
        ok.append(cmd_check(True, stamps, pkg, repo) == 1)          # a manifest edit is STALE
        shutil.rmtree(stamps)
    print("self-test", "PASS" if all(ok) else f"FAIL {ok}")
    return 0 if all(ok) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", nargs="?", choices=("write", "write-merged", "check"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--development-cache", type=Path, help="development artifact/receipt cache (or MOJOLEARN_BINCACHE_DEVELOPMENT_DIR)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.cmd == "write":
        if len(a.args) != 2:
            ap.error("write <build script> <output .so>")
        cmd_write(a.args[0], a.args[1])
        return 0
    if a.cmd == "write-merged":
        if len(a.args) != 2:
            ap.error("write-merged <output .so> <member,member,...>")
        cmd_write_merged(a.args[0], a.args[1])
        return 0
    if a.cmd == "check":
        return cmd_check(a.quiet, development_cache=a.development_cache)
    ap.error("write or check")


if __name__ == "__main__":
    sys.exit(main())
