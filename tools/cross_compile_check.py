#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Cross-compile the changed Linux bindings on the Mac BEFORE a release rents a
GPU box (2026-09-25).

    pixi run cross-compile-check                         # vs the last v* tag
    python3 tools/cross_compile_check.py --ref v0.8.15 [--limit 3] [--archs sm_89]
    python3 tools/cross_compile_check.py --only _mojolearn_svm --tiers fast

WHY. 0.8.19's `fast` svm binding never finished compiling for NVPTX or AMDGPU:
every Linux leg sat on it for more than 33 minutes and was killed by its build
bound. The Mac reproduces it (`mojo build --target-accelerator sm_89` needs no
NVIDIA device), so the leg's rental bought nothing a local compile could not
have said first.

WHAT IT BUILDS. The bindings whose identity inputs changed between --ref and
--head, read the way tools/release_reuse.py reads them: the Mojo import
closure of the file each build script compiles, plus the script and
pixi.toml/pixi.lock (`bincache.source_digest` over `git archive` of each
commit). Each changed binding is compiled once per requested accelerator and
per tier it ships in on Linux (packaging/linux/pack_wheel.py `tier_names`),
with the flags the release build sets for that tier and column: the tier
define and `-D MOJOLEARN_COLUMN_NVIDIA|AMD`, `--target-accelerator <arch>`,
`-j 1`, `--emit shared-lib`, from the WORKING TREE. Host bindings are not a
cross compile and are skipped. The host CPU is the Mac's; the GPU half, which
is what hung, is the Linux target's.

ONE AT A TIME. The Mac runs one Mojo compile at a time; so does this.

VERDICTS, one line per job:
  PASS     the compiler exited 0 and wrote the library
  FAIL     the compiler exited non-zero (the first error lines follow)
  TIMEOUT  still running at --timeout seconds (default CROSS_COMPILE_SECONDS in
           tools/release_limits.sh); the process group is killed
  STALLED  the compiler's CPU time did not advance for --stall seconds. The
           0.8.19 svm hang is exactly this: 30 s of CPU, then the main thread
           asleep on a semaphore with no worker running. Waiting out the
           full timeout on it proves nothing more. --stall 0 disables it.
Exit status is 0 only when every job PASSes; a selection that cannot be read
(a build script whose compile line is not literal) refuses by name, it never
widens to "build everything".
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import release_limits  # noqa: E402

TIER_DEFINES = {
    "identical": ["-D", "MOJOLEARN_NUMERIC_IDENTICAL=1"],
    "deterministic": ["-D", "MOJOLEARN_NUMERIC_DETERMINISTIC=1"],
    "fast": [],
}
DEFAULT_ARCHS = ("sm_89", "gfx942")


def column_of(arch):
    return "NVIDIA" if arch.startswith("sm_") else "AMD"


# ---------------------------------------------------------------- selection
def changed_scripts(ref_facts, head_facts):
    """{script: reason} for every build script whose closure differs, is new,
    or cannot be read at head. Pure: facts are tools/release_reuse.tree_facts
    dicts."""
    out = {}
    for script, cur in sorted(head_facts["closures"].items()):
        prev = ref_facts["closures"].get(script)
        if cur is None:
            continue  # no such build script at head
        if prev is None:
            out[script] = "new at head"
        elif prev.get("digest") != cur.get("digest") or prev.get("scope") != cur.get("scope"):
            changed = sorted(set(cur.get("sources", [])) ^ set(prev.get("sources", [])))
            out[script] = "closure changed" + (f" ({len(changed)} files added/removed)" if changed else "")
    return out


def linux_rows():
    """[(name, script, tier)] for every non-host binding the Linux sets ship,
    from the same list the pack reads."""
    import release_reuse
    seen, rows = set(), []
    for b in release_reuse.bindings(release_reuse.LINUX):
        if b.host:
            continue
        key = (b.name, b.script, b.tier)
        if key not in seen:
            seen.add(key)
            rows.append(key)
    return rows


def plan_jobs(scripts, rows, archs, tiers=None, only=None, limit=0):
    """[(name, script, tier, arch)] in a stable order: bindings by name, then
    arch, then tier as the Linux lists order them. `limit` caps BINDINGS,
    not jobs."""
    names = []
    for name, script, _tier in rows:
        if script in scripts and name not in names and (not only or name in only):
            names.append(name)
    names.sort()
    if limit:
        names = names[:limit]
    jobs = []
    for name in names:
        for arch in archs:
            for rname, script, tier in rows:
                if rname == name and (not tiers or tier in tiers):
                    jobs.append((name, script, tier, arch))
    return jobs


def compile_cmd(mojo, root_file, incs, tier, arch, out):
    cmd = list(mojo) + ["build", "-j", "1", "--emit", "shared-lib", "--target-accelerator", arch]
    cmd += TIER_DEFINES[tier] + ["-D", "MOJOLEARN_COLUMN_" + column_of(arch)]
    for inc in incs:
        cmd += ["-I", str(inc)]
    return cmd + [str(root_file), "-o", str(out)]


# ---------------------------------------------------------------- running
def group_cpu_seconds(pgid):
    """Summed CPU time of a process group (ps TIME column), or None."""
    try:
        o = subprocess.run(["ps", "-g", str(pgid), "-o", "time="], capture_output=True, text=True,
                           timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    total = 0.0
    for tok in o.split():
        v = 0.0
        for part in tok.replace("-", ":").split(":"):
            try:
                v = v * 60 + float(part)
            except ValueError:
                return None
        total += v
    return total


def run_one(cmd, out, timeout, stall, cwd=ROOT, poll=2.0, cpu_probe=group_cpu_seconds, clock=time.monotonic,
            sleep=time.sleep):
    """(verdict, seconds, log_tail). The compiler runs in its own process
    group so a kill takes every child with it."""
    log = tempfile.TemporaryFile(mode="w+")
    t0 = clock()
    p = subprocess.Popen(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    last_cpu, last_move = -1.0, t0
    verdict = None
    while verdict is None:
        rc = p.poll()
        now = clock()
        if rc is not None:
            verdict = "PASS" if rc == 0 and Path(out).is_file() else "FAIL"
            break
        if now - t0 >= timeout:
            verdict = "TIMEOUT"
            break
        if stall:
            cpu = cpu_probe(p.pid)
            if cpu is not None and cpu > last_cpu + 0.5:
                last_cpu, last_move = cpu, now
            elif cpu is not None and now - last_move >= stall:
                verdict = "STALLED"
                break
        sleep(poll)
    if p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()
    secs = clock() - t0
    log.seek(0)
    text = log.read()
    log.close()
    tail = [ln for ln in text.splitlines() if "error" in ln][:6] if verdict == "FAIL" else []
    if verdict == "FAIL" and not tail:
        tail = text.splitlines()[-6:]
    if verdict == "STALLED":
        tail = [f"compiler CPU flat at {max(last_cpu, 0):.0f} s for {stall:.0f} s"]
    return verdict, secs, tail


# ---------------------------------------------------------------- main
def last_tag(root=ROOT):
    return subprocess.run(["git", "-C", str(root), "describe", "--tags", "--abbrev=0", "--match", "v*"],
                          capture_output=True, text=True, check=True).stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", default="", help="compare against this commit or tag (default: the last v* tag)")
    ap.add_argument("--head", default="HEAD", help="the commit whose changes select (the WORKING TREE is compiled)")
    ap.add_argument("--archs", default=",".join(DEFAULT_ARCHS))
    ap.add_argument("--tiers", default="", help="comma list; default every tier the Linux lists ship")
    ap.add_argument("--only", default="", help="comma list of binding names (still must be changed)")
    ap.add_argument("--limit", type=int, default=0, help="at most this many bindings (0: all)")
    ap.add_argument("--timeout", type=float, default=float(release_limits.get("CROSS_COMPILE_SECONDS")))
    ap.add_argument("--stall", type=float, default=240.0, help="seconds of flat compiler CPU = STALLED (0: off)")
    ap.add_argument("--mojo", default="pixi run mojo", help="the compiler command")
    ap.add_argument("--cache", default="", help="identity cache (default <evidence>/release/identities)")
    ap.add_argument("--json", default="", help="write the results here")
    ap.add_argument("--dry-run", action="store_true", help="print the jobs, compile nothing")
    a = ap.parse_args(argv)

    import bincache
    import release_reuse
    ev = Path(os.environ.get("MOJOLEARN_EVIDENCE_ROOT", os.path.expanduser("~/mojolearn-evidence")))
    cache = Path(a.cache) if a.cache else ev / "release" / "identities"
    ref = a.ref or last_tag()
    print(f"cross-compile-check: {ref} .. {a.head} (working tree compiled), archs {a.archs}, "
          f"timeout {a.timeout:.0f} s, stall {a.stall:.0f} s")
    scripts = changed_scripts(release_reuse.facts_for_commit(ref, cache),
                              release_reuse.facts_for_commit(a.head, cache))
    rows = linux_rows()
    shipped = {s for _n, s, _t in rows}
    for script, why in sorted(scripts.items()):
        if script in shipped:
            print(f"  changed  bindings/{script}: {why}")
    archs = [x for x in a.archs.split(",") if x]
    jobs = plan_jobs(scripts, rows, archs, tiers=set(filter(None, a.tiers.split(","))),
                     only=set(filter(None, a.only.split(","))), limit=a.limit)
    if not jobs:
        print("no changed Linux GPU binding: nothing to cross-compile")
        return 0
    plans = {}
    for script in sorted({j[1] for j in jobs}):
        plans[script] = bincache.script_plan(ROOT, str(ROOT / "bindings" / script), [])
    unreadable = [s for s, pl in plans.items() if pl is None or len(pl[0]) != 1]
    if unreadable:
        for s in unreadable:
            print(f"REFUSED: bindings/{s}: its compile line cannot be read literally (one root .mojo); "
                  f"fix the script or tools/bincache.py script_plan, nothing was compiled")
        return 2
    mojo = a.mojo.split()
    results = []
    scratch = Path(tempfile.mkdtemp(prefix="mojolearn-xcc-"))
    try:
        for name, script, tier, arch in jobs:
            roots, incs, _shells = plans[script]
            out = scratch / f"{name}-{tier}-{arch}.so"
            cmd = compile_cmd(mojo, roots[0].relative_to(ROOT),
                              [os.path.relpath(i, ROOT) for i in incs], tier, arch, out)
            if a.dry_run:
                print(f"  would    {name:34s} {tier:13s} {arch:7s} {' '.join(cmd)}")
                continue
            verdict, secs, tail = run_one(cmd, out, a.timeout, a.stall)
            print(f"  {verdict:8s} {name:34s} {tier:13s} {arch:7s} {secs:7.1f} s", flush=True)
            for ln in tail:
                print(f"           {ln[:240]}")
            results.append(dict(name=name, script=script, tier=tier, arch=arch, verdict=verdict,
                                seconds=round(secs, 1), detail=tail))
            out.unlink(missing_ok=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if a.dry_run:
        return 0
    bad = [r for r in results if r["verdict"] != "PASS"]
    print(f"cross-compile-check: {len(results) - len(bad)} PASS, {len(bad)} not "
          f"({', '.join(sorted({r['verdict'] for r in bad})) or 'none'})")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(dict(ref=ref, head=a.head, results=results), indent=1) + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
