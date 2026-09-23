#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The whole local release verification, CPU and Apple GPU AT ONCE
(2026-09-21).

    pixi run -e test release-check

`release-check` ran `cpu-pass` and then `apple-pass`, one after the other:
602 s and then 591 s for 0.8.12, twenty minutes during which the GPU sat idle
for the first ten and four of the five CPU slots for the last ten. The two
passes share nothing but the Mac's slot scheduler (tools/mac_slot.py), so
they run together here: the Apple pass holds the Metal lock and ONE of the
MAC_SLOTS (default 5), and the CPU pass takes the rest (MAC_SLOTS - 1 shards,
through MOJOLEARN_CPU_PASS_SLOTS). The Apple release budget of five cores is
the scheduler's own limit, so it cannot be exceeded. The CPU slots run at
nice 19 and the Metal job at normal priority, so the CPU pass yields to the
GPU's host thread rather than starving it.

Each pass keeps its own records and resume behaviour
(~/mojolearn-evidence/release-check/<commit>/{cpu,metal}/), writes its
transcript to <that directory>/../release-check.<pass>.log, and the exit
status is non-zero if either pass is incomplete. Run the two tasks separately
to keep the old serial behaviour.

    pixi run -e test release-check --plan     what the default would run, per backend
                                              (CPU, Apple, NVIDIA, AMD), and why; runs nothing
    ... release-check --plan --paths=a,b      the same for a hypothetical change to those paths
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def plan(out=print, backends=None, paths=None):
    """WHAT THE DEFAULT RELEASE VERIFICATION WOULD RUN, per backend, without
    running anything (2026-09-22): the lanes each pass selects with no flags,
    its cell count (lanes x base,denormal,odd, one fit each), why each lane is
    in, and, when a rule selects every lane, the exact paths that made it.
    CPU and Apple run here; NVIDIA and AMD run in the Linux build rentals
    (`pixi run release`). The CPU pass covers the union, because its column is
    the reference the other three are diffed against."""
    sys.path.insert(0, str(ROOT / "tools"))
    import lane_select
    import verify_lanes
    fixtures = verify_lanes.APPLE_PASS_FIXTURES
    nfix = len(fixtures.split(","))
    sources, _ = lane_select.lane_sources()
    total = len(sources)
    backends = backends or verify_lanes.RELEASE_BACKENDS
    picked, refused = {}, {}
    for b in backends:
        try:
            picked[b] = verify_lanes.pass_selection(b, fixtures, sources, paths=paths)
        except verify_lanes.UnattributedPaths as exc:
            refused[b] = exc.sel
    if refused:
        for b, sel in refused.items():
            out(f"\n== {b}: REFUSED")
            lane_select.refuse_unattributed(sel, out=lambda m: out("   " + m))
        out("\n== the default release verification cannot run until every changed path is attributed")
        return 3
    if "cpu" in picked:
        lanes, sel, covers, how = picked["cpu"]
        union = set(lanes)
        for b, (ol, _, _, _) in picked.items():
            union |= set(ol)
        skip = set(sel["dropped"])
        import lane_applicability
        skip |= set(lane_applicability.degenerate("cpu-host"))
        picked["cpu"] = ([n for n in lane_select.all_lanes() if n in union and n not in skip], sel, covers,
                         how + "; widened to the union of every backend's selection (the reference)")
    label = dict(cpu="CPU (this Mac)", metal="Apple GPU (this Mac)", cuda="NVIDIA H100 (build rental)",
                 hip="AMD MI325X (build rental)")
    grand = 0
    for b, (lanes, sel, covers, how) in picked.items():
        cells = len(lanes) * nfix
        grand += cells
        out(f"\n== {label.get(b, b)}: {len(lanes)} of {total} lanes x {nfix} fixtures = {cells} cells, "
            f"each fitted once ({how})")
        out(f"   {len(sel.get('changed', []))} changed path(s); {len(sel.get('inert', []))} reach no lane")
        if sel.get("every_rules"):
            out(f"   EVERY LANE, by rule, because of {len(sel['every_rules'])} path(s):")
            for p, why in sel["every_rules"].items():
                out(f"     {p}: {why}")
        else:
            out("   every changed path attributed to specific lanes; no every-lane rule fired")
        why = {}
        for p, hit in (sel.get("by_path") or {}).items():
            for n in hit or ():
                why.setdefault(n, []).append(p)
        for n in lanes:
            src = why.get(n)
            out(f"     {n}: " + (", ".join(src[:3]) + (f" (+{len(src) - 3} more)" if len(src) > 3 else "")
                              if src else "the union with another backend's selection"))
        if sel.get("dropped"):
            out(f"   left out, cannot run on {b}: {len(sel['dropped'])} lane(s): {','.join(sel['dropped'])}")
    out(f"\n== total {grand} cells across {len(picked)} backend(s); nothing ran")
    return 0


def main():
    if "--plan" in sys.argv[1:]:
        backends = None
        for a in sys.argv[1:]:
            if a.startswith("--backends="):
                backends = tuple(a.split("=", 1)[1].split(","))
        paths = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--paths=")), None)
        return plan(backends=backends, paths=paths)
    sys.path.insert(0, str(ROOT / "tools"))
    import verify_lanes
    slots = int(os.environ.get("MAC_SLOTS", "5"))
    if slots < 2:
        raise SystemExit("REFUSING: release-check needs MAC_SLOTS >= 2 (one for Metal, one for the CPU pass)")
    logs = Path(verify_lanes.pass_out_dir("metal")).parent
    logs.mkdir(parents=True, exist_ok=True)
    runs = {}
    started = time.monotonic()
    for name, flag, env in (("apple", "--apple-pass", {}),
                            ("cpu", "--cpu-pass", {"MOJOLEARN_CPU_PASS_SLOTS": str(slots - 1)})):
        log = logs / f"release-check.{name}.log"
        fh = open(log, "w")
        proc = subprocess.Popen([sys.executable, str(ROOT / "tools" / "verify_lanes.py"), flag],
                                cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ, **env))
        runs[name] = (proc, fh, log)
        print(f"# {name} pass started, transcript {log}", flush=True)
    codes = {}
    try:
        for name, (proc, fh, log) in runs.items():
            codes[name] = proc.wait()
            fh.close()
            print(f"# {name} pass exited {codes[name]} after {time.monotonic() - started:.0f} s", flush=True)
    except KeyboardInterrupt:
        for proc, fh, _ in runs.values():
            if proc.poll() is None:
                proc.terminate()
        for name, (proc, fh, _) in runs.items():
            codes[name] = proc.wait()
            fh.close()
    for name, (_, _, log) in runs.items():
        tail = [l for l in log.read_text(errors="replace").splitlines() if l.startswith("# verdict")
                or l.startswith("# FAIL") or l.startswith("# nothing to check")][-6:]
        for line in tail:
            print(f"  {name}: {line}")
    ok = all(c == 0 for c in codes.values())
    print(f"# release-check {'COMPLETE' if ok else 'INCOMPLETE'} in {time.monotonic() - started:.0f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
