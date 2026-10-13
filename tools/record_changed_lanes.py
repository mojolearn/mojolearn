#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The changed lanes since the admitted table, and the exact lq lines that record them on NVIDIA and AMD.

    python3 tools/record_changed_lanes.py --branch lane/x [--tag 0845] [--all-builds]

Andrew, 2026-10-13: releases use no GPUs, so the changed lanes are recorded BEFORE merge, through lq, when
the boxes are up for other work. This tool only PRINTS; it queues nothing, rents nothing and never ssh-es:

  1. the changed lanes of THIS checkout (it should be the branch's head) since the commit each lane's admitted
     rows rest on: `tools/identity_columns.py changed-lanes --scope routine --backend cuda` and `--backend hip`,
     their union (NVIDIA == AMD needs both columns over the same lanes). MOJOLEARN_RELEASE_BUILD_ONLY is
     honored through tools/lane_select.py. No lane: prints NOTHING-CHANGED and exits 0;
  2. the bindings those lanes load (tools/lane_select.py lane evidence; the core and the host twins too),
     as lq's BUILDS= list, so the job builds only what the record runs (--all-builds: BUILDS=all);
  3. one `lq add nv CMD ...` and one `lq add amd CMD ...` line: REPEATS=1 tools/record_identity_column.sh
     with LANES=<changed>, writing under /root/lq/out/ (which the box archive brings home);
  4. the admit line to run here once both records are home (tools/admit_identity_columns.sh).

Identity runs ONCE (Andrew 2026-10-10): one record per vendor; a mismatch is a bug to fix, never a rerun.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

#: the held lq boxes (CLAUDE.md: nv is the RunPod L40S, amd the DigitalOcean MI325X) and their record labels
BOXES = (("nv", "cuda", "nvidia-l40s-sm_89"), ("amd", "hip", "amd-mi325x-gfx942"))
LQ = os.path.expanduser("~/mojolearn-evidence/lq/lq")


def changed(backend, since):
    """(lanes, the selector's CHANGED-LANES line) for one backend."""
    cmd = [sys.executable, str(ROOT / "tools" / "identity_columns.py"), "changed-lanes", "--scope", "routine",
           "--backend", backend] + (["--since", since] if since else [])
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"record_changed_lanes: identity_columns.py changed-lanes --backend {backend} exited "
                         f"{p.returncode}: {p.stderr.strip()[-400:]}")
    line = next((l for l in p.stderr.splitlines() if "CHANGED-LANES" in l), "")
    text = (p.stdout.strip().splitlines() or [""])[-1]
    return [x for x in text.split(",") if x], line.lstrip("# ")


def builds_for(lanes):
    """lq BUILDS= names (build script basenames without .sh) for every binding the lanes load, their host
    twins and the core; None when a lane's bindings cannot be read (then BUILDS=all)."""
    import lane_select
    import lane_map_import_graph
    _, why = lane_select.lane_sources()
    scripts = lane_map_import_graph.build_scripts()
    names = {"_mojolearn"}
    for lane in lanes:
        ev = why.get(lane)
        if ev is None:
            return None
        names |= set(ev.get("bindings") or ()) | set(ev.get("binding_use") or {}) | set(ev.get("declared") or ())
    names |= {n + "_host" for n in list(names) if n + "_host" in scripts}
    out = sorted({Path(scripts[n]).name[:-3] for n in names if n in scripts})
    return out or None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--branch", required=True, help="the pushed branch lq checks out (this checkout's head)")
    ap.add_argument("--tag", default="", help="tag stem for the lq jobs and the record directories")
    ap.add_argument("--since", default="", help="one commit for every lane instead of each lane's admitted row")
    ap.add_argument("--all-builds", action="store_true", help="BUILDS=all instead of the lanes' bindings")
    ap.add_argument("--nv-label", default=BOXES[0][2])
    ap.add_argument("--amd-label", default=BOXES[1][2])
    a = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", a.branch):
        ap.error("--branch: a branch name")
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    remote = subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"origin/{a.branch}"], capture_output=True,
                            text=True).stdout.strip()
    if remote != head:
        print(f"# WARNING: this checkout is at {head[:12]}, origin/{a.branch} at {remote[:12] or '(none)'}; "
              "the records will name the branch head lq checks out, so push and run this from it")
    per = {}
    for _, backend, _ in BOXES:
        per[backend], line = changed(backend, a.since)
        print(f"# {backend}: {line or len(per[backend])}")
    lanes = list(dict.fromkeys(per["cuda"] + per["hip"]))
    if not lanes:
        print("NOTHING-CHANGED: no lane's bits can have moved since the admitted table; nothing to record")
        return 0
    builds = None if a.all_builds else builds_for(lanes)
    tag = a.tag or head[:9]
    print(f"CHANGED-LANES {len(lanes)}: {','.join(lanes)}")
    print(f"# BUILDS: {'all' if builds is None else ','.join(builds)}")
    labels = {"nv": a.nv_label, "amd": a.amd_label}
    dirs = []
    for box, _, _ in BOXES:
        out = f"/root/lq/out/rec-{tag}-{box}"
        dirs.append(out)
        print(f"{LQ} add {box} CMD {a.branch} rec-{tag}-{box} LANES={','.join(lanes)} REPEATS=1 "
              f"bash tools/record_identity_column.sh {labels[box]} {out} "
              f"BUILDS={'all' if builds is None else ','.join(builds)}")
    print(f"# when both RECORDED lines read admissible=yes (`{LQ} results nv rec-{tag}` / amd) and the box "
          "archive has brought /root/lq/out home:")
    print(f"#   tools/admit_identity_columns.sh <home>/rec-{tag}-nv <home>/rec-{tag}-amd   "
          "(NVIDIA == AMD, 0 conflicts; then commit the table and the records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
