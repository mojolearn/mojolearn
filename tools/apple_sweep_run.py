#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A CLEAN, PER-LANE AGREEMENT SWEEP on one Mac (lane/m3-sweep, 2026-09-28).

    pixi run -e default python -u tools/apple_sweep_run.py --lanes a,b,c --out DIR \\
        [--budget-s 5400] [--deadline 2026-09-28T21:00:00Z]

Most identity lanes were recorded on ONE Apple machine (an M4). This runs the
one lane check's CLEAN stage (tools/algos_lane_check.py: build what is stale,
fit on Metal and on the CPU host bindings, diff) lane by lane, with NO
sabotage and no seam drivers: the sweep is about agreement on this chip, not
about checks that can fail. Unlike the lane check, a failure never ends the
batch: every lane gets its own line,

    SWEEP: <lane>: AGREE | DISAGREE | NOTHING COMPARED | BUILD FAIL | ARM FAIL | ERROR | NOT RUN: <detail>

and a row in DIR/results.jsonl, written as each lane finishes. A binding that
failed to build is not rebuilt for the next lane that needs it (BUILD FAIL,
cached). No lane STARTS after --budget-s of this batch or after --deadline
(UTC): those read NOT RUN, so the caller resubmits them. It is submitted to a
Mac as an apple_steward speed job (pinned to one Mac, run alone on its GPU).
Exit 0 whenever the sweep itself ran; the verdicts are the lines.
"""
import argparse
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import algos_lane_check as c  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lanes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, default=5400.0)
    ap.add_argument("--deadline", default="", help="UTC, e.g. 2026-09-28T21:00:00Z: no lane starts after it")
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rows = out / "results.jsonl"
    deadline = (datetime.strptime(a.deadline, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
                if a.deadline else None)
    t_batch = time.time()
    backend = c.gpu_backend()
    ib = c.load_harness()
    host = platform.node()
    lanes = [x for x in a.lanes.split(",") if x]
    print(f"SWEEP START {host} {platform.machine()} backend={backend} lanes={len(lanes)} "
          f"store={c.STORE or 'none'}", flush=True)
    failed_bindings = {}
    for lane in lanes:
        now = time.time()
        row = {"lane": lane, "host": host, "started": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        if now - t_batch > a.budget_s or (deadline and now > deadline):
            row.update(verdict="NOT RUN", detail="batch budget spent" if now - t_batch > a.budget_s
                       else "past the deadline")
        elif lane not in ib.LANES:
            row.update(verdict="ERROR", detail="not a registered lane at this commit")
        else:
            d = out / lane
            d.mkdir(parents=True, exist_ok=True)
            log = d / "lane_check.log"
            try:
                needed = c.needed_bindings([lane])[lane]
                row["bindings"] = needed
                bad = [b for b in needed if b in failed_bindings]
                if bad:
                    raise c.Fail(f"BUILDFAIL {bad[0]} did not build earlier in this batch: {failed_bindings[bad[0]]}")
                c.ensure_portable_math(log)
                for b in sorted(needed):
                    try:
                        c.ensure_built({b}, log, publish=True)
                    except c.Fail as exc:
                        failed_bindings[b] = str(exc)[:300]
                        raise c.Fail(f"BUILDFAIL {b}: {exc}")
                gj, cj = d / "gpu.json", d / "cpu.json"
                for kind, j in (("gpu", gj), ("cpu", cj)):
                    try:
                        c.run_arm(kind, lane, backend, "", j, log, moved_ok=True)
                    except c.Fail as exc:
                        raise c.Fail(f"ARMFAIL {exc}")
                verdict, detail = c.compare(ib, lane, gj, cj, "", log, backend)
                row.update(verdict=verdict, detail=detail)
            except c.Fail as exc:
                msg = str(exc)
                if msg.startswith("BUILDFAIL "):
                    row.update(verdict="BUILD FAIL", detail=msg[len("BUILDFAIL "):][:1500])
                elif msg.startswith("ARMFAIL "):
                    row.update(verdict="ARM FAIL", detail=msg[len("ARMFAIL "):][:1500])
                else:
                    row.update(verdict="ERROR", detail=msg[:1500])
            except Exception:
                row.update(verdict="ERROR", detail=traceback.format_exc()[-1500:])
            row["wall_s"] = round(time.time() - now, 1)
        with open(rows, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"SWEEP: {lane}: {row['verdict']}: {row.get('detail', '')[:400]}", flush=True)
    print(f"SWEEP END {host} after {time.time() - t_batch:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
