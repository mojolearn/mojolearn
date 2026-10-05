#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Fold an M3 IDENTICAL sweep (~/mq/out/race-ident-*) into the canonical M3
board: replace each race's `ours` IDENTICAL cell, recompute ratios, re-render.

    python3 tools/af_board_ident_update.py BOARD_DIR SWEEP_GLOB HEAD [--dry-run]

Box (M3). Keeps BOARD_DIR/board.before-ident-<date>.json. Opponent and FAST
cells are untouched. A sweep row that errored keeps the old cell and is listed.
Then re-renders BOARD.md, BOARD_M3_FAST.md and BOARD_M3_IDENTICAL.md (tools/af_board_render.py)
and runs the render check.
"""
import copy, glob, json, os, shutil, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_board as bb

bdir, pat, head = sys.argv[1], sys.argv[2], sys.argv[3]
dry = "--dry-run" in sys.argv
path = os.path.join(bdir, "board.json")
board = json.load(open(path))
index = {}
for rid, rec in board["races"].items():
    for c in rec.get("cells", []):
        if c.get("library") == "mojolearn" and c.get("mode") == "identical":
            index[(c["lane"], c["dataset"])] = (rid, c)
done, kept, missing = [], [], []
for d in sorted(glob.glob(pat)):
    res = glob.glob(os.path.join(d, "A-1", "res", "*.json"))
    if not res:
        missing.append(os.path.basename(d)); continue
    r = json.load(open(res[0])); o = r["arms"].get("ours") or {}
    key = (r["lane"], r["dataset"])
    if key not in index:
        missing.append("%s/%s (no board race)" % key); continue
    rid, c = index[key]
    if o.get("status") != "ok" or not o.get("median_ms"):
        kept.append("%s/%s" % key); continue
    old = c["median_ms"]
    # status too: a cell that was REFUSED in 0.8.34 and now has a time is ok (it was left REFUSED before)
    c.update(status="ok", median_ms=o["median_ms"], min_ms=min(o["ms"]), max_ms=max(o["ms"]), rounds=len(o["ms"]),
             quality=(r.get("quality") or {}).get("ours", c.get("quality")),
             hash=(o.get("digests") or [c.get("hash")])[-1],
             source={"sweep": os.path.basename(d), "head": head, "finished": r.get("finished"),
                     "previous_median_ms": old, "previous_hash": c.get("hash"),
                     "previous_status": c.get("status")})
    bb.add_ratios(board["races"][rid]["cells"])
    done.append((key, old, o["median_ms"]))
print("updated=%d kept_old(error)=%d unmatched=%d" % (len(done), len(kept), len(missing)))
for k in kept: print("KEPT", k)
for m in missing: print("UNMATCHED", m)
if not dry:
    shutil.copy2(path, os.path.join(bdir, "board.before-ident-%s.json" % time.strftime("%Y%m%d")))
    board["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    board.setdefault("box_history", []).append({"ident_sweep": pat, "head": head, "updated_cells": len(done)})
    with open(path, "w") as fh:
        json.dump(board, fh, indent=1)
    # board.json is the one M3 board: re-render BOARD.md and both docs pages, then check them
    import af_board_render as R
    R.write_all(bdir, R.DOCS, board)
    bad = R.check(bdir, R.DOCS)
    for b in bad:
        print(b)
    if bad:
        sys.exit("RENDER CHECK FAIL: the pages do not match board.json")
    print("RENDER CHECK OK")
