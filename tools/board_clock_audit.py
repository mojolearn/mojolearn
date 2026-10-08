#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE BOARD'S TWO CLOCKS: which clock every stored board cell is, and both
clocks wherever the stored fields give them (lane board-two-clocks, 2026-10-08).

    python3 tools/board_clock_audit.py --board nvidia-l40s=PATH/board.json \
        [--board amd-mi325x=PATH/board.json ...] --out ~/mojolearn-evidence/board-clocks

Standard library only, no sibling imports (it runs under `python3 -I` against
untrusted board snapshots, and tools/bench_board.py loads it by path to render
the same derived fields on the board).

THE RULE (AGENTS.md, measurement process item 6, Andrew 2026-10-07/08). Our
board fit clock includes the host-to-device copy of the inputs. The torch GPU
columns exclude it (tensors already on the device); cuML/sklearn columns take
NumPy and compare as is. The board shows our whole-operation AND kernel-only
clocks; ratios against a torch GPU arm use the kernel-only clock on both
sides; ratios against everything else use the whole-operation clock. No
opponent is re-scored.

TWO CLOCKS PER CELL, NEVER INVENTED
  whole_ms   the operation including the host-to-device copy of its inputs
  kernel_ms  the same operation with its inputs already on the device
  copy_ms    the difference, ONLY from a stored field:
    * ours: `span.upload_ms_separate` (tools/classical_two_datasets.py
      `_ours_upload_probe`, a separate upload of the same input timed outside
      every clock, recorded since 2026-10-07): kernel_ms = median - copy.
    * a GPU opponent whose inputs went up before its clock:
      `span.upload_ms_untimed` (cuML/cuVS/cuGraph `_cuml_setup`, torch
      `_torch_setup`): whole_ms = median + copy. That upload covers every array
      the arm's worker put on the device before the clock (it can include the
      predict rows), so whole_ms is an upper bound of the fit's own copy, and
      it is never used for an inference cell (the fit inputs are in it too).
    * an arm on the CPU: no host-to-device copy exists, so both clocks are the
      stored median (copy 0, source `cpu-arm`).
  A clock the stored fields cannot give stays None and says why.

WHAT EACH STORED CLOCK IS (`stored`)
  whole    ours everywhere (the public call uploads inside its clock); every
           tree arm (bench/speed/forest_speed_arm.py fits host NumPy, the
           xgboost CUDA predictor uploads with cupy.asarray inside its clock);
           the neural family on both sides (tools/bench_board_neural.py: "host
           inputs to the device, the call, the result back on the host,
           synchronized" -- the neural torch twins INCLUDE the copy, so their
           like-for-like comparison is whole/whole); an opponent whose span
           says input_home=host.
  kernel   a GPU opponent with a pre-clock upload (`upload_ms_untimed`) or a
           span saying input_home=device (tools/bench_board_algos.py torch
           arms upload before the clock and record no upload time: kernel only).
  unknown  no span and no family rule.

RATIOS (`pair_ratio`). `ours / arm` on the opponent's preferred clock (kernel
for a torch GPU arm, whole otherwise) when both sides have it; else on the
other clock when both sides have that one (labelled as the fallback); else
the two stored medians, labelled MIXED with each side's clock. The board never
states a direction; these are numbers and labels only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

#: Opponent libraries whose GPU column is compared on the kernel-only clock
#: (AGENTS.md item 6: "torch ratios use the kernel-only clock"). gpytorch is
#: torch underneath and sets up its tensors on the device the same way.
KERNEL_PREFERRED_LIBRARIES = ("torch", "gpytorch")

WHOLE, KERNEL, UNKNOWN = "whole", "kernel", "unknown"


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _span(cell):
    return ((cell.get("comparability") or {}).get("span")) or {}


def _ok_median(cell):
    m = _num(cell.get("median_ms"))
    if cell.get("status") != "ok" or m is None or m <= 0:
        return None
    return m


def _phase(cell):
    return "infer" if cell.get("phase") == "infer" else "fit"


def stored_clock(cell):
    """(clock, reason): which clock the stored median of this cell is."""
    fam = cell.get("family")
    span = _span(cell)
    lib = cell.get("library")
    if lib == "mojolearn":
        if span.get("input_home") == "device":
            return KERNEL, "our span says input_home=device"
        return WHOLE, "ours: the public call uploads its host inputs inside the clock"
    if cell.get("device") == "cpu":
        return WHOLE, "CPU arm: host in, host out, no device copy"
    if fam == "trees":
        return WHOLE, ("trees: every arm takes host NumPy in the timed call "
                       "(bench/speed/forest_speed_arm.py; xgboost CUDA predict uploads inside its clock)")
    if fam == "neural":
        return WHOLE, ("neural: host inputs to the device inside the clock on every arm "
                       "(tools/bench_board_neural.py SPAN)")
    if _num(span.get("upload_ms_untimed")) is not None:
        return KERNEL, "inputs uploaded before the clock (upload_ms_untimed recorded)"
    home = span.get("input_home")
    if home == "device":
        return KERNEL, "span says input_home=device (uploaded before the clock, no upload time recorded)"
    if home == "host":
        return WHOLE, "span says input_home=host (the copy is inside the clock)"
    return UNKNOWN, "no clock span recorded and no family rule"


def cell_clock(cell):
    """Both clocks of one stored cell, from stored fields only."""
    stored, why = stored_clock(cell)
    m = _ok_median(cell)
    span = _span(cell)
    out = {"stored": stored, "stored_why": why, "median_ms": m, "whole_ms": None,
           "kernel_ms": None, "copy_ms": None, "copy_source": None, "copy_share": None,
           "note": None}
    if span.get("pre_clock_fit"):
        out["note"] = "fit before its clock (pre_clock_fit)"
    if m is None:
        out["derivable"] = "none"
        out["note"] = "no completed time (status %s)" % str(cell.get("status", "?")).split("(")[0]
        return out
    lib = cell.get("library")
    infer = _phase(cell) == "infer"
    if stored == WHOLE:
        out["whole_ms"] = m
        if lib != "mojolearn" and cell.get("device") == "cpu":
            out.update(kernel_ms=m, copy_ms=0.0, copy_source="cpu-arm")
        elif lib == "mojolearn" and not infer:
            up = _num(span.get("upload_ms_separate"))
            if up is not None:
                if 0 <= up < m:
                    out.update(kernel_ms=m - up, copy_ms=up, copy_source="upload_ms_separate")
                else:
                    out["note"] = ("upload_ms_separate %.3f ms is not below the scored median; "
                                   "kernel clock withheld" % up)
    elif stored == KERNEL:
        out["kernel_ms"] = m
        up = _num(span.get("upload_ms_untimed"))
        if up is not None and not infer and up >= 0:
            out.update(whole_ms=m + up, copy_ms=up, copy_source="upload_ms_untimed")
        elif up is not None and infer:
            out["note"] = "upload_ms_untimed covers the fit inputs too; the predict copy is not separable"
    if out["copy_ms"] is not None:
        out["copy_share"] = out["copy_ms"] / m
    have = (out["whole_ms"] is not None, out["kernel_ms"] is not None)
    out["derivable"] = {(True, True): "both", (True, False): "whole",
                        (False, True): "kernel", (False, False): "none"}[have]
    return out


def preferred_clock(cell):
    """The clock a ratio against this opponent cell is read on."""
    if cell.get("library") in KERNEL_PREFERRED_LIBRARIES and cell.get("device") != "cpu":
        return KERNEL
    return WHOLE


def pair_ratio(ours_clock, opp_clock, opp_cell):
    """`ours / arm` with the clock it is read on. Never invents a clock."""
    pref = preferred_clock(opp_cell)
    other = WHOLE if pref == KERNEL else KERNEL
    for clock, kind in ((pref, "preferred"), (other, "fallback")):
        a, b = ours_clock.get(clock + "_ms"), opp_clock.get(clock + "_ms")
        if a is not None and b is not None and b > 0:
            return {"value": a / b, "clock": clock, "kind": kind, "preferred": pref,
                    "label": "%s/%s" % (clock, clock) + ("" if kind == "preferred" else
                                                         " (%s not derivable)" % pref)}
    a, b = ours_clock.get("median_ms"), opp_clock.get("median_ms")
    if a is not None and b is not None and b > 0:
        return {"value": a / b, "clock": "mixed", "kind": "mixed", "preferred": pref,
                "label": "MIXED ours %s / arm %s" % (ours_clock["stored"], opp_clock["stored"])}
    return {"value": None, "clock": None, "kind": "none", "preferred": pref, "label": "-"}


def _ours(cells, mode):
    for c in cells:
        if c.get("library") == "mojolearn" and c.get("mode") == mode and _ok_median(c) is not None \
                and c.get("device") != "cpu":
            return c
    return None


def annotate_cells(cells):
    """Set `clock` on every cell and `ratio_ours_identical_clock` /
    `ratio_ours_fast_clock` on every opponent cell, in place. The stored
    `ratio_ours_*_over` fields are left exactly as they are."""
    for c in cells:
        c["clock"] = cell_clock(c)
    groups = {}
    for c in cells:
        groups.setdefault(c.get("batch") if _phase(c) == "infer" else None, []).append(c)
    for g in groups.values():
        ours = {m: _ours(g, m) for m in ("identical", "fast")}
        for c in g:
            for m in ("identical", "fast"):
                c["ratio_ours_%s_clock" % m] = None
                o = ours[m]
                if c.get("library") == "mojolearn" or o is None or _ok_median(c) is None:
                    continue
                c["ratio_ours_%s_clock" % m] = pair_ratio(o["clock"], c["clock"], c)
    return cells


def annotate_result(result):
    """annotate_cells over every race (and page-only extra race) of a board."""
    for group in ("races", "extra_races"):
        for rr in (result.get(group) or {}).values():
            annotate_cells(rr.get("cells") or [])
            annotate_cells(rr.get("infer_cells") or [])
    return result


# ---------------------------------------------------------------------------
# Audit (per board: a markdown table, a json, the rerun candidates)
# ---------------------------------------------------------------------------

def _f(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return "%.*f" % (nd, v)
    return str(v).replace("|", "/")


def _arm_ms(cell):
    """The arm's own stored time in this race: warm-up plus timed rounds."""
    t = sum(x for x in (cell.get("times_ms") or []) if _num(x) is not None)
    w = _num(cell.get("warmup_ms")) or 0.0
    return t + w


def audit_board(name, result):
    annotate_result(result)
    rows, pairs = [], []
    for group in ("races", "extra_races"):
        for rid in sorted(result.get(group) or {}):
            rr = result[group][rid]
            for kind in ("cells", "infer_cells"):
                cells = rr.get(kind) or []
                for c in cells:
                    k = c["clock"]
                    row = {"board": name, "race": rid, "phase": _phase(c), "family": c.get("family"),
                           "lane": c.get("lane"), "dataset": c.get("dataset"), "batch": c.get("batch"),
                           "arm": c.get("arm"), "library": c.get("library"), "device": c.get("device"),
                           "mode": c.get("mode"), "status": str(c.get("status", "?")).split("(")[0],
                           "arm_ms": _arm_ms(c), "race_wall_s": _num(rr.get("wall_s"))}
                    row.update({x: k[x] for x in ("stored", "stored_why", "median_ms", "whole_ms",
                                                  "kernel_ms", "copy_ms", "copy_source", "copy_share",
                                                  "derivable", "note")})
                    rows.append(row)
                    for mode in ("identical", "fast"):
                        r = c.get("ratio_ours_%s_clock" % mode)
                        if not r:
                            continue
                        ours = _ours([x for x in cells if x.get("batch") == c.get("batch")], mode)
                        pairs.append({"race": rid, "phase": row["phase"], "batch": c.get("batch"),
                                      "arm": c.get("arm"), "library": c.get("library"), "ours_mode": mode,
                                      "preferred": r["preferred"], "kind": r["kind"], "label": r["label"],
                                      "value": r["value"],
                                      "ours_arm": ours.get("arm") if ours else None,
                                      "ours_derivable": ours["clock"]["derivable"] if ours else None,
                                      "ours_arm_ms": _arm_ms(ours) if ours else None,
                                      "race_wall_s": _num(rr.get("wall_s"))})
    return rows, pairs


def summarize(rows, pairs):
    s = {"cells": len(rows), "fit_cells": sum(r["phase"] == "fit" for r in rows),
         "infer_cells": sum(r["phase"] == "infer" for r in rows)}
    for d in ("both", "whole", "kernel", "none"):
        s["derivable_" + d] = sum(r["derivable"] == d for r in rows)
    s["derivable_one"] = s["derivable_whole"] + s["derivable_kernel"]
    s["none_no_time"] = sum(r["derivable"] == "none" and r["median_ms"] is None for r in rows)
    s["none_with_time"] = sum(r["derivable"] == "none" and r["median_ms"] is not None for r in rows)
    s["stored"] = {k: sum(r["stored"] == k for r in rows) for k in (WHOLE, KERNEL, UNKNOWN)}
    s["pairs"] = {k: sum(p["kind"] == k for p in pairs) for k in ("preferred", "fallback", "mixed", "none")}
    return s


def rerun_candidates(rows, pairs):
    """(1) cells with a completed time and NEITHER clock (the clock kind itself
    is unknown): the only cells a rerun of that arm would fix; (2) the MIXED
    pairs: no common clock, and since opponents are never re-scored the side
    to rerun is ours, with a separate-upload probe in its driver."""
    neither = [r for r in rows if r["derivable"] == "none" and r["median_ms"] is not None]
    mixed = [p for p in pairs if p["kind"] == "mixed"]
    ours_needed = {}
    for p in mixed:
        key = (p["race"], p["phase"], p["batch"], p["ours_arm"])
        ours_needed[key] = p
    est = {"neither_cells": len(neither),
           "neither_arm_s": sum(r["arm_ms"] for r in neither) / 1000.0,
           "neither_race_wall_s": sum({r["race"]: r["race_wall_s"] or 0.0 for r in neither}.values()),
           "mixed_pairs": len(mixed), "our_cells_to_rerun": len(ours_needed),
           "our_arm_s": sum((p["ours_arm_ms"] or 0.0) for p in ours_needed.values()) / 1000.0,
           "our_races": len({k[0] for k in ours_needed}),
           "our_race_wall_s_upper": sum({k[0]: p["race_wall_s"] or 0.0
                                         for k, p in ours_needed.items()}.values())}
    by_lane = {}
    for (race, phase, batch, arm), p in sorted(ours_needed.items()):
        fam_lane = "/".join(race.split("/")[:2])
        by_lane.setdefault(fam_lane, set()).add(race.split("/")[2] if race.count("/") >= 2 else "?")
    est["our_lanes"] = {k: sorted(v) for k, v in sorted(by_lane.items())}
    return neither, mixed, est


def render_md(name, rows, pairs, summary, neither, mixed, est):
    L = ["# Board clocks: %s" % name, "",
         "Generated by tools/board_clock_audit.py from the stored board.json; nothing re-measured.", "",
         "## Summary", "",
         "| field | value |", "|---|---|"]
    for k in ("cells", "fit_cells", "infer_cells", "derivable_both", "derivable_one", "derivable_whole",
              "derivable_kernel", "derivable_none", "none_no_time", "none_with_time"):
        L.append("| %s | %s |" % (k, summary[k]))
    L.append("| stored clock | %s |" % ", ".join("%s %d" % kv for kv in summary["stored"].items()))
    L.append("| opponent ratios | %s |" % ", ".join("%s %d" % kv for kv in summary["pairs"].items()))
    L += ["", "## Rerun estimate", "",
          "- cells with a time and neither clock: %d (arm time %.1f s; their races' stored wall %.1f s)"
          % (est["neither_cells"], est["neither_arm_s"], est["neither_race_wall_s"]),
          "- MIXED opponent ratios (no common clock): %d, needing %d of our cells in %d races "
          "(our arm time %.1f s; those races' stored wall %.1f s, an upper bound)"
          % (est["mixed_pairs"], est["our_cells_to_rerun"], est["our_races"], est["our_arm_s"],
             est["our_race_wall_s_upper"]), ""]
    for k, v in est["our_lanes"].items():
        L.append("  - %s: %s" % (k, ", ".join(v)))
    L += ["", "## Cells", "",
          "| phase | race | batch | arm | device | status | stored | median ms | whole ms | kernel ms | "
          "copy ms | copy source | copy share | derivable | note |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["phase"], r["race"], _f(r["batch"]), r["arm"], _f(r["device"]), r["status"], r["stored"],
            _f(r["median_ms"]), _f(r["whole_ms"]), _f(r["kernel_ms"]), _f(r["copy_ms"], 2),
            _f(r["copy_source"]), _f(r["copy_share"], 3), r["derivable"], _f(r["note"])))
    L += ["", "## Opponent ratios and their clock", "",
          "| phase | race | batch | arm | ours | preferred | ratio | clock | our cell derivable |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p in pairs:
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            p["phase"], p["race"], _f(p["batch"]), p["arm"], p["ours_mode"], p["preferred"], _f(p["value"], 3),
            p["label"], _f(p["ours_derivable"])))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--board", action="append", required=True, metavar="NAME=PATH",
                    help="a board name and its board.json (repeatable)")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    overall = {}
    for spec in args.board:
        name, _, path = spec.partition("=")
        if not path:
            ap.error("--board wants NAME=PATH, got %r" % spec)
        with open(path) as fh:
            result = json.load(fh)
        rows, pairs = audit_board(name, result)
        summary = summarize(rows, pairs)
        neither, mixed, est = rerun_candidates(rows, pairs)
        with open(os.path.join(args.out, name + ".json"), "w") as fh:
            json.dump({"board": name, "source": os.path.abspath(path), "summary": summary,
                       "rerun_estimate": est, "neither": neither, "cells": rows, "pairs": pairs},
                      fh, indent=1, sort_keys=True)
        with open(os.path.join(args.out, name + ".md"), "w") as fh:
            fh.write(render_md(name, rows, pairs, summary, neither, mixed, est))
        overall[name] = {"summary": summary, "rerun_estimate": est}
        print("CLOCKS board=%s cells=%d both=%d one=%d (whole %d, kernel %d) none=%d "
              "(no_time %d, with_time %d) pairs=%s mixed_our_cells=%d our_arm_s=%.1f"
              % (name, summary["cells"], summary["derivable_both"], summary["derivable_one"],
                 summary["derivable_whole"], summary["derivable_kernel"], summary["derivable_none"],
                 summary["none_no_time"], summary["none_with_time"],
                 ",".join("%s:%d" % kv for kv in summary["pairs"].items()),
                 est["our_cells_to_rerun"], est["our_arm_s"]))
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(overall, fh, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
