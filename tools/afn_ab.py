#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""tools/afn_ab.sh's helper: the summary lines and the output judge.

    afn_ab.py summary --log LOG --tag TAG --lane L --shape S --rounds N \\
        --a-defines "..." --b-defines "..."
        Reads the AFN-AB-RUN lines afn_ab.sh appended to LOG and prints one
        `AFN-AB <tag> arm=A|B lane= shape= median_ms= rounds= defines=''` line
        per arm (the median of that arm's race medians) and
        `AFN-DEF-SUMMARY <tag> A=<ms> B=<ms> ratio=<B/A>`.

    afn_ab.py compare-outputs --tag TAG --lane L --a DIR_A --b DIR_B \\
        [--rel-tol 1e-4] [--loss-tol 1e-3]
        DIR_A and DIR_B are two `bench_board_neural.py race --keep-outputs`
        result directories (one arm each). Train lanes: the per-step losses
        (same init, same batches) of the first and the last step, |a - b| /
        |a| against --loss-tol. Forward lanes: max |a - b| and max |a - b| /
        max |a| of the output against --rel-tol. Prints one
        `AFN-QUALITY <tag> ... status=OK|DIFF|NONE` line (NONE: an output is
        missing) and, when the race JSON carries them, the race's own
        quality numbers per arm (max_rel_err_vs_fp64 on the GEMM lanes,
        mean_nll on the logit lanes).

TOLERANCES (f32 reassociation noise, never a quality bar on their own; the
quality bar is tools/neural_fast_quality.py's rule on the train lanes):
  --rel-tol 1e-4   forward lanes. A free-order f32 fold of K = 4096 (the GEMM)
                   or 2048 tokens x 8 layers (the LM) moves single elements by
                   a few ulps of the largest output; 1e-4 of max|a| is well
                   above that and well below any wrong-kernel error.
  --loss-tol 1e-3  train lanes. The loss is a mean over B x L tokens; two
                   fold orders of the same arithmetic differ by ~1e-6
                   relative at step 1 and drift through the optimizer over the
                   raced steps; 1e-3 relative after a handful of steps is
                   noise, a wrong gradient is 1e-1.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_LANES = ("lm-train-step", "samba-train-step", "mlp-train-step")


def _median(xs):
    return round(statistics.median(xs), 3) if xs else None


def cmd_summary(args):
    runs = {"A": [], "B": []}
    status = {"A": set(), "B": set()}
    pat = re.compile(r"^AFN-AB-RUN %s arm=([AB]) rep=(\d+) lane=%s shape=%s .*median_ms=(\S+)"
                     % (re.escape(args.tag), re.escape(args.lane), re.escape(args.shape)))
    with open(args.log) as fh:
        for line in fh:
            m = pat.match(line.strip())
            if not m:
                continue
            st = re.search(r"status=(\S+)", line)
            status[m.group(1)].add(st.group(1) if st else "none")
            try:
                runs[m.group(1)].append(float(m.group(4)))
            except ValueError:
                pass
    med = {}
    for arm, defs in (("A", args.a_defines), ("B", args.b_defines)):
        med[arm] = _median(runs[arm])
        print("AFN-AB %s arm=%s lane=%s shape=%s median_ms=%s rounds=%d reps=%d runs=%s status=%s "
              "defines='%s'" % (args.tag, arm, args.lane, args.shape, med[arm], args.rounds,
                                len(runs[arm]), [round(x, 1) for x in runs[arm]],
                                ",".join(sorted(status[arm])) or "none", defs), flush=True)
    ratio = (round(med["B"] / med["A"], 4) if med["A"] and med["B"] else None)
    print("AFN-DEF-SUMMARY %s A=%s B=%s ratio=%s (B/A; below 1.0 arm B's median is the lower one)"
          % (args.tag, med["A"], med["B"], ratio), flush=True)
    return 0 if ratio is not None else 1


def _outputs(d):
    """(npz path, race json path, arm) of one --keep-outputs result dir."""
    npz = sorted(glob.glob(os.path.join(d, "*.outputs.npz")))
    js = sorted(p for p in glob.glob(os.path.join(d, "*.json")) if not p.endswith(".params.json"))
    arm = None
    if npz:
        base = os.path.basename(npz[0])[:-len(".outputs.npz")]
        arm = base.split("-", 2)[-1] if base.count("-") >= 2 else None
    return (npz[0] if npz else None), (js[0] if js else None), arm


def _race_quality(js):
    if not js:
        return {}
    try:
        with open(js) as fh:
            r = json.load(fh)
    except (OSError, ValueError):
        return {}
    return r.get("quality") or {}


def cmd_compare_outputs(args):
    import numpy as np
    pa, ja, arm_a = _outputs(args.a)
    pb, jb, arm_b = _outputs(args.b)
    tag, lane = args.tag, args.lane
    for arm, js, label in ((arm_a, ja, "A"), (arm_b, jb, "B")):
        q = _race_quality(js).get(arm or "", {})
        keep = {k: v for k, v in q.items() if k in ("max_rel_err_vs_fp64", "mean_nll",
                                                     "loss_first_step", "loss_last_step")}
        if keep:
            print("AFN-QUALITY-RACE %s arm=%s %s" % (
                tag, label, " ".join("%s=%s" % (k, v) for k, v in sorted(keep.items()))), flush=True)
    if pa is None or pb is None:
        print("AFN-QUALITY %s lane=%s status=NONE missing_outputs=%s" % (
            tag, lane, ",".join(l for l, p in (("A", pa), ("B", pb)) if p is None)), flush=True)
        return 2
    with np.load(pa) as z:
        a = {k: z[k] for k in z.files}
    with np.load(pb) as z:
        b = {k: z[k] for k in z.files}
    if lane in TRAIN_LANES:
        if "losses" not in a or "losses" not in b:
            print("AFN-QUALITY %s lane=%s status=NONE no_losses" % (tag, lane), flush=True)
            return 2
        la, lb = a["losses"].astype(np.float64), b["losses"].astype(np.float64)
        n = min(len(la), len(lb))
        if n == 0:
            print("AFN-QUALITY %s lane=%s status=NONE empty_losses" % (tag, lane), flush=True)
            return 2
        la, lb = la[:n], lb[:n]
        first = abs(la[0] - lb[0]) / (abs(la[0]) or 1.0)
        last = abs(la[-1] - lb[-1]) / (abs(la[-1]) or 1.0)
        worst = float(np.max(np.abs(la - lb) / np.maximum(np.abs(la), 1e-30)))
        ok = max(first, last) <= args.loss_tol and np.all(np.isfinite(lb))
        print("AFN-QUALITY %s lane=%s steps=%d loss_first=%.6f vs %.6f loss_last=%.6f vs %.6f "
              "rel_first=%.3e rel_last=%.3e rel_worst=%.3e tol=%g status=%s"
              % (tag, lane, n, la[0], lb[0], la[-1], lb[-1], first, last, worst, args.loss_tol,
                 "OK" if ok else "DIFF"), flush=True)
        return 0 if ok else 1
    if "y" not in a or "y" not in b:
        print("AFN-QUALITY %s lane=%s status=NONE no_output" % (tag, lane), flush=True)
        return 2
    ya, yb = a["y"].astype(np.float64), b["y"].astype(np.float64)
    if ya.shape != yb.shape:
        print("AFN-QUALITY %s lane=%s status=DIFF shape=%s vs %s" % (
            tag, lane, list(ya.shape), list(yb.shape)), flush=True)
        return 1
    scale = float(np.max(np.abs(ya))) or 1.0
    diff = float(np.max(np.abs(ya - yb)))
    rel = diff / scale
    ok = rel <= args.rel_tol and bool(np.all(np.isfinite(yb)))
    print("AFN-QUALITY %s lane=%s elements=%d max_abs_diff=%.3e max_rel_diff=%.3e scale=%.3e tol=%g "
          "status=%s" % (tag, lane, ya.size, diff, rel, scale, args.rel_tol, "OK" if ok else "DIFF"),
          flush=True)
    return 0 if ok else 1


def build_parser():
    p = argparse.ArgumentParser(prog="afn_ab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--log", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--lane", required=True)
    s.add_argument("--shape", required=True)
    s.add_argument("--rounds", type=int, required=True)
    s.add_argument("--a-defines", default="")
    s.add_argument("--b-defines", default="")
    c = sub.add_parser("compare-outputs")
    c.add_argument("--tag", required=True)
    c.add_argument("--lane", required=True)
    c.add_argument("--a", required=True, help="arm A's race result dir (--keep-outputs)")
    c.add_argument("--b", required=True, help="arm B's race result dir (--keep-outputs)")
    c.add_argument("--rel-tol", type=float, default=1e-4)
    c.add_argument("--loss-tol", type=float, default=1e-3)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return {"summary": cmd_summary, "compare-outputs": cmd_compare_outputs}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
