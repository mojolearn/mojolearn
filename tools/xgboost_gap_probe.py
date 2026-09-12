#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane xgboost-gap: EXPLAIN the taxi lossguide gap before trying to close it.

The gap under test (bench/OPPONENT_REFERENCE.md, September 12, pod
u4elzj1eo486ps): at 1,000,000 taxi rows, 100 trees, depth 6, max_leaves 64,
our IDENTICAL arm fits lossguide in 950.0 ms against XGBoost GPU's 476.2, a
ratio of 1.995x. That is the largest single gap on the GBDT board. Quality is
within a thousandth, so there is nothing to explain it away with.

WHY THIS FILE RATHER THAN gbdt_fairness_probe.py. That probe attacks the
CatBoost claim and its decompose arm is hardcoded to CatBoost. The lens it
established is what is borrowed here: a ratio at one tree count says almost
nothing, because an arm that wins only on the intercept is not a faster
learner. That lane found ours at 86.1 ms fixed + 2.297 ms/tree against
CatBoost's 346.0 ms fixed + 2.809 ms/tree -- a 4.0x advantage before the first
tree and only 1.22x per tree. The same decomposition against XGBoost is the
first question this lane has to answer, because it decides whether the 1.995x
is a startup problem or a boosting-loop problem, and those have nothing to do
with each other.

    python3 tools/xgboost_gap_probe.py decompose --dataset taxi
    python3 tools/xgboost_gap_probe.py decompose --dataset istella
    python3 tools/xgboost_gap_probe.py stages    --dataset taxi

SUBCOMMANDS

`decompose` Each arm timed at 1, 10 and 100 trees, interleaved, in one
        process and one heat window. The slope over trees is the per-tree
        cost and the intercept is what the arm pays before it boosts at all
        (pool build, quantization, CUDA setup, the QuantileDMatrix on their
        side). Both arms get a REAL device drain, so neither clock stops with
        work in flight -- the asymmetry the fairness lane had to rule out.

`stages` Our own per-stage ledger for ONE lossguide fit
        (MOJOLEARN_STAGE_TIMES=1), so the intercept and the slope can be
        attributed to named stages rather than guessed at. Ours only: the
        ledger is our instrumentation and XGBoost has no equivalent to line
        up against it.

Every line begins `XGAP ` so one log can be parsed on the Mac. Nothing here
writes to bench/results; the leg fetches the logs home.

A NOTE ON WHAT THE `hash=` FIELD IS. The harness's digest
(`speed_gbdt_arm.hash_predictions`) is a sha256 over the PREDICTION vector's
dtype, shape and bytes, truncated to 16 hex -- not a hash of the model
structure. It is the right identity witness for this lane anyway: if a change
moved any split or any leaf value, the predictions move and the digest moves
with them. It is a same-device witness, not a cross-vendor one.
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "python"),
           os.path.join(_ROOT, "bench", "speed")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import speed_gbdt_arm as spec              # noqa: E402
import gbdt_fairness_probe as fair         # noqa: E402


#: The lane under investigation, and the default for `--lane`.
LANE = "gbdt-lossguide"


def xgb_arm(lane, cfg, data, n_estimators):
    """The published arm, not a re-spelling of it.

    `spec.xgboost_arms` is what produced the 476.2 ms row, including
    `max_bin = borders + 1` (DEVIATION 1832), the subsample/colsample pins
    (1833), the categorical framing for taxi's declared columns, and
    `max_leaves` under lossguide. Re-deriving those here would be a second
    place for them to drift, so the arm is built from the same function with
    only the tree count moved."""
    sub = dict(cfg)
    sub["n_estimators"] = n_estimators
    arms = spec.xgboost_arms(lane, sub, data, ["gpu"])
    if len(arms) != 1:
        raise RuntimeError("expected one xgboost gpu arm, got %d" % len(arms))
    return arms[0]


def cmd_decompose(args):
    lane = args.lane
    data, cfg, _size = fair.load(args.dataset, args.rows, lane)
    drain = fair.build_device_sync()
    ladder = [int(v) for v in args.trees.split(",")]

    # One untimed warm-up per library, the harness's own contract.
    fair.our_estimator(cfg, data, n_estimators=1).fit(data._ours_X, data._ours_y)
    drain()
    _warm = xgb_arm(lane, cfg, data, 1)
    _warm.fit(_warm.make(), data)
    drain()

    medians = {}
    for n in ladder:
        m = fair.timed(lambda: fair.our_estimator(cfg, data, n_estimators=n)
                       .fit(data._ours_X, data._ours_y), drain, args.reps)
        medians[("ours", n)] = fair.report("ours trees=%d" % n, m)

        arm = xgb_arm(lane, cfg, data, n)
        m = fair.timed(lambda: arm.fit(arm.make(), data), drain, args.reps)
        medians[("xgboost", n)] = fair.report("xgboost trees=%d" % n, m)

    lo, hi = min(ladder), max(ladder)
    for who in ("ours", "xgboost"):
        a, b = medians[(who, lo)], medians[(who, hi)]
        per_tree = (b - a) / float(hi - lo)
        intercept = a - per_tree * lo
        print("XGAP DECOMPOSE lane=%s dataset=%s %s fixed_ms=%.1f "
              "per_tree_ms=%.3f at%d=%.1f at%d=%.1f"
              % (lane, args.dataset, who, intercept, per_tree, lo, a, hi, b),
              flush=True)

    o_lo, x_lo = medians[("ours", lo)], medians[("xgboost", lo)]
    o_hi, x_hi = medians[("ours", hi)], medians[("xgboost", hi)]
    print("XGAP DECOMPOSE-VERDICT lane=%s dataset=%s one_tree_ratio=%.4f "
          "full_ratio=%.4f" % (lane, args.dataset, o_lo / x_lo, o_hi / x_hi),
          flush=True)


def cmd_stages(args):
    """One lossguide fit under our stage ledger, ours only.

    The env is set before the fit because the ledger reads it once per fit
    (`StageTimes`, gbdt/methods/doc_parallel_boosting.mojo). A stage-timed run
    drains per stage, so the table is a SPLIT of the fit, not a timing of it:
    the total here is not comparable to a clean round and is not quoted as
    one."""
    os.environ["MOJOLEARN_STAGE_TIMES"] = "1"
    data, cfg, _size = fair.load(args.dataset, args.rows, args.lane)
    drain = fair.build_device_sync()
    fair.our_estimator(cfg, data, n_estimators=1).fit(data._ours_X, data._ours_y)
    drain()
    print("XGAP STAGES lane=%s dataset=%s trees=%d begin"
          % (args.lane, args.dataset, args.stage_trees), flush=True)
    fair.our_estimator(cfg, data, n_estimators=args.stage_trees).fit(
        data._ours_X, data._ours_y)
    drain()
    print("XGAP STAGES dataset=%s end" % args.dataset, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--dataset", default="taxi", choices=("taxi", "istella"))
        q.add_argument("--rows", type=int, default=1000000)
        # gbdt-depthwise is the DISCRIMINATING cell, not a second opinion.
        # Both policies grow the same number of nodes at depth 6; only the
        # number of SEQUENTIAL expansion steps differs (37 for lossguide on
        # taxi, 6 levels for depthwise). If our per-tree cost is the per-split
        # chain, depthwise must be much cheaper per tree on the same box.
        q.add_argument("--lane", default=LANE,
                       choices=("gbdt-lossguide", "gbdt-depthwise"))

    q = sub.add_parser("decompose")
    common(q)
    q.add_argument("--trees", default="1,10,100")
    q.add_argument("--reps", type=int, default=3)
    q.set_defaults(fn=cmd_decompose)

    q = sub.add_parser("stages")
    common(q)
    q.add_argument("--stage-trees", type=int, default=100)
    q.set_defaults(fn=cmd_stages)

    args = p.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
