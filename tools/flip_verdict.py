#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The default-flip verdict of ENGINEERING_RULES.md section 9, from FSPEED logs.

    python3 tools/flip_verdict.py --taxi-before B.log --taxi-after A.log \
        --istella-before B.log --istella-after A.log [--lane L] [--arm A] [--rows N]
    python3 tools/flip_verdict.py --self-test
The last line is FLIP or NO FLIP. Exit 0 FLIP, 1 NO FLIP, 2 parse or selection error.
"""

import argparse
import collections
import contextlib
import io
import math
import os
import re
import statistics
import sys
import tempfile

EXIT_FLIP, EXIT_NO_FLIP, EXIT_ERROR = 0, 1, 2
DATASETS = ("taxi", "istella")
SIDES = ("before", "after")

# Only these two line kinds are read. FSPEED-WARMUP, -HEADER, -ORDER, -NOTE
# and the rest carry other prefixes, so a warm-up never enters a median.
LINE_RE = re.compile(r"^(FSPEED|FSPEED-ACC) (.*)$")
KV_RE = re.compile(r"(\w+)=(\S*)")
# tools/speed_gbdt_arm.py tags a shape "<dataset>-<rows>x<features>". The
# classical Mojo lanes start the tag at "<rows>x<features>". Neural tags
# carry no row count and no dataset name.
NAMED_SHAPE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)-\d+x\d+")
ROWS_RE = re.compile(r"(?:^|-)(\d+)x\d+")

DEFAULT_QUALITY_TOL = 1e-4
LOWER_IS_BETTER = frozenset((
    "logloss", "rmse", "mse", "mae", "loss", "nll", "perplexity", "ppl",
    "bpb", "bpc", "error"))
HIGHER_IS_BETTER = frozenset((
    "auc", "accuracy", "acc", "ndcg", "map", "mrr", "recall", "precision",
    "f1", "r2"))

Round = collections.namedtuple("Round", "lane arm shape ms hash where")
Acc = collections.namedtuple("Acc", "lane arm shape metric value where")


class FlipError(Exception):
    """A parse or selection problem. Exit 2, never a verdict."""


# ---------------------------------------------------------------- parsing

def _fields(kind, rest, need, where):
    kv = dict(KV_RE.findall(rest))
    missing = [k for k in need if not kv.get(k)]
    if missing:
        raise FlipError("%s: %s line lacks %s" % (where, kind, ", ".join(missing)))
    return kv


def _number(kv, key, where):
    try:
        return float(kv[key])
    except ValueError:
        raise FlipError("%s: %s=%s is not a number" % (where, key, kv[key]))


def parse_files(paths):
    """(rounds, accs) from the FSPEED and FSPEED-ACC lines of `paths`."""
    rounds, accs = [], []
    for path in paths:
        try:
            with open(path, "r", errors="replace") as fh:
                lines = fh.read().splitlines()
        except OSError as exc:
            raise FlipError("cannot read %s: %s" % (path, exc))
        # FSPEED-ACC carries no shape. speed_gbdt_arm.py times one shape per
        # process and prints accuracy after that process's rounds, so an ACC
        # line belongs to the shape of the last FSPEED line of its lane and
        # arm earlier in the same file.
        last_shape = {}
        for no, line in enumerate(lines, 1):
            m = LINE_RE.match(line)
            if not m:
                continue
            kind, rest = m.groups()
            where = "%s:%d" % (path, no)
            if kind == "FSPEED":
                kv = _fields(kind, rest, ("lane", "arm", "shape", "ms"), where)
                ms = _number(kv, "ms", where)
                if not (math.isfinite(ms) and ms >= 0.0):
                    raise FlipError("%s: ms=%s is not a finite nonnegative time"
                                    % (where, kv["ms"]))
                last_shape[(kv["lane"], kv["arm"])] = kv["shape"]
                rounds.append(Round(kv["lane"], kv["arm"], kv["shape"], ms,
                                    kv.get("hash") or "-", where))
            else:
                kv = _fields(kind, rest, ("lane", "arm", "metric", "value"), where)
                accs.append(Acc(kv["lane"], kv["arm"],
                                last_shape.get((kv["lane"], kv["arm"])),
                                kv["metric"], _number(kv, "value", where), where))
    return rounds, accs


# -------------------------------------------------------------- selection

def choose(label, present, wanted, default=None):
    present = sorted(present)
    if wanted is not None:
        if wanted in present:
            return wanted
        raise FlipError("no FSPEED line has %s=%s; present: %s"
                        % (label, wanted, ", ".join(present) or "none"))
    if len(present) == 1:
        return present[0]
    if default is not None:
        return default
    raise FlipError("the logs carry %d %ss, pass --%s; present: %s"
                    % (len(present), label, label, ", ".join(present)))


def default_arm(arms):
    """Our arm when its name leaves no doubt: `ours`, else the one ours*identical* arm.

    The mode rides only in FSPEED-HEADER, which this tool does not read, so
    before and after must come from runs in the same mode.
    """
    if "ours" in arms:
        return "ours"
    hits = [a for a in arms if a.startswith("ours") and "identical" in a]
    return hits[0] if len(hits) == 1 else None


def rows_of(shape):
    m = ROWS_RE.search(shape)
    return int(m.group(1)) if m else None


def select_side(ds, side, lane, arm, rounds, accs, args):
    """(shape, rounds, accs) of one side at exactly one shape."""
    wanted = getattr(args, "%s_shape" % ds)
    keep = [r for r in rounds
            if (args.rows is None or rows_of(r.shape) == args.rows)
            and (wanted is None or r.shape == wanted)]
    shapes = sorted({r.shape for r in keep})
    if len(shapes) != 1:
        raise FlipError(
            "--%s-%s: %s for lane=%s arm=%s, select with --rows or --%s-shape; "
            "shapes present: %s"
            % (ds, side, "%d shapes" % len(shapes) if shapes
               else "no shape matches the selectors", lane, arm, ds,
               ", ".join(sorted({r.shape for r in rounds}))))
    shape = shapes[0]
    named = NAMED_SHAPE_RE.match(shape)
    if named and ds not in named.group(1).lower():
        raise FlipError("--%s-%s: shape=%s is not a %s shape, a log of another "
                        "dataset was passed" % (ds, side, shape, ds))
    orphans = [a for a in accs if a.shape is None]
    if orphans:
        raise FlipError("%s: FSPEED-ACC precedes every FSPEED line of lane=%s "
                        "arm=%s in its file, so its shape is unknown"
                        % (orphans[0].where, lane, arm))
    return (shape, [r for r in keep if r.shape == shape],
            [a for a in accs if a.shape == shape])


# ---------------------------------------------------------------- quality

def parse_directions(items):
    out = {}
    for item in items:
        name, _, way = item.partition("=")
        if not name or way not in ("lower", "higher"):
            raise FlipError("--metric-direction wants METRIC=lower or "
                            "METRIC=higher, got %r" % item)
        out[name] = way == "lower"
    return out


def lower_is_better(metric, overrides):
    if metric in overrides:
        return overrides[metric]
    base = re.split(r"[@:]", metric.lower())[0]
    if base in LOWER_IS_BETTER:
        return True
    if base in HIGHER_IS_BETTER:
        return False
    raise FlipError("metric=%s has no known direction; pass --metric-direction "
                    "%s=lower or %s=higher" % (metric, metric, metric))


def judge_quality(ds, before, after, tol_rel, overrides):
    """(worse, unmeasured) for one dataset, one printed line per metric."""
    (b_rounds, b_accs), (a_rounds, a_accs) = before, after
    b_metrics, a_metrics = {}, {}
    for accs, into in ((b_accs, b_metrics), (a_accs, a_metrics)):
        for a in accs:
            into.setdefault(a.metric, []).append(a)
    if not b_metrics and not a_metrics:
        hashes = {r.hash for r in b_rounds} | {r.hash for r in a_rounds}
        if len(hashes) == 1 and "-" not in hashes:
            print("quality %s: no FSPEED-ACC lines; every round before and after "
                  "hashed %s, the outputs are equal" % (ds, sorted(hashes)[0]))
            return False, False
        print("quality %s: no FSPEED-ACC lines and the round hashes do not show "
              "equal outputs, quality is unmeasured" % ds)
        return False, True
    worse = unmeasured = False
    for metric in sorted(set(b_metrics) | set(a_metrics)):
        if metric not in b_metrics or metric not in a_metrics:
            print("quality %s %s: only in the %s logs, unmeasured on the other side"
                  % (ds, metric, "before" if metric in b_metrics else "after"))
            unmeasured = True
            continue
        lower = lower_is_better(metric, overrides)
        b_vals = [x.value for x in b_metrics[metric]]
        a_vals = [x.value for x in a_metrics[metric]]
        for x in b_metrics[metric]:
            if not math.isfinite(x.value):
                raise FlipError("%s: before value of %s is not finite, nothing to "
                                "compare against" % (x.where, metric))
        b = statistics.median(b_vals)
        # Several before values (repeat runs of the same arm) are the
        # round-to-round noise, and a change inside that spread is not worse.
        tol = max(tol_rel * abs(b), max(b_vals) - min(b_vals))
        if all(math.isfinite(v) for v in a_vals):
            a = statistics.median(a_vals)
            bad = ((a - b) if lower else (b - a)) > tol
            shown = "after=%.6f delta=%+.6f" % (a, a - b)
        else:
            bad = True
            shown = "after=not-finite"
        print("quality %s %s: before=%.6f %s n=%d/%d %s-is-better tol=%.3g %s"
              % (ds, metric, b, shown, len(b_vals), len(a_vals),
                 "lower" if lower else "higher", tol, "WORSE" if bad else "ok"))
        worse = worse or bad
    return worse, unmeasured


# ---------------------------------------------------------------- verdict

def judge_dataset(ds, lane, arm, parsed, args, overrides):
    """(ratio, worse, unmeasured); ratio is None when the dataset is missing."""
    sides = {}
    for side in SIDES:
        paths, rounds, accs = parsed[(ds, side)]
        rounds = [r for r in rounds if r.lane == lane and r.arm == arm]
        accs = [a for a in accs if a.lane == lane and a.arm == arm]
        if not rounds:
            print("time %s: %s" % (ds, "no --%s-%s log given" % (ds, side)
                                   if not paths else
                                   "no FSPEED line for lane=%s arm=%s in the %s logs"
                                   % (lane, arm, side)))
            return None, False, False
        sides[side] = select_side(ds, side, lane, arm, rounds, accs, args)
    (b_shape, b_rounds, b_accs) = sides["before"]
    (a_shape, a_rounds, a_accs) = sides["after"]
    if b_shape != a_shape:
        raise FlipError("%s: before timed shape=%s and after timed shape=%s"
                        % (ds, b_shape, a_shape))
    b_med = statistics.median([r.ms for r in b_rounds])
    a_med = statistics.median([r.ms for r in a_rounds])
    if b_med <= 0.0 or a_med <= 0.0:
        raise FlipError("%s: a median time is not positive (before %.3f ms, "
                        "after %.3f ms)" % (ds, b_med, a_med))
    ratio = a_med / b_med
    print("time %s shape=%s: before median=%.3f ms n=%d, after median=%.3f ms "
          "n=%d, ratio=%.3f" % (ds, b_shape, b_med, len(b_rounds), a_med,
                                len(a_rounds), ratio))
    worse, unmeasured = judge_quality(ds, (b_rounds, b_accs), (a_rounds, a_accs),
                                      args.quality_tol, overrides)
    return ratio, worse, unmeasured


def _fmt(x):
    return "-" if x is None else "%.3f" % x


def verdict(args):
    if not (math.isfinite(args.quality_tol) and args.quality_tol >= 0.0):
        raise FlipError("--quality-tol must be a finite number >= 0")
    overrides = parse_directions(args.metric_direction)
    parsed = {}
    for ds in DATASETS:
        for side in SIDES:
            paths = getattr(args, "%s_%s" % (ds, side))
            parsed[(ds, side)] = (paths,) + parse_files(paths)
    everything = [r for (_, rounds, _) in parsed.values() for r in rounds]
    if not everything:
        raise FlipError("no FSPEED line in any input log")
    lane = choose("lane", {r.lane for r in everything}, args.lane)
    arms = {r.arm for r in everything if r.lane == lane}
    arm = choose("arm", arms, args.arm, default_arm(arms))
    print("selected lane=%s arm=%s" % (lane, arm))

    ratios, missing, quality = {}, [], []
    for ds in DATASETS:
        ratio, worse, unmeasured = judge_dataset(ds, lane, arm, parsed, args,
                                                 overrides)
        ratios[ds] = ratio
        if ratio is None or unmeasured:
            missing.append("missing:" + ds)
        if worse:
            quality.append("quality:" + ds)
    reasons = missing + quality
    geomean = None
    if all(ratios[ds] is not None for ds in DATASETS):
        geomean = math.sqrt(ratios["taxi"] * ratios["istella"])
        print("time geomean=%.6f (decided unrounded, below 1 wins)" % geomean)
        if not geomean < 1.0:
            reasons.append("time")
    values = (_fmt(geomean), _fmt(ratios["taxi"]), _fmt(ratios["istella"]))
    if reasons:
        if len(reasons) > 1:
            print("reasons: " + " ".join(reasons))
        print("NO FLIP geomean=%s taxi=%s istella=%s reason=%s" % (values + (reasons[0],)))
        return EXIT_NO_FLIP
    print("FLIP geomean=%s taxi=%s istella=%s quality=ok" % values)
    return EXIT_FLIP


def build_parser():
    p = argparse.ArgumentParser(
        description="ENGINEERING_RULES.md section 9 default-flip verdict. Reads "
                    "only FSPEED and FSPEED-ACC lines.")
    for ds in DATASETS:
        for side in SIDES:
            p.add_argument("--%s-%s" % (ds, side), nargs="+", action="extend",
                           default=[], metavar="LOG")
    p.add_argument("--lane", help="lane= value, required when the logs carry several")
    p.add_argument("--arm", help="arm= value; default ours, or the one "
                                 "ours*identical* arm")
    p.add_argument("--rows", type=int, help="row count in the shape tag")
    p.add_argument("--taxi-shape", help="exact shape= tag for the taxi logs")
    p.add_argument("--istella-shape", help="exact shape= tag for the Istella-S logs")
    p.add_argument("--quality-tol", type=float, default=DEFAULT_QUALITY_TOL,
                   help="relative tolerance on a quality metric; the before "
                        "spread across repeat runs widens it (default %g)"
                        % DEFAULT_QUALITY_TOL)
    p.add_argument("--metric-direction", action="append", default=[],
                   metavar="METRIC=lower|higher",
                   help="direction of better for a metric this tool does not know")
    p.add_argument("--self-test", action="store_true",
                   help="check the verdicts on synthetic logs in a temp dir")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    try:
        return verdict(args)
    except FlipError as exc:
        print("flip_verdict: ERROR %s" % exc, file=sys.stderr)
        return EXIT_ERROR


# -------------------------------------------------------------- self-test

TAXI, ISTELLA = "taxi-1000000x16", "istella-1000000x220"
TAXI_ACC = (("logloss", 0.525912), ("auc", 0.616793))
ISTELLA_ACC = (("logloss", 0.145578), ("auc", 0.964548))


def _log(shape, ms, acc, lane="rf", arm="ours", digest="0123456789abcdef",
         opp_logloss=0.5):
    """Lines in the speed_gbdt_arm.py format, with an opponent arm beside ours."""
    lines = [
        "FSPEED-NOTE lane=%s arms=%s metric=devices delta=1.000000 reason=self-test"
        % (lane, arm),
        "FSPEED-HEADER family=forest lane=%s arm=%s mode=IDENTICAL device=selftest "
        "rounds=%d size=shipped" % (lane, arm, len(ms)),
        # Far slower than any round, so a warm-up leaking into a median moves
        # every ratio below.
        "FSPEED-WARMUP lane=%s arm=%s shape=%s ms=%.3f" % (lane, arm, shape, 1.0e7),
    ]
    for i, t in enumerate(ms, 1):
        lines.append("FSPEED-ORDER lane=%s round=%d arms=%s,catboost-gpu" % (lane, i, arm))
        lines.append("FSPEED lane=%s arm=%s shape=%s round=%d ms=%.3f hash=%s"
                     % (lane, arm, shape, i, t, digest))
        # The opponent is flat across before and after: picking it by
        # mistake gives ratio 1.000 and no FLIP.
        lines.append("FSPEED lane=%s arm=catboost-gpu shape=%s round=%d ms=50.000 "
                     "hash=fedcba9876543210" % (lane, shape, i))
    for metric, value in acc:
        lines.append("FSPEED-ACC lane=%s arm=%s metric=%s value=%.6f"
                     % (lane, arm, metric, value))
    lines.append("FSPEED-ACC lane=%s arm=catboost-gpu metric=logloss value=%.6f"
                 % (lane, opp_logloss))
    lines.append("FSPEED-SCALE stage=after rows=1 features=1 input_mib=0.0 "
                 "large_candidate=false")
    return lines


def self_test():
    results = []
    with tempfile.TemporaryDirectory(prefix="flip_verdict_") as tmp:
        def write(name, *blocks):
            path = os.path.join(tmp, name)
            with open(path, "w") as fh:
                for block in blocks:
                    fh.write("\n".join(block) + "\n")
            return path

        def run(label, argv, want_code, want_last=None):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                try:
                    code = main(argv)
                except SystemExit as exc:
                    code = exc.code
            lines = out.getvalue().strip().splitlines()
            last = lines[-1] if lines else ""
            ok = code == want_code and (want_last is None or last == want_last)
            results.append(ok)
            print("SELF-TEST %s %s" % ("ok  " if ok else "FAIL", label))
            if not ok:
                print("    exit %r, last line %r, stderr %r" % (code, last, err.getvalue().strip()))
                print("    wanted exit %r, last line %r" % (want_code, want_last))

        # Medians 1000 -> 900 (0.900) and 5000 -> 4000 (0.800). Means, or a
        # leaked warm-up, give other ratios.
        tb = write("taxi.before.log", _log(TAXI, [1000, 1100, 990], TAXI_ACC))
        ta = write("taxi.after.log", _log(TAXI, [900, 905, 800], TAXI_ACC, opp_logloss=0.9))
        ib = write("istella.before.log", _log(ISTELLA, [5000, 5050, 4950], ISTELLA_ACC))
        ia = write("istella.after.log", _log(ISTELLA, [4000, 4040, 3960], ISTELLA_ACC,
                                             opp_logloss=0.9))
        flip = "FLIP geomean=0.849 taxi=0.900 istella=0.800 quality=ok"

        def four(taxi_before=tb, taxi_after=ta, istella_before=ib, istella_after=ia):
            argv = []
            for flag, paths in (("--taxi-before", taxi_before), ("--taxi-after", taxi_after),
                                ("--istella-before", istella_before),
                                ("--istella-after", istella_after)):
                if paths:
                    argv += [flag] + (paths if isinstance(paths, list) else [paths])
            return argv

        run("FLIP on both datasets", four(), EXIT_FLIP, flip)

        ta_08 = write("taxi.after08.log", _log(TAXI, [800, 880, 792], TAXI_ACC))
        ia_11 = write("istella.after11.log", _log(ISTELLA, [5500, 5555, 5445], ISTELLA_ACC))
        run("FLIP on a 20% win and a 10% loss (section 9 example)",
            four(taxi_after=ta_08, istella_after=ia_11), EXIT_FLIP,
            "FLIP geomean=0.938 taxi=0.800 istella=1.100 quality=ok")

        ia_13 = write("istella.after13.log", _log(ISTELLA, [6500, 6565, 6435], ISTELLA_ACC))
        run("NO FLIP on time", four(taxi_after=ta_08, istella_after=ia_13), EXIT_NO_FLIP,
            "NO FLIP geomean=1.020 taxi=0.800 istella=1.300 reason=time")

        ia_ll = write("istella.after.logloss.log",
                      _log(ISTELLA, [4000, 4040, 3960], (("logloss", 0.150000), ("auc", 0.964548))))
        run("NO FLIP on Istella-S logloss up", four(istella_after=ia_ll), EXIT_NO_FLIP,
            "NO FLIP geomean=0.849 taxi=0.900 istella=0.800 reason=quality:istella")

        ta_auc = write("taxi.after.auc.log",
                       _log(TAXI, [900, 905, 800], (("logloss", 0.525912), ("auc", 0.606793))))
        run("NO FLIP on taxi AUC down (higher is better)", four(taxi_after=ta_auc),
            EXIT_NO_FLIP, "NO FLIP geomean=0.849 taxi=0.900 istella=0.800 reason=quality:taxi")

        ta_better = write("taxi.after.better.log",
                          _log(TAXI, [900, 905, 800], (("logloss", 0.400000), ("auc", 0.700000))))
        ia_noise = write("istella.after.noise.log",
                         _log(ISTELLA, [4000, 4040, 3960], (("logloss", 0.145579), ("auc", 0.964600))))
        run("FLIP when quality improves or moves inside tolerance",
            four(taxi_after=ta_better, istella_after=ia_noise), EXIT_FLIP, flip)

        ib2 = write("istella.before2.log",
                    _log(ISTELLA, [5000, 5050, 4950], (("logloss", 0.145978), ("auc", 0.964548))))
        ia_spread = write("istella.after.spread.log",
                          _log(ISTELLA, [4000, 4040, 3960], (("logloss", 0.145900), ("auc", 0.964548))))
        run("FLIP when the change sits inside the before spread",
            four(istella_before=[ib, ib2], istella_after=ia_spread), EXIT_FLIP, flip)
        run("NO FLIP on the same change without the second before run",
            four(istella_after=ia_spread), EXIT_NO_FLIP,
            "NO FLIP geomean=0.849 taxi=0.900 istella=0.800 reason=quality:istella")

        run("NO FLIP on Istella-S unmeasured", four(istella_after=None), EXIT_NO_FLIP,
            "NO FLIP geomean=- taxi=0.900 istella=- reason=missing:istella")
        ta_et = write("taxi.after.et.log", _log(TAXI, [900, 905, 800], TAXI_ACC, lane="et"))
        run("NO FLIP on taxi carrying only another lane",
            four(taxi_after=ta_et) + ["--lane", "rf"], EXIT_NO_FLIP,
            "NO FLIP geomean=- taxi=- istella=0.800 reason=missing:taxi")
        run("ERROR on two lanes and no --lane", four(taxi_after=ta_et), EXIT_ERROR)

        ta_two = write("taxi.after.two.log", _log(TAXI, [900, 905, 800], TAXI_ACC),
                       _log("taxi-2000000x16", [1900, 1905, 1800],
                            (("logloss", 5.0), ("auc", 0.1))))
        run("ERROR on two shapes and no --rows", four(taxi_after=ta_two), EXIT_ERROR)
        run("FLIP with --rows, ACC attached to its own shape",
            four(taxi_after=ta_two) + ["--rows", "1000000"], EXIT_FLIP, flip)

        run("ERROR on datasets swapped", four(ib, ia, tb, ta), EXIT_ERROR)
        run("ERROR on an arm that is not there", four() + ["--arm", "nope"], EXIT_ERROR)
        bad_ms = write("taxi.after.badms.log", _log(TAXI, [900, 905, 800], TAXI_ACC),
                       ["FSPEED lane=rf arm=ours shape=%s round=4 ms=fast hash=-" % TAXI])
        run("ERROR on an unparseable ms", four(taxi_after=bad_ms), EXIT_ERROR)

        hb = write("taxi.before.hash.log", _log(TAXI, [1000, 1100, 990], ()))
        ha = write("taxi.after.hash.log", _log(TAXI, [900, 905, 800], ()))
        ihb = write("istella.before.hash.log", _log(ISTELLA, [5000, 5050, 4950], ()))
        iha = write("istella.after.hash.log", _log(ISTELLA, [4000, 4040, 3960], ()))
        iha_moved = write("istella.after.moved.log",
                          _log(ISTELLA, [4000, 4040, 3960], (), digest="1111111111111111"))
        run("FLIP with no ACC lines and equal output hashes", four(hb, ha, ihb, iha),
            EXIT_FLIP, flip)
        run("NO FLIP with no ACC lines and moved output hashes", four(hb, ha, ihb, iha_moved),
            EXIT_NO_FLIP, "NO FLIP geomean=0.849 taxi=0.900 istella=0.800 reason=missing:istella")

        wb = write("istella.before.weird.log", _log(ISTELLA, [5000, 5050, 4950], (("weird", 1.0),)))
        wa = write("istella.after.weird.log", _log(ISTELLA, [4000, 4040, 3960], (("weird", 0.9),)))
        run("ERROR on a metric with no known direction",
            four(istella_before=wb, istella_after=wa), EXIT_ERROR)
        run("FLIP once --metric-direction names it",
            four(istella_before=wb, istella_after=wa) + ["--metric-direction", "weird=lower"],
            EXIT_FLIP, flip)

    passed = sum(results)
    print("SELF-TEST %s %d/%d" % ("PASS" if passed == len(results) else "FAIL",
                                  passed, len(results)))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
