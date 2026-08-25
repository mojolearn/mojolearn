"""Turn the FSPEED lines every speed arm prints into ONE ratio table.

    python3 tools/fast_speed_table.py <run-dir> [--out TABLE.md]

WHAT THIS READS
===============
Every arm in the FAST speed lane, Mojo or Python, ours or a vendor's, prints
the same six line kinds and nothing else that this file looks at:

    FSPEED-HEADER  family= lane= arm= mode= device= rounds= size=
    FSPEED         lane= arm= shape= round= ms= hash=
    FSPEED-WARMUP  lane= arm= shape= ms=
    FSPEED-ACC     lane= arm= metric= value=
    FSPEED-NOTE    lane= arm= <free text>
    FSPEED-REFUSED lane= arm= reason=

The run directory is a flat set of `<family>.<lane>.<arm>.log` files, one per
process, exactly as the leg body writes them. This file never runs anything
and never touches a GPU; it is a parser, so it is the one piece of this lane
that can be developed and trusted on a laptop.

WHY THE WARM-UP IS PARSED AND THEN EXCLUDED
===========================================
It is excluded from every statistic and printed in its own column. A reader
who cannot see what the first call paid cannot tell a cold-compile artifact
from a real cost, and this lane's opponents (torch especially) pay enormous
first-call prices for autotuning and lazy module init. Hiding that number
makes the median look like the whole story. `bench/lanes_price_main.mojo`
made the same choice for the same reason.

THE STATISTIC IS THE MEDIAN, NOT THE MEAN
=========================================
A rented box is shared and throttled. One descheduled round moves a mean and
does not move a median, and this repository has measured a box drifting
1.7x inside twenty minutes. The min is printed beside it so a reader can see
how far apart they are: when median and min diverge a lot, the box was busy
and the row deserves less weight, which is a judgment the table should let a
reader make rather than making it for them.

THE RATIO IS OURS OVER THEIRS AND IT IS PRINTED THAT WAY ROUND
==============================================================
`ratio = median(ours) / median(theirs)`, so **above 1.0 means we are SLOWER**
and the column is labelled `ours/theirs` on every table so nobody has to
remember. The temptation is to flip it whenever we win; a column whose
direction depends on the result is a column that will be misread.

MODE IS READ FROM THE HEADER THE BINARY PRINTED, NEVER FROM THE FILENAME
========================================================================
The whole point of this run is the FAST path. A table that says FAST because
the file was named fast, while the binary was built IDENTICAL, is the exact
failure the compile-time mode witness exists to prevent, and three
mislabelled measurements were caught by that witness on 2026-08-23. So every
`ours` row carries the mode its own header reported, any row whose mode is
not FAST is flagged loudly, and a run containing a mixture is reported as
mixed rather than quietly averaged.
"""

import argparse
import os
import re
import statistics
import sys


KIND = re.compile(r"^FSPEED(-[A-Z]+)?\s")


def parse_kv(rest):
    """`k=v` pairs, values allowed to contain spaces up to the next ` k=`."""
    out = {}
    for m in re.finditer(r"(\w+)=(.*?)(?=\s+\w+=|$)", rest.strip()):
        out[m.group(1)] = m.group(2).strip()
    return out


class Run:
    def __init__(self):
        self.headers = []      # dicts
        self.rounds = []       # dicts with float ms
        self.warmups = []
        self.acc = []
        self.notes = []
        self.refused = []
        self.agree = []

    def add_line(self, line, source):
        if not KIND.match(line):
            return
        head, _, rest = line.partition(" ")
        kv = parse_kv(rest)
        kv["_source"] = source
        if head == "FSPEED":
            try:
                kv["ms"] = float(kv.get("ms", "nan"))
                kv["round"] = int(kv.get("round", "-1"))
            except ValueError:
                return
            self.rounds.append(kv)
        elif head == "FSPEED-HEADER":
            self.headers.append(kv)
        elif head == "FSPEED-WARMUP":
            try:
                kv["ms"] = float(kv.get("ms", "nan"))
            except ValueError:
                return
            self.warmups.append(kv)
        elif head == "FSPEED-ACC":
            self.acc.append(kv)
        elif head == "FSPEED-REFUSED":
            self.refused.append(kv)
        elif head == "FSPEED-AGREE":
            # ITS OWN SECTION, not a note. A speed number for a block that
            # computes something different from its opponent is worthless,
            # and this is the only line that says whether the two sides
            # agreed. Burying it among free-text notes is how it gets
            # skipped by the reader who most needs it.
            self.agree.append(kv)
        else:
            kv["text"] = rest.strip()
            self.notes.append(kv)


def load(run_dir):
    run = Run()
    files = []
    for dirpath, _, names in os.walk(run_dir):
        for n in sorted(names):
            if n.endswith(".log") or n.endswith(".txt"):
                files.append(os.path.join(dirpath, n))
    for path in files:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                run.add_line(line.rstrip("\n"), os.path.basename(path))
    return run, files


def key(kv):
    return (kv.get("lane", "?"), kv.get("arm", "?"), kv.get("shape", "-"))


def summarize(run):
    """(lane, arm, shape) -> stats. The warm-up never enters a statistic."""
    buckets = {}
    for r in run.rounds:
        buckets.setdefault(key(r), []).append(r["ms"])
    warm = {}
    for w in run.warmups:
        warm.setdefault(key(w), []).append(w["ms"])
    hashes = {}
    for r in run.rounds:
        hashes.setdefault(key(r), set()).add(r.get("hash", "-"))

    stats = {}
    for k, ms in buckets.items():
        ms = sorted(ms)
        stats[k] = {
            "n": len(ms),
            "median": statistics.median(ms),
            "min": ms[0],
            "max": ms[-1],
            "warmup": max(warm.get(k, [float("nan")])),
            "hashes": hashes.get(k, set()),
        }
    return stats


def mode_of(run, lane):
    modes = {h.get("mode") for h in run.headers
             if h.get("lane") == lane and h.get("arm") == "ours"}
    modes.discard(None)
    if not modes:
        return "?"
    if len(modes) > 1:
        return "MIXED:" + "/".join(sorted(modes))
    return modes.pop()


def fmt(x, places=3):
    if x != x:
        return "-"
    return f"{x:.{places}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run, files = load(args.run_dir)
    if not files:
        print(f"no .log files under {args.run_dir}", file=sys.stderr)
        return 2
    stats = summarize(run)

    lines = []
    w = lines.append

    device = sorted({h.get("device", "?") for h in run.headers})
    w("# The FAST path against the vendor, one rented NVIDIA box")
    w("")
    w(f"Parsed from `{args.run_dir}`, {len(files)} arm logs.")
    w("")
    w(f"Device(s) reported by the arms themselves: {', '.join(device) or 'none'}")
    w("")
    w("`ratio = median(ours) / median(theirs)`. **Above 1.0 means WE ARE SLOWER.**")
    w("The warm-up round is excluded from every statistic and printed on its own,")
    w("because torch pays an enormous first call and hiding that makes the median")
    w("read as the whole story. `min` is printed beside the median so a reader can")
    w("see when the box was busy: the further they are apart, the less the row is worth.")
    w("")

    lanes = sorted({k[0] for k in stats})
    ours_modes = {lane: mode_of(run, lane) for lane in lanes}
    bad_mode = [l for l, m in ours_modes.items() if m not in ("FAST",)]
    if bad_mode:
        w("## MODE WARNING")
        w("")
        w("These lanes did NOT report FAST from their own compile-time witness.")
        w("This whole run is supposed to be the FAST path. Do not quote them as FAST.")
        w("")
        for l in bad_mode:
            w(f"- `{l}` reported mode `{ours_modes[l]}`")
        w("")

    w("## Every arm, as measured")
    w("")
    w("| lane | arm | shape | n | median ms | min ms | max ms | warmup ms | hash |")
    w("|---|---|---|---|---|---|---|---|---|")
    for k in sorted(stats):
        lane, arm, shape = k
        s = stats[k]
        h = sorted(s["hashes"])
        hs = h[0] if len(h) == 1 else f"MOVED({len(h)})"
        w(f"| {lane} | {arm} | {shape} | {s['n']} | {fmt(s['median'])} | "
          f"{fmt(s['min'])} | {fmt(s['max'])} | {fmt(s['warmup'])} | {hs} |")
    w("")

    w("## Ours against each opponent")
    w("")
    w("| lane | shape | opponent | ours ms | theirs ms | ratio ours/theirs | verdict |")
    w("|---|---|---|---|---|---|---|")
    any_row = False
    for lane in lanes:
        shapes = sorted({k[2] for k in stats if k[0] == lane})
        for shape in shapes:
            ours = stats.get((lane, "ours", shape))
            if ours is None:
                continue
            opponents = sorted(k[1] for k in stats
                               if k[0] == lane and k[2] == shape and k[1] != "ours")
            if not opponents:
                w(f"| {lane} | {shape} | (none ran) | {fmt(ours['median'])} | - | - | "
                  f"NO OPPONENT ON THIS BOX |")
                any_row = True
                continue
            for opp in opponents:
                t = stats[(lane, opp, shape)]
                ratio = ours["median"] / t["median"] if t["median"] else float("nan")
                if ratio != ratio:
                    verdict = "-"
                elif ratio <= 1.0:
                    verdict = f"we are {1.0 / ratio:.2f}x FASTER"
                else:
                    verdict = f"we are {ratio:.2f}x SLOWER"
                w(f"| {lane} | {shape} | {opp} | {fmt(ours['median'])} | "
                  f"{fmt(t['median'])} | {fmt(ratio, 2)} | {verdict} |")
                any_row = True
    if not any_row:
        w("| - | - | - | - | - | - | nothing parsed |")
    w("")

    if run.acc:
        w("## Accuracy, because a faster learner that fits worse has not won")
        w("")
        w("| lane | arm | metric | value |")
        w("|---|---|---|---|")
        for a in run.acc:
            w(f"| {a.get('lane','?')} | {a.get('arm','?')} | "
              f"{a.get('metric','?')} | {a.get('value','?')} |")
        w("")

    # WHERE TO SPEND THE NEXT DAY, ranked, so the optimization target is a
    # measurement rather than an intuition. Only the BEST opponent per row
    # counts: losing 70x to a TF32 arm and 12x to an FP32 arm is one fact,
    # not two, and the honest headline is the smaller gap we could close
    # against the strongest thing that actually ran.
    worst = []
    for lane in lanes:
        for shape in sorted({k[2] for k in stats if k[0] == lane}):
            ours = stats.get((lane, "ours", shape))
            if ours is None:
                continue
            opps = [(stats[(lane, o, shape)]["median"], o)
                    for o in sorted(k[1] for k in stats
                                    if k[0] == lane and k[2] == shape and k[1] != "ours")]
            opps = [(t, o) for t, o in opps if t > 0]
            if not opps:
                continue
            best_t, best_o = min(opps)
            worst.append((ours["median"] / best_t, lane, shape, best_o,
                          ours["median"], best_t))
    worst.sort(reverse=True)
    if worst:
        w("## Where we lose most, ranked")
        w("")
        w("Against the FASTEST opponent that actually ran on each row, because")
        w("losing 70x to a TF32 arm and 12x to an FP32 arm is one fact and not")
        w("two. This is the optimization queue: it is a measurement, not an")
        w("intuition about which kernel feels slow.")
        w("")
        w("| rank | lane | shape | ours ms | best opponent | their ms | we are |")
        w("|---|---|---|---|---|---|---|")
        for i, (r, lane, shape, o, om, tm) in enumerate(worst[:25], 1):
            verdict = ("%.2fx SLOWER" % r) if r > 1.0 else ("%.2fx FASTER" % (1.0 / r))
            w("| %d | %s | %s | %s | %s | %s | **%s** |"
              % (i, lane, shape, fmt(om), o, fmt(tm), verdict))
        if len(worst) > 25:
            w("")
            w("%d further rows not listed; %d rows in total have an opponent."
              % (len(worst) - 25, len(worst)))
        wins = [x for x in worst if x[0] <= 1.0]
        w("")
        w("**%d of %d rows with an opponent are wins for us.**"
          % (len(wins), len(worst)))
        w("")

    if run.agree:
        w("## Did the two sides compute the same thing")
        w("")
        w("A speed number for an arm that computes something different from")
        w("its opponent is worthless. This is that check, reported and not")
        w("gated: a large difference does not fail the run, it disqualifies")
        w("the ROW, and the row has to be readable to be disqualified.")
        w("")
        w("| lane | max abs diff | max rel diff | n | source |")
        w("|---|---|---|---|---|")
        for a in run.agree:
            w("| %s | %s | %s | %s | %s |" % (
                a.get("lane", "?"), a.get("max_abs_diff", "-"),
                a.get("max_rel_diff", "-"), a.get("n", "-"),
                a.get("_source", "?")))
        w("")

    if run.refused:
        w("## Refused arms, kept rather than dropped")
        w("")
        w("An arm that could not run is a result about this box and this image.")
        w("Deleting the row would make the table read as full coverage.")
        w("")
        w("| lane | arm | reason |")
        w("|---|---|---|")
        for r in run.refused:
            w(f"| {r.get('lane','?')} | {r.get('arm','?')} | {r.get('reason','?')} |")
        w("")

    moved = [k for k, s in stats.items() if len(s["hashes"]) > 1]
    w("## Determinism, reported and not judged")
    w("")
    w("This is the FAST path. A hash that moves between rounds is EXPECTED here")
    w("and is recorded, not failed. It is the direct evidence for what the")
    w("IDENTICAL mode buys, measured on the same box in the same hour.")
    w("")
    if moved:
        w("| lane | arm | shape | distinct hashes across rounds |")
        w("|---|---|---|---|")
        for k in sorted(moved):
            w(f"| {k[0]} | {k[1]} | {k[2]} | {len(stats[k]['hashes'])} |")
    else:
        w("No arm's output hash moved across its rounds in this run.")
    w("")

    if run.notes:
        w("## Notes the arms printed")
        w("")
        for n in run.notes:
            w(f"- `{n.get('_source','?')}`: {n.get('text','')}")
        w("")

    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
