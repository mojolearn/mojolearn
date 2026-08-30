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
            # OUR OWN SECOND KERNEL IS NOT AN OPPONENT.
            #
            # A lane may A/B two of ITS OWN arms -- `knn` times the shipped
            # AUTO choice as `ours` and both explicit arms beside it as
            # `ours-fused` and `ours-tiled`, because DEVIATION 36's arm
            # choice was measured entirely on an M4 and the H100 row came in
            # 23.9x behind cuML. Those alternates must not enter the vendor
            # comparison: counting one as an opponent would let us "beat"
            # ourselves into the win tally, and picking the fastest opponent
            # per row would silently compare our slow arm against our fast
            # one instead of against cuML.
            opps = [(stats[(lane, o, shape)]["median"], o)
                    for o in sorted(k[1] for k in stats
                                    if k[0] == lane and k[2] == shape
                                    and not k[1].startswith("ours"))]
            opps = [(t, o) for t, o in opps if t > 0]
            if not opps:
                continue
            best_t, best_o = min(opps)
            worst.append((ours["median"] / best_t, lane, shape, best_o,
                          ours["median"], best_t))
    worst.sort(reverse=True)

    # THROUGHPUT ROWS AND FIXED-COST ROWS ARE NOT THE SAME KIND OF NUMBER,
    # AND MIXING THEM IS HOW A TABLE LIES.
    #
    # A lane at 4,000,000 rows measures arithmetic. A lane at 96 rows -- or
    # 16, which is what `krr` ships -- measures what one fit costs end to
    # end, and that is mostly launch and dispatch latency on both sides.
    # Both come out as milliseconds and both divide into a ratio, so nothing
    # in the shape of the data distinguishes them.
    #
    # On 2026-08-25 a summary of the first classical leg against cuML
    # reported ELEVEN WINS OUT OF FOURTEEN. Ten were at 8 to 4,000 rows.
    # Exactly one -- kmeans at 4,000,000 x 32 -- was a claim about kernels.
    # The write-up was not fabricated; it read a table that had no way of
    # telling it the difference.
    #
    # There is a second asymmetry that only bites the small rows: the vendor
    # arms are called through PYTHON inside the timed region while our arm
    # is a compiled binary. At sub-millisecond totals a real share of the
    # ratio is Python call overhead. That is something a cuML user genuinely
    # pays, and it is NOT a fact about anybody's CUDA kernels.
    #
    # So the lanes declare which kind they are (`scale=` in the FSPEED
    # header, defaulting to `fixed` so an unconsidered lane comes out
    # unclaimable), the two kinds get separate tables, and THE HEADLINE
    # COUNTS ONLY THROUGHPUT ROWS.
    scale_of = {}
    for h in run.headers:
        if h.get("arm") == "ours" and h.get("scale"):
            scale_of[h.get("lane")] = h["scale"]
    thr = [x for x in worst if scale_of.get(x[1]) == "throughput"]
    fixd = [x for x in worst if scale_of.get(x[1]) != "throughput"]
    # A RUN THAT PREDATES THE FIELD MUST NOT BE SILENTLY CALLED FIXED-COST.
    # No `scale=` anywhere means the logs are older than the declaration, not
    # that every lane is small -- and one of them (kmeans, 4,000,000 x 32)
    # demonstrably is not. Saying "unknown" is the honest answer; saying
    # "fixed" would be a second wrong label replacing the first.
    unscaled = not scale_of

    def _rank_table(rows, title, blurb, headline):
        if not rows:
            return
        w("## " + title)
        w("")
        for line in blurb:
            w(line)
        w("")
        w("| rank | lane | shape | ours ms | best opponent | their ms | we are |")
        w("|---|---|---|---|---|---|---|")
        for i, (r, lane, shape, o, om, tm) in enumerate(rows[:25], 1):
            verdict = ("%.2fx SLOWER" % r) if r > 1.0 else ("%.2fx FASTER" % (1.0 / r))
            w("| %d | %s | %s | %s | %s | %s | **%s** |"
              % (i, lane, shape, fmt(om), o, fmt(tm), verdict))
        if len(rows) > 25:
            w("")
            w("%d further rows not listed; %d rows in this table."
              % (len(rows) - 25, len(rows)))
        if headline:
            wins = [x for x in rows if x[0] <= 1.0]
            w("")
            w("**%d of %d THROUGHPUT rows with an opponent are wins for us.**"
              % (len(wins), len(rows)))
        w("")

    _rank_table(
        thr,
        "Where we lose most, ranked -- THROUGHPUT rows",
        ["These are the rows big enough for the ratio to be about",
         "arithmetic. THIS IS THE ONLY TABLE ANY SPEED CLAIM MAY BE DRAWN",
         "FROM. Against the fastest opponent that actually ran on each row,",
         "because losing 70x to a TF32 arm and 12x to an FP32 arm is one",
         "fact and not two."],
        True,
    )
    if unscaled:
        w("## Scale UNKNOWN for every row in this run")
        w("")
        w("No arm in these logs emitted `scale=` in its FSPEED header, so")
        w("this run is OLDER than the throughput/fixed-cost declaration and")
        w("the split below could not be made. The rows are listed under")
        w("FIXED-COST because that is the conservative bucket, but the label")
        w("is NOT a measurement here -- some of these lanes are genuinely")
        w("large (kmeans ships 4,000,000 x 32) and some are genuinely tiny")
        w("(krr ships 16 rows). Check each shape tag by hand before quoting")
        w("anything from this file, or re-run so the arms declare it.")
        w("")
    _rank_table(
        fixd,
        ("Rows, ranked -- scale UNDECLARED, check each shape by hand"
         if unscaled else "FIXED-COST rows -- NOT a speed claim"),
        ["Every row here is a lane whose fixture is small enough that both",
         "arms are dominated by launch and dispatch latency, plus the Python",
         "call overhead the vendor arm pays inside the clock and ours does",
         "not. Read each as WHAT ONE FIT COSTS END TO END on this box.",
         "",
         "A ratio here is not wrong, it is UNCLAIMABLE: it does not tell you",
         "whose kernel is faster. Some of these lanes cannot be made bigger",
         "for stated reasons -- `hdbscan`'s dense mutual-reachability arm",
         "materializes an m x m matrix, and inventing a larger fixture to",
         "make the number look like throughput would be inventing a dataset.",
         "Others simply have no size knob yet, and that is owed work.",
         "",
         "They are ranked and kept rather than deleted because the fixed",
         "cost is a real thing a user pays on a small problem."],
        False,
    )
    if thr or fixd:
        w("%d rows in total have an opponent: %d throughput, %d fixed-cost."
          % (len(worst), len(thr), len(fixd)))
        w("")

    # OUR OWN ARMS, SIDE BY SIDE. Separate from every vendor table because
    # it answers a different question: not "are we fast" but "is our own
    # dispatch picking the right kernel ON THIS COLUMN".
    ab = sorted({(k[0], k[2]) for k in stats if k[1].startswith("ours-")})
    if ab:
        w("## Our own arms, A/B")
        w("")
        w("Two of OUR kernels on the same row. This is not a speed claim")
        w("against anyone; it asks whether our own dispatch is choosing the")
        w("right arm on THIS vendor. An arm chosen by a measurement taken on")
        w("a different vendor is the failure this table exists to catch.")
        w("")
        w("| lane | shape | arm | median ms | vs shipped `ours` |")
        w("|---|---|---|---|---|")
        for lane, shape in ab:
            base = stats.get((lane, "ours", shape))
            arms = sorted(k[1] for k in stats
                          if k[0] == lane and k[2] == shape
                          and k[1].startswith("ours"))
            for a in arms:
                st = stats[(lane, a, shape)]
                if base and base["median"] and a != "ours":
                    rel = st["median"] / base["median"]
                    tag = ("%.2fx slower" % rel) if rel > 1.0 else (
                        "**%.2fx FASTER**" % (1.0 / rel))
                else:
                    tag = "(the shipped choice)" if a == "ours" else "-"
                w("| %s | %s | %s | %s | %s |"
                  % (lane, shape, a, fmt(st["median"]), tag))
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

    # THE STATUS SECTION GOES FIRST, AND ITS VERDICT IS THIS PROCESS'S EXIT.
    #
    # A ratio table is only evidence if the arms behind it ran. Until
    # 2026-08-30 this file returned 0 unconditionally and rendered a leg that
    # had lost half its arms exactly like a complete one, which is how nine
    # silent defects survived a whole day of boards on 2026-08-28. The
    # section is PREPENDED rather than appended because the failure a reader
    # needs is the one they see before they start reading numbers.
    status_rc = 0
    try:
        import leg_status
        rep = leg_status.survey(args.run_dir)
        status_rc = leg_status.SEVERITY_EXIT[rep["worst"]]
        text = leg_status.render(rep) + "\n" + text
    except Exception as exc:                      # noqa: BLE001
        # A status reader that throws must not delete the board, and must
        # not quietly pass either.
        text = ("## Leg status: UNAVAILABLE\n\n"
                "`tools/leg_status.py` raised while surveying this run:"
                " `%s`. Nothing below has been checked for whether its arms"
                " ran.\n\n" % exc) + text
        status_rc = 1

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return status_rc


if __name__ == "__main__":
    raise SystemExit(main())
