"""Two FAST speed runs, and what moved between them.

    python3 tools/fast_speed_diff.py <before-logs-dir> <after-logs-dir> [--out F.md]

WHY A DIFFER AND NOT TWO TABLES SIDE BY SIDE
=============================================
A performance fix is a CLAIM ABOUT A CHANGE, and a change is not readable
from two sixty-row tables printed one after the other. What is wanted is the
per-row delta, sorted, with the rows that did NOT move listed as well --
because a fix aimed at five rows that moves forty of them has done something
nobody predicted, and a fix that moves none of the rows it was aimed at has
failed regardless of what else improved.

THE HASH COLUMN IS THE OTHER HALF AND IT IS THE STRICTER ONE
=============================================================
`DEVIATION 1873` removed one of two byte-identical transposes, so the
arithmetic must not move: the four TN row hashes have to be EQUAL across the
two runs. `DEVIATIONS 1876` and `1877` change which kernel FAST uses, so
those rows' hashes are EXPECTED to move and a row that did not move is the
suspicious one -- it would mean the new path was not taken.

So this file never says "the hashes differ, therefore red". It says WHICH
rows moved and lets the reader check that against what was supposed to
happen. A differ that decided for you would have to know the intent of every
commit, and it does not.

NEITHER RUN IS TREATED AS AUTHORITATIVE ON TIMING. A rented box is shared and
throttled and this repository has measured one drifting 1.7x inside twenty
minutes, so a delta under about 1.15x either way is reported as NOISE rather
than as an improvement or a regression. That threshold is stated on the page
rather than hidden here.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_speed_table import load, summarize  # noqa: E402

NOISE = 1.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--out", default=None)
    ap.add_argument("--noise", type=float, default=NOISE)
    args = ap.parse_args()

    rb, fb = load(args.before)
    ra, fa = load(args.after)
    if not fb or not fa:
        print("one of the directories has no .log files", file=sys.stderr)
        return 2
    sb, sa = summarize(rb), summarize(ra)

    lines = []
    w = lines.append
    w("# What moved between two FAST speed runs")
    w("")
    w("before: `%s` (%d arm logs)" % (args.before, len(fb)))
    w("after:  `%s` (%d arm logs)" % (args.after, len(fa)))
    w("")
    w("`speedup = before / after`, so **above 1.0 is FASTER after**.")
    w("A delta inside %.2fx either way is reported as NOISE: a rented box is"
      % args.noise)
    w("shared and throttled and this repository has measured one drifting")
    w("1.7x inside twenty minutes.")
    w("")

    common = sorted(set(sb) & set(sa))
    only_b = sorted(set(sb) - set(sa))
    only_a = sorted(set(sa) - set(sb))

    rows = []
    for k in common:
        bm, am = sb[k]["median"], sa[k]["median"]
        if am <= 0 or bm <= 0:
            continue
        rows.append((bm / am, k, bm, am,
                     sorted(sb[k]["hashes"]), sorted(sa[k]["hashes"])))
    rows.sort(reverse=True)

    moved = [r for r in rows if r[0] >= args.noise or r[0] <= 1.0 / args.noise]
    w("## Timing, ranked by how much it moved")
    w("")
    if moved:
        w("| lane | arm | shape | before ms | after ms | speedup | |")
        w("|---|---|---|---|---|---|---|")
        for sp, k, bm, am, _, _ in moved:
            tag = "**FASTER**" if sp > 1.0 else "**SLOWER**"
            w("| %s | %s | %s | %.3f | %.3f | %.2fx | %s |"
              % (k[0], k[1], k[2], bm, am, sp, tag))
    else:
        w("Nothing moved outside the noise band.")
    w("")
    w("%d of %d shared rows moved outside %.2fx; %d stayed inside it."
      % (len(moved), len(rows), args.noise, len(rows) - len(moved)))
    w("")

    hmoved, hsame = [], []
    for sp, k, bm, am, hb, ha in rows:
        if hb == ["-"] and ha == ["-"]:
            continue
        (hmoved if hb != ha else hsame).append((k, hb, ha))
    w("## Output bits")
    w("")
    w("Read this against what the commits CLAIMED, because both answers are")
    w("correct for different rows. A fix that only removes redundant work must")
    w("leave the hash EQUAL; a fix that changes which kernel runs must move")
    w("it, and a row that did not move is then the suspicious one.")
    w("")
    if hmoved:
        w("| lane | arm | shape | before | after |")
        w("|---|---|---|---|---|")
        for k, hb, ha in hmoved:
            w("| %s | %s | %s | `%s` | `%s` |"
              % (k[0], k[1], k[2], ",".join(hb)[:34], ",".join(ha)[:34]))
    else:
        w("No hashed row changed its output.")
    w("")
    w("%d hashed rows moved, %d are bit-identical across the two runs."
      % (len(hmoved), len(hsame)))
    w("")

    if only_b or only_a:
        w("## Rows present in only one run")
        w("")
        w("A row that VANISHED is as much a result as a row that got slower.")
        w("")
        for k in only_b:
            w("- gone:  `%s / %s / %s`" % k)
        for k in only_a:
            w("- new:   `%s / %s / %s`" % k)
        w("")

    text = "\n".join(lines) + "\n"
    if args.out:
        open(args.out, "w").write(text)
        print("wrote " + args.out)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
