#!/bin/sh
# What IDENTICAL costs: the two modes, ALTERNATED, arm by arm.
#
#   pixi run price-unsupervised-identity          # 3 rounds
#   MOJOLEARN_PRICE_ROUNDS=6 pixi run price-unsupervised-identity
#
# `bench/identity_price_main.mojo` carries the reasoning; the mechanics are
# here. One round is FAST-then-IDENTICAL, and the rounds alternate rather
# than running all of one mode and then all of the other, because the M4's
# GPU governor drifts up to 1.7x across twenty minutes
# (`[[mojolearn-box-drifts]]`) and a block-of-A-then-block-of-B design
# measures the drift instead of the modes.
#
# Two modes are two BINARIES -- `GLOBAL_NUMERIC_MODE` is comptime -- so this
# cannot interleave inside one process the way `bench/` interleaves two arms
# of one build. Process-level alternation is the closest available thing and
# the numbers should be read as a RATIO WITH A WIDE BAND, not as a
# measurement to four figures. Report the median of each arm and say how
# many rounds it came from.
set -e

cd "$(dirname "$0")/.."
ROUNDS="${MOJOLEARN_PRICE_ROUNDS:-3}"
OUT="${MOJOLEARN_PRICE_OUT:-/tmp/mojolearn_identity_price.txt}"
: > "$OUT"

i=1
while [ "$i" -le "$ROUNDS" ]; do
    echo "== round $i/$ROUNDS: FAST =="
    pixi run mojo run -I . bench/identity_price_main.mojo \
        | grep '^PRICE' | tee -a "$OUT"
    echo "== round $i/$ROUNDS: IDENTICAL =="
    tools/with_identical_mode.sh pixi run mojo run -I . \
        bench/identity_price_main.mojo | grep '^PRICE' | tee -a "$OUT"
    i=$((i + 1))
done

echo
echo "== medians over $ROUNDS rounds =="
/usr/bin/python3 - "$OUT" <<'PY'
import sys, collections
rows = collections.defaultdict(list)
for line in open(sys.argv[1]):
    parts = line.split()
    if len(parts) == 4 and parts[0] == "PRICE":
        rows[(parts[2], parts[1])].append(float(parts[3]))


def med(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


arms = sorted({a for a, _ in rows})
print("%-12s %10s %10s %8s %s" % ("arm", "FAST ms", "IDENT ms", "ratio",
                                  "samples"))
for a in arms:
    f = rows.get((a, "FAST"), [])
    d = rows.get((a, "IDENTICAL"), [])
    if not f or not d:
        continue
    mf, md = med(f), med(d)
    print("%-12s %10.2f %10.2f %7.2fx %d/%d"
          % (a, mf, md, md / mf if mf else float("nan"), len(f), len(d)))
print()
print("Ratios above 1 are what identity costs. Read them as a band: the")
print("modes are two binaries and cannot be interleaved inside a process.")
PY
