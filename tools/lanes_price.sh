#!/bin/sh
# What IDENTICAL costs on the six frozen lanes: two binaries, ALTERNATED.
#
#   tools/lanes_price.sh                                  # 5 rounds, six lanes
#   MOJOLEARN_LANES_PRICE_ROUNDS=7 tools/lanes_price.sh
#   MOJOLEARN_LANES_PRICE_LANES="kde svm" tools/lanes_price.sh
#   MOJOLEARN_LANES_PRICE_SMOKE=1 MOJOLEARN_LANES_PRICE_ROUNDS=1 tools/lanes_price.sh
#   MOJOLEARN_LANES_PRICE_SABOTAGE_SWAP=1 ... tools/lanes_price.sh   # must ABORT (exit 3)
#
# `bench/lanes_price_main.mojo` carries the per-lane entries and the hash;
# `bench/LANES_PRICE.md` carries the clean-window procedure and the table;
# the mechanics are here. Modelled on `tools/gemm_price.sh` (the mode
# witness, the A5 concurrency check, the discarded-leg rule) with two
# changes that file's docstring invites:
#
#   BUILT ONCE PER MODE, RUN MANY TIMES. `tools/gemm_price.sh` recompiles
#   every leg through `mojo run`, so its alternation window carries a
#   compile between every two measurements. Here the FAST binary is built
#   bare and the IDENTICAL binary through `tools/with_identical_mode.sh`'s
#   `mojo build` form (the `-D MOJOLEARN_NUMERIC_IDENTICAL=1` define is
#   injected right after `mojo build`), BOTH before the first measurement,
#   and the window then alternates two ready binaries: F I F I F I ... per
#   lane. The GPU kernels are still compiled on first launch inside each
#   process, which is what the driver's untimed `warmup` round pays.
#
#   IT REFUSES RATHER THAN WARNS. A number taken beside another GPU job on
#   this box is not a contaminated number, it is the governor's number
#   (`[[mojolearn-box-drifts]]`: heat pins the M4's GPU at MINIMUM clock for
#   96% of an 11-second trace while 93.8% busy, up to 1.7x across twenty
#   minutes). So: exit 2 if another process holds the build lock, and exit 2
#   if a `mojo`/`pixi` executable is running, unless
#   MOJOLEARN_LANES_PRICE_ALLOW_BUSY=1 says the operator knows what it is.
#   (`tools/with_build_lock.sh` has no busy detector of its own -- it only
#   blocks -- so the detector here is `tools/gemm_price.sh`'s, matched on
#   the EXECUTABLE and not the command line, for the reason that file
#   gives.) MOJOLEARN_LANES_PRICE_WAIT_LOCK=1 waits for the lock instead of
#   refusing, which is for builds and smoke runs, never for a number.
#
# EVERY LEG IS READ BACK. The binary prints the mode it COMPILED in (the
# `_mode_name()` comptime witness) in its header and on every `LPRICE`
# line; this script asserts both against the mode the leg asked for and
# ABORTS THE WHOLE RUN on a mismatch. Not "discards the leg": a binary that
# says IDENTICAL when built as FAST means the build step is wrong, and every
# later leg of that binary would be wrong the same way. Three mislabeled
# measurements were caught by this witness in this tree on 2026-08-23.
#
# A PRICE RUN IS A HASH RUN. The driver prints the FNV-1a64 of each round's
# output; this script compares it across legs (same lane, same mode) and
# across modes, and prints HASH-MOVED / HASH-STABLE per lane. Under
# IDENTICAL a move between legs is a FINDING and is printed as one; it does
# not abort, because the seconds beside it are still a price and the finding
# is the more valuable of the two lines.
#
# OUTPUT: bench/results/lanes_price/<timestamp>_<sha>[_SMOKE]/
#   bin/lanes_price.FAST, bin/lanes_price.IDENTICAL   the two binaries
#                   (deleted at the end unless MOJOLEARN_LANES_PRICE_KEEP_BIN=1)
#   <lane>.FAST.log, <lane>.IDENTICAL.log              every leg, appended
#   summary.tsv     lane  mode  round  seconds  hash   (timed rounds)
#   warmup.tsv      the same for the untimed warm-up of every leg
#   ratio.tsv       lane  n  fast_med  ident_med  ratio_of_medians
#                   ratio_min  ratio_max  fast_min  fast_max  ident_min
#                   ident_max  fast_hash  ident_hash  hash_note
#   env.txt         sha, date, host, cpu, rounds, smoke, busy-check result
#
# WHAT MAY NOT BE CONCLUDED. A SMOKE run (tiny sizes) proves the build, the
# witness and the hash and NOTHING about cost: its seconds are launch
# overhead. A non-smoke run is a price on ONE M4 laptop in the thermal state
# it was in; the ratio is a BAND (min/max over rounds) and the median is its
# center, not a figure to four places. Nothing here is a certified timing
# and nothing here ranks vendors.
set -e

cd "$(dirname "$0")/.."
REPO="$(pwd)"
ROUNDS="${MOJOLEARN_LANES_PRICE_ROUNDS:-5}"
LANES="${MOJOLEARN_LANES_PRICE_LANES:-cd kde linkage svm metrics gemm}"
SMOKE="${MOJOLEARN_LANES_PRICE_SMOKE:-0}"
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
STAMP="$(date +%Y-%m-%d_%H%M%S)_${SHA}"
[ "$SMOKE" = "1" ] && STAMP="${STAMP}_SMOKE"
OUT="${MOJOLEARN_LANES_PRICE_OUT:-bench/results/lanes_price/$STAMP}"
DRIVER="bench/lanes_price_main.mojo"

# ---------------------------------------------------------------------------
# 0. Refuse beside another job. The lock probe is non-blocking and is the
#    same fcntl lock `tools/with_build_lock.sh` takes; a holder's child is
#    marked by MOJOLEARN_BUILD_LOCK_HELD and is not "another process".
# ---------------------------------------------------------------------------
if [ "${MOJOLEARN_BUILD_LOCK_HELD:-}" != "1" ]; then
    if ! /usr/bin/python3 -c '
import fcntl, sys
f = open("/tmp/cbsym-build.lock", "w")
try:
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    sys.exit(1)
'; then
        if [ "${MOJOLEARN_LANES_PRICE_WAIT_LOCK:-0}" = "1" ]; then
            echo "!! build lock is held by another process; WAITING (MOJOLEARN_LANES_PRICE_WAIT_LOCK=1)"
        else
            echo "!! REFUSED (exit 2): another process holds the build lock"
            echo "!! (/tmp/cbsym-build.lock). A price taken behind an unknown"
            echo "!! GPU job is the governor's number. Wait for it, or set"
            echo "!! MOJOLEARN_LANES_PRICE_WAIT_LOCK=1 for a build or a smoke."
            exit 2
        fi
    fi
    # Hold the lock for the WHOLE run, re-entering this script under it,
    # the way tools/with_identical_mode.sh does.
    exec tools/with_build_lock.sh "$0" "$@"
fi

busy=$(ps -Ao pid=,comm= | grep -E '(^|/)(mojo|pixi)$' || true)
BUSY_NOTE="no concurrent mojo/pixi process at start"
if [ -n "$busy" ]; then
    BUSY_NOTE="CONCURRENT mojo/pixi process at start: $(echo "$busy" | tr '\n' ';')"
    if [ "${MOJOLEARN_LANES_PRICE_ALLOW_BUSY:-0}" != "1" ]; then
        echo "!! REFUSED (exit 2): a mojo/pixi executable is running:"
        echo "$busy" | while read -r pid _; do
            echo "!!   $(ps -p "$pid" -o command= 2>/dev/null | cut -c1-96)"
        done
        echo "!! The M4's governor makes a number taken beside it fiction."
        echo "!! Set MOJOLEARN_LANES_PRICE_ALLOW_BUSY=1 only if you know the"
        echo "!! job is not on the GPU (a smoke run may)."
        exit 2
    fi
    echo "!! $BUSY_NOTE (allowed by MOJOLEARN_LANES_PRICE_ALLOW_BUSY=1)"
fi

mkdir -p "$OUT/bin"
echo "== lanes_price: rounds $ROUNDS, lanes [$LANES], smoke $SMOKE -> $OUT =="
{
    echo "sha $SHA"
    echo "head $(git rev-parse HEAD 2>/dev/null || echo nogit)"
    echo "date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host $(hostname -s)"
    echo "uname $(uname -srm)"
    echo "cpu $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
    echo "rounds $ROUNDS"
    echo "lanes $LANES"
    echo "smoke $SMOKE"
    echo "busy_check_start $BUSY_NOTE"
    echo "loadavg_start $(sysctl -n vm.loadavg 2>/dev/null || uptime)"
} > "$OUT/env.txt"

# ---------------------------------------------------------------------------
# 1. Build both binaries, under the lock, before any measurement.
# ---------------------------------------------------------------------------
FAST_BIN="$OUT/bin/lanes_price.FAST"
IDENT_BIN="$OUT/bin/lanes_price.IDENTICAL"
echo "== build FAST -> $FAST_BIN =="
if ! pixi run mojo build -I . "$DRIVER" -o "$FAST_BIN" > "$OUT/build.FAST.log" 2>&1; then
    echo "!! FAST build FAILED; $OUT/build.FAST.log:"
    grep -v "warning:" "$OUT/build.FAST.log" | grep -A6 "error" | head -60
    exit 1
fi
echo "== build IDENTICAL (tools/with_identical_mode.sh, mojo build form) -> $IDENT_BIN =="
if ! tools/with_identical_mode.sh pixi run mojo build -I . "$DRIVER" -o "$IDENT_BIN" > "$OUT/build.IDENTICAL.log" 2>&1; then
    echo "!! IDENTICAL build FAILED; $OUT/build.IDENTICAL.log:"
    grep -v "warning:" "$OUT/build.IDENTICAL.log" | grep -A6 "error" | head -60
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. One leg: run one binary for one lane, one timed round, read the mode
#    back from the header AND from every LPRICE line, abort on mismatch.
# ---------------------------------------------------------------------------
printf 'lane\tmode\tround\tseconds\thash\n' > "$OUT/summary.tsv"
printf 'lane\tmode\tround\tseconds\thash\n' > "$OUT/warmup.tsv"
LEG="$OUT/.leg"

run_leg() {
    lane="$1"; want="$2"; round="$3"; bin="$4"
    log="$OUT/$lane.$want.log"
    echo "== leg lane=$lane mode=$want round=$round $(date +%T) ==" >> "$log"
    if ! env MOJOLEARN_LANES_PRICE_LANE="$lane" \
             MOJOLEARN_LANES_PRICE_ROUNDS=1 \
             MOJOLEARN_LANES_PRICE_SMOKE="$SMOKE" \
             "$bin" > "$LEG" 2>&1; then
        cat "$LEG" >> "$log"
        echo "!! leg FAILED (lane=$lane mode=$want round=$round); output:"
        cat "$LEG"
        echo "!! ABORT: a failed leg is a broken driver or a raised finding,"
        echo "!! not a missing sample."
        exit 1
    fi
    cat "$LEG" >> "$log"
    got=$(grep -o '^== bench/lanes_price_main.mojo \[[A-Z]*\]' "$LEG" \
          | sed 's/.*\[\(.*\)\]/\1/' || true)
    if [ "$got" != "$want" ]; then
        echo "!! MODE MISMATCH (lane=$lane round=$round): asked for $want,"
        echo "!! the binary's header says '$got'. ABORT. A correctly-labelled"
        echo "!! sample in the wrong column is the failure the witness exists"
        echo "!! to catch, and it means the BUILD step is wrong, not one leg."
        exit 3
    fi
    bad=$(grep '^LPRICE ' "$LEG" | awk -v w="$want" '$3 != w' || true)
    if [ -n "$bad" ]; then
        echo "!! MISLABELLED LPRICE LINES in a $want leg (lane=$lane):"
        echo "$bad" | sed 's/^/!!   /'
        echo "!! ABORT."
        exit 3
    fi
    # LPRICE <lane> <mode> <round|warmup> <size> <seconds> <hash>
    grep '^LPRICE ' "$LEG" | awk -v r="$round" -v OFS='\t' \
        -v s="$OUT/summary.tsv" -v w="$OUT/warmup.tsv" '
        $4 == "warmup" { print $2, $3, r, $6, $7 >> w; next }
        { print $2, $3, r, $6, $7 >> s }'
    grep -E '^(HASH-MOVED|HASH-STABLE)' "$LEG" | sed "s/^/   round $round: /"
}

# ---------------------------------------------------------------------------
# 3. The window: per lane, F I F I F I ... ROUNDS times, nothing between.
# ---------------------------------------------------------------------------
for lane in $LANES; do
    echo
    echo "== lane $lane: $ROUNDS rounds, FAST then IDENTICAL per round =="
    r=1
    while [ "$r" -le "$ROUNDS" ]; do
        run_leg "$lane" FAST "$r" "$FAST_BIN"
        # SABOTAGE SWITCH for the witness itself: hand the FAST binary to
        # the IDENTICAL leg and watch run_leg abort with MODE MISMATCH.
        # A check that cannot be made to fail is not a check.
        if [ "${MOJOLEARN_LANES_PRICE_SABOTAGE_SWAP:-0}" = "1" ]; then
            run_leg "$lane" IDENTICAL "$r" "$FAST_BIN"
        else
            run_leg "$lane" IDENTICAL "$r" "$IDENT_BIN"
        fi
        r=$((r + 1))
    done
done
rm -f "$LEG"

busy_end=$(ps -Ao pid=,comm= | grep -E '(^|/)(mojo|pixi)$' || true)
if [ -n "$busy_end" ]; then
    echo "!! a mojo/pixi process was running AT THE END; the later rounds"
    echo "!! may have been taken beside it. Noted in env.txt."
    echo "busy_check_end CONCURRENT: $(echo "$busy_end" | tr '\n' ';')" >> "$OUT/env.txt"
else
    echo "busy_check_end none" >> "$OUT/env.txt"
fi
echo "loadavg_end $(sysctl -n vm.loadavg 2>/dev/null || uptime)" >> "$OUT/env.txt"

# ---------------------------------------------------------------------------
# 4. The table: per lane, median IDENTICAL / median FAST, with the spread
#    of the per-round paired ratios and of each mode's seconds, and the
#    hash verdicts across legs and across modes.
# ---------------------------------------------------------------------------
echo
echo "== ratio table (written to $OUT/ratio.tsv) =="
/usr/bin/python3 - "$OUT/summary.tsv" "$OUT/ratio.tsv" "$SMOKE" "$ROUNDS" <<'PY'
import sys, collections

summary, ratio_path, smoke, rounds = sys.argv[1], sys.argv[2], sys.argv[3] == "1", int(sys.argv[4])
secs = collections.defaultdict(dict)    # (lane, mode) -> {round: seconds}
hashes = collections.defaultdict(dict)  # (lane, mode) -> {round: hash}
lanes = []
with open(summary) as fh:
    next(fh)
    for line in fh:
        lane, mode, rnd, s, h = line.rstrip("\n").split("\t")
        if lane not in lanes:
            lanes.append(lane)
        secs[(lane, mode)][int(rnd)] = float(s)
        hashes[(lane, mode)][int(rnd)] = h


def med(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


cols = ["lane", "n", "fast_med_s", "ident_med_s", "ratio_med", "ratio_min",
        "ratio_max", "fast_min_s", "fast_max_s", "ident_min_s", "ident_max_s",
        "fast_hash", "ident_hash", "hash_note"]
rows = []
for lane in lanes:
    f, d = secs.get((lane, "FAST"), {}), secs.get((lane, "IDENTICAL"), {})
    hf, hd = hashes.get((lane, "FAST"), {}), hashes.get((lane, "IDENTICAL"), {})
    common = sorted(set(f) & set(d))
    if not common:
        rows.append([lane, 0] + ["-"] * 11 + ["ONE MODE ONLY"])
        continue
    paired = [d[r] / f[r] for r in common if f[r] > 0]
    mf, md = med([f[r] for r in common]), med([d[r] for r in common])
    notes = []
    fset, dset = set(hf.values()), set(hd.values())
    if len(fset) > 1:
        notes.append("FAST HASH MOVED across legs (%d distinct)" % len(fset))
    if len(dset) > 1:
        notes.append("IDENTICAL HASH MOVED across legs (%d distinct) -- FINDING" % len(dset))
    if len(fset) == 1 and len(dset) == 1:
        notes.append("fast==ident bits" if fset == dset else "fast!=ident bits")
    rows.append([
        lane, len(common), "%.6f" % mf, "%.6f" % md,
        "%.3f" % (md / mf if mf else float("nan")),
        "%.3f" % min(paired), "%.3f" % max(paired),
        "%.6f" % min(f[r] for r in common), "%.6f" % max(f[r] for r in common),
        "%.6f" % min(d[r] for r in common), "%.6f" % max(d[r] for r in common),
        (sorted(fset)[0] if len(fset) == 1 else "MOVED"),
        (sorted(dset)[0] if len(dset) == 1 else "MOVED"),
        "; ".join(notes),
    ])

with open(ratio_path, "w") as fh:
    fh.write("\t".join(cols) + "\n")
    for r in rows:
        fh.write("\t".join(str(c) for c in r) + "\n")

w = max(len(r[0]) for r in rows) if rows else 8
print("%-*s %2s %11s %11s %7s %7s %7s  %-16s %-16s %s"
      % (w, "lane", "n", "FAST med s", "IDENT med s", "ratio", "min", "max",
         "FAST hash", "IDENT hash", "hash note"))
for r in rows:
    print("%-*s %2s %11s %11s %7s %7s %7s  %-16s %-16s %s"
          % (w, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[11], r[12], r[13]))
print()
if smoke:
    print("SMOKE RUN: tiny sizes, %d round(s). The seconds above are launch" % rounds)
    print("overhead and prove only that both binaries run, the mode reads back,")
    print("and the hash is stable. NO NUMBER HERE IS A PRICE.")
else:
    print("READ THE RATIO AS A BAND: min/max are the per-round paired ratios")
    print("(IDENTICAL_r / FAST_r) over %d alternated rounds on THIS box." % rounds)
    print("The median is the band's center, not a figure to four places. Copy")
    print("the row into bench/LANES_PRICE.md with the date, the sha and the")
    print("machine beside it.")
    # THE CAVEAT MUST NAME THE BOX IT APPLIES TO. This block read "on one M4
    # whose governor drifts 1.7x under heat" unconditionally, so the AMD leg of
    # 2026-08-31 printed a thermal caveat about an Apple laptop underneath
    # numbers taken on an MI325X in a Linux datacenter. A caveat attached to
    # the wrong machine is worse than none: it invites the reader to discount a
    # number for a reason that does not apply to it.
    import platform as _pf
    if _pf.system() == "Darwin":
        print("")
        print("THIS BOX IS THE LAPTOP: heat pins its GPU at MINIMUM clock and")
        print("the governor drifts 1.7x within one session, so alternate the")
        print("arms INSIDE one window or the number is fiction.")
PY
# The binaries are rebuilt by every run and are not results; they stay out
# of the committed tree unless asked for.
if [ "${MOJOLEARN_LANES_PRICE_KEEP_BIN:-0}" != "1" ]; then
    rm -rf "$OUT/bin"
fi
echo
echo "== done: $OUT =="
