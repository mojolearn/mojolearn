#!/bin/sh
# What `mojolearn.identical.gemm.fp32.v1` COSTS: the two modes, ALTERNATED.
#
#   tools/gemm_price.sh                         # 3 rounds, every measurement
#   MOJOLEARN_GEMM_PRICE_ROUNDS=6 tools/gemm_price.sh
#   MOJOLEARN_GEMM_PRICE_ONLY=m4 tools/gemm_price.sh    # one measurement
#   MOJOLEARN_GEMM_PRICE_ARM=device tools/gemm_price.sh # the Phase 2b stub
#
# `bench/gemm_price_main.mojo` carries the reasoning and contract section 13.6
# carries the specification; the mechanics are here. Modelled on
# `tools/price_linalg_identity.sh` deliberately -- same `PRICE <mode> <arm>
# <ms>` line format, so the median table is SHARED and a v1 number is directly
# comparable with the shipped kernels' numbers rather than living in its own
# units.
#
# ============================================================================
# TIMING ON THIS BOX WHILE ANOTHER LANE RUNS GPU WORK IS MEANINGLESS.
# ============================================================================
# Contract 13.5's assumption A5, and it is not a formality: the M4's GPU
# governor drifts up to 1.7x across twenty minutes under heat
# (`[[mojolearn-box-drifts]]`, where the mechanism was measured -- the
# governor pins at MINIMUM clock for 96% of an 11-second trace while 93.8%
# busy). Two lanes were live in this tree on 2026-08-23 and a third was
# starting. So this script LOOKS for a concurrent `mojo` or `pixi` process
# before it starts and prints what it found. It does not refuse -- the arms
# this harness can run today are HOST arms, which contend for CPU rather than
# for the GPU and are therefore less fragile than a device timing, not immune
# -- but a number produced beside a leg is reported as contaminated and the
# banner is printed with it rather than remembered.
#
# ONE ROUND IS FAST-THEN-IDENTICAL, AND THE ROUNDS ALTERNATE rather than
# running all of one mode and then all of the other. A block of A then a block
# of B measures the drift instead of the modes. Two modes are two BINARIES --
# `GLOBAL_NUMERIC_MODE` is comptime -- so this cannot interleave inside one
# process; process-level alternation is the closest available thing and the
# ratio is A BAND, not a figure to four places. The ARMS inside one mode do
# better: `oracle` and `serial` live in the same binary and alternate CALL BY
# CALL inside the timed loop, so their ratio does not need this treatment.
#
# BOTH ARMS TAKE THE BUILD LOCK, AND IT IS NOT OPTIONAL.
# Measured in this tree on 2026-08-23: of five FAST invocations WITHOUT the
# lock, ONE compiled inside a concurrent session's IDENTICAL window and
# emitted `PRICE IDENTICAL ...` -- a correctly-labelled sample that landed in
# the wrong arm's column. `tools/price_unsupervised_identity.sh` still has
# that hole; this script does not, and it does not merely take the lock:
#
#   EVERY LEG IS READ BACK. The binary prints the mode it COMPILED in and
#   labels every `PRICE` line with it. This script asserts that the header
#   mode AND every single PRICE line's mode field match the mode the leg
#   asked for, and DISCARDS the leg if they do not. Do not trust the flip.
#
# WHAT MAY NOT BE CONCLUDED FROM THESE NUMBERS -- contract 13.7:
#   - not that the balanced fold is faster or slower than the serial fold.
#     What runs today is a HOST comparison of the ORACLE's fold spelling,
#     which allocates a List per tree level; a device kernel allocates
#     nothing, and contract 13.4's question is about LAUNCHES, which a host
#     does not have.
#   - not the old ~15 GFLOP/s hand-written contraction number, generalized to
#     anything. It is not this design.
#   - not the k-NN lane's 2.85x nor the linalg lane's 4.7x as universal.
#     Two arms, two shapes.
set -e

cd "$(dirname "$0")/.."
ROUNDS="${MOJOLEARN_GEMM_PRICE_ROUNDS:-3}"
OUT="${MOJOLEARN_GEMM_PRICE_OUT:-/tmp/mojolearn_gemm_price.txt}"
LEG="${OUT}.leg"
: > "$OUT"

echo "== mojolearn.identical.gemm.fp32.v1 -- Phase 4 price, contract 13.6 =="
echo "   rounds $ROUNDS, samples -> $OUT"

# ---------------------------------------------------------------------------
# A5: is another lane on the GPU right now?
# ---------------------------------------------------------------------------
# MATCHED ON THE EXECUTABLE, NOT ON THE COMMAND LINE. The first version of
# this check grepped `ps -o command=` for "pixi run", which matched every
# SHELL WRAPPER whose argv merely CONTAINED that string -- including the very
# shell invoking this script, and including an unrelated agent editing
# pixi.toml. A concurrency check that fires on itself teaches the reader to
# ignore it, which is worse than not having one. `comm` is the executable
# path, so only a real `mojo` or `pixi` process matches.
a5_check() {
    when="$1"
    busy=$(ps -Ao pid=,comm= | grep -E '/(mojo|pixi)$' || true)
    if [ -n "$busy" ]; then
        echo
        echo "!! CONCURRENT BUILD OR RUN DETECTED ($when). Contract 13.5"
        echo "!! assumption A5: no timing produced on this box is"
        echo "!! trustworthy while another lane runs a leg, and the M4"
        echo "!! drifts 1.7x in twenty minutes under heat. The arms this"
        echo "!! harness runs today are HOST arms, so they contend for CPU"
        echo "!! rather than for the GPU -- contaminated, not void. Every"
        echo "!! number below inherits this banner."
        echo "$busy" | while read -r pid _; do
            echo "!!   $(ps -p "$pid" -o command= 2>/dev/null | cut -c1-96)"
        done
        echo
        return 1
    fi
    echo "   no concurrent mojo/pixi process seen $when (A5 check)"
    return 0
}

# CHECKED AT BOTH ENDS. A lane that starts a leg AFTER this script begins
# contaminates every remaining round, and a start-of-run check alone would
# report the run as clean. The end check runs while this script's own pixi
# has exited, so it sees only other people's work.
A5_START=0
a5_check "at start" || A5_START=1

# ---------------------------------------------------------------------------
# One leg: run in one mode, verify the mode came back, keep the samples.
# ---------------------------------------------------------------------------
run_leg() {
    want="$1"      # FAST | IDENTICAL
    shift
    if ! "$@" > "$LEG" 2>&1; then
        echo "!! leg FAILED ($want); its output follows and nothing was kept"
        cat "$LEG"
        return 1
    fi

    # READ THE MODE BACK, twice: the header the binary printed, and every
    # single sample line's own label. The header alone is not enough -- it is
    # printed once at start-up and a sample is what lands in the table.
    got=$(grep -o '^== bench/gemm_price_main.mojo \[[A-Z]*\]' "$LEG" \
          | sed 's/.*\[\(.*\)\]/\1/' || true)
    if [ "$got" != "$want" ]; then
        echo "!! MODE MISMATCH: asked for $want, the binary reports '$got'."
        echo "!! The leg is DISCARDED. A correctly-labelled sample in the"
        echo "!! wrong column is the exact failure the build lock exists to"
        echo "!! prevent, and it has happened in this tree."
        return 1
    fi
    bad=$(grep '^PRICE ' "$LEG" | awk -v w="$want" '$2 != w' || true)
    if [ -n "$bad" ]; then
        echo "!! MISLABELLED SAMPLES in a $want leg; the leg is DISCARDED:"
        echo "$bad" | sed 's/^/!!   /'
        return 1
    fi

    grep '^PRICE ' "$LEG" >> "$OUT" || true
    # The non-PRICE output is the per-shape detail, the workspace table and
    # the gates. Shown, not swallowed: a driver that printed only the medians
    # would hide the 13.6.5 table, which is the finding 13.5 calls bigger
    # than the fold question.
    grep -v '^PRICE ' "$LEG" || true
}

FAILED=0
i=1
while [ "$i" -le "$ROUNDS" ]; do
    echo
    echo "== round $i/$ROUNDS: FAST =="
    # THE FAST ARM TAKES THE BUILD LOCK TOO. Under the lock the mode define
    # cannot move while this round compiles.
    # A discarded leg does not abort the run: the remaining rounds are still
    # worth having and the median table refuses outright if NOTHING survived.
    # It is counted and reported, because a quietly short sample count reads
    # as a noisy machine rather than as a broken driver.
    run_leg FAST tools/with_build_lock.sh \
        pixi run mojo run -I . bench/gemm_price_main.mojo \
        || FAILED=$((FAILED + 1))
    echo
    echo "== round $i/$ROUNDS: IDENTICAL =="
    # with_identical_mode.sh takes the same lock itself before injecting
    # `-D MOJOLEARN_NUMERIC_IDENTICAL=1`.
    run_leg IDENTICAL tools/with_identical_mode.sh \
        pixi run mojo run -I . bench/gemm_price_main.mojo \
        || FAILED=$((FAILED + 1))
    i=$((i + 1))
done

echo
A5_END=0
a5_check "at end" || A5_END=1
if [ "$A5_START" -ne 0 ] || [ "$A5_END" -ne 0 ]; then
    echo "!! A5 FIRED. The medians below were measured beside other work on"
    echo "!! this box. Quote them with that sentence attached or rerun on a"
    echo "!! quiet machine."
fi
if [ "$FAILED" -gt 0 ]; then
    echo "!! $FAILED of $((ROUNDS * 2)) legs were DISCARDED. Their reasons are"
    echo "!! above. The medians below are over what survived."
fi
echo "== medians over $ROUNDS rounds =="
/usr/bin/python3 - "$OUT" "$ROUNDS" <<'PY'
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
if not arms:
    print("NO SAMPLES. Every leg was discarded or the run emitted no PRICE")
    print("lines. That is a failure, not an empty result -- do not read it")
    print("as 'the arms cost the same'.")
    raise SystemExit(1)

width = max(len(a) for a in arms) + 2

# ---------------------------------------------------------------------------
# TABLE 1: what IDENTICAL costs, per arm. The linalg driver's table, same
# shape, so the two are read side by side.
# ---------------------------------------------------------------------------
print()
print("-- what IDENTICAL costs, per arm --")
print("%-*s %10s %10s %8s %s"
      % (width, "arm", "FAST ms", "IDENT ms", "ratio", "samples"))
for a in arms:
    f = rows.get((a, "FAST"), [])
    d = rows.get((a, "IDENTICAL"), [])
    if not f or not d:
        # Reported, never dropped: an arm present in one mode and absent in
        # the other is a broken leg, and a silently omitted row looks exactly
        # like an arm that cost nothing.
        print("%-*s %10s %10s %8s %d/%d  ONE MODE ONLY"
              % (width, a, "%.3f" % med(f) if f else "-",
                 "%.3f" % med(d) if d else "-", "-", len(f), len(d)))
        continue
    mf, md = med(f), med(d)
    print("%-*s %10.3f %10.3f %7.2fx %d/%d"
          % (width, a, mf, md, md / mf if mf else float("nan"),
             len(f), len(d)))

# ---------------------------------------------------------------------------
# TABLE 2: the two arms against each other, INSIDE one mode. This is the
# comparison the harness alternates call by call, so it is the tighter of the
# two and does not carry the two-binaries caveat.
# ---------------------------------------------------------------------------
pairs = []
for a in arms:
    if a.endswith(".oracle"):
        other = a[: -len(".oracle")] + ".serial"
        if other in arms:
            pairs.append((a, other, "oracle", "serial"))
    elif a.endswith(".tree"):
        other = a[: -len(".tree")] + ".serial"
        if other in arms:
            pairs.append((a, other, "tree", "serialfold"))

if pairs:
    print()
    print("-- the two topologies against each other, WITHIN one mode --")
    print("   (alternated call by call inside one process: no drift caveat,")
    print("    but see 13.7 -- this is a HOST ratio between two host")
    print("    functions and says nothing about launches or about 13.4)")
    print("%-*s %11s %11s %11s %11s"
          % (width, "pair", "FAST a/b", "IDENT a/b", "FAST a ms",
             "IDENT a ms"))
    for a, b, na, nb in pairs:
        out = [a[: -len("." + na)]]
        cells = []
        for mode in ("FAST", "IDENTICAL"):
            va, vb = rows.get((a, mode), []), rows.get((b, mode), [])
            cells.append(med(va) / med(vb) if va and vb and med(vb) else None)
        base = []
        for mode in ("FAST", "IDENTICAL"):
            va = rows.get((a, mode), [])
            base.append(med(va) if va else None)
        print("%-*s %11s %11s %11s %11s"
              % (width, out[0],
                 "%.3fx" % cells[0] if cells[0] is not None else "-",
                 "%.3fx" % cells[1] if cells[1] is not None else "-",
                 "%.3f" % base[0] if base[0] is not None else "-",
                 "%.3f" % base[1] if base[1] is not None else "-"))

# ---------------------------------------------------------------------------
# TABLE 3: 13.6.4's two subtractions, done here rather than left to the
# reader, because the whole point of four arms is the differences between
# them and a reader doing the arithmetic by eye will do it once.
# ---------------------------------------------------------------------------
def one(arm, mode):
    v = rows.get((arm, mode), [])
    return med(v) if v else None


if ("m4.leaf.plain", "FAST") in rows or ("m4.leaf.plain", "IDENTICAL") in rows:
    print()
    print("-- 13.6.4: the ftz cost separated from the fma cost --")
    for mode in ("FAST", "IDENTICAL"):
        p = one("m4.leaf.plain", mode)
        f = one("m4.leaf.fma", mode)
        c = one("m4.leaf.contract", mode)
        d = one("m4.leaf.denorm", mode)
        if p is None or f is None or c is None:
            continue
        print("   %-10s plain %.3f  fma %.3f  contract %.3f  denorm %s"
              % (mode, p, f, c, "%.3f" % d if d is not None else "-"))
        print("   %-10s   fma cost = fma - plain      = %+.3f ms (%+.1f%%)"
              % ("", f - p, 100.0 * (f - p) / p if p else 0.0))
        print("   %-10s   ftz cost = contract - fma   = %+.3f ms (%+.1f%%)"
              % ("", c - f, 100.0 * (c - f) / f if f else 0.0))
        if mode == "FAST":
            drift = abs(c - p) / p if p else 0.0
            verdict = "OK" if drift < 0.10 else "SUSPECT"
            print("   %-10s   MODE WITNESS: under FAST both pins compile"
                  " away, so contract should sit on plain." % "")
            print("   %-10s   |contract - plain| / plain = %.1f%%  -> %s"
                  % ("", 100.0 * drift, verdict))
            if verdict == "SUSPECT":
                print("   %-10s   A FAST build whose contract arm does NOT"
                      " collapse onto its plain arm means the mode did not"
                      " reach the leaf loop, and every IDENTICAL number"
                      " above is measuring something other than the pins."
                      % "")

print()
print("READ THE MODE RATIOS AS A BAND. The two modes are two BINARIES and")
print("cannot be interleaved inside one process; %d rounds is %d samples per"
      % (int(sys.argv[2]), int(sys.argv[2])))
print("cell and the M4 drifts up to 1.7x across twenty minutes. The")
print("within-mode table does not carry that caveat -- those arms alternate")
print("call by call -- but it carries contract 13.7's instead: it is a HOST")
print("ratio between two HOST functions, the device kernel is Phase 2b, and")
print("nothing here says the balanced fold is faster or slower on a GPU.")
PY
