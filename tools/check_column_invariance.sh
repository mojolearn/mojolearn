#!/bin/sh
# THE COLUMN MAY NOT REACH THE ANSWER: one machine, one mode, three vendors.
#
#   pixi run check-column-invariance
#   MOJOLEARN_COLUMNS="APPLE AMD" pixi run check-column-invariance
#
# WHAT THIS IS FOR, AND WHAT IT HONESTLY IS NOT
# ---------------------------------------------
# `archive/research/UNSUPERVISED_IDENTITY.md`'s first owed item is "a second vendor", and the
# only way to discharge it completely is E1: the same commit on an M4 and on
# an MI300X, cards diffed. That needs hardware, costs money, and takes a day.
#
# This gate is the half of that debt which does NOT need hardware, and it is
# a bigger half than it looks. `checks/kernel_matrix.mojo` makes the
# vendor a COMPTIME COLUMN: block sizes, lane widths, shared-memory budgets,
# occupancy and every grid computed from them are constants chosen by
# `TARGET_COLUMN`, and `-D MOJOLEARN_COLUMN_AMD` compiles this source against
# AMD's numbers on whatever device is attached. So the question
#
#     does the CHOICE OF VENDOR COLUMN change the answer?
#
# can be asked here, today, for free. That question is most of what rows 8,
# 21 and 22 are about -- a 64-wide fold, a block size feeding a float sum, a
# grid shape reaching an argmin -- because those defects live in the SOURCE'S
# response to a column, not in the silicon.
#
# WHAT IT CANNOT SEE, stated plainly because a green run here will be
# tempting to over-read:
#
#   - the arithmetic is still Metal's. Contraction (row 9), denormal policy
#     (row 10) and the vendor's transcendental implementations (row 12) are
#     properties of the BACKEND, and this gate compiles three columns onto
#     one backend. It cannot fail for any of them.
#   - a kernel that would not launch on real AMD (an LDS budget that fits
#     Metal but not CDNA, or the reverse) is not exercised.
#   - A KERNEL COUPLED TO THE HARDWARE'S LANE WIDTH CANNOT BE ANSWERED HERE
#     AT ALL, and this is the sharpest limit. `vote` returns one bit per
#     lane of the REAL wavefront, so a 64-lane ballot compiled for
#     COLUMN_AMD and executed on a 32-lane Metal warp is not an
#     approximation of AMD's answer, it is garbage: the top half never
#     sets and every `pop_count(mask & lid_mask)` position is wrong. Such
#     checks call `column_is_simulated()` and report NOT ANSWERABLE.
#     Before DEVIATION 515 they did something worse -- `RBC_LANES` was the
#     literal 32, so an "AMD" build silently compiled a 32-lane ball cover
#     and the check PASSED, testing the Apple kernel under an AMD header.
#   - REFUSALS ARE COLUMN-CORRECT ANSWERS, not failures: the fused k-NN arm
#     raises on a 64-lane column by design (row 23), which is why the k-NN
#     arm here goes through the default and DEVIATION 509 pins that to the
#     tiled selector on every column.
#
# A green run says: the source's column-dependence is neutralized under
# IDENTICAL. A red run is a cross-vendor identity bug found without renting
# anything, which is the whole point.
#
# THE MODE FLIP is session-local through `tools/with_identical_mode.sh`,
# which holds the shared build lock for its whole window (DEVIATION 514) and
# reverts on EXIT/INT/TERM. It must never be committed.
set -e

cd "$(dirname "$0")/.."

COLUMNS="${MOJOLEARN_COLUMNS:-APPLE NVIDIA AMD}"
ARMS="${MOJOLEARN_ARMS:-kmeans knn dbscan}"
OUT="${MOJOLEARN_COLUMN_OUT:-bench/results/column_invariance/$(date +%Y-%m-%d_%H%M%S)}"
mkdir -p "$OUT"

if [ "${MOJOLEARN_COLINV_INNER:-}" != "1" ]; then
    echo "== column invariance, NUMERIC_IDENTICAL =="
    echo "   columns: $COLUMNS"
    echo "   arms:    $ARMS"
    echo "   out:     $OUT"
    git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || true
    MOJOLEARN_COLINV_INNER=1 MOJOLEARN_COLUMN_OUT="$OUT" \
        exec tools/with_identical_mode.sh "$0"
fi

# ---- re-entered with the tree flipped and the build lock held -------------

# REPEATS ARE PART OF THE CLAIM, NOT A FLAKE FILTER. Every (arm, column)
# runs `MOJOLEARN_REPEATS` times and every repeat must produce the same card
# as the first run of the first column. Two different properties are being
# gated by one comparison and both are load-bearing:
#
#   across REPEATS   the same binary on the same device gives the same
#                    answer. Not a formality: measured 2026-08-23, the FAST
#                    k-NN default returned THREE DIFFERENT sorted index sets
#                    in three consecutive runs of this exact fixture on one
#                    M4 -- `updateSortedWarpQ`'s mutex merge resolving an
#                    equidistant tie by arrival order (IDENTITY_PATHS row
#                    11). A gate that compared columns but not repeats could
#                    have called that green by luck.
#   across COLUMNS   the vendor's constants do not reach the answer.
#
# Under IDENTICAL both must hold. The same six runs under FAST produce
# several answers, which is what makes this gate a measurement rather than a
# tautology.
REPEATS="${MOJOLEARN_REPEATS:-2}"

fail=0
for arm in $ARMS; do
    base=""
    for col in $COLUMNS; do
      rep=1
      while [ "$rep" -le "$REPEATS" ]; do
        tag="${col}_r${rep}"
        card="$OUT/${arm}_${tag}.card"
        log="$OUT/${arm}_${tag}.log"
        : > "$card"
        if MOJOLEARN_IDENTITY_TRACE="$card" MOJOLEARN_UNSUP_ARM="$arm" \
            pixi run mojo run ${MOJOLEARN_MOJO_DEFINES:-} -D "MOJOLEARN_COLUMN_${col}=1" -I . \
            bench/unsupervised_trace_main.mojo > "$log" 2>&1; then
            # THE CONTAMINATION GUARD (DEVIATION 514's other half).
            # `checks/numerics.mojo` is a SHARED FILE that this gate
            # rewrites for the length of its run, and this checkout is
            # worked by parallel sessions. A build that lands inside
            # another session's flip window compiles the OTHER arm and says
            # nothing about it, because the mode is a comptime constant and
            # every label in the binary agrees with the binary. So the mode
            # is not assumed from the flip -- it is READ BACK from the run,
            # which prints what it was compiled against.
            #
            # This is not defensive decoration. It caught a real
            # contamination the first time it ran: a verification pass that
            # was supposed to be the FAST arm reported `mode IDENTICAL` on
            # both columns, because a concurrent session held the flip.
            got_mode=$(grep "^mode " "$log" | head -1 | awk '{print $2}')
            if [ "$got_mode" != "IDENTICAL" ]; then
                echo "  $arm/$tag: CONTAMINATED -- compiled as ${got_mode:-<none>},"
                echo "      not IDENTICAL. Another session is flipping"
                echo "      checks/numerics.mojo. Re-run when it is idle."
                fail=1
                rep=$((rep + 1))
                continue
            fi
            # NOT `^column`: that line REPORTS which column was built and
            # differs by construction. What must not move is the inputs,
            # the outputs and the shapes derived from them.
            grep -E "^input\.|^output\.|^query_tile" "$log" \
                > "$OUT/${arm}_${tag}.hashes" || true
            echo "  $arm/$tag: ran ($got_mode), $(grep -c '	' "$card") stages"
        else
            # A column that does not RUN is a finding and is reported as
            # one; it is not silently skipped and it is not a pass.
            echo "  $arm/$tag: FAILED TO RUN --"
            grep -E "Unhandled|error:" "$log" | head -3 | sed 's/^/      /'
            fail=1
            rep=$((rep + 1))
            continue
        fi
        if [ -z "$base" ]; then
            base="$tag"
            rep=$((rep + 1))
            continue
        fi
        # THE CARD DIFF is the claim; the console hashes are the summary.
        # Both are compared, because a card can agree stage by stage while
        # the caller-visible output differs (a divergence after the last
        # recorded stage) and the reverse is the row-9 lesson: outputs that
        # match by argmax margin over stages that do not.
        if python3 tools/identity_trace_diff.py \
            "$OUT/${arm}_${base}.card" "$OUT/${arm}_${tag}.card" \
            > "$OUT/${arm}_${base}_vs_${tag}.diff" 2>&1; then
            cards="cards IDENTICAL"
        else
            cards="CARDS DIVERGE"
            fail=1
        fi
        if diff -q "$OUT/${arm}_${base}.hashes" "$OUT/${arm}_${tag}.hashes" \
            > /dev/null 2>&1; then
            outs="outputs IDENTICAL"
        else
            outs="OUTPUTS DIFFER"
            fail=1
        fi
        echo "  $arm: $base vs $tag -- $cards, $outs"
        rep=$((rep + 1))
      done
    done
done

echo
if [ "$fail" -ne 0 ]; then
    echo "column invariance: RED. See $OUT for the cards and diffs."
    echo "A divergence here is a source-level cross-vendor identity defect:"
    echo "the arithmetic was the same backend's, so only the column moved."
    exit 1
fi
echo "column invariance: every column produced the same cards and the same"
echo "outputs, on one device, under IDENTICAL. Artifacts in $OUT."
echo "This is NOT a cross-vendor measurement -- the arithmetic was this"
echo "machine's on every arm. E1 is still owed; see archive/evidence/E1_RUNBOOK.md."
