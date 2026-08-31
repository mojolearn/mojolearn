#!/bin/sh
# THE COLUMN MAY NOT REACH THE PRODUCT: one machine, one mode, three vendors.
#
#   pixi run check-linalg-column-invariance
#   MOJOLEARN_COLUMNS="APPLE AMD" pixi run check-linalg-column-invariance
#
# IDENTITY_PATHS rows 27 and 28. The sibling of
# `tools/check_column_invariance.sh`, which does this for rows 19-26, and it
# is deliberately the same shape: same env vars, same contamination guard,
# same repeats-are-part-of-the-claim reasoning. THE DEBT IS NAMED, NOT
# HIDDEN -- the two drivers differ only in which trace main they run and
# which arms they name, and they should become one driver with an arm
# registry when either lane next touches the other's file. They are separate
# today because that file is the unsupervised lane's and is under edit.
#
# WHAT THIS IS FOR, AND WHAT IT HONESTLY IS NOT
# ---------------------------------------------
# `checks/kernel_matrix.mojo` makes the vendor a COMPTIME COLUMN, so
# `-D MOJOLEARN_COLUMN_AMD` compiles this source against AMD's constants on
# whatever device is attached. That answers, for free and today,
#
#     does the CHOICE OF VENDOR COLUMN change the product?
#
# which is most of what rows 27a and 27b are about: an arm selected from the
# device's column, and a k partition derived from its core count. Those
# defects live in the SOURCE'S RESPONSE TO A COLUMN, not in the silicon, and
# this gate sees them.
#
# It has already earned its keep on this lane. Before DEVIATION 521 the AMD
# column under FAST partitions k into 880 chunks where Apple takes 240, and
# `gram_splitk_applies(32, 32, ...)` answers FALSE on it -- the shipped
# PCA/OLS Gram shape does not enter the pinned kernel at all on that column.
# Both facts are printed by every run below, so a reader does not have to
# take the claim on trust.
#
# WHAT IT CANNOT SEE, stated plainly because a green run is tempting to
# over-read:
#
#   - the arithmetic is still Metal's. Contraction (row 9), denormal policy
#     (row 10) and `sqrt`'s rounding (row 10's NVIDIA half, corrected
#     2026-08-23) are properties of the BACKEND, and this gate compiles
#     three columns onto one backend. It cannot fail for any of them, and
#     the NVIDIA sqrt defect is the standing proof that a column can be
#     wrong in a way only its own silicon shows.
#   - a kernel that would not launch on real AMD (an LDS budget that fits
#     Metal but not CDNA) is not exercised.
#   - REFUSALS ARE COLUMN-CORRECT ANSWERS, not failures.
#
# A green run says: the source's column-dependence is neutralized under
# IDENTICAL for the matrix products. A red run is a cross-vendor identity
# bug found without renting anything, which is the whole point. E1 still
# owns the rest.
set -e

cd "$(dirname "$0")/.."

COLUMNS="${MOJOLEARN_COLUMNS:-APPLE NVIDIA AMD}"
ARMS="${MOJOLEARN_LINALG_ARMS:-gram nt gemv}"
OUT="${MOJOLEARN_COLUMN_OUT:-bench/results/linalg_column_invariance/$(date +%Y-%m-%d_%H%M%S)}"
REPEATS="${MOJOLEARN_REPEATS:-2}"
mkdir -p "$OUT"

if [ "${MOJOLEARN_LIN_COLINV_INNER:-}" != "1" ]; then
    echo "== linalg column invariance, NUMERIC_IDENTICAL =="
    echo "   columns: $COLUMNS"
    echo "   arms:    $ARMS"
    echo "   out:     $OUT"
    git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || true
    MOJOLEARN_LIN_COLINV_INNER=1 MOJOLEARN_COLUMN_OUT="$OUT" \
        exec tools/with_identical_mode.sh "$0"
fi

# ---- re-entered with the tree flipped and the build lock held -------------
#
# REPEATS ARE PART OF THE CLAIM, NOT A FLAKE FILTER. Two properties are
# gated by one comparison and both are load-bearing:
#   across REPEATS   the same binary on the same device gives the same
#                    product. Not a formality -- the unsupervised lane
#                    measured the FAST k-NN default returning three
#                    different answers in three consecutive runs on one M4.
#   across COLUMNS   the vendor's constants do not reach the product.

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
        if MOJOLEARN_IDENTITY_TRACE="$card" MOJOLEARN_LINALG_ARM="$arm" \
            pixi run mojo run ${MOJOLEARN_MOJO_DEFINES:-} -D "MOJOLEARN_COLUMN_${col}=1" -I . \
            bench/linalg_trace_main.mojo > "$log" 2>&1; then

            # THE CONTAMINATION GUARD. The mode is READ BACK from the run
            # rather than assumed from the flip: a build that lands inside
            # another session's flip window compiles the other arm and every
            # label inside it agrees with the binary, so the flip proves
            # nothing (DEVIATION 514).
            got_mode=$(grep "^mode " "$log" | head -1 | awk '{print $2}')
            if [ "$got_mode" != "IDENTICAL" ]; then
                echo "  $arm/$tag: CONTAMINATED -- compiled as ${got_mode:-<none>},"
                echo "      not IDENTICAL. Another session is flipping"
                echo "      checks/numerics.mojo. Re-run when it is idle."
                fail=1
                rep=$((rep + 1))
                continue
            fi

            chunks=$(grep "^gram_chunks " "$log" | head -1 | awk '{print $2}')
            armtaken=$(grep "^gram_splitk_arm " "$log" | head -1 | awk '{print $2}')
            gotcol=$(grep "^column " "$log" | head -1 | awk '{print $2}')

            if [ -z "$base" ]; then
                base="$card"
                echo "  $arm/$tag: BASE (column $gotcol, chunks $chunks, split-K arm $armtaken)"
            elif cmp -s "$base" "$card"; then
                echo "  $arm/$tag: match (column $gotcol, chunks $chunks, split-K arm $armtaken)"
            else
                echo "  $arm/$tag: DIVERGED (column $gotcol, chunks $chunks, split-K arm $armtaken)"
                echo "      first divergence:"
                /usr/bin/python3 tools/identity_trace_diff.py "$base" "$card" \
                    2>&1 | sed 's/^/      /' | head -20 || true
                fail=1
            fi
        else
            echo "  $arm/$tag: RUN FAILED"
            tail -5 "$log" | sed 's/^/      /'
            fail=1
        fi
        rep=$((rep + 1))
      done
    done
done

echo
if [ "$fail" -ne 0 ]; then
    echo "linalg column invariance: RED. Cards in $OUT"
    exit 1
fi
echo "linalg column invariance: every column and repeat produced the SAME"
echo "card, for every arm. Cards in $OUT"
echo
echo "This is ONE BACKEND compiled against three columns. It cannot see"
echo "contraction, denormal policy or sqrt rounding, which are the"
echo "backend's and not the column's -- see the header. E1 on real"
echo "hardware still owns those."
