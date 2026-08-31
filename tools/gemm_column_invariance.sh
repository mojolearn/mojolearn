#!/bin/sh
# THE VENDOR COLUMN MAY NOT REACH THE PRODUCT: one Mac, one mode, three
# vendors, one card.
#
#   tools/gemm_column_invariance.sh
#   MOJOLEARN_COLUMNS="APPLE AMD" tools/gemm_column_invariance.sh
#   MOJOLEARN_GEMM_CARD_ARM=device tools/gemm_column_invariance.sh
#
# Phase 3's exit criterion is *"Apple/NVIDIA/AMD hashes agree"*. Renting three
# machines answers that properly and this does not. What this answers, for
# free and today, is the half of it that lives in OUR SOURCE:
# `checks/kernel_matrix.mojo` makes the vendor a COMPTIME COLUMN, so
# `-D MOJOLEARN_COLUMN_AMD=1` compiles this source against AMD's constants on
# whatever silicon is attached, and the question
#
#     does the CHOICE OF VENDOR COLUMN change the product?
#
# gets a yes/no on one desk. Contract section 6 forbids the k partition to
# depend on the core count, the SM count, the warp width or the vendor, and
# section 7's fold topology is fixed by logical leaf index and not by any
# physical index. Those are properties of the SOURCE'S RESPONSE TO A COLUMN,
# and this gate sees them. This is the sibling of
# `tools/check_linalg_column_invariance.sh` and is deliberately the same
# shape: same env-var names, same read-back guard, same honesty section.
#
# WHAT IT CANNOT SEE, stated up here rather than buried, because a green run
# is very tempting to over-read
# -------------------------------------------------------------------------
#   - THE ARITHMETIC IS STILL METAL'S. Three columns compile onto ONE
#     backend. FMA contraction (IDENTITY_PATHS row 9), denormal policy
#     (row 10) and `sqrt` rounding belong to the BACKEND, not the column, so
#     this gate CANNOT FAIL for any of them no matter how wrong they are.
#     The standing proof that a column can be wrong in a way only its own
#     silicon shows: NVIDIA's `sqrt` is not correctly rounded -- 180,714 of
#     2^20 patterns disagree with the correctly-rounded result, 176,577 of
#     them on normals. No amount of green here would have found that.
#   - A kernel that would not LAUNCH on real AMD -- an LDS budget that fits
#     Metal but not CDNA, a wavefront assumption -- is never exercised.
#   - A refusal is a column-correct answer, not a failure.
#
# AND, TODAY, IT PASSES TRIVIALLY. Said plainly because a gate that passes
# for free and does not admit it is worse than no gate at all: the `oracle`
# and `serial` arms are HOST-ONLY scalar code. They never read
# `kernel_matrix.mojo`, so `-D MOJOLEARN_COLUMN_NVIDIA=1` changes nothing
# they can observe and three identical cards are the only possible outcome.
# The run is still worth having -- it is the harness proving itself against a
# case whose answer is known, and it is what makes the REAL run a one-env-var
# change -- but it is not evidence about the profile. The gate becomes a real
# test the moment Phase 2b's device kernel lands, because that kernel does
# read the column, for its tile shape and its launch geometry, and the
# contract says none of that may reach the bits.
#
# WHEN PHASE 2b LANDS, THE ONLY CHANGE IS ONE ENVIRONMENT VARIABLE:
#
#     MOJOLEARN_GEMM_CARD_ARM=device tools/gemm_column_invariance.sh
#
# Nothing in this file, and nothing in `tools/gemm_card.sh`, needs an edit.
# The trivial-pass banner below switches itself off on any arm that is not a
# host arm, so the gate stops claiming to be free the moment it stops being
# free.
#
# ENVIRONMENT
#   MOJOLEARN_COLUMNS            columns to sweep  (default "APPLE NVIDIA AMD")
#   MOJOLEARN_REPEATS            runs per column   (default 2)
#   MOJOLEARN_GEMM_CARD_ARM      arm under test    (default oracle)
#   MOJOLEARN_COLUMN_OUT         output directory
#   MOJOLEARN_GEMM_COLINV_SABOTAGE
#                                a tag (e.g. "AMD_r2") whose card is
#                                deliberately corrupted after it is emitted.
#                                THE SABOTAGE HANDLE. Today every card on
#                                this desk is identical by construction, so
#                                the DIVERGED branch -- the entire reason
#                                this gate exists -- would never once have
#                                executed before Phase 2b. A branch that has
#                                never run is not a branch that is gated
#                                (`reached-but-inert`). Set this and the run
#                                MUST go red and MUST name the corrupted
#                                stage. If it stays green, this gate is
#                                decoration.
set -e

cd "$(dirname "$0")/.."

COLUMNS="${MOJOLEARN_COLUMNS:-APPLE NVIDIA AMD}"
REPEATS="${MOJOLEARN_REPEATS:-2}"
ARM="${MOJOLEARN_GEMM_CARD_ARM:-oracle}"
OUT="${MOJOLEARN_COLUMN_OUT:-bench/results/gemm_column_invariance/$(date +%Y-%m-%d_%H%M%S)}"

mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || true

case "$ARM" in
    oracle|serial) HOST_ONLY=1 ;;
    *)             HOST_ONLY=0 ;;
esac

echo "== gemm.fp32.v1 column invariance, NUMERIC_IDENTICAL =="
echo "   profile: mojolearn.identical.gemm.fp32.v1"
echo "   arm:     $ARM"
echo "   columns: $COLUMNS"
echo "   repeats: $REPEATS per column"
echo "   out:     $OUT"
echo
if [ "$HOST_ONLY" = "1" ]; then
    echo "   TRIVIAL PASS EXPECTED. The '$ARM' arm is host-only scalar code"
    echo "   that never reads checks/kernel_matrix.mojo, so the column"
    echo "   defines cannot reach the answer and three matching cards prove"
    echo "   nothing about the profile. This run exercises the HARNESS. The"
    echo "   gate becomes real at MOJOLEARN_GEMM_CARD_ARM=device."
    echo
fi

# REPEATS ARE PART OF THE CLAIM, NOT A FLAKE FILTER. One comparison carries
# two separate properties and both are load-bearing:
#   across REPEATS   the same binary on the same device returns the same
#                    product. Not a formality: the unsupervised lane measured
#                    the FAST k-NN default returning three different answers
#                    in three consecutive runs on one M4, so "we ran it once
#                    and it was right" is a claim this repo has already been
#                    burned by.
#   across COLUMNS   the vendor's comptime constants do not reach the
#                    product. That is contract section 6's "not the vendor,
#                    not the core count, not the warp width" made falsifiable.
# Collapsing repeats to one would still compare columns and would silently
# stop testing the first property.

fail=0
base=""
base_tag=""

for col in $COLUMNS; do
    rep=1
    while [ "$rep" -le "$REPEATS" ]; do
        tag="${col}_r${rep}"
        card="$OUT/${ARM}_${tag}.card"

        # Driven through tools/gemm_card.sh rather than calling `mojo run`
        # here, so the mode read-back guard and the skipped-shape accounting
        # are written ONCE and both gates get the same ones. That script
        # takes the build lock per run, which is what keeps this sweep
        # interleavable with the other three sessions on this checkout.
        if MOJOLEARN_GEMM_CARD_DEFINES="-D MOJOLEARN_COLUMN_${col}=1" \
           MOJOLEARN_GEMM_CARD_OUT="$OUT" \
           tools/gemm_card.sh "$ARM" "$card" > "$OUT/${tag}.drv" 2>&1; then
            sed -n 's/^  arm /  '"$tag"': arm /p' "$OUT/${tag}.drv"
        else
            echo "  $tag: FAILED"
            sed 's/^/      /' "$OUT/${tag}.drv" | tail -14
            fail=1
            rep=$((rep + 1))
            continue
        fi

        if [ "${MOJOLEARN_GEMM_COLINV_SABOTAGE:-}" = "$tag" ]; then
            # Flip one hex digit of one stage hash. Not a truncation and not
            # a deletion: those would also trip the differ's PARSE and
            # ALIGNMENT steps, and a red that comes from step 1 does not
            # prove step 3 works.
            awk 'NR==3 && NF==5 { $5 = ($5 ~ /^0/ ? "1" : "0") substr($5,2) }
                 { print }' OFS='\t' "$card" > "$card.sab" && mv "$card.sab" "$card"
            echo "      SABOTAGED (MOJOLEARN_GEMM_COLINV_SABOTAGE=$tag)"
        fi

        if [ -z "$base" ]; then
            base="$card"
            base_tag="$tag"
            echo "      BASE for the sweep"
        elif cmp -s "$base" "$card"; then
            echo "      identical to $base_tag"
        else
            echo "      DIVERGED from $base_tag -- first divergence:"
            /usr/bin/python3 tools/identity_trace_diff.py "$base" "$card" \
                --labels "$base_tag","$tag" 2>&1 \
                | sed 's/^/        /' | head -24 || true
            fail=1
        fi
        rep=$((rep + 1))
    done
done

echo
if [ -z "$base" ]; then
    echo "gemm column invariance: RED -- not one run produced a card."
    exit 1
fi
if [ "$fail" -ne 0 ]; then
    echo "gemm column invariance: RED. Cards in $OUT"
    exit 1
fi

echo "gemm column invariance: every column and every repeat produced the"
echo "SAME card for arm '$ARM'. Cards in $OUT"
echo
echo "READ THIS BEFORE QUOTING THE GREEN:"
if [ "$HOST_ONLY" = "1" ]; then
    echo "  * This was the TRIVIAL case. Arm '$ARM' is host-only and cannot"
    echo "    observe a column define, so this result was determined before"
    echo "    the first build. It says the harness works. It says NOTHING"
    echo "    about cross-vendor identity of the profile."
else
    echo "  * The source's column-dependence is neutralized under IDENTICAL"
    echo "    for arm '$ARM': the k partition, the fold topology and the"
    echo "    output bits did not move when the vendor's comptime constants"
    echo "    did."
fi
echo "  * ONE BACKEND, three columns. FMA contraction, denormal policy and"
echo "    sqrt rounding are the BACKEND's, not the column's, and this gate"
echo "    cannot fail for any of them. NVIDIA's sqrt disagrees with the"
echo "    correctly-rounded result on 180,714 of 2^20 patterns (176,577 on"
echo "    normals) and this gate would stay green through all of it."
echo "  * A kernel that would not launch on real AMD is not exercised here."
echo "  * Real Apple/NVIDIA/AMD hardware still owns the Phase 3 criterion."
