#!/usr/bin/env bash
# THE EMBEDDING LANE'S SABOTAGE ARM (workstream D, 2026-09-14), PREPARED.
#
# embedding/checks/embedding_check.mojo has run clause (a) on Apple and AMD
# (cards md5 c7f824c3, 6,887 cells) and ITS OWN HEADER SAYS NO SABOTAGE ARM
# WAS EVER BUILT, so the gate has never been shown capable of failing. The
# sixteen arms exist in embedding/checks/embedding_identical.mojo as
# `is_defined` switches (MOJOLEARN_EMB_SABOTAGE_*) and the check refuses a
# binary whose armed name is not the one the caller expected
# (MOJOLEARN_EMB_EXPECT_SABOTAGE, DEVIATION 1510), which is what makes a
# misspelled -D a failure instead of a silent clean build.
#
# This script builds the clean check once and REQUIRES IT TO PASS, then
# builds every arm below and REQUIRES EACH TO FAIL with its own name in the
# log. A run in which an arm passes is the finding this file exists to
# make, and it is printed by name rather than folded into a count.
#
#     tools/embedding_sabotage_arm.sh OUTPUT_DIRECTORY [ARM ...]
#
# Run inside an activated IDENTICAL toolchain, or set MOJO to its mojo
# binary (the shape of tools/check_embedding_plan_sort.sh). With no ARM
# arguments every arm runs. NEVER RUN ON THE MAC UNDER THE NO-HEAVY-LOCAL-
# COMPUTE RULE; the owed legs are the NVIDIA one (which has never run this
# lane at all) and a rerun of the AMD and Apple columns with the arms.
#
# THE ARMS AND THEIR WITNESS FIXTURES are embedding_check.mojo's: two of
# the eighteen contract arms are not built (its header's findings (1) and
# (2) name FOLD_READS_LAUNCH's block-size dependence and RANK_BY_ARRIVAL's
# two-block requirement, both of which the check drives itself), and the
# sort negative control has its own script. Sixteen defines below.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out=${1:?usage: embedding_sabotage_arm.sh OUTPUT_DIRECTORY [ARM ...]}
shift || true
mkdir -p "$out"
out=$(cd "$out" && pwd)
mojo_bin=${MOJO:-mojo}
jobs=${MOJOLEARN_COMPILE_JOBS:-2}

# One line, on purpose: python/mojolearn/tests/test_expose_d_manifest.py
# reads this string and holds it equal to the is_defined names in
# embedding/checks/embedding_identical.mojo.
ALL_ARMS="FOLD_DESCENDING FOLD_BALANCED_TREE SEED_SEEDLESS SINGLE_RUN_BYPASS EMPTY_ROW_SKIPPED EMPTY_ROW_NEG_ZERO FOLD_READS_LAUNCH RANK_BY_ARRIVAL SORT_TIE_REVERSED PAD_ROW_CONTRIBUTES PAD_ROW_NEG_ZERO NO_FLUSH_ACC GATHER_NO_FLUSH GATHER_CLAMP_OOR ACCUM_BY_ADD ACCUM_REFILLS"
arms=${*:-$ALL_ARMS}

build() {
    # $1 output binary, then the extra defines
    local bin=$1; shift
    "$mojo_bin" build -j "$jobs" -D MOJOLEARN_NUMERIC_IDENTICAL=1 "$@" \
        -I "$repo" "$repo/embedding/checks/embedding_check.mojo" -o "$bin"
}

echo "== clean arm (must PASS)"
build "$out/embedding-check-clean"
if ! MOJOLEARN_EMB_EXPECT_SABOTAGE=none MOJOLEARN_IDENTITY_TRACE="$out/clean.card" \
        "$out/embedding-check-clean" > "$out/clean.log" 2>&1; then
    echo "FAIL: the clean embedding check did not pass; nothing below means anything" >&2
    tail -20 "$out/clean.log" >&2
    exit 1
fi
echo "   clean: PASS ($(wc -l < "$out/clean.log") log lines)"

failed=0
passed_arms=""
for arm in $arms; do
    echo "== sabotage $arm (must FAIL)"
    bin="$out/embedding-check-$arm"
    if ! build "$bin" -D "MOJOLEARN_EMB_SABOTAGE_$arm=1" > "$out/$arm.build.log" 2>&1; then
        echo "   BUILD FAILED for $arm (see $out/$arm.build.log); that is a finding, not a bite" >&2
        failed=$((failed + 1))
        continue
    fi
    if MOJOLEARN_EMB_EXPECT_SABOTAGE="$arm" MOJOLEARN_IDENTITY_TRACE="$out/$arm.card" \
            "$bin" > "$out/$arm.log" 2>&1; then
        echo "   $arm: PASSED WITH THE ARM COMPILED IN. The gate cannot see this arm." >&2
        passed_arms="$passed_arms $arm"
        failed=$((failed + 1))
    elif ! grep -q "sabotage: $arm" "$out/$arm.log"; then
        echo "   $arm: failed, and the log does not name the arm (DEVIATION 1510: a misspelled -D?)" >&2
        failed=$((failed + 1))
    else
        echo "   $arm: FAILED as required ($(grep -c . "$out/$arm.log") log lines)"
    fi
done

{
    echo "embedding sabotage arm, $(date -u +%Y-%m-%dT%H:%M:%SZ), mojo $("$mojo_bin" --version 2>/dev/null | head -1)"
    echo "clean: PASS"
    for arm in $arms; do
        if [ -f "$out/$arm.log" ]; then
            if printf '%s' "$passed_arms" | grep -qw "$arm"; then echo "$arm: PASSED (inert, a finding)"; else echo "$arm: bit"; fi
        else
            echo "$arm: build failed"
        fi
    done
} > "$out/verdict.txt"
cat "$out/verdict.txt"
if [ "$failed" -ne 0 ]; then
    echo "FAIL: $failed arm(s) did not bite or did not build; the inert ones by name:$passed_arms" >&2
    exit 1
fi
echo "PASS: the clean check passes and every requested arm fails by name"
