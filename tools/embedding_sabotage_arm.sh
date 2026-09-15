#!/usr/bin/env bash
# THE EMBEDDING LANE'S SABOTAGE ARM (workstream D, 2026-09-14). First run on an H100 and an
# MI300X at b163b76ba (bench/results/ivf_embed_km_legs_2026-09-14/README.md).
#
# embedding/checks/embedding_check.mojo carries clause (a) (card md5
# c7f824c3, 6,887 cells, identical on Apple, NVIDIA and AMD) and clause (g),
# the sabotage verdicts. The sixteen arms exist in
# embedding/checks/embedding_identical.mojo as `is_defined` switches
# (MOJOLEARN_EMB_SABOTAGE_*) and the check refuses a binary whose armed name
# is not the one the caller expected (MOJOLEARN_EMB_EXPECT_SABOTAGE,
# DEVIATION 1510), which is what makes a misspelled -D a failure instead of
# a silent clean build.
#
# This script builds the clean check once and REQUIRES IT TO PASS, then
# builds every arm below and requires clause (g) to show each one BIT, or,
# for NO_FLUSH_ACC on a device that flushes the raw add, asserted INERT. An
# arm that raises is the finding this file exists to make, and it is printed
# by name rather than folded into a count.
#
#     tools/embedding_sabotage_arm.sh OUTPUT_DIRECTORY [ARM ...]
#
# Run inside an activated IDENTICAL toolchain, or set MOJO to its mojo
# binary (the shape of tools/check_embedding_plan_sort.sh). With no ARM
# arguments every arm runs. On the Mac run it on one core
# (MOJOLEARN_COMPILE_JOBS=1): clean plus sixteen arms took 122 seconds on
# the M4 on 2026-09-14. Per-column verdicts:
# bench/results/ivf_embed_km_legs_2026-09-14/ (the first round, five
# findings) and bench/results/embedding_sabotage_2026-09-14/ (after they
# were resolved).
#
# THE ARMS AND THEIR WITNESS FIXTURES are embedding_check.mojo's: all
# eighteen rows of contract 11.1 since 2026-09-15, when
# EMB_FOLD_VIA_GEMM_ONEHOT (the backward through identical_gemm over a one-hot
# matrix, checked cell by cell against the host GEMM oracle) and
# EMB_SORT_KEY_ID_ONLY_UNSTABLE (PLAN_SORT's compare pass on the id half of
# the key; clause (a) runs PLAN_SORT on that build) were built. The sort
# negative control has its own script. Eighteen defines below.
#
# THE CLEAN RUN is every clause the check has: (a), (b), (c) and (f) always,
# and (d) and (e) turned on here, so "clean: PASS" means all six passed.
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
ALL_ARMS="FOLD_DESCENDING FOLD_BALANCED_TREE SEED_SEEDLESS SINGLE_RUN_BYPASS EMPTY_ROW_SKIPPED EMPTY_ROW_NEG_ZERO FOLD_READS_LAUNCH RANK_BY_ARRIVAL SORT_TIE_REVERSED PAD_ROW_CONTRIBUTES PAD_ROW_NEG_ZERO NO_FLUSH_ACC GATHER_NO_FLUSH GATHER_CLAMP_OOR ACCUM_BY_ADD ACCUM_REFILLS FOLD_VIA_GEMM_ONEHOT SORT_KEY_ID_ONLY_UNSTABLE"
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
        MOJOLEARN_EMB_CHECK_CLAUSE_D=1 MOJOLEARN_EMB_CHECK_CLAUSE_E=1 \
        "$out/embedding-check-clean" > "$out/clean.log" 2>&1 \
        || ! grep -q "^embedding_check: GREEN, clauses (a), (b), (c), (f), (d), (e) PASS" "$out/clean.log"; then
    echo "FAIL: the clean embedding check did not pass; nothing below means anything" >&2
    tail -20 "$out/clean.log" >&2
    exit 1
fi
echo "   clean: PASS, $(grep -m1 "^embedding_check: GREEN" "$out/clean.log")"

# HOW A SABOTAGE BUILD REPORTS (read from the check, 2026-09-14 legs). Under a
# sabotage build embedding_check.mojo INVERTS its own verdict: clause (g)
# requires the armed switch to MOVE its witness case, FIRST at the stage its
# own clause writes, and to leave its predicted-inert control unmoved, and the
# binary EXITS 0 when all of that holds ("clause (g): <ARM> BIT on <case>").
# It RAISES when the arm is inert on its witness, moves its control, or moves
# an earlier stage. So an exit 0 WITH the BIT line is the arm caught, and a
# raise is the finding, printed with its message. The first version of this
# script required a non-zero exit and read every caught arm as inert; the
# first H100 and MI300X runs of it at b163b76ba showed that.
#   ACCUM_BY_ADD  no kernel reads its switch (the wrong spelling lives in the
#                 caller), so the armed build only names it; clause (e) of the
#                 CLEAN build computes the by-add result itself and asserts it
#                 moves the straddling rows and nothing else. Its verdict is
#                 the clean binary run with MOJOLEARN_EMB_CHECK_CLAUSE_E=1.
#   NO_FLUSH_ACC  on a column whose DEVICE flushes the raw add (Apple) the
#                 check asserts the arm inert on every case (contract 9.3)
#                 and exits 0 with an "INERT ON THIS COLUMN" line; recorded
#                 as inert on this column, not counted as bit and not a
#                 failure. Its reach is shown on NVIDIA and AMD.
#   GATHER_CLAMP_OOR  runnable since the arm also drops the device entry
#                 point's id refusal (it was unrunnable while the refusal
#                 stood in front of the clamp); an older binary that still
#                 says NO RUNNABLE WITNESS is recorded as not runnable.
failed=0
inert_arms=""
bit_arms=""
raised_arms=""
unshown_arms=""
norun_arms=""
for arm in $arms; do
    echo "== sabotage $arm (clause (g) must show it BIT)"
    bin="$out/embedding-check-$arm"
    if ! build "$bin" -D "MOJOLEARN_EMB_SABOTAGE_$arm=1" > "$out/$arm.build.log" 2>&1; then
        echo "   BUILD FAILED for $arm (see $out/$arm.build.log)" >&2
        echo "$arm: build failed" >> "$out/verdicts.tmp"
        failed=$((failed + 1))
        continue
    fi
    if [ "$arm" = ACCUM_BY_ADD ]; then
        MOJOLEARN_EMB_EXPECT_SABOTAGE="$arm" MOJOLEARN_IDENTITY_TRACE="$out/$arm.card" "$bin" > "$out/$arm.log" 2>&1 || true
        if MOJOLEARN_EMB_CHECK_CLAUSE_E=1 MOJOLEARN_EMB_EXPECT_SABOTAGE=none MOJOLEARN_IDENTITY_TRACE="$out/$arm.clause_e.card" \
                "$out/embedding-check-clean" > "$out/$arm.clause_e.log" 2>&1 \
                && grep -q "^clause (e) by-add control:" "$out/$arm.clause_e.log"; then
            echo "   $arm: BIT under clause (e) of the clean build ($(grep -m1 "^clause (e) by-add control:" "$out/$arm.clause_e.log" | cut -c28-140))"
            echo "$arm: bit (clause (e) by-add control, clean build)" >> "$out/verdicts.tmp"
            bit_arms="$bit_arms $arm"
        else
            msg=$(grep -m1 "Unhandled exception" "$out/$arm.clause_e.log" | cut -c1-400 || true)
            echo "   $arm: clause (e) did not show it: ${msg:-no by-add control line}" >&2
            echo "$arm: NOT SHOWN (clause (e)): ${msg:-no by-add control line}" >> "$out/verdicts.tmp"
            unshown_arms="$unshown_arms $arm"
            failed=$((failed + 1))
        fi
        continue
    fi
    if MOJOLEARN_EMB_EXPECT_SABOTAGE="$arm" MOJOLEARN_IDENTITY_TRACE="$out/$arm.card" \
            "$bin" > "$out/$arm.log" 2>&1; then
        if grep -q "^clause (g): $arm BIT on" "$out/$arm.log"; then
            echo "   $arm: BIT ($(grep -m1 "^clause (g): $arm BIT on" "$out/$arm.log" | cut -c13-120))"
            echo "$arm: bit" >> "$out/verdicts.tmp"
            bit_arms="$bit_arms $arm"
        elif grep -q "^clause (g): $arm INERT ON THIS COLUMN" "$out/$arm.log"; then
            echo "   $arm: INERT ON THIS COLUMN, asserted ($(grep -m1 "^device probe:" "$out/$arm.log" | cut -c1-120))"
            echo "$arm: inert on this column (device flushes the raw add, asserted on every case)" >> "$out/verdicts.tmp"
            inert_arms="$inert_arms $arm"
        else
            echo "   $arm: exit 0 but no clause (g) BIT line; NOT SHOWN" >&2
            echo "$arm: NOT SHOWN (exit 0, no BIT line)" >> "$out/verdicts.tmp"
            unshown_arms="$unshown_arms $arm"
            failed=$((failed + 1))
        fi
    elif ! grep -q "sabotage: $arm" "$out/$arm.log"; then
        echo "   $arm: failed, and the log does not name the arm (DEVIATION 1510: a misspelled -D?)" >&2
        echo "$arm: failed unnamed" >> "$out/verdicts.tmp"
        failed=$((failed + 1))
    elif grep -q "HAS NO RUNNABLE WITNESS" "$out/$arm.log"; then
        echo "   $arm: NOT RUNNABLE (the check refuses it by name: no runnable witness)"
        echo "$arm: not runnable (refused by the check)" >> "$out/verdicts.tmp"
        norun_arms="$norun_arms $arm"
    else
        msg=$(grep -m1 "Unhandled exception" "$out/$arm.log" | cut -c1-400 || true)
        echo "   $arm: RAISED: $msg" >&2
        echo "$arm: RAISED: $msg" >> "$out/verdicts.tmp"
        raised_arms="$raised_arms $arm"
        failed=$((failed + 1))
    fi
done

{
    echo "embedding sabotage arm, $(date -u +%Y-%m-%dT%H:%M:%SZ), mojo $("$mojo_bin" --version 2>/dev/null | head -1)"
    echo "clean: PASS"
    cat "$out/verdicts.tmp" 2>/dev/null
} > "$out/verdict.txt"
rm -f "$out/verdicts.tmp"
cat "$out/verdict.txt"
if [ "$failed" -ne 0 ]; then
    echo "FAIL: $failed arm(s) not shown; raised:${raised_arms:- none}; not shown:${unshown_arms:- none}; inert on this column:${inert_arms:- none}; not runnable (not counted):${norun_arms:- none}" >&2
    exit 1
fi
echo "PASS: the clean check passes; BIT:${bit_arms:- none}; inert on this column (asserted, contract 9.3):${inert_arms:- none}; not runnable (not counted):${norun_arms:- none}"
