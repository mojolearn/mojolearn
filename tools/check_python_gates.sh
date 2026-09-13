#!/usr/bin/env bash
# Run EVERY gate-style Python test under python/mojolearn/tests/.
#
# WHY THIS EXISTS. There are two shapes of test in that directory and only one
# of them was ever invoked by anything.
#
#   - pytest files (37 of 58): collected by `pixi run -e test test-python`.
#   - GATE-STYLE files (13): they define no test functions at all. They run as
#     MODULES, print their own arm-by-arm verdicts and exit non-zero on RED.
#     pytest collects NOTHING from them, which is correct and is also why they
#     were invisible: until 2026-09-13 exactly ONE pixi task ran exactly ONE of
#     them (check-arima-surface), and no CI workflow ran any.
#
# That is not hypothetical rot. test_samba_surface and test_transformer_hd128
# sat RED for days after the numpy-free migration took `.view`/`.mean` off
# Array, and nothing looked. Same disease as the packaging tests in 88c0c33b.
#
# THE CARD. test_linalg_identity is the one gate with an EXTERNAL reference: it
# compares the Python surface against gemm_oracle's own answer and FAILS rather
# than skips when no card is supplied, because every other arm in it is a
# RELATIVE comparison that could be self-consistently wrong everywhere. The card
# is ~15 s of `mojo run` and needs no GPU, so this script just makes one.
#
# GATE_FULL. Without it the linalg gate DEFERS 4 shapes over its element budget
# and says so: "not skipped quietly and not counted as agreeing". Running them
# costs about 2 s (112 checks -> 116) so this script always asks for them.
#
#   bash tools/check_python_gates.sh          # all gates
#   MOJOLEARN_GATES_SKIP_CARD=1 bash ...      # reuse $MOJOLEARN_GEMM_CARD
set -u

cd "$(dirname "$0")/.." || exit 2
repo="$PWD"

export MOJOLEARN_NUMERIC_MODE=identical

# The card, unless the caller supplied one already.
if [ "${MOJOLEARN_GATES_SKIP_CARD:-0}" != "1" ]; then
    card="${MOJOLEARN_GEMM_CARD:-/tmp/mojolearn_gates_oracle.card}"
    printf '== oracle card ==\n'
    if tools/gemm_card.sh oracle "$card" >/tmp/mojolearn_gates_card.log 2>&1; then
        printf '  %s\n' "$(tail -1 /tmp/mojolearn_gates_card.log)"
        export MOJOLEARN_GEMM_CARD="$card"
    else
        printf '  CARD GENERATION FAILED (log /tmp/mojolearn_gates_card.log)\n'
        printf '  test_linalg_identity will report RED, which is the correct\n'
        printf '  outcome for an identity gate with no external reference.\n'
    fi
fi
export MOJOLEARN_LINALG_GATE_FULL=1

# Discover them rather than listing them, so a new gate file is covered the day
# it lands instead of the day someone remembers to add it here.
cd python || exit 2
gates=$(
    for f in mojolearn/tests/test_*.py; do
        grep -q 'import pytest' "$f" && continue
        grep -q 'unittest.TestCase' "$f" && continue
        basename "$f" .py
    done
)

printf '\n== gate-style python tests ==\n'
pass=0
fail=0
failed=""
for g in $gates; do
    if out=$(python3 -m "mojolearn.tests.$g" 2>&1); then
        pass=$((pass + 1))
        printf '  PASS  %-34s %s\n' "$g" "$(printf '%s' "$out" | tail -1 | cut -c1-58)"
    else
        fail=$((fail + 1))
        failed="$failed $g"
        printf '  FAIL  %-34s %s\n' "$g" "$(printf '%s' "$out" | tail -1 | cut -c1-58)"
        printf '%s\n' "$out" | tail -12 | sed 's/^/          /'
    fi
done

printf '\n  pass=%d fail=%d\n' "$pass" "$fail"
if [ "$fail" -ne 0 ]; then
    printf '  RED:%s\n' "$failed"
    exit 1
fi
printf '  ALL GATES GREEN\n'
