#!/bin/sh
# tools/linalg_python_gate.sh -- the Python linalg identity gate, the way it
# is meant to be run (2026-09-13). `python/mojolearn/tests/test_linalg_identity.py`
# FAILS, by design, when no oracle card is supplied ("a silently skipped
# identity test is worse than no test"), and the card is emitted by
# tools/gemm_card.sh oracle from the NORMATIVE host oracle, which builds the
# GEMM sources. This script does the three steps in order so nobody rediscovers
# the card:
#
#   1. build the IDENTICAL linalg extension (the only tier linalg ships,
#      DEVIATION 2490: the FAST half of the old recipe is retired)
#   2. emit the oracle card into bench/results/gemm_card/<stamp>/oracle.card
#   3. run the gate under IDENTICAL with MOJOLEARN_GEMM_CARD pointing at it
#
#   pixi run check-linalg-python
#   MOJOLEARN_GEMM_CARD=<existing card> sh tools/linalg_python_gate.sh   # skips 2
set -eu
cd "$(dirname "$0")/.."
MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_linalg.sh
if [ -z "${MOJOLEARN_GEMM_CARD:-}" ]; then
    STAMP=$(date +%Y-%m-%d_%H%M%S)
    OUT="bench/results/gemm_card/$STAMP"
    mkdir -p "$OUT"
    sh tools/gemm_card.sh oracle "$OUT/oracle.card"
    MOJOLEARN_GEMM_CARD="$PWD/$OUT/oracle.card"
    export MOJOLEARN_GEMM_CARD
fi
echo "== linalg python gate: MOJOLEARN_GEMM_CARD=$MOJOLEARN_GEMM_CARD"
cd python && MOJOLEARN_NUMERIC_MODE=identical python3 -m mojolearn.tests.test_linalg_identity
