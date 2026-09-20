#!/bin/bash
# THE WATCHED CPU-ONLY RUN A PROMOTION NEEDS, run on a RunPod CPU pod by
# tools/runpod_cpu_leg.sh (lane/verifier-full-exposure, 2026-09-20).
#
# `host_surface.public_reference_lanes()` says what makes a lane public: a
# covered lane, in a release record's scope, with a reference in the shipped
# table, WATCHED TO READ CLEAN BY A CPU-ONLY RUN AT THIS COMMIT. For the
# thirteen lanes below the first three clauses are already true and the fourth
# is the whole remaining blocker. It needs no GPU, which is why this is a CPU
# pod and not a rental of anything scarce.
#
# It is deliberately TWO runs, in this order:
#
#   1. the thirteen held lanes, with --batch-checks so every part the shipped
#      table carries for them is compared, not just the five default ones. A
#      promotion rests on this one.
#   2. the whole public set as a user gets it, which is the run whose verdict
#      and 256-lane accounting are the thing being claimed.
#
# The first runs first on purpose: if the lease runs out during the second,
# the evidence the promotion needs is already written.
set -u
: "${LEG_OUT:?LEG_OUT must be set by the leg}"

HELD="mamba3,transformer,transformer-window,samba,samba-untied-dropout-accum"
HELD="$HELD,grad-accumulation,hf-causal-lm,hf-checkpoint,hf-tokenizer"
HELD="$HELD,linalg-qr,linalg-eigh,linalg-svdvals,lowbit-conversions"

echo "### commit $MOJOLEARN_COMMIT  mode ${MOJOLEARN_NUMERIC_MODE:-unset}"
python3 -c "import mojolearn, mojolearn._backend as b; print('vendor', b.vendor(), 'mode', b.numeric_mode())" \
    2>&1 | tee "$LEG_OUT/backend.txt"
python3 -m mojolearn verify --coverage --json-out "$LEG_OUT/coverage.json" \
    > "$LEG_OUT/coverage.txt" 2>&1
echo "coverage exit $?"

echo "### 1. the thirteen held lanes, all nine fixtures, all parts"
python3 -m mojolearn verify --all --include-pending --batch-checks --no-models \
    --lanes "$HELD" --json-out "$LEG_OUT/held13.json" \
    > "$LEG_OUT/held13.txt" 2>&1
echo "held13 exit $?"
tail -40 "$LEG_OUT/held13.txt"

echo "### 2. the public set as a user gets it"
python3 -m mojolearn verify --all --json-out "$LEG_OUT/public.json" \
    > "$LEG_OUT/public.txt" 2>&1
echo "public exit $?"
tail -60 "$LEG_OUT/public.txt"

echo "### 3. the gates that must agree with all of it"
python3 tools/lane_accounting.py --self-test > "$LEG_OUT/lane_accounting_selftest.txt" 2>&1
echo "lane_accounting --self-test exit $?"
python3 tools/lane_accounting.py --check > "$LEG_OUT/lane_accounting_check.txt" 2>&1
echo "lane_accounting --check exit $?"
tail -5 "$LEG_OUT/lane_accounting_check.txt"
exit 0
