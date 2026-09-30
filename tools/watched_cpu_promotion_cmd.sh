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

# THE LIBRARY NO BINDING BUILD PRODUCES, AND `import mojolearn` CANNOT SKIP.
# Measured here on 2026-09-20 and already documented in tools/gap_column_leg.sh
# after an AMD leg died the same way: every binding built, not one cell
# recorded, one second in, at
#
#   OSError: python/mojolearn/.libs/libMojolearnMath.so: cannot open shared
#   object file: No such file or directory
#
# `_training_impl.py`'s `def kaiming_uniform(self, shape, fan_in,
# a=math.sqrt(5.0))` evaluates that sqrt as a DEFAULT ARGUMENT at class
# definition time, so the dlopen happens during import and no lane can avoid
# it. Nothing under bindings/ builds it, `python/mojolearn/.libs/` is
# gitignored so nothing ships it, and a developer Mac has it lying around from
# some past wheel build and never notices. This calls the tree's OWN recipe
# rather than retyping its compiler flags, because those flags
# (-ffp-contract=off, -fno-fast-math, -nostdlib) ARE the arithmetic contract
# and a second copy of them is a second answer to it.
PYTHONPATH=packaging/portable_math python3 -c \
    "import pathlib, stage; print(stage.build(pathlib.Path('python/mojolearn/.libs/libMojolearnMath.so')))" \
    > "$LEG_OUT/portable_math.log" 2>&1
echo "portable_math build exit $?"
ls -l python/mojolearn/.libs/ >> "$LEG_OUT/portable_math.log" 2>&1
tail -3 "$LEG_OUT/portable_math.log"
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
