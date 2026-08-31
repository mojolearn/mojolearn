#!/bin/sh
# The stage-hash instrument, gated as ONE tool: writer, then reader.
#
#   pixi run check-identity-trace
#
# `original/identity_trace_check.mojo` proves the WRITER is honest (it is
# self-consistent, it localizes a one-cell perturbation, it hashes contents
# and not the grid width, it can see a denormal, and its dumps agree with its
# hashes). This script then proves the READER agrees with that writer, which
# no amount of testing either half alone can establish: a writer and a reader
# that were only ever tested apart are two programs, not one instrument.
#
# The last step is the END-TO-END one. The writer leaves a clean/dirty pair
# whose only difference is +0.0 against the smallest denormal at one
# histogram cell -- the SHAPE of a real Metal-versus-CUDA divergence, since
# Metal flushes subnormals and CUDA does not. The differ must (a) name the
# stage, (b) trust the dumps, and (c) classify the cell as DENORMAL-vs-ZERO
# rather than as ULP noise. If it reported the divergence without the class,
# a real cross-vendor run would send the reader hunting for an accumulation
# order that was never the problem.
set -e

echo "== writer =="
mojo run ${MOJOLEARN_MOJO_DEFINES:-} -I . original/identity_trace_check.mojo

echo
echo "== reader, own selftest =="
python3 tools/identity_trace_diff.py --selftest

echo
echo "== end to end: writer's pair through the reader =="
OUT=$(python3 tools/identity_trace_diff.py \
        /tmp/mojolearn_it_pair_clean.trace \
        /tmp/mojolearn_it_pair_dirty.trace \
        --labels METAL,CUDA || true)

fail=0
echo "$OUT" | grep -q "FIRST DIVERGENCE: fixture.hist" || {
    echo "  FAIL the differ did not name fixture.hist as the first divergence"
    fail=1
}
echo "$OUT" | grep -q "Both dumps re-hash to their recorded hashes" || {
    echo "  FAIL the differ could not verify the dumps against their records"
    fail=1
}
echo "$OUT" | grep -q "DENORMAL-vs-ZERO" || {
    echo "  FAIL the differ did not classify the planted cell as DENORMAL-vs-ZERO"
    fail=1
}
if [ "$fail" -ne 0 ]; then
    echo
    echo "$OUT"
    exit 1
fi
echo "  ok   stage named, dumps verified, cell classified DENORMAL-vs-ZERO"
echo
echo "identity trace: writer and reader agree. PASS"
