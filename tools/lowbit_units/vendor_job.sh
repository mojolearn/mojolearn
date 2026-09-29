#!/bin/bash
# tools/lowbit_units/vendor_job.sh -- lane/lowbit-units, the COMPARISON arms:
# the vendor library's strict fp32, TF32 (CUDA only) and bf16 products at the
# transformer rows of bench/gemm_shapes.mojo, through
# tools/vendor_gemm_price.py. COMPARISON ONLY: no identity claim, no hash.
#
# Linux (NVIDIA, AMD): tools/remote_vendor_torch.sh builds a throwaway venv
# with the box's own torch wheel. Apple: the pixi `skgpu` environment, whose
# torch reaches Metal through MPS.
#
# Writes bench/results/lowbit_units/<box>/vendor/{vendor.log,vendor_price.json}
# and prints the log, so a steward's stdout carries it home.
set -u
cd "$(dirname "$0")/../.." || exit 9
BOX=${MOJOLEARN_LOWBIT_BOX:-$(hostname -s)}
OUT="$PWD/bench/results/lowbit_units/$BOX/vendor"
mkdir -p "$OUT"
export PATH="$HOME/.pixi/bin:$PATH"
ARGS="--bf16 --only llama8b"
if [ "$(uname -s)" = Darwin ]; then
    pixi run -e skgpu python tools/vendor_gemm_price.py --repeats "${VENDOR_REPEATS:-10}" --warmup 3 \
        $ARGS --out "$OUT/vendor_price.json" > "$OUT/vendor.log" 2>&1
    rc=$?
else
    VENDOR_EXTRA_ARGS="$ARGS" VENDOR_MAX_MACS="${VENDOR_MAX_MACS:-1e12}" \
        VENDOR_PRICE_OUT="$OUT/vendor_price.json" bash tools/remote_vendor_torch.sh > "$OUT/vendor.log" 2>&1
    rc=$?
fi
cat "$OUT/vendor.log"
echo "vendor_job: box=$BOX exit=$rc"
exit $rc
