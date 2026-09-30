#!/bin/sh
# Final AMD default qualification, including every tuned loader affected.
set -eu
cd "${MOJOLEARN_GEMM_STEP_ROOT:-/root/mojolearn}"
OUT=/root/gemm_leg_out/stage-default
mkdir -p "$OUT"
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_GPU_ARCHS=gfx942
build() {
    pixi run mojo build -j 2 --target-accelerator "$MOJOLEARN_GPU_ARCHS" \
        -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . "$@"
}
for gate in gather tuned; do
    source=gemm/checks/gemm_gather_ftz_check.mojo
    defect=MOJOLEARN_GEMM_SABOTAGE_GATHER_FTZ
    if [ "$gate" = tuned ]; then
        source=gemm/checks/gemm_stage_ftz_check.mojo
        defect=MOJOLEARN_GEMM_SABOTAGE_TUNED_STAGE_FTZ
    fi
    build -D "$defect=1" "$source" -o "$OUT/check" > "$OUT/build-negative-$gate.log" 2>&1
    if "$OUT/check" > "$OUT/negative-$gate.log" 2>&1; then
        echo "FAIL: omitted $gate operand flush accepted"; exit 9
    fi
    grep 'got=.*flat=.*oracle=' "$OUT/negative-$gate.log"
    for arm in base stage; do
        EXTRA=
        [ "$arm" = stage ] || EXTRA='-D MOJOLEARN_GEMM_LEGACY_STAGE_FTZ=1'
        build $EXTRA "$source" -o "$OUT/check" > "$OUT/build-$gate-$arm.log" 2>&1
        "$OUT/check" > "$OUT/check-$gate-$arm.log" 2>&1
    done
done
for arm in base stage; do
    EXTRA=
    [ "$arm" = stage ] || EXTRA='-D MOJOLEARN_GEMM_LEGACY_STAGE_FTZ=1'
    build $EXTRA -D MOJOLEARN_GEMM_ARM_TRIAL=1 bench/gemm_step_price_main.mojo \
        -o "$OUT/price-$arm" > "$OUT/build-price-$arm.log" 2>&1
done
build -D MOJOLEARN_GEMM_ARM_TRIAL=1 -D MOJOLEARN_GEMM_PRICE_SABOTAGE=1 \
    bench/gemm_step_price_main.mojo -o "$OUT/price-negative" > "$OUT/build-price-negative.log" 2>&1
if MOJOLEARN_GEMM_ARM=kpack_hg MOJOLEARN_GEMM_STEP_CALLS=proj_fwd \
    "$OUT/price-negative" > "$OUT/price-negative.log" 2>&1; then
    echo 'FAIL: sabotaged price arm accepted'; exit 9
fi
grep '^BITS .* MOVED' "$OUT/price-negative.log"
for entry in base-1 stage-1 stage-2 base-2; do
    arm=${entry%-*}
    MOJOLEARN_GEMM_ARM=shipped MOJOLEARN_GEMM_STEP_ROUNDS=11 \
        "$OUT/price-$arm" > "$OUT/price-$entry.log" 2>&1
done
python3 tools/gemm_class_price_summary.py "$OUT"/price-base-*.log "$OUT"/price-stage-*.log > "$OUT/summary.log"
python3 - "$OUT" <<'PYCODE'
import re, sys
from pathlib import Path
root = Path(sys.argv[1])
def check(text, expected):
    rows = re.findall(r'^STAGE_FTZ enabled=(True|False)$', text, re.M)
    assert rows == [expected], (rows, expected)
for arm, expected in [('base', 'False'), ('stage', 'True')]:
    for bad in ('', 'STAGE_FTZ enabled='+str(expected == 'False'),
                ('STAGE_FTZ enabled='+expected+'\n')*2):
        try: check(bad, expected)
        except AssertionError as e: print('EXPECTED FAIL stage dispatch', e)
        else: raise AssertionError('blind dispatch gate')
    for p in [root/('check-tuned-'+arm+'.log'), *root.glob('price-'+arm+'-*.log')]:
        check(p.read_text(), expected)
        print('MATCH', p.name, 'STAGE_FTZ', expected)
PYCODE
rm "$OUT/check" "$OUT/price-negative" "$OUT/price-base" "$OUT/price-stage"
echo COMPLETE
