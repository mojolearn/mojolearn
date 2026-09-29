#!/bin/bash
# lane/lowbit-default: where the resident generate time goes, on ONE box,
# ALONE on it. generate from 512 tokens with 1 new token (the prefill and one
# pick) and with 32, both profiles alternated: decode per token is
# (t32 - t1) / 31. The bindings are the ones the resident job built.
cd "$(dirname "$0")/../.." || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical
unset MOJOLEARN_NUMERIC_PROFILE
M=/root/models/SmolLM2-360M
BOX=${LB_BOX:-$(hostname -s)}
OUT=$PWD/bench/results/lowbit_default/$BOX/split
mkdir -p "$OUT"
for n in 1 32; do
    pixi run -e default python tools/lowbit_default/default_gate.py --model $M --phases generate --new $n \
        --rounds ${LB_ROUNDS:-7} --box "$BOX-new$n" --out "$OUT" 2>&1 | grep -E "RESULT|GATE|rror|Traceback"
done
