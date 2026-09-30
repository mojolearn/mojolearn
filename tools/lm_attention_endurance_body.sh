#!/bin/sh
# The shipped/default build, B4 target, 2000 steps, pinned R2 corpus.
set -eu
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/lm-attention-endurance
mkdir -p "$OUT"
cd "$ROOT"
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_GPU_ARCHS=sm_90a
export PYTHONPATH="$ROOT/python:$ROOT"
CORPUS=training/corpus/enwik8/input.txt
printf '%s  %s\n' 2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8 "$CORPUS" | sha256sum -c - > "$OUT/corpus_verified.log"
test "$(wc -c < "$CORPUS" | tr -d ' ')" = 100000000
rm -f python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_byte_lm.so
MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/build_base.log" 2>&1
MOJOLEARN_BUILD_EXTRA_DEFINES="" sh bindings/build_byte_lm.sh > "$OUT/build_default.log" 2>&1
pixi run python tools/lm_ce_alias_probe.py --out "$OUT/batch4" --shape 4 2048 768 12 12 64 2048 12 50257 --steps 2000 --tail 0 --corpus "$CORPUS" --smi-every 10 --witness-every 0 > "$OUT/batch4.log" 2>&1
pixi run python tools/lm_attention_endurance_check.py "$OUT/batch4/result.json" > "$OUT/verdict.log" 2>&1
cat "$OUT/verdict.log"
