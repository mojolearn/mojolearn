#!/bin/sh
set -eu
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/lm-attention-integrated
mkdir -p "$OUT"
cd "$ROOT"
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_GPU_ARCHS=sm_90a
export PYTHONPATH="$ROOT/python:$ROOT"
sh tools/fetch_corpus_enwik8.sh --check > "$OUT/corpus_enwik8.log"
pixi run python tools/lm_attention_report_check.py > "$OUT/report_check.log" 2>&1
rm -f python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_byte_lm.so
MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/build_base.log" 2>&1
DEFS="-D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_NVIDIA"
for site in Z DQ; do
    if pixi run mojo run -j 2 --target-accelerator sm_90a $DEFS -D MOJOLEARN_ATTN_REPAIR_SAB_${site}=1 -I . transformer/checks/attention_masked_tail_check.mojo > "$OUT/sabotage_$site.log" 2>&1; then
        echo "BLIND: $site replay sabotage passed"; exit 1
    fi
    if [ "$site" = Z ]; then pattern='zdot repair differs'; else pattern='dq repair differs'; fi
    grep "$pattern" "$OUT/sabotage_$site.log"
done
pixi run mojo run -j 2 --target-accelerator sm_90a $DEFS -I . transformer/checks/attention_masked_tail_check.mojo > "$OUT/native_gate.log" 2>&1
sh bindings/build_byte_lm.sh > "$OUT/build_default.log" 2>&1
pixi run python tools/lm_ce_alias_probe.py --out "$OUT/repaired" --steps 700 --tail 0 --corpus training/corpus/enwik8/input.txt --smi-every 10 --witness-every 699 > "$OUT/repaired.log" 2>&1
# The source transfer intentionally excludes bench/results. Compare the
# fetched result on the host, where the committed legacy references live:
# python tools/lm_attention_integrated_compare.py <fetched>/repaired/result.json
echo training_complete_host_comparison_required > "$OUT/status.txt"
