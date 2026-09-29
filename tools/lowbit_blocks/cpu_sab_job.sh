#!/bin/bash
# lane/lowbit-blocks, NVIDIA pod: (1) the byte_lm binding rebuilt (it imports
# modeling_llama); (2) THE CPU HOST PATH: the model's full logits under the
# profile with MOJOLEARN_VENDOR=cpu (host bindings only), device cpu;
# (3) THE WHOLE-MODEL SABOTAGE ARM: a copy of this tree with
# gemm/checks/sabotage/int15_device_value_flip.patch and
# tools/lowbit_blocks/int15_tuned_epilogue_flip.patch applied (every cell the
# int15 reference kernels and the tuned epilogue store flipped), linalg and transformer rebuilt there, the same
# identity run: its profile hash MUST differ from the clean one and its
# fp32_v1 hash MUST equal the clean one.
set -u
cd "$(dirname "$0")/../.." || exit 9
T=$PWD
export PATH="$HOME/.pixi/bin:$PATH"
BOX=${LB_BOX:-$(hostname -s)}
OUT=$T/bench/results/lowbit_blocks/$BOX/cpu_sab
mkdir -p "$OUT"
PX="pixi run --manifest-path $T/pixi.toml -e default"
ARCH=sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
echo "== build build_byte_lm ($ARCH)"
env MOJOLEARN_GPU_ARCHS=$ARCH MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 $PX sh bindings/build_byte_lm.sh > "$OUT/build_byte_lm.log" 2>&1; echo "build byte_lm exit $?"; tail -2 "$OUT/build_byte_lm.log"
echo "== sabotage tree"
S=/root/lb/sabtree
rm -rf "$S"; mkdir -p "$S"
tar --exclude=./.pixi --exclude=./bench/results --exclude=./.git -cf - . | tar -xf - -C "$S"
(cd "$S" && patch -p1 < gemm/checks/sabotage/int15_device_value_flip.patch && patch -p1 < tools/lowbit_blocks/int15_tuned_epilogue_flip.patch) || { echo "patch failed"; exit 4; }
cat "$S/gemm/checks/gemm_int15.mojo" "$S/gemm/checks/gemm_int15_tuned.mojo" | grep -c "comptime if True:  # SABOTAGE" | sed 's/^/sabotage sites in the copy: /'
for b in build_linalg build_transformer; do
    (cd "$S" && env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 $PX sh bindings/$b.sh > "$OUT/sab_$b.log" 2>&1); echo "sab build $b exit $?"
done
for prof in fixed15_v1 fp32_v1; do
    echo "== sabotaged tree, $prof"
    (cd "$S" && $PX python tools/lowbit_blocks/model_logits.py --model /root/models/SmolLM2-360M --profile $prof --phases identity --box "$BOX-sabotage" --out "$OUT") 2>&1 | grep -v tcmalloc
done
echo "== CPU host path (MOJOLEARN_VENDOR=cpu, device cpu), fixed15_v1 then fp32_v1"
for prof in fixed15_v1 fp32_v1; do
    MOJOLEARN_VENDOR=cpu CUDA_VISIBLE_DEVICES=-1 $PX python tools/lowbit_blocks/model_logits.py --model /root/models/SmolLM2-360M --profile $prof --device cpu --phases identity --box "$BOX-cpu" --out "$OUT" 2>&1 | grep -v tcmalloc
done
echo "out $OUT"
