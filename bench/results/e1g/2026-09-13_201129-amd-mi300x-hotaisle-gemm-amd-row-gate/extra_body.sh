# Shipped-build gate for the AMD kernel-body row (DEVIATION 2707, brief 18.2,
# AMD half): the card under the new shipped path for the Mac to diff against
# the M4's, then tools/gemm_kernel_leg.sh with `shipped` (the new body at
# row 1), `ksplit` (EXACTLY the row-0 dispatch it replaced, at S=110) and
# `kpack_hg` (the arm it was flipped from, which must now read 1.00 against
# shipped), price and LM on both corpora, every witness equal.
set -u
cd /root/mojolearn || exit 9
export MOJOLEARN_NUMERIC_MODE=identical
G=/root/gemm_leg_out/gate; mkdir -p "$G"
sh tools/gemm_card.sh device "$G/amd.card" > "$G/card_driver.log" 2>&1
echo "card_exit=$?" > "$G/gate.txt"
MOJOLEARN_GPU_ARCHS=${MOJOLEARN_GPU_ARCHS:-gfx942} \
MOJOLEARN_TARGET_COLUMN=amd \
MOJOLEARN_COMPILE_JOBS=8 \
MOJOLEARN_GEMM_STEP_LEG_ARMS=shipped,ksplit,kpack_hg \
MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=ksplit,kpack_hg \
MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS=shipped,ksplit,kpack_hg \
MOJOLEARN_GEMM_STEP_LEG_OUT=/root/gemm_leg_out/gemm-amd-gate \
    sh tools/gemm_kernel_leg.sh
echo "kernel_leg_exit=$?" >> "$G/gate.txt"
