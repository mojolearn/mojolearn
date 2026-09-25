#!/bin/sh
# tools/nvidia_step_time/arms.sh -- lane/nvidia-step-time: every GEMM step
# trial arm through bench/gemm_excp_ab_main.mojo at the T3 shapes on the H100,
# hashes held to the branch build's (ab/branch.hashes). Runs on the box.
set -u
cd /root/mojolearn || exit 9
PATH="$HOME/.pixi/bin:/usr/local/cuda/bin:$PATH"; export PATH
O=/root/gemm_leg_out/nv-step-time/ab
B=/root/nv_bin
[ -x $B/ab_trial ] || pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_GEMM_ARM_TRIAL=1 \
    --target-accelerator sm_90a -I . bench/gemm_excp_ab_main.mojo -o $B/ab_trial > $O/trial.build.log 2>&1
grep '^EXCP_AB call' $O/branch.log | grep ordinary | sed 's/ ms=.*//' > $O/branch.ord.hashes
for arm in ${ARMS:-shipped tuned128 lfold half half_ks16 quarter kpack kpack_wide kpack_hg ksplit ksplit_leaf kfoldv}; do
    MOJOLEARN_EXCP_AB_KINDS=ordinary MOJOLEARN_EXCP_AB_ROUNDS=3 MOJOLEARN_GEMM_ARM=$arm $B/ab_trial > $O/arm-$arm.log 2>&1
    grep '^EXCP_AB call' $O/arm-$arm.log | sed 's/ ms=.*//' > $O/arm-$arm.hashes
    if cmp -s $O/branch.ord.hashes $O/arm-$arm.hashes; then v=IDENTICAL; else v=DIFFER; fi
    echo "arm $arm vs_branch=$v $(grep '^EXCP_AB call' $O/arm-$arm.log | sed 's/.*call=\([a-zA-Z_]*\).* ms=\([0-9.]*\).*/\1=\2/' | tr '\n' ' ')"
done
