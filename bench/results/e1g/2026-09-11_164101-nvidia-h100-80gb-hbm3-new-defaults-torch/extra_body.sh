#!/bin/sh
# Neural H100 leg body: the new shipped defaults against the old ones, then every
# torch column, on ONE pod (RunPod H100 pods ran at two speeds on 2026-09-11, so
# the ratio to torch must be same-pod).
#
# 1. tools/attention_flip_r3_leg.sh: LM step for stash_tiled (the old attention
#    default) and the new default (attention stash_tiled_fgrid_r32_qres_pf,
#    DEVIATION 2534; its binding also carries the GEMM ksplit default, 2595) on
#    enwik8 and Pile GitHub, plus the shipped fused check.
# 2. tools/torch_lm_step_opponent_leg.sh: eager_fp32, eager_tf32, eager_bf16,
#    compile_fp32, compile_tf32, compile_bf16 on the same corpora.
set -u
cd /root/mojolearn || exit 9
sh tools/attention_flip_r3_leg.sh
a=$?
echo "attention_flip_leg_exit=$a" >> /root/gemm_leg_out/leg.txt
sh tools/torch_lm_step_opponent_leg.sh
t=$?
echo "torch_leg_exit=$t" >> /root/gemm_leg_out/leg.txt
[ "$a" = 0 ] && [ "$t" = 0 ]
