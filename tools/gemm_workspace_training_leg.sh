#!/bin/sh
set -eu
ROOT=${MOJOLEARN_GEMM_STEP_ROOT:-/root/mojolearn}
sh "$ROOT/tools/gemm_stage_training_leg.sh" enwik8 workspace
sh "$ROOT/tools/gemm_stage_training_leg.sh" pile_github workspace
python3 "$ROOT/tools/gemm_training_compare.py" /root/gemm_leg_out/gemm-workspace-training --kind workspace
