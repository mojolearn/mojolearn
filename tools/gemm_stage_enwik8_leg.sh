#!/bin/sh
set -eu
exec sh "${MOJOLEARN_GEMM_STEP_ROOT:-/root/mojolearn}/tools/gemm_stage_training_leg.sh" enwik8
