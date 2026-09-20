#!/bin/sh
# MOJOLEARN_GEMM_LEG_EXTRA for lane/gpu-confirm-never-launched, leg 5:
# THE stepfull PART FOR SEVEN LANES THAT HAVE IT ON NO VENDOR AT ALL.
#
# Measured over all 559 admissible columns under bench/results/identity_break
# and the shipped python/mojolearn/verify_reference/table.json, 2026-09-20:
# 29 (lane, class) pairs are in the shipped table for a lane the harness
# declares a STEPFULL probe for, with no `stepfull` entry, AND no committed
# column carries one either. Seven of those lanes are missing it on ALL THREE
# vendor classes:
#
#   mamba1                  amd apple nvidia
#   mamba2                  amd apple nvidia
#   par-samba               amd apple nvidia
#   par-samba-clip          amd apple nvidia
#   par-byte-lm             amd apple nvidia
#   par-byte-lm-model-pool  amd apple nvidia
#   par-byte-lm-offload     amd apple nvidia
#
# `stepfull` is the DECODE-STATE part: forward(x) against allocate_state +
# step, position by position. So decode identity for Mamba-1 and Mamba-2 --
# core shipped blocks -- has never been verified on any vendor, ever. It was
# at least visible as a gap until OWED stopped gating; now it costs a run
# nothing, so it can sit indefinitely. This column takes the NVIDIA third.
#
# WHY THE PART WAS NEVER THERE, WHICH IS NOT A FAILED RUN. Nobody skipped it.
# `identity_break` runs train/infer/model/batch/rlpair at its DEFAULTS and
# needs --step-full, --batch-grad, --batch-scale and --ragged to be asked.
# Only 249 of 559 committed columns carry stepfull at all, and the flagship
# records -- 2026-09-14_118-lanes, 166-lanes, 120-lanes-2711flip -- are
# themselves four-part. The part was never asked for, on any of them.
#
# ONE DEVICE, DELIBERATELY, FOR THE par-* LANES TOO. MOJOLEARN_GAP_TWO_DEVICE
# is NOT set. These five par-* lanes are degenerate on the DEVICE axis here
# and that is fine: the gap being closed is `stepfull`, whose claim is about
# decode state and not about sharding, and the lane's own in-cell comparison
# against the plain fit still fires. A one-device par column records
# par_devices="0" and `admit()` admits it by the default rule, which is the
# rule the shipped table reads.
MOJOLEARN_GAP_LANES=mamba1,mamba2,par-samba,par-samba-clip,par-byte-lm,par-byte-lm-model-pool,par-byte-lm-offload
MOJOLEARN_GAP_SLUG=stepfull-never-recorded-2026-09-20
MOJOLEARN_GAP_COMMIT_DIR=bench/results/identity_break/2026-09-20_gpu-confirm-never-launched/
MOJOLEARN_GAP_PARTS="--step-full --batch-grad --batch-scale --ragged"
MOJOLEARN_COMPILE_JOBS=16
export MOJOLEARN_GAP_LANES MOJOLEARN_GAP_SLUG MOJOLEARN_GAP_COMMIT_DIR \
       MOJOLEARN_GAP_PARTS MOJOLEARN_COMPILE_JOBS
exec sh /root/mojolearn/tools/gap_column_leg.sh
