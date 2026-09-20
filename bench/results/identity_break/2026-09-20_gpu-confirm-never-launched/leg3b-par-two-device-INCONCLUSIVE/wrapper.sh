#!/bin/sh
# MOJOLEARN_GEMM_LEG_EXTRA wrapper for lane/gpu-confirm-never-launched, leg 3.
#
# THE par-* HALF OF THE LANES ADDED SINCE v0.8.8, ON TWO DEVICES.
# Nine lanes, every one of them starting `par-`, which is what the
# two-device phase requires -- ONLY a par-* lane may be credited from a
# column `admit(..., par_axis=True)` admits.
#
# FIVE OF THE NINE HAVE NO GPU COLUMN OF ANY KIND: par-gpc-fit,
# par-gpc-predict, par-causal-lm and par-cross-val read gpu=[] and par_two=[]
# in tools/verification_matrix.py today, and par-ordered / par-border-types
# carry apple and nvidia but the two new gpc lanes and the two Python-side
# drivers have never run on a GPU at all.
#
# WHAT THE TWO-DEVICE PHASE ACTUALLY STATES. gap_column_leg.sh runs the
# one-device column FIRST (par_devices="0", the REFERENCE) and then the same
# lane list at MOJOLEARN_PAR_DEVICES=0,1 (the EVIDENCE), on the same build,
# the same box and the same commit, and diffs them on the box while it can
# still be asked again. A par-* driver's claim is not stateable on one
# device -- lane_applicability.degenerate('nvidia-1gpu') holds every one of
# them on the device axis -- and the default admit rule refuses a two-device
# column, so this pair is the only shape that says anything.
#
# THIS LEG NEEDS MOJOLEARN_GEMM_LEG_GPU_COUNT=2. The body counts visible
# GPUs and REFUSES the phase by name on a one-GPU box rather than running a
# second copy of the one-device column and calling the tautology agreement.
#
# DEFAULT PARTS ON PURPOSE. The par-* axis is the equality of two columns,
# and every committed par two-device column is five parts wide; widening
# only this one would make the pair incomparable to the 55-lane par matrix
# it has to join.
MOJOLEARN_GAP_LANES=par-ordered,par-border-types,par-forecast-arima,par-forecast-holtwinters,par-ivf,par-gpc-fit,par-gpc-predict,par-causal-lm,par-cross-val
MOJOLEARN_GAP_TWO_DEVICE=1
MOJOLEARN_GAP_SLUG=par-two-device-new-since-088-2026-09-20
MOJOLEARN_GAP_COMMIT_DIR=bench/results/identity_break/2026-09-20_gpu-confirm-never-launched/
MOJOLEARN_COMPILE_JOBS=16
# The runner's work bound is MINUTES*60-600 = 3000 s and it polls to
# deadline_epoch-300. 2850 leaves the body room to write its sentinel and the
# runner room to fetch. The default 2500 was set for a leg whose derivation
# names a handful of families; this one builds all of them up front.
MOJOLEARN_GAP_BUDGET=2850
export MOJOLEARN_GAP_LANES MOJOLEARN_GAP_TWO_DEVICE MOJOLEARN_GAP_SLUG \
       MOJOLEARN_GAP_COMMIT_DIR MOJOLEARN_COMPILE_JOBS MOJOLEARN_GAP_BUDGET
exec sh /root/mojolearn/tools/gap_column_leg.sh
