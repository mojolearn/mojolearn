#!/bin/sh
# MOJOLEARN_GEMM_LEG_EXTRA wrapper for lane/gpu-confirm-never-launched, leg 1.
#
# THE OWED NVIDIA COLUMN FOR FIVE NEURAL LANES, AT ALL NINE PARTS, plus the
# two decode-session lanes that have never run on any hardware at all.
#
# Measured on the committed tree, 2026-09-20:
#   mamba3                      20 nvidia columns, ZERO at norms-near-one-1
#   transformer                 2 admissible at norms-near-one-1, 5/9 parts
#   transformer-window          2 admissible at norms-near-one-1, 5/9 parts
#   samba                       2 admissible at steps-1-1,        5/9 parts
#   samba-untied-dropout-accum  2 admissible at steps-3-1,        5/9 parts
# and all four of those columns sit under bench/results/attention_replay_*,
# outside bench/results/identity_break/, the only tree build_table walks.
# So this is a fresh column, not a repair.
#
#   mamba1-decode-session       no GPU column anywhere, ever
#   transformer-decode-session  no GPU column anywhere, ever
# Their constructors live only in bindings/_mojolearn_mamba.mojo and
# bindings/_mojolearn_transformer.mojo and refuse BY NAME on the host route,
# so a GPU column is the only place their proposition is stateable.
#
# This file sets the variables and execs the body OUT OF THE PINNED ARCHIVE,
# so the body that runs is the committed one. tools/gemm_remote_leg.sh runs
# the extra body with NO ENVIRONMENT PASSTHROUGH; naming gap_column_leg.sh
# directly rents a box, runs with an empty lane list and exits 8.
MOJOLEARN_GAP_LANES=mamba3,transformer,transformer-window,samba,samba-untied-dropout-accum,mamba1-decode-session,transformer-decode-session
MOJOLEARN_GAP_SLUG=neural-nine-parts-2026-09-20
MOJOLEARN_GAP_COMMIT_DIR=bench/results/identity_break/2026-09-20_gpu-confirm-never-launched/
# The four opt-in parts. Without these the column is five of nine wide and
# the stepfull gap these lanes actually have stays open while reading closed.
MOJOLEARN_GAP_PARTS="--step-full --batch-grad --batch-scale --ragged"
MOJOLEARN_COMPILE_JOBS=16
export MOJOLEARN_GAP_LANES MOJOLEARN_GAP_SLUG MOJOLEARN_GAP_COMMIT_DIR \
       MOJOLEARN_GAP_PARTS MOJOLEARN_COMPILE_JOBS
exec sh /root/mojolearn/tools/gap_column_leg.sh
