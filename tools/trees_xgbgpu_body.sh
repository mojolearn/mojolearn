#!/bin/sh
# The amd_xgboost GPU follow-up of lane trees-hotaisle for runners with no
# body environment (tools/gemm_remote_leg.sh on RunPod AMD, 2026-09-11):
# taxi then Istella-S, depthwise and lossguide only, ours interleaved with
# XGBoost on the GPU in one pod. AMD ships amd_xgboost only as manylinux_2_39
# wheels, so launch it with
# MOJOLEARN_GEMM_LEG_IMAGE_AMD=rocm/dev-ubuntu-24.04:6.4.1-complete (glibc
# 2.39). The body refuses before any timing unless the GPU probe passed.
# Runs from /root/mojolearn.
MOJOLEARN_TREES_HA_LEG=taxi,istella
MOJOLEARN_TREES_HA_CELLS=xgb
export MOJOLEARN_TREES_HA_LEG MOJOLEARN_TREES_HA_CELLS
exec sh tools/trees_hotaisle_body.sh
