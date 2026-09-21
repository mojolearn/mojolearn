#!/bin/sh
# One warm AMD lease, two independent repeated-training candidates. Cloudflare
# R2 is staged once by the provider wrapper; scaler finishes before RF starts.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out
mkdir -p "$OUT"
cd "$ROOT" || exit 9

echo "combined_training_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/combined-training.txt"
sh tools/standard_fit_transform_perf_leg.sh
scaler_rc=$?
echo "scaler_exit=$scaler_rc" >> "$OUT/combined-training.txt"

MOJOLEARN_RF_FUSED_RUN_GUARD=R2_TAXI_ISTELLA \
MOJOLEARN_RF_FUSED_OUT="$OUT/rf-fused-bootstrap" \
MOJOLEARN_RF_FUSED_ROOT="$ROOT" \
  sh tools/rf_fused_bootstrap_leg.sh
rf_rc=$?
echo "rf_exit=$rf_rc" >> "$OUT/combined-training.txt"
echo "combined_training_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/combined-training.txt"

[ "$scaler_rc" -eq 0 ] && [ "$rf_rc" -eq 0 ]
