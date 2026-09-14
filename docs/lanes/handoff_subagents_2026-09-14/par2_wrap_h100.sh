# two-device leg of the multi-GPU drivers: the sixteen par-* lanes with MOJOLEARN_PAR_DEVICES=0,1
# on a 2xH100 pod, at the 136-lane record's commit; the column must hash equal to the one-device cells
set -u
cd /root/mojolearn
echo 4048e1b513fc3e443926dafc798de8649f9bd1c5 > commit.txt
OUT=/root/gemm_leg_out; mkdir -p "$OUT"
nvidia-smi -L | tee "$OUT/gpus.txt"
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1); case "$_cc" in 9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;; *) MOJOLEARN_GPU_ARCHS="sm_$(echo "$_cc" | tr -d .)" ;; esac; export MOJOLEARN_GPU_ARCHS; fi
echo "archs=$MOJOLEARN_GPU_ARCHS"
built=0; failed=""
for s in bindings/build*.sh; do
    n=$(basename "$s" .sh); [ "$n" = build_host_family ] && continue
    case "$n" in build_*_host) E="env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu";; *) E="env";; esac
    if $E MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=8 sh "$s" > "$OUT/build_$n.log" 2>&1; then built=$((built+1)); else failed="$failed $n"; fi
done
echo "bindings_built=$built failed=${failed:-none}"
LANES=par-forest,par-forest-et,par-boosting,par-kmeans,par-gram,par-logistic,par-cd,par-svm,par-gp,par-dbscan,par-scaler,par-arima,par-mlp,par-samba,par-byte-lm,par-iforest
env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMMIT=4048e1b513fc3e443926dafc798de8649f9bd1c5 MOJOLEARN_PAR_DEVICES=0,1 PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py --lanes "$LANES" --vendor nvidia-2xh100-sm_90a --json "$OUT/identity_break.nvidia-2xh100-sm_90a.json" > "$OUT/identity_par2.log" 2>&1
echo "identity exit $?"; grep -E "^cells=|^REFUSED" "$OUT/identity_par2.log" | head -8
exit 0
