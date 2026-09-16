#!/bin/sh
# tools/predict_kmeans_amd_leg.sh: the on-box body
# (MOJOLEARN_GEMM_LEG_EXTRA for tools/gemm_remote_leg.sh amd) of
# lane/classical-host-recordings, 2026-09-16.
#
#   MOJOLEARN_GPU_ARCHS=gfx942 \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/predict_kmeans_amd_leg.sh \
#   sh tools/gemm_remote_leg.sh amd --rent --local-card <card>
#
# WHAT IS OWED HERE. lane/saved-model-reference-gaps took the NVIDIA column
# and the recording for dbscan, agglomerative, spectral and
# spectral-precomputed and wrote down that the AMD ones were owed, because
# AMD was being left alone by a standing instruction. That instruction is
# lifted. lane/kmeans-save took no AMD column either. So this box answers
# both, in that order: the older debt first, and a box that dies late still
# comes home with it.
#
# THE ARCH IS NOT AUTODETECTED FROM A DRIVER QUERY THE WAY sm_NN IS. The
# runner requires an explicit MOJOLEARN_GPU_ARCHS for AMD; this body reads
# rocminfo as a fallback and REFUSES to guess a family it cannot see.
#
# POSIX sh. `set -u`, deliberately not `set -e`: a phase that fails is a
# finding and its log must come home.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/predict-kmeans-amd
mkdir -p "$OUT/logs"
G="$OUT/gate.txt"
PREDICT_REC=bench/results/classical_host/2026-09-16-amd-predict
KMEANS_REC=bench/results/classical_host/2026-09-16-amd-kmeans
PREDICT_GATE=dbscan,agglomerative,spectral,spectral-precomputed
PREDICT_ID=dbscan,dbscan-brute-l1,dbscan-weighted,agglomerative,spectral,spectral-precomputed
KMEANS_GATE=kmeans,kmeans-random,kmeans-array,kmeans-weighted,kmeans-sqrt,kmeans-classic-pp
KMEANS_ID=kmeans,kmeans-random,kmeans-array,kmeans-weighted,kmeans-sqrt,kmeans-classic-pp,kmeans-cosine
SAB=/root/host-sabotage
JOBS="${MOJOLEARN_COMPILE_JOBS:-8}"

say() { echo "$@" >> "$G"; }
run() {
    _n=$1; shift
    _t0=$(date +%s)
    "$@" > "$OUT/logs/$_n.log" 2>&1
    _e=$?
    echo "$_n	$_e	$(( $(date +%s) - _t0 ))" >> "$OUT/status.tsv"
    return "$_e"
}
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ ! -s /root/mojolearn/commit.txt ]; then
    _c=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)
    case "$_c" in
        [0-9a-f][0-9a-f]*) printf '%s\n' "$_c" > /root/mojolearn/commit.txt ;;
        *) say "NO COMMIT WITNESS: leg.txt gave '$_c'; the identity phases will refuse" ;;
    esac
fi
say "commit=$(head -1 /root/mojolearn/commit.txt 2>/dev/null)"
rocm-smi --showproductname > "$OUT/logs/device.txt" 2>&1
say "device=$(grep -i 'card series\|Card model\|Device Name' "$OUT/logs/device.txt" | head -2 | tr '\n' ' ')"
(grep -m1 'model name' /proc/cpuinfo; uname -m) > "$OUT/logs/cpu.txt" 2>&1
say "cpu=$(tr '\n' ' ' < "$OUT/logs/cpu.txt")"

if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    _gfx=$(rocminfo 2>/dev/null | sed -n 's/.*Name:  *\(gfx[0-9a-f]*\).*/\1/p' | head -1)
    case "$_gfx" in
        gfx[0-9]*) MOJOLEARN_GPU_ARCHS="$_gfx"; export MOJOLEARN_GPU_ARCHS ;;
        *) say "NO GPU ARCH: rocminfo gave '$_gfx' and none was passed in. REFUSING to guess."
           say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; exit 2 ;;
    esac
fi
say "gpu_archs=$MOJOLEARN_GPU_ARCHS"
LABEL="amd-$(grep -i 'card series\|Card model' "$OUT/logs/device.txt" | head -1 | sed 's/.*:[[:space:]]*//' | tr 'A-Z ' 'a-z-' | tr -cd 'a-z0-9-' | cut -c1-20)-$MOJOLEARN_GPU_ARCHS"
case "$LABEL" in amd--*|amd--*) LABEL="amd-$MOJOLEARN_GPU_ARCHS" ;; esac
say "vendor_label=$LABEL"

BUILD="env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=$JOBS"
# `-u` BEFORE any assignment, always: `env FOO=1 -u BAR` does not unset BAR.
HOSTBUILD="env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS=$JOBS"

for fam in base estimators solver metrics; do
    case "$fam" in
        base) script=bindings/build.sh ;;
        *)    script="bindings/build_${fam}.sh" ;;
    esac
    run "build-$fam" $BUILD sh "$script" || say "BUILD FAILED: $script"
done
say "vendor=$(env PYTHONPATH=/root/mojolearn/python pixi run python -c 'import mojolearn; print(mojolearn.vendor())' 2>&1 | tail -1)"
IB="env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python"

# ======================== 1. THE OLDER DEBT: the four predict lanes on AMD
run record-predict $IB pixi run python tools/classical_host_gate.py record "$PREDICT_REC" --lanes "$PREDICT_GATE"
say "record_predict_exit=$(awk -F'\t' '$1=="record-predict"{print $2}' "$OUT/status.tsv")"
tail -20 "$OUT/logs/record-predict.log" >> "$G" 2>/dev/null
rm -rf "$OUT/recording-predict"; cp -R "$PREDICT_REC" "$OUT/recording-predict" 2>/dev/null

run identity-predict $IB pixi run python tools/identity_break.py --lanes "$PREDICT_ID" \
    --repeats 2 --vendor "$LABEL" --json "$OUT/$LABEL.predict.json"
grep -E '^cells=|^summary|MOVED|DIVERGENT|REFUSED' "$OUT/logs/identity-predict.log" | head -30 >> "$G"

# ============================================ 2. THE K-MEANS COLUMN ON AMD
run record-kmeans $IB pixi run python tools/classical_host_gate.py record "$KMEANS_REC" --lanes "$KMEANS_GATE"
say "record_kmeans_exit=$(awk -F'\t' '$1=="record-kmeans"{print $2}' "$OUT/status.tsv")"
tail -20 "$OUT/logs/record-kmeans.log" >> "$G" 2>/dev/null
rm -rf "$OUT/recording-kmeans"; cp -R "$KMEANS_REC" "$OUT/recording-kmeans" 2>/dev/null

run identity-kmeans $IB pixi run python tools/identity_break.py --lanes "$KMEANS_ID" \
    --repeats 2 --vendor "$LABEL" --json "$OUT/$LABEL.kmeans.json"
grep -E '^cells=|^summary|MOVED|DIVERGENT|REFUSED' "$OUT/logs/identity-kmeans.log" | head -30 >> "$G"

# ------------------------------------------- the host set and the gate checks
run build-core-host       $HOSTBUILD sh bindings/build_core_host.sh
run build-estimators-host $HOSTBUILD sh bindings/build_estimators_host.sh
run build-metrics-host    $HOSTBUILD sh bindings/build_metrics_host.sh
run check-predict $IB pixi run python tools/classical_host_gate.py check "$PREDICT_REC" --report "$OUT/check.predict.json"
say "check_predict_exit=$(awk -F'\t' '$1=="check-predict"{print $2}' "$OUT/status.tsv")"
tail -20 "$OUT/logs/check-predict.log" >> "$G" 2>/dev/null
run check-kmeans $IB pixi run python tools/classical_host_gate.py check "$KMEANS_REC" --report "$OUT/check.kmeans.json"
say "check_kmeans_exit=$(awk -F'\t' '$1=="check-kmeans"{print $2}' "$OUT/status.tsv")"
tail -20 "$OUT/logs/check-kmeans.log" >> "$G" 2>/dev/null

# ------------------------------------------------------------ the sabotage arms
mkdir -p "$SAB"
# The k-means saved-model arm: the dedicated define, which moves BOTH halves
# of the (predict, transform) pair. The family define moves the transform half
# alone on purpose, so identity_break's `predict(X) == labels_` assertion still
# holds; see cluster/host/kmeans_oracle.mojo.
run build-core-host-sabotage env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_HOST_OUTDIR="$SAB" \
    MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_KMEANS_PREDICT_SABOTAGE=1" \
    MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS="$JOBS" sh bindings/build_core_host.sh
run build-estimators-host-sabotage env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_HOST_OUTDIR="$SAB" \
    MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_TRANSDUCTIVE_PREDICT_SABOTAGE=1" \
    MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS="$JOBS" sh bindings/build_estimators_host.sh
run build-metrics-host-sabotage env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_HOST_OUTDIR="$SAB" \
    MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_SPECTRAL_PREDICT_SABOTAGE=1" \
    MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS="$JOBS" sh bindings/build_metrics_host.sh
ls -l "$SAB" >> "$G" 2>&1
# REFUSE the arm rather than emit cells that cannot fail.
if [ -s "$SAB/_mojolearn_estimators_host.so" ] && [ -s "$SAB/_mojolearn_metrics_host.so" ]; then
    run sabotage-predict env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
        MOJOLEARN_HOST_DIR="$SAB" MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
        pixi run python tools/classical_host_gate.py check "$PREDICT_REC" \
        --expect-mismatch --every-fixture --report "$OUT/sabotage.predict.json"
    say "sabotage_predict_exit=$(awk -F'\t' '$1=="sabotage-predict"{print $2}' "$OUT/status.tsv")"
    tail -30 "$OUT/logs/sabotage-predict.log" >> "$G" 2>/dev/null
else
    say "PREDICT SABOTAGE ARM NOT TAKEN: a sabotage binding is missing from $SAB."
fi
if [ -s "$SAB/_mojolearn_core_host.so" ]; then
    run sabotage-kmeans env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
        MOJOLEARN_HOST_DIR="$SAB" MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
        pixi run python tools/classical_host_gate.py check "$KMEANS_REC" \
        --expect-mismatch --every-fixture --report "$OUT/sabotage.kmeans.json"
    say "sabotage_kmeans_exit=$(awk -F'\t' '$1=="sabotage-kmeans"{print $2}' "$OUT/status.tsv")"
    tail -30 "$OUT/logs/sabotage-kmeans.log" >> "$G" 2>/dev/null
else
    say "KMEANS SABOTAGE ARM NOT TAKEN: no $SAB/_mojolearn_core_host.so."
fi

rm -rf "$OUT/recording-predict" "$OUT/recording-kmeans"
cp -R "$PREDICT_REC" "$OUT/recording-predict" 2>/dev/null
cp -R "$KMEANS_REC" "$OUT/recording-kmeans" 2>/dev/null
cat "$OUT/status.tsv" >> "$G" 2>/dev/null
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
