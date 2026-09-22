# The on-box body of the reference-regen NVIDIA leg: the twelve parallel-driver
# lanes the algorithm inventory maps and no committed column carries a hash for.
# One device (par_devices unset, so the harness records par_devices=0 and the
# column is admissible to the reference table).
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/identity
mkdir -p "$OUT/logs"
G="$OUT/gate.txt"
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
# THE COMMIT WITNESS. The RunPod runner passes no environment to this body
# and ships a git archive with no .git, so identity_break refuses ("no commit
# witness") unless the sha reaches the box in the body itself. The first
# attempt (2026-09-17 20:24Z) built all ten bindings and then lost the run to
# exactly that. The sha below is substituted by git rev-parse when this file
# is generated, never typed.
MOJOLEARN_COMMIT="${MOJOLEARN_COMMIT:-68cac62a5197e4bf2703293833f352569ab537c0}"
export MOJOLEARN_COMMIT
echo "$MOJOLEARN_COMMIT" > /root/mojolearn/commit.txt
say "commit=$(cat /root/mojolearn/commit.txt 2>/dev/null | head -1 || echo unknown)"
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
        case "$_cc" in 9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;; 8.9) MOJOLEARN_GPU_ARCHS=sm_89 ;; 8.6) MOJOLEARN_GPU_ARCHS=sm_86 ;; 8.0) MOJOLEARN_GPU_ARCHS=sm_80 ;; 12.0) MOJOLEARN_GPU_ARCHS=sm_120a ;; *) MOJOLEARN_GPU_ARCHS="sm_$(echo "$_cc" | tr -d .)" ;; esac
    fi
    export MOJOLEARN_GPU_ARCHS
fi
say "gpu_archs_resolved=${MOJOLEARN_GPU_ARCHS:-unset}"
BUILD_ENV="env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-8}"
built=0; failed=""
for n in build build_linalg build_estimators build_trees build_rf build_gbdt build_solver build_svm build_preprocessing build_metrics; do
    s="bindings/$n.sh"
    [ -f "$s" ] || { failed="$failed $n(missing)"; continue; }
    if run "$n" $BUILD_ENV sh "$s"; then built=$((built + 1)); else failed="$failed $n"; fi
done
say "bindings_built=$built failed=${failed:-none}"
sha256sum python/mojolearn/identical/*.so >> "$G" 2>/dev/null
VENDOR_LABEL="nvidia-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr 'A-Z ' 'a-z-' | tr -cd 'a-z0-9-' | cut -c1-24)-${MOJOLEARN_GPU_ARCHS:-sm}"
LANES=par-forest-reg,par-forest-et-clf,par-boosting-clf,par-boosting-reg,par-gram-ols,par-gram-pca,par-gram-tsvd,par-cd-elasticnet,par-svm-svr,par-scaler-minmax,par-queries-nn,par-graph-umap
run identity_break env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/identity_break.py --lanes "$LANES" --repeats 2 --vendor "$VENDOR_LABEL" \
    --json "$OUT/identity_break.$VENDOR_LABEL.json"
say "identity_break_exit=$(awk -F'\t' '$1=="identity_break"{print $2}' "$OUT/status.tsv")"
grep -E "^# CELL|REFUSED|MOVED|DIVERGENT" "$OUT/logs/identity_break.log" 2>/dev/null | tail -60 >> "$G"
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
