#!/bin/sh
# tools/saved_model_predict_nvidia_leg.sh: the on-box body
# (MOJOLEARN_GEMM_LEG_EXTRA for tools/gemm_remote_leg.sh nvidia) of
# lane/saved-model-reference-gaps, 2026-09-16.
#
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/saved_model_predict_nvidia_leg.sh \
#   sh tools/gemm_remote_leg.sh nvidia --rent
#
# NO GPU IS PINNED. Any NVIDIA card answers this leg; the runner's default
# (RTX 4090) is the most available one, and a Hot Aisle leg starved on a
# pinned spec for thirty minutes this morning and created nothing.
#
# WHAT THIS BOX IS FOR. DBSCAN.predict, AgglomerativeClustering.predict
# (DEVIATION 2740) and SpectralClustering.predict on both affinities
# (DEVIATION 2860) ship, and no GPU column and no saved-model recording
# exists for any of them. The Apple/Metal and x86 CPU columns were taken on
# 2026-09-15; NVIDIA is the owed one. AMD is left alone entirely.
#
# THE ORDER IS THE VALUE ORDER. Builds, then the recording (the deliverable),
# then the NVIDIA identity column, then the host check, then the two sabotage
# arms. A box that dies late still comes home with the recording.
#
# POSIX sh: the pod's /bin/sh is dash. `set -u` and deliberately NOT `set -e`:
# a phase that fails is a FINDING and its log must come home.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/saved-model-gaps
mkdir -p "$OUT/logs"
G="$OUT/gate.txt"
REC=bench/results/classical_host/2026-09-16-nvidia-predict
GATE_LANES=dbscan,agglomerative,spectral,spectral-precomputed
ID_LANES=dbscan,dbscan-brute-l1,dbscan-weighted,agglomerative,spectral,spectral-precomputed
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
# THE COMMIT WITNESS, OR identity_break REFUSES. The box has no `.git`, and
# the gemm payload does not write /root/mojolearn/commit.txt for this
# payload: the 2026-09-16 run reached the identity phase with an empty
# witness and it had to be written by hand over ssh mid-build. The runner
# DOES record the commit in /root/gemm_leg_out/leg.txt, so take it from
# there, and never guess.
if [ ! -s /root/mojolearn/commit.txt ]; then
    _c=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)
    case "$_c" in
        [0-9a-f][0-9a-f]*) printf '%s\n' "$_c" > /root/mojolearn/commit.txt ;;
        *) say "NO COMMIT WITNESS: leg.txt gave '$_c'; the identity phase will refuse" ;;
    esac
fi
say "commit=$(head -1 /root/mojolearn/commit.txt 2>/dev/null)"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader > "$OUT/logs/device.txt" 2>&1
say "device=$(tr '\n' ' ' < "$OUT/logs/device.txt")"
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
    case "$_cc" in
        9.0)  MOJOLEARN_GPU_ARCHS=sm_90a ;;
        8.9)  MOJOLEARN_GPU_ARCHS=sm_89 ;;
        8.6)  MOJOLEARN_GPU_ARCHS=sm_86 ;;
        8.0)  MOJOLEARN_GPU_ARCHS=sm_80 ;;
        12.0) MOJOLEARN_GPU_ARCHS=sm_120a ;;
        *)    MOJOLEARN_GPU_ARCHS="sm_$(echo "$_cc" | tr -d .)" ;;
    esac
    export MOJOLEARN_GPU_ARCHS
fi
say "gpu_archs=$MOJOLEARN_GPU_ARCHS"
LABEL="nvidia-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr 'A-Z ' 'a-z-' | tr -cd 'a-z0-9-' | cut -c1-24)-$MOJOLEARN_GPU_ARCHS"
say "vendor_label=$LABEL"

# ---------------------------------------------------------------- GPU builds
# base carries the fixtures and the shared kernels; estimators holds DBSCAN and
# AgglomerativeClustering, metrics holds SpectralClustering, solver holds the
# linkage solve the agglomerative fit runs through.
for fam in base estimators solver metrics; do
    case "$fam" in
        base) script=bindings/build.sh ;;
        *)    script="bindings/build_${fam}.sh" ;;
    esac
    run "build-$fam" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
        MOJOLEARN_COMPILE_JOBS="$JOBS" sh "$script"
done
sha256sum python/mojolearn/identical/_mojolearn_estimators.so python/mojolearn/identical/_mojolearn_metrics.so >> "$G" 2>/dev/null

# ------------------------------------------------------- THE DELIVERABLE
# `record` refuses a CPU-only install, so this is the step that can only
# happen here. Nine fixtures per lane, the model saved and reloaded on the
# GPU path and required to predict the same bits before anything is written.
run record env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/classical_host_gate.py record "$REC" --lanes "$GATE_LANES"
say "record_exit=$?"
tail -40 "$OUT/logs/record.log" >> "$G" 2>/dev/null
# COPIED HERE, NOT ONLY AT THE END. The fetch takes whatever is under
# /root/gemm_leg_out when the poll deadline is reached, and the deliverable
# must be inside it from the moment it exists, not after every later phase.
rm -rf "$OUT/recording"; cp -R "$REC" "$OUT/recording" 2>/dev/null

# --------------------------------------------------- the NVIDIA identity column
run identity env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/identity_break.py --lanes "$ID_LANES" --repeats 2 \
    --vendor "$LABEL" --json "$OUT/$LABEL.json"
grep -E '^cells=|^summary|MOVED|DIVERGENT|REFUSED' "$OUT/logs/identity.log" | head -30 >> "$G"

# The batch part's own negative control, on this column, base fixture.
run identity-batch-sabotage env MOJOLEARN_NUMERIC_MODE=identical \
    MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1 PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/identity_break.py --lanes "$ID_LANES" --fixtures base --repeats 1 \
    --vendor "$LABEL" --json "$OUT/$LABEL.batch-sabotage.json"
run diff-batch-sabotage env PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py \
    --diff "$OUT/$LABEL.json" "$OUT/$LABEL.batch-sabotage.json" --lanes "$ID_LANES"
cp "$OUT/logs/diff-batch-sabotage.log" "$OUT/diff.nvidia-vs-batch-sabotage.txt" 2>/dev/null
grep -E 'BATCH_MOVED|summary' "$OUT/logs/diff-batch-sabotage.log" | head -20 >> "$G"

# THE THREE-COLUMN DIFFS ARE NOT RUN HERE. `git archive` ships the source
# only, so bench/results/ does not exist on the box and the reference columns
# are not here to diff against; the 2026-09-16 leg's two diff phases died with
# FileNotFoundError for exactly that. They cost nothing at home, against the
# fetched column, and that is where they run.

# ------------------------------------------------- the CPU host route, here
# `check` runs the host subclasses: `_bind` answers the CPU binding and
# everything else is the GPU class's own Python. Only two host families are
# reachable from these four lanes, so only two are built.
#
# `env -u MOJOLEARN_GPU_ARCHS` IS NOT DECORATION. This file exports that
# variable for the GPU builds above, and bindings/build_host_family.sh refuses
# a Linux CPU build that carries one ("a CPU build takes no
# MOJOLEARN_GPU_ARCHS"). On 2026-09-16 all four host builds exited 2 in zero
# seconds for exactly that reason and took the check and both sabotage arms
# down with them.
run build-estimators-host env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS="$JOBS" \
    sh bindings/build_estimators_host.sh
run build-metrics-host env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS="$JOBS" \
    sh bindings/build_metrics_host.sh
# No --gpu-column here either, and for the same reason: those JSONs are under
# bench/results/, which the archive does not carry. The comparison this phase
# makes is the one that matters on this box, the GPU recording against the CPU
# host binding beside it.
run check env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/classical_host_gate.py check "$REC" \
    --report "$OUT/check.json"
say "check_exit=$?"
tail -60 "$OUT/logs/check.log" >> "$G" 2>/dev/null

# ------------------------------------------------------------ the sabotage arm
# PROVE THE CHECK CAN FAIL BEFORE ANY PASS IS BELIEVED. Two predict-only
# defines, one per family, into a SEPARATE host directory; the production set
# above is untouched. --every-fixture is the strong rule: every recorded cell
# must move. A fixture that does NOT move is a finding, and its recording is
# dropped at home rather than shipped as coverage.
mkdir -p "$SAB"
run build-estimators-host-sabotage env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS="$JOBS" \
    MOJOLEARN_HOST_OUTDIR="$SAB" \
    MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_TRANSDUCTIVE_PREDICT_SABOTAGE=1" \
    sh bindings/build_estimators_host.sh
run build-metrics-host-sabotage env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS="$JOBS" \
    MOJOLEARN_HOST_OUTDIR="$SAB" \
    MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_SPECTRAL_PREDICT_SABOTAGE=1" \
    sh bindings/build_metrics_host.sh
ls -l "$SAB" >> "$G" 2>/dev/null
run sabotage-every-fixture env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    MOJOLEARN_HOST_DIR="$SAB" MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
    pixi run python tools/classical_host_gate.py check "$REC" --expect-mismatch --every-fixture \
    --report "$OUT/sabotage.every-fixture.json"
say "sabotage_every_fixture_exit=$?"
run sabotage-every-lane env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    MOJOLEARN_HOST_DIR="$SAB" MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
    pixi run python tools/classical_host_gate.py check "$REC" --expect-mismatch --every-lane \
    --report "$OUT/sabotage.every-lane.json"
say "sabotage_every_lane_exit=$?"
tail -80 "$OUT/logs/sabotage-every-fixture.log" >> "$G" 2>/dev/null

# --------------------------------------------------------------- bring it home
cp -R "$REC" "$OUT/recording" 2>/dev/null
cat "$OUT/status.tsv" >> "$G" 2>/dev/null
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
