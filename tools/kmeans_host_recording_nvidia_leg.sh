#!/bin/sh
# tools/kmeans_host_recording_nvidia_leg.sh: the on-box body
# (MOJOLEARN_GEMM_LEG_EXTRA for tools/gemm_remote_leg.sh nvidia) of
# lane/classical-host-recordings, 2026-09-16.
#
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/kmeans_host_recording_nvidia_leg.sh \
#   sh tools/gemm_remote_leg.sh nvidia --rent --local-card <card>
#
# NO GPU IS PINNED HERE, and the caller must not pin one either. The runner
# pins a single `gpuTypeIds` and has no retry; its RTX 4090 default answered
# "There are no instances currently available" twice on 2026-09-16. The retry
# that walks a cheapest-first list and stops at the first pod is
# ~/mojolearn-evidence/kmeans-save/rent_retry.sh. Keep
# `allowedCudaVersions: ["13.0"]`: that is the pinned MAX compiler's driver
# floor, not a preference.
#
# THREE THINGS ARE OWED AND THEY SHARE THIS BOX.
#
#  1. THE K-MEANS RECORDING. `bench/results/classical_host/` had no k-means
#     entry, so the six fitted k-means lanes were the last of
#     SAVED_MODEL_INFERENCE_OWED. `record` refuses a CPU-only install, so
#     this is the phase that can only happen here. It runs FIRST: a box that
#     dies late still comes home with the deliverable.
#
#  2. THE K-MEANS SABOTAGE ARM, WHICH NEVER RAN. lane/kmeans-save's L40S leg
#     came home with `train IDENTICAL=7, infer 12, model 12, batch 6` and NO
#     negative control: `env FOO=1 -u BAR cmd` does not unset BAR (env stops
#     parsing options at the first assignment), the sabotage build exited
#     127, MOJOLEARN_HOST_DIR pointed at an empty directory, and every infer
#     and model cell of that arm read REFUSED. A REFUSED cell is not a
#     DIVERGENT cell and a missing binary is indistinguishable from a pass.
#     Every `-u` below comes BEFORE any assignment, and the arm REFUSES TO
#     BE TAKEN when the binary is absent rather than emitting cells that
#     cannot fail.
#
#  3. SPECTRAL'S x86 CPU IDENTITY COLUMN AT THE PUBLISHED 512-ROW SIZE.
#     lane/saved-model-reference-gaps found that the spectral lane was shrunk
#     from 2000 rows to 512 at e2bb9e541, which is NOT an ancestor of either
#     the CPU column (0a6957015) or the Metal column (b886dbc97), so the
#     diff tool's LANE_REVISIONS excluded both and read them as absent. Its
#     NVIDIA column was the first at the published size and it retook the
#     Apple one; the CPU one it left owed. The column is taken here, on this
#     box, from a package copy with the GPU bindings removed, so the
#     arithmetic is the same `MOJOLEARN_TARGET_COLUMN=cpu` binaries a CPU pod
#     would build, and a second rental is not needed for it.
#
# POSIX sh (the pod's /bin/sh is dash). `set -u` and deliberately NOT `set -e`:
# a phase that fails is a FINDING and its log must come home.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/kmeans-host-recording
mkdir -p "$OUT/logs"
G="$OUT/gate.txt"
REC=bench/results/classical_host/2026-09-16-nvidia-kmeans
GATE_LANES=kmeans,kmeans-random,kmeans-array,kmeans-weighted,kmeans-sqrt,kmeans-classic-pp
# kmeans-cosine IS in the identity arms (its cell is the refusal sentence, and
# a column that hashes there means the refusal was lifted) and is NOT in the
# gate lanes (a refused fit has no model to save).
ID_LANES=kmeans,kmeans-random,kmeans-array,kmeans-weighted,kmeans-sqrt,kmeans-classic-pp,kmeans-cosine
SPECTRAL_LANES=spectral,spectral-precomputed
SAB=/root/host-sabotage
CPUPKG=/root/cpu-only-pkg
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

# THE COMMIT WITNESS, OR identity_break REFUSES TO WRITE A JSON. The box has
# no .git and the gemm payload does not write commit.txt for this payload; the
# runner does record the commit in /root/gemm_leg_out/leg.txt, so take it from
# there and never guess.
if [ ! -s /root/mojolearn/commit.txt ]; then
    _c=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)
    case "$_c" in
        [0-9a-f][0-9a-f]*) printf '%s\n' "$_c" > /root/mojolearn/commit.txt ;;
        *) say "NO COMMIT WITNESS: leg.txt gave '$_c'; the identity phases will refuse" ;;
    esac
fi
say "commit=$(head -1 /root/mojolearn/commit.txt 2>/dev/null)"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader > "$OUT/logs/device.txt" 2>&1
say "device=$(tr '\n' ' ' < "$OUT/logs/device.txt")"
(grep -m1 'model name' /proc/cpuinfo; uname -m) > "$OUT/logs/cpu.txt" 2>&1
say "cpu=$(tr '\n' ' ' < "$OUT/logs/cpu.txt")"

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
CPULABEL="cpu-$(sed -n 's/^model name[^:]*: //p' /proc/cpuinfo | head -1 | tr 'A-Z ' 'a-z-' | tr -cd 'a-z0-9-' | cut -c1-40)"
say "vendor_label=$LABEL"
say "cpu_label=$CPULABEL"

BUILD="env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=$JOBS"
# `env -u` MUST COME BEFORE ANY ASSIGNMENT, and this leg exports
# MOJOLEARN_GPU_ARCHS above; bindings/build_host_family.sh refuses a Linux CPU
# build that carries one, and on 2026-09-16 that took four host builds, a
# check and both sabotage arms down with it in zero seconds.
HOSTBUILD="env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS=$JOBS"

# ---------------------------------------------------------------- GPU builds
# The base binding carries kmeans_fit, kmeans_predict and kmeans_transform;
# metrics carries SpectralClustering, whose CPU column this leg also takes.
run build-base    $BUILD sh bindings/build.sh         || say "BUILD FAILED: bindings/build.sh"
run build-metrics $BUILD sh bindings/build_metrics.sh || say "BUILD FAILED: bindings/build_metrics.sh"
sha256sum python/mojolearn/identical/_mojolearn.so >> "$G" 2>/dev/null
say "vendor=$(env PYTHONPATH=/root/mojolearn/python pixi run python -c 'import mojolearn; print(mojolearn.vendor())' 2>&1 | tail -1)"

IB="env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python"

# ============================================================ 1. THE RECORDING
run record $IB pixi run python tools/classical_host_gate.py record "$REC" --lanes "$GATE_LANES"
say "record_exit=$(awk -F'\t' '$1=="record"{print $2}' "$OUT/status.tsv")"
tail -20 "$OUT/logs/record.log" >> "$G" 2>/dev/null
# COPIED THE MOMENT IT EXISTS, not after every later phase: the fetch takes
# whatever is under /root/gemm_leg_out when the poll deadline is reached.
rm -rf "$OUT/recording"; cp -R "$REC" "$OUT/recording" 2>/dev/null

# ------------------------------------------------------------- the host set
run build-core-host    $HOSTBUILD sh bindings/build_core_host.sh
run build-metrics-host $HOSTBUILD sh bindings/build_metrics_host.sh
mkdir -p "$SAB"
run build-core-host-sabotage env -u MOJOLEARN_GPU_ARCHS \
    MOJOLEARN_HOST_OUTDIR="$SAB" \
    MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_HOST_SABOTAGE=1" \
    MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS="$JOBS" \
    sh bindings/build_core_host.sh
run build-metrics-host-sabotage env -u MOJOLEARN_GPU_ARCHS \
    MOJOLEARN_HOST_OUTDIR="$SAB" \
    MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_SPECTRAL_PREDICT_SABOTAGE=1" \
    MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS="$JOBS" \
    sh bindings/build_metrics_host.sh
ls -l "$SAB" >> "$G" 2>&1
sha256sum python/mojolearn/host/_mojolearn_core_host.so "$SAB/_mojolearn_core_host.so" >> "$G" 2>&1
if [ ! -s "$SAB/_mojolearn_core_host.so" ]; then
    say "SABOTAGE ARM NOT TAKEN: no $SAB/_mojolearn_core_host.so."
    say "  Every sabotage cell below is NOT a negative control. Do not read one as one."
    SABOTAGE_READY=0
else
    SABOTAGE_READY=1
fi
say "sabotage_ready=$SABOTAGE_READY"

# ===================================== 2. THE GATE CHECK, AND ITS SABOTAGE ARM
run check $IB pixi run python tools/classical_host_gate.py check "$REC" --report "$OUT/check.json"
say "check_exit=$(awk -F'\t' '$1=="check"{print $2}' "$OUT/status.tsv")"
tail -30 "$OUT/logs/check.log" >> "$G" 2>/dev/null
if [ "$SABOTAGE_READY" = 1 ]; then
    # --every-fixture is the strong rule: EVERY recorded cell must move, not
    # one per lane. `unmoved` EMPTY is the sentence that matters.
    run sabotage-every-fixture env MOJOLEARN_NUMERIC_MODE=identical \
        PYTHONPATH=/root/mojolearn/python MOJOLEARN_HOST_DIR="$SAB" \
        MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
        pixi run python tools/classical_host_gate.py check "$REC" \
        --expect-mismatch --every-fixture --report "$OUT/sabotage.every-fixture.json"
    say "sabotage_every_fixture_exit=$(awk -F'\t' '$1=="sabotage-every-fixture"{print $2}' "$OUT/status.tsv")"
    run sabotage-every-lane env MOJOLEARN_NUMERIC_MODE=identical \
        PYTHONPATH=/root/mojolearn/python MOJOLEARN_HOST_DIR="$SAB" \
        MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
        pixi run python tools/classical_host_gate.py check "$REC" \
        --expect-mismatch --every-lane --report "$OUT/sabotage.every-lane.json"
    say "sabotage_every_lane_exit=$(awk -F'\t' '$1=="sabotage-every-lane"{print $2}' "$OUT/status.tsv")"
    tail -40 "$OUT/logs/sabotage-every-fixture.log" >> "$G" 2>/dev/null
fi

# ============================ 3. THE IDENTITY ARMS THE L40S LEG LEFT UNPROVED
# A: infer on the GPU. B: the same fits, infer through mojolearn.host_model on
# the CPU host binding. C: --diff A B, the deliverable, every cell IDENTICAL.
# D: B again on the SABOTAGED host binding. E: --diff A D, which must NOT read
# IDENTICAL and must not read REFUSED either.
run A_gpu_infer $IB pixi run python tools/identity_break.py --lanes "$ID_LANES" \
    --fixtures base --vendor "$LABEL" --json "$OUT/gpu.json"
run B_host_infer $IB MOJOLEARN_IDENTITY_HOST_INFER=1 pixi run python tools/identity_break.py \
    --lanes "$ID_LANES" --fixtures base --vendor "$LABEL" --json "$OUT/host.json"
run C_diff_gpu_host env PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py \
    --diff "$OUT/gpu.json" "$OUT/host.json"
if [ "$SABOTAGE_READY" = 1 ]; then
    run D_host_sabotage $IB MOJOLEARN_IDENTITY_HOST_INFER=1 MOJOLEARN_HOST_DIR="$SAB" \
        MOJOLEARN_HOST_ALLOW_SABOTAGE=1 pixi run python tools/identity_break.py \
        --lanes "$ID_LANES" --fixtures base --vendor "$LABEL-sabotage" --json "$OUT/host_sabotage.json"
    run E_diff_gpu_sabotage env PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py \
        --diff "$OUT/gpu.json" "$OUT/host_sabotage.json"
    # A diff that EXITS 0 here is the failure: the sabotaged arithmetic moved
    # nothing, so the control controlled nothing.
    if [ "$(awk -F'\t' '$1=="E_diff_gpu_sabotage"{print $2}' "$OUT/status.tsv")" = 0 ]; then
        say "SABOTAGE ARM DID NOT FIRE: the diff against the sabotaged host read clean."
    fi
    if grep -q "REFUSED" "$OUT/logs/E_diff_gpu_sabotage.log" 2>/dev/null; then
        say "SABOTAGE ARM REFUSED at least one cell; a REFUSED cell is not a DIVERGENT cell."
    fi
fi
say "--- C (must be IDENTICAL on every cell) ---"
grep -E "IDENTICAL|DIVERGENT|REFUSED|RELOAD|MOVED|^cells=|^summary" "$OUT/logs/C_diff_gpu_host.log" 2>/dev/null | head -40 >> "$G"
say "--- E (must be DIVERGENT: the control has to be able to fail) ---"
grep -E "IDENTICAL|DIVERGENT|REFUSED|RELOAD|MOVED|^cells=|^summary" "$OUT/logs/E_diff_gpu_sabotage.log" 2>/dev/null | head -40 >> "$G"

# F: the boundary stated directly in bytes, with the one-float32-ULP control
# that must move an answer and must NAME the outputs that did not move.
run F_boundary_bytes env PYTHONPATH=/root/mojolearn/python MOJOLEARN_NUMERIC_MODE=identical \
    pixi run python tools/kmeans_save_boundary_check.py
say "F_boundary_bytes exit=$(awk -F'\t' '$1=="F_boundary_bytes"{print $2}' "$OUT/status.tsv")"
cat "$OUT/logs/F_boundary_bytes.log" >> "$G" 2>/dev/null

# ------------------------- the NVIDIA k-means identity column, all nine fixtures
run kmeans-column $IB pixi run python tools/identity_break.py --lanes "$ID_LANES" \
    --repeats 2 --vendor "$LABEL" --json "$OUT/$LABEL.kmeans.json"
grep -E '^cells=|^summary|MOVED|DIVERGENT|REFUSED' "$OUT/logs/kmeans-column.log" | head -30 >> "$G"
run kmeans-batch-sabotage env MOJOLEARN_NUMERIC_MODE=identical \
    MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1 PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/identity_break.py --lanes "$ID_LANES" --fixtures base --repeats 1 \
    --vendor "$LABEL" --json "$OUT/$LABEL.kmeans.batch-sabotage.json"
run kmeans-diff-batch-sabotage env PYTHONPATH=/root/mojolearn/python pixi run python \
    tools/identity_break.py --diff "$OUT/$LABEL.kmeans.json" "$OUT/$LABEL.kmeans.batch-sabotage.json" \
    --lanes "$ID_LANES"
grep -E 'BATCH_MOVED|summary' "$OUT/logs/kmeans-diff-batch-sabotage.log" | head -20 >> "$G"

# =========================== 4. SPECTRAL'S x86 CPU COLUMN AT 512 ROWS
# A package copy with `identical/` removed routes every call to the host
# bindings built above, which are the same `MOJOLEARN_TARGET_COLUMN=cpu`
# binaries a CPU-only pod builds; the GPU on this box is simply not reachable
# from this PYTHONPATH. `mojolearn.vendor()` is asserted to read `cpu` before
# the column is taken, so a GPU column can never be mislabelled as a CPU one.
rm -rf "$CPUPKG"; mkdir -p "$CPUPKG"
cp -R python/mojolearn "$CPUPKG/mojolearn"
rm -rf "$CPUPKG/mojolearn/identical" "$CPUPKG/mojolearn/deterministic" "$CPUPKG/mojolearn/__pycache__"
rm -f "$CPUPKG/mojolearn"/*.so
_cv=$(env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$CPUPKG" pixi run python -c \
    'import mojolearn; print(mojolearn.vendor())' 2>&1 | tail -1)
say "cpu_pkg_vendor=$_cv"
if [ "$_cv" != cpu ]; then
    say "CPU COLUMN NOT TAKEN: the package copy reports vendor '$_cv', not 'cpu'."
else
    CPUIB="env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=$CPUPKG"
    run spectral-cpu-column $CPUIB pixi run python tools/identity_break.py \
        --lanes "$SPECTRAL_LANES" --repeats 2 --vendor "$CPULABEL" --json "$OUT/$CPULABEL.spectral.json"
    grep -E '^cells=|^summary|MOVED|DIVERGENT|REFUSED' "$OUT/logs/spectral-cpu-column.log" | head -30 >> "$G"
    # Its own negative control, on the SAME package copy: the spectral predict
    # sabotage define, in a separate host directory.
    if [ -s "$SAB/_mojolearn_metrics_host.so" ]; then
        run spectral-cpu-host-sabotage $CPUIB MOJOLEARN_HOST_DIR="$SAB" \
            MOJOLEARN_HOST_ALLOW_SABOTAGE=1 pixi run python tools/identity_break.py \
            --lanes "$SPECTRAL_LANES" --fixtures base,ties --repeats 1 \
            --vendor "$CPULABEL-sabotage" --json "$OUT/$CPULABEL.spectral.host-sabotage.json"
        run spectral-cpu-diff-sabotage env PYTHONPATH="$CPUPKG" pixi run python \
            tools/identity_break.py --diff "$OUT/$CPULABEL.spectral.json" \
            "$OUT/$CPULABEL.spectral.host-sabotage.json" --lanes "$SPECTRAL_LANES"
        grep -E 'IDENTICAL|DIVERGENT|REFUSED|^summary' "$OUT/logs/spectral-cpu-diff-sabotage.log" | head -20 >> "$G"
    else
        say "SPECTRAL CPU SABOTAGE ARM NOT TAKEN: no $SAB/_mojolearn_metrics_host.so."
    fi
fi

# --------------------------------------------------------------- bring it home
rm -rf "$OUT/recording"; cp -R "$REC" "$OUT/recording" 2>/dev/null
cat "$OUT/status.tsv" >> "$G" 2>/dev/null
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
