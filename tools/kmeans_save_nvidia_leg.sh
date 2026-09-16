#!/bin/sh
# tools/kmeans_save_nvidia_leg.sh: the on-box body of lane/kmeans-save's NVIDIA
# leg (MOJOLEARN_GEMM_LEG_EXTRA for tools/gemm_remote_leg.sh), 2026-09-16.
#
# WHAT THIS LEG CLAIMS, AND WHY IT IS NOT AN APPLE CELL.
# The claim is that a k-means model FITTED ON A GPU, written to a
# `mojolearn-kmeans-1` file and loaded on a machine with no GPU through
# `mojolearn.host_model`, predicts and transforms THE SAME BITS. That claim is
# about a saved model crossing a DEVICE BOUNDARY. NVIDIA is a device boundary.
# The Apple cell for this lane is a release-record cell taken once per release
# with the other GPU columns, not a per-lane gate, and the single M4 is the
# scarce column (a Metal wait costs about 2815 us against 164 us to enqueue,
# and one long-lived process crossing Apple's ~512 command-queue limit
# degrades about 20x inside its own lifetime).
#
# SIX CELLS, IN ORDER, AND ONE OF THEM MUST FAIL.
#   A  identity_break on the seven k-means lanes, base fixture: infer on the
#      GPU, and the model column (the saved file's hash, and a reload that
#      must predict what the model in memory predicted).
#   B  the SAME fits, infer through `mojolearn.host_model` on
#      `_mojolearn_core_host` (MOJOLEARN_IDENTITY_HOST_INFER=1).
#   C  --diff A B. Every cell must read IDENTICAL. This is the deliverable.
#   D  B again against a core host binding built -D MOJOLEARN_HOST_SABOTAGE=1.
#      `cluster/host/kmeans_oracle.mojo` is the k-means arm of that define, so
#      the control reaches THIS lane's arithmetic and not a neighbor's.
#   E  --diff A D. It must read DIVERGENT. A comparison that has not been
#      watched to fail is not a comparison.
#   F  the same boundary stated directly, in bytes: the sha256 of the GPU
#      estimator's predict and transform against the loaded host model's, and
#      a one-float32-ULP rewrite of one centroid inside the saved file, which
#      must move the answer and which NAMES THE OUTPUTS THAT DID NOT MOVE as
#      well as those that did.
#
# The lanes' fixtures are identity_break's own hashed synthetic stream, so
# this leg stages no dataset and touches no bucket.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/kmeans-save
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
LANES=kmeans,kmeans-sqrt,kmeans-random,kmeans-array,kmeans-weighted,kmeans-classic-pp,kmeans-cosine

say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# identity_break REQUIRES a commit witness: a JSON with an empty commit is not
# written. The runner ships a git archive with no .git, so the commit reaches
# the box as MOJOLEARN_COMMIT or already baked into commit.txt.
if [ -n "${MOJOLEARN_COMMIT:-}" ] && [ ! -s /root/mojolearn/commit.txt ]; then echo "$MOJOLEARN_COMMIT" > /root/mojolearn/commit.txt; fi
say "commit=$(cat /root/mojolearn/commit.txt /root/mojolearn/COMMIT 2>/dev/null | head -1 || echo unknown)"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader > "$OUT/logs/device.txt" 2>&1
say "device=$(tr '\n' ' ' < "$OUT/logs/device.txt")"

# The runner passes no environment to the body, and bindings/build.sh needs an
# explicit architecture. Derive it from the device rather than assuming one.
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
    case "$_cc" in
        9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;;
        8.9) MOJOLEARN_GPU_ARCHS=sm_89 ;;
        8.6) MOJOLEARN_GPU_ARCHS=sm_86 ;;
        8.0) MOJOLEARN_GPU_ARCHS=sm_80 ;;
        12.0) MOJOLEARN_GPU_ARCHS=sm_120a ;;
        *) MOJOLEARN_GPU_ARCHS="sm_$(echo "$_cc" | tr -d .)" ;;
    esac
    export MOJOLEARN_GPU_ARCHS
fi
say "gpu_archs_resolved=${MOJOLEARN_GPU_ARCHS:-unset}"

JOBS=${MOJOLEARN_COMPILE_JOBS:-8}
BUILD_ENV="env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=$JOBS"
HOSTBUILD="env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=$JOBS"

# Three binaries carry this lane. The GPU core extension fits k-means; the
# core host binding is what a CPU-only install predicts on; the sabotage build
# of the same source is the negative control.
run build_core_gpu   $BUILD_ENV sh bindings/build.sh          || say "BUILD FAILED: build.sh"
run build_core_host  $HOSTBUILD sh bindings/build_core_host.sh || say "BUILD FAILED: build_core_host.sh"
mkdir -p /root/sabotage-host
# `-u` MUST COME BEFORE ANY ASSIGNMENT. `env FOO=1 -u BAR cmd` does not unset
# BAR: env stops parsing options at the first non-option argument, so `-u` is
# taken as a file to execute and the whole call dies `env: '-u': No such file
# or directory`, exit 127. That is exactly what happened on the 2026-09-16
# L40S leg: the sabotage binary was never built, MOJOLEARN_HOST_DIR pointed at
# an empty directory, and every cell of arm D read REFUSED. A REFUSED cell is
# not a failed cell, so arm E read ONE-COLUMN and proved NOTHING. The whole
# point of the arm is that it must be able to fail, and a missing binary makes
# it unable to either fail or pass.
run build_core_host_sabotage env -u MOJOLEARN_GPU_ARCHS \
    MOJOLEARN_HOST_OUTDIR=/root/sabotage-host \
    MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_HOST_SABOTAGE=1" \
    MOJOLEARN_TARGET_COLUMN=cpu MOJOLEARN_NUMERIC_MODE=identical \
    MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=$JOBS sh bindings/build_core_host.sh \
    || say "BUILD FAILED: sabotage core host"
# And REFUSE TO RUN THE ARM AT ALL if the binary is not there, rather than
# letting a missing file masquerade as a control.
if [ ! -s /root/sabotage-host/_mojolearn_core_host.so ]; then
    say "SABOTAGE ARM NOT TAKEN: no /root/sabotage-host/_mojolearn_core_host.so."
    say "  Arms D and E below are NOT a negative control. Do not read them as one."
    SABOTAGE_READY=0
else
    SABOTAGE_READY=1
fi
sha256sum python/mojolearn/identical/_mojolearn.so python/mojolearn/host/_mojolearn_core_host.so \
    /root/sabotage-host/_mojolearn_core_host.so >> "$G" 2>&1

# `import mojolearn` needs at least one identical binding and tolerates the
# rest as stubs, but if it does not import there is nothing to measure, so
# build the remaining bindings rather than lose the leg.
if ! env PYTHONPATH=/root/mojolearn/python pixi run python -c "import mojolearn" > "$OUT/logs/import_probe.log" 2>&1; then
    say "import_probe=FAILED, building every binding"
    for s in bindings/build*.sh; do
        n=$(basename "$s" .sh)
        [ "$n" = build_host_family ] && continue
        case "$n" in
            build_core|build) continue ;;
            build_*_host) run "$n" $HOSTBUILD sh "$s" ;;
            *) run "$n" $BUILD_ENV sh "$s" ;;
        esac
    done
else
    say "import_probe=ok"
fi
say "vendor=$(env PYTHONPATH=/root/mojolearn/python pixi run python -c 'from mojolearn import _backend; print(_backend.vendor())' 2>&1 | tail -1)"

IB="env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python"
LABEL="nvidia-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr 'A-Z ' 'a-z-' | tr -cd 'a-z0-9-' | cut -c1-24)-${MOJOLEARN_GPU_ARCHS}"
say "vendor_label=$LABEL"

# A: infer on the GPU.
run A_gpu_infer $IB pixi run python tools/identity_break.py --lanes "$LANES" --fixtures base \
    --vendor "$LABEL" --json "$OUT/gpu.json"
# B: the same fits, infer through mojolearn.host_model on the CPU host binding.
run B_host_infer $IB MOJOLEARN_IDENTITY_HOST_INFER=1 pixi run python tools/identity_break.py \
    --lanes "$LANES" --fixtures base --vendor "$LABEL" --json "$OUT/host.json"
# C: the deliverable. Every cell must read IDENTICAL.
run C_diff_gpu_host env PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py \
    --diff "$OUT/gpu.json" "$OUT/host.json"
# D and E: the negative control, and the diff that must NOT read IDENTICAL.
if [ "$SABOTAGE_READY" = 1 ]; then
    run D_host_sabotage $IB MOJOLEARN_IDENTITY_HOST_INFER=1 MOJOLEARN_HOST_DIR=/root/sabotage-host \
        MOJOLEARN_HOST_ALLOW_SABOTAGE=1 pixi run python tools/identity_break.py \
        --lanes "$LANES" --fixtures base --vendor "$LABEL-sabotage" --json "$OUT/host_sabotage.json"
    run E_diff_gpu_sabotage env PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py \
        --diff "$OUT/gpu.json" "$OUT/host_sabotage.json"
    # A diff that EXITS 0 here is the failure: the sabotaged arithmetic did not
    # move a byte, so the control did not control anything.
    if [ "$(awk -F'\t' '$1=="E_diff_gpu_sabotage"{print $2}' "$OUT/status.tsv")" = 0 ]; then
        say "SABOTAGE ARM DID NOT FIRE: the diff against the sabotaged host read clean."
    fi
    # And a cell that REFUSED is not a cell that differed.
    if grep -q "REFUSED" "$OUT/logs/E_diff_gpu_sabotage.log" 2>/dev/null; then
        say "SABOTAGE ARM REFUSED at least one cell; a REFUSED cell is not a DIVERGENT cell."
    fi
fi

say "sabotage_ready=$SABOTAGE_READY"
for c in A_gpu_infer B_host_infer C_diff_gpu_host D_host_sabotage E_diff_gpu_sabotage; do
    say "$c exit=$(awk -F'\t' -v n="$c" '$1==n{print $2}' "$OUT/status.tsv")"
done
say "--- C (must be IDENTICAL on every cell) ---"
grep -E "IDENTICAL|DIVERGENT|REFUSED|RELOAD|MOVED|^cells=|^summary" "$OUT/logs/C_diff_gpu_host.log" 2>/dev/null | head -40 >> "$G"
say "--- E (must be DIVERGENT: the control has to be able to fail) ---"
grep -E "IDENTICAL|DIVERGENT|REFUSED|RELOAD|MOVED|^cells=|^summary" "$OUT/logs/E_diff_gpu_sabotage.log" 2>/dev/null | head -40 >> "$G"

# F: the boundary stated directly in bytes, and the one-ULP control.
run F_boundary_bytes env PYTHONPATH=/root/mojolearn/python MOJOLEARN_NUMERIC_MODE=identical \
    pixi run python tools/kmeans_save_boundary_check.py
say "F_boundary_bytes exit=$(awk -F'\t' '$1=="F_boundary_bytes"{print $2}' "$OUT/status.tsv")"
cat "$OUT/logs/F_boundary_bytes.log" >> "$G" 2>/dev/null

say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
