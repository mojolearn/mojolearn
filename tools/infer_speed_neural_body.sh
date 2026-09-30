#!/bin/sh
# tools/infer_speed_neural_body.sh -- lane/infer-speed-neural: faster neural
# INFERENCE in IDENTICAL mode with no output bit moved. RUNS ON THE POD from a
# checkout root, one phase per call, under nohup:
#
#   INFER_PHASE=setup    ROOT=/root/mojolearn sh tools/infer_speed_neural_body.sh
#   INFER_PHASE=identity ROOT=<tree> LABEL=<before|after> sh tools/infer_speed_neural_body.sh
#   INFER_PHASE=bytelm   ROOT=<tree> LABEL=<before|after> sh tools/infer_speed_neural_body.sh
#   INFER_PHASE=build    ROOT=<tree> sh tools/infer_speed_neural_body.sh
#
# PHASES
#   setup     box record; pixi install; MOJOLEARN_GPU_ARCHS from the compute
#             capability; the three GPU bindings (transformer, mamba, byte_lm)
#             and the two host bindings (neural, byte_lm) in IDENTICAL mode;
#             SHA-256 of every .so
#   build     the same five bindings on ROOT (a candidate tree)
#   identity  tools/identity_break.py on the neural lanes, fixtures
#             base,ties,odd, --step-full, repeats 2, ONE cuda column and ONE
#             cpu column (a Python-only staging of the package with
#             MOJOLEARN_HOST_DIR pointing at ROOT's host bindings, the route
#             tools/identity_iterate.py stages)
#   bytelm    tools/byte_lm_gpu_logits_sweep.py (GPU against the CPU
#             reference path) and tools/byte_lm_host_gate.py on ROOT
# Each step appends name, exit, seconds to $OUT/status.tsv. POSIX sh, set -u.
set -u
ROOT=${ROOT:-/root/mojolearn}
OUT=${OUT:-/root/infer_out}
LABEL=${LABEL:-run}
L="$OUT/logs"
mkdir -p "$L"
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1
export MOJOLEARN_CORPUS_SOURCE_DIR=${MOJOLEARN_CORPUS_SOURCE_DIR:-/root/datasets/corpus}
export PYTHONPATH="$ROOT/python:$ROOT/tools"
# The commit witness identity_break and the byte LM sweep refuse to run
# without: a COMMIT file names a candidate tree, SHIPPED_COMMIT.txt the base.
if [ -s "$ROOT/COMMIT" ]; then MOJOLEARN_COMMIT=$(cat "$ROOT/COMMIT"); else MOJOLEARN_COMMIT=$(cat "$ROOT/SHIPPED_COMMIT.txt" 2>/dev/null || echo unknown); fi
export MOJOLEARN_COMMIT MOJOLEARN_GATE_COMMIT="$MOJOLEARN_COMMIT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NEURAL_LANES="mlp,transformer,transformer-window,mamba1,mamba2,mamba2-dtlimit,mamba3,samba,samba-untied-dropout-accum,byte-lm,byte-lm-resident,byte-lm-host-infer"
# The CPU column carries the lanes the host bindings cover: the decode lanes,
# mlp and the byte LM host inference lane (the resident session is a GPU
# object and the rlpair part needs the training binding on either column).
CPU_LANES="mlp,transformer,transformer-window,mamba1,mamba2,mamba2-dtlimit,mamba3,samba,samba-untied-dropout-accum,byte-lm-host-infer"
note() { echo "$* $(date -u +%H:%M:%S)" | tee -a "$OUT/progress.txt"; }

step() {
    _n=$1; _cap=$2; shift 2
    _t=$(date +%s)
    timeout -k 30 "$_cap" "$@" > "$L/$_n.log" 2>&1
    _rc=$?
    printf '%s\t%s\t%s\n' "$_n" "$_rc" "$(( $(date +%s) - _t ))" >> "$OUT/status.tsv"
    note "$_n=$_rc"
    return $_rc
}

gpu_arch() {
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')
    case "$cap" in 9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;; *) MOJOLEARN_GPU_ARCHS=sm_$(printf '%s' "$cap" | tr -d '.') ;; esac
    export MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=nvidia
}

build_all() {
    _tag=$1
    gpu_arch
    export MOJOLEARN_COMPILE_JOBS=8
    step "build_transformer_$_tag" 1500 sh bindings/build_transformer.sh
    step "build_mamba_$_tag" 1500 sh bindings/build_mamba.sh
    step "build_byte_lm_$_tag" 1500 sh bindings/build_byte_lm.sh
    step "build_neural_host_$_tag" 1500 env -u MOJOLEARN_GPU_ARCHS -u MOJOLEARN_TARGET_COLUMN sh bindings/build_neural_host.sh
    step "build_byte_lm_host_$_tag" 1500 env -u MOJOLEARN_GPU_ARCHS -u MOJOLEARN_TARGET_COLUMN sh bindings/build_byte_lm_host.sh
    # The lanes' other bindings, unchanged by this lane, so every neural lane
    # has a column: training (samba, the resident rlpair) and linalg (mlp)
    # on the GPU; the reference host families the CPU column trains through.
    if [ "${INFER_BUILD_EXTRA:-1}" = 1 ]; then
        step "build_training_$_tag" 1500 sh bindings/build_training.sh
        step "build_linalg_$_tag" 1500 sh bindings/build_linalg.sh
        for fam in transformer mamba training linalg; do
            step "build_${fam}_host_$_tag" 1500 env -u MOJOLEARN_GPU_ARCHS -u MOJOLEARN_TARGET_COLUMN sh bindings/build_host_family.sh "$fam"
        done
    fi
    sha256sum python/mojolearn/identical/*.so python/mojolearn/host/*.so > "$OUT/so_sha256_$_tag.txt" 2>&1
}

cpu_stage() {
    # A Python-only copy of the package beside ROOT's host bindings, the
    # CPU-only install route (tools/identity_iterate.py::cpu_package).
    _stage="$ROOT/../cpu-stage-$(basename "$ROOT")"
    rm -rf "$_stage"; mkdir -p "$_stage/mojolearn"
    (cd "$ROOT/python/mojolearn" && find . -name '*.py' -not -path '*/__pycache__/*' | while read -r f; do
        mkdir -p "$_stage/mojolearn/$(dirname "$f")"; cp "$f" "$_stage/mojolearn/$f"; done)
    printf '%s' "$_stage"
}

case "${INFER_PHASE:-}" in
setup)
    note start commit="$(cat "$ROOT/SHIPPED_COMMIT.txt" 2>/dev/null)"
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader > "$OUT/gpu.txt" 2>&1
    { nproc; grep -m1 'model name' /proc/cpuinfo; uname -a; python3 --version; free -g | head -2; } > "$OUT/box.txt" 2>&1
    [ -x "$HOME/.pixi/bin/pixi" ] || curl -fsSL https://pixi.sh/install.sh | sh > "$L/pixi_get.log" 2>&1
    step pixi_install 1500 pixi install
    pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
    pixi run python -c "import numpy; print('numpy', numpy.__version__)" >> "$OUT/mojo_version.txt" 2>&1
    build_all base
    step import_identical 300 pixi run python -c "import mojolearn; print('import OK', mojolearn.vendor(), mojolearn.numeric_mode())"
    note setup_done
    : > "$OUT/setup.done"
    ;;
build)
    build_all "$LABEL"
    : > "$OUT/build_$LABEL.done"
    ;;
identity)
    gpu_arch
    rm -f "$OUT/identity_cuda_$LABEL.json" "$OUT/identity_cpu_$LABEL.json"
    step "identity_cuda_$LABEL" 3000 pixi run python -u tools/identity_break.py \
        --lanes "$NEURAL_LANES" --fixtures base,ties,odd --step-full --no-rlpair --repeats 2 \
        --require-backend cuda --fail-on-refused --json "$OUT/identity_cuda_$LABEL.json"
    _stage=$(cpu_stage)
    step "identity_cpu_$LABEL" 3000 env PYTHONPATH="$_stage:$ROOT/tools" MOJOLEARN_HOST_DIR="$ROOT/python/mojolearn/host" \
        MOJOLEARN_CPU_THREADS=4 pixi run python -u tools/identity_break.py \
        --lanes "$CPU_LANES" --fixtures base,ties,odd --step-full --no-rlpair --repeats 2 \
        --require-cpu --require-backend cpu --fail-on-refused --json "$OUT/identity_cpu_$LABEL.json"
    : > "$OUT/identity_$LABEL.done"
    ;;
bytelm)
    gpu_arch
    rm -f "$OUT/bytelm_sweep_$LABEL.json" "$OUT/bytelm_host_gate_$LABEL.json"
    step "bytelm_sweep_$LABEL" 2400 pixi run python -u tools/byte_lm_gpu_logits_sweep.py \
        --stateless-every 8 --report "$OUT/bytelm_sweep_$LABEL.json"
    step "bytelm_host_gate_$LABEL" 2400 pixi run python -u tools/byte_lm_host_gate.py \
        --steps every:16 --report "$OUT/bytelm_host_gate_$LABEL.json"
    : > "$OUT/bytelm_$LABEL.done"
    ;;
sabotage)
    # The controls that must be SEEN TO FAIL. (1) A copy of ROOT with the GPU
    # byte LM binding rebuilt under -D MOJOLEARN_BYTE_LM_LOGITS_SABOTAGE=1 and
    # the host binding CLEAN: the GPU sweep must exit non-zero on a real digest
    # difference. (2) The same copy with the host binding then rebuilt under
    # -D MOJOLEARN_BYTE_LM_HOST_SABOTAGE=1: the host gate must mismatch
    # (--expect-mismatch exits 0 only then; the loader admits the build only
    # under MOJOLEARN_BYTE_LM_HOST_ALLOW_SABOTAGE=1). (3) identity_break on the
    # cuda column under MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1, diffed against the
    # AFTER column: the batch and stepfull cells must read BATCH_MOVED.
    gpu_arch
    SAB="$ROOT/../sabotage-$(basename "$ROOT")"
    rm -rf "$SAB"; cp -a "$ROOT" "$SAB"
    rm -f "$SAB/python/mojolearn/identical/_mojolearn_byte_lm.so"
    (cd "$SAB" && step sabotage_build_byte_lm_gpu 1500 env MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_BYTE_LM_LOGITS_SABOTAGE=1" sh bindings/build_byte_lm.sh)
    rm -f "$OUT/sabotage_sweep.json" "$OUT/sabotage_host_gate.json" "$OUT/identity_cuda_sabotage.json"
    (cd "$SAB" && step sabotage_sweep_must_fail 2400 env PYTHONPATH="$SAB/python:$SAB/tools" pixi run python -u tools/byte_lm_gpu_logits_sweep.py \
        --stateless-every 8 --report "$OUT/sabotage_sweep.json")
    rm -f "$SAB/python/mojolearn/host/_mojolearn_byte_lm_host.so"
    (cd "$SAB" && step sabotage_build_byte_lm_host 1500 env -u MOJOLEARN_GPU_ARCHS -u MOJOLEARN_TARGET_COLUMN MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_BYTE_LM_HOST_SABOTAGE=1" sh bindings/build_byte_lm_host.sh)
    (cd "$SAB" && step sabotage_host_gate_expect_mismatch 2400 env PYTHONPATH="$SAB/python:$SAB/tools" MOJOLEARN_BYTE_LM_HOST_ALLOW_SABOTAGE=1 pixi run python -u tools/byte_lm_host_gate.py \
        --steps every:16 --expect-mismatch --report "$OUT/sabotage_host_gate.json")
    if [ "${INFER_SABOTAGE_IDENTITY:-1}" = 1 ]; then
        step identity_cuda_sabotage 3000 env MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1 pixi run python -u tools/identity_break.py \
            --lanes "transformer,transformer-window,mamba1,mamba2,mamba3" --fixtures base --step-full --no-rlpair --repeats 2 \
            --require-backend cuda --json "$OUT/identity_cuda_sabotage.json"
        step identity_diff_sabotage_must_move 600 pixi run python -u tools/identity_break.py --diff "$OUT/identity_cuda_after.json" "$OUT/identity_cuda_sabotage.json"
    fi
    : > "$OUT/sabotage.done"
    ;;
diff)
    step identity_diff_cuda 600 pixi run python -u tools/identity_break.py --diff "$OUT/identity_cuda_before.json" "$OUT/identity_cuda_after.json"
    step identity_diff_cpu 600 pixi run python -u tools/identity_break.py --diff "$OUT/identity_cpu_before.json" "$OUT/identity_cpu_after.json"
    : > "$OUT/diff.done"
    ;;
*)
    echo "INFER_PHASE must be setup, build, identity, bytelm, sabotage or diff" >&2
    exit 2
    ;;
esac
