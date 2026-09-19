#!/bin/sh
# tools/gpu_class_gaps_nvidia_leg.sh: the on-box body (MOJOLEARN_GEMM_LEG_EXTRA
# for tools/gemm_remote_leg.sh nvidia) of the NVIDIA identity column that closes
# the "GPU column on fewer than three classes" gap, 2026-09-19.
#
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/gpu_class_gaps_nvidia_leg.sh \
#   sh tools/gemm_remote_leg.sh nvidia --rent \
#      --local-card <an apple.card from a previous leg>
#
# WHY THIS FILE AND NOT tools/identity_three_columns_leg.sh. That body builds
# EVERY binding and runs EVERY lane. It did so on 2026-09-14 at 166 lanes, in
# 1010 s of builds plus 1006 s of identity run SPLIT ACROSS TWO GPUs
# (bench/results/identity_break/2026-09-14_166-lanes). The harness carries 241
# lanes today and this leg has ONE GPU and a SIXTY MINUTE HARD LEASE. Serially
# that does not fit, and a lease that expires mid-run fetches nothing at all.
# So this body is the same body, scoped: the same builds, the same
# tools/identity_break.py, the same `--vendor`/`--json` column, over the lanes
# that are actually owed an NVIDIA column.
#
# WHICH LANES, AND WHY EXACTLY THESE. docs/VERIFICATION_MATRIX.md reports 49
# lanes with a GPU column on fewer than three classes. Of those 49:
#
#   * 18 have apple AND amd and are missing ONLY nvidia. One single-device
#     NVIDIA column takes each of them to three classes. They are PHASE A,
#     below, together with the four gbdt lanes in the next bullet, which cost
#     nothing extra once _mojolearn_gbdt is built.
#   * 18 have apple ALONE. An NVIDIA column takes them to two classes, not
#     three (the third would be AMD, which this leg does not touch). Four are
#     gbdt lanes and ride along in PHASE A; the other fourteen need the linalg,
#     mamba, transformer and training families and are PHASE B.
#   * 13 are `par-*` multi-GPU driver lanes. NOTHING HERE CAN ANSWER THEM and
#     nothing here pretends to: their claim is written in
#     identity_break._par_devices and needs two devices, and a column with
#     par_devices set is REFUSED by python/mojolearn/_verify_reference.admit
#     ("par_devices N"). They need a two-device leg, which is a different box
#     and a different lease.
#
# THE COLUMN MUST BE ADMISSIBLE OR IT IS NOT EVIDENCE. `admit` wants identical
# mode, a real commit, the default fixture size, one device, and a name that is
# not a sabotage/partial/probe/smoke run. Hence: no --fixtures cap (the default
# set, at the default size), no MOJOLEARN_IDENTITY_*_SABOTAGE anywhere in this
# file, the commit witness taken from the runner's own leg.txt rather than
# guessed, and file names with none of the excluded tokens in them.
#
# EVERY PHASE IS BOUNDED BY `timeout`. Not decoration: the body must FINISH so
# the runner can fetch. A phase that hangs past its bound is a finding in
# status.tsv and the rest of the leg still comes home.
#
# POSIX sh: the pod's /bin/sh is dash. `set -u` and deliberately NOT `set -e`:
# a phase that fails is a FINDING and its log must come home.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/gpu-class-gaps
mkdir -p "$OUT/logs"
G="$OUT/gate.txt"
JOBS="${MOJOLEARN_COMPILE_JOBS:-8}"

# PHASE A: the 18 lanes that reach three classes with this column, plus the
# four gbdt lanes that share _mojolearn_gbdt with gbdt-query-rmse.
LANES_A=gbdt-query-rmse,gbdt-categorical-ctr-tables,gbdt-pair-logit,gbdt-tensor-ctr-tables,gbdt-yeti-rank,gmm-random-init-sample,gmm-sample,gp-normalize-y,gp-optimize,gp-optimize-restarts,gp-sample-y,gp-sample-y-normalize,gpc,gpc-multiclass,ivf-extend,kernel-ridge-laplacian,kernel-ridge-poly,kernel-ridge-sigmoid,nystroem-laplacian,nystroem-poly,nystroem-sigmoid,svc-poly
# PHASE B: the apple-only weight-format and gemm lanes. Two classes, not three.
LANES_B=gemm-bf16,gemm-int8,mamba1-bf16w,mamba1-int8w,mamba2-bf16w,mamba2-int8w,mamba3-bf16w,mamba3-int8w,mlp-bf16w,mlp-int8w,samba-bf16w,samba-int8w,transformer-bf16w,transformer-int8w

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
say "lanes_a=$LANES_A"
say "lanes_b=$LANES_B"

# THE COMMIT WITNESS, OR identity_break REFUSES TO WRITE A JSON. The box has no
# .git (the runner ships `git archive` at a pinned sha) and the gemm payload
# does not write commit.txt for the extra body. The runner DOES record the
# commit in /root/gemm_leg_out/leg.txt. Take it from there, and never guess.
if [ ! -s /root/mojolearn/commit.txt ]; then
    _c=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1 | awk '{print $1}')
    case "$_c" in
        [0-9a-f][0-9a-f]*) printf '%s\n' "$_c" > /root/mojolearn/commit.txt ;;
        *) say "NO COMMIT WITNESS: leg.txt gave '$_c'; every identity phase will refuse" ;;
    esac
fi
say "commit=$(head -1 /root/mojolearn/commit.txt 2>/dev/null)"

# ------------------------------------------------------------ the box itself
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader > "$OUT/logs/device.txt" 2>&1
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
say "gpu_archs=${MOJOLEARN_GPU_ARCHS:-unset}"
LABEL="nvidia-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr 'A-Z ' 'a-z-' | tr -cd 'a-z0-9-' | cut -c1-24)-${MOJOLEARN_GPU_ARCHS}"
say "vendor_label=$LABEL"

build() {
    run "$1" timeout 420 env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
        MOJOLEARN_COMPILE_JOBS="$JOBS" sh "bindings/$1.sh"
}

# ------------------------------------------- THE HOST MATH LIBRARY, OR NOTHING IMPORTS
# MEASURED ON POD 9laka4vs9h2zli, 2026-09-19: both identity phases died ONE
# SECOND in, before any GPU work, at
#
#   OSError: python/mojolearn/.libs/libMojolearnMath.so: cannot open shared
#   object file: No such file or directory
#
# and the run came home with every binding built and NOT ONE CELL.
#
# `python/mojolearn/_portable_math.py` dlopens that library, and
# `_training_impl.py`'s `def kaiming_uniform(self, shape, fan_in,
# a=math.sqrt(5.0))` evaluates it as a DEFAULT ARGUMENT, at class definition
# time. So `import mojolearn` needs it unconditionally -- it is not lazy and
# no lane can avoid it.
#
# Nothing under bindings/ builds it, `python/mojolearn/.libs/` and `.dylibs/`
# are gitignored so `git archive` ships nothing, and the only thing in the
# tree that ever compiles it is `packaging/macos/build_release_wheel.sh`, a
# RELEASE WHEEL script that does not run on Linux. A developer Mac has the
# file sitting in its checkout from some past wheel build and never notices;
# a freshly rented box cannot import the package at all.
#
# This calls the tree's OWN recipe, `packaging/portable_math/stage.py`'s
# build(), rather than retyping its compiler flags here -- those flags
# (-ffp-contract=off, -fno-fast-math, -march=x86-64-v3, -nostdlib) are the
# arithmetic contract, and a second copy of them is a second answer.
run portable_math timeout 300 env PYTHONPATH=/root/mojolearn/packaging/portable_math \
    pixi run python -c "import pathlib, stage; stage.build(pathlib.Path('/root/mojolearn/python/mojolearn/.libs/libMojolearnMath.so'))"
say "portable_math_exit=$(awk -F'	' '$1=="portable_math"{print $2}' "$OUT/status.tsv")"
ls -l python/mojolearn/.libs/ >> "$G" 2>&1
# FAIL FAST. If the package still cannot import, every phase below is a
# one-second traceback and the lease is spent finding that out twice.
run import_probe timeout 300 env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python -c "import mojolearn; print('IMPORT_OK', mojolearn.__version__)"
say "import_probe=$(tail -1 "$OUT/logs/import_probe.log" 2>/dev/null)"

# --------------------------------------------------- PHASE A builds, then the column
# bindings/build.sh is the shared kernels and fixtures every lane reaches; the
# other six are exactly the families phase A's lanes bind:
#   gbdt-*      -> _mojolearn_gbdt        gmm-*        -> _mojolearn_mixture
#   gp-*, gpc-* -> _mojolearn_gp          ivf-extend   -> _mojolearn_ivf
#   svc-poly    -> _mojolearn_svm
#   kernel-ridge-*, nystroem-* -> _mojolearn_kernel_methods
for b in build build_gbdt build_mixture build_gp build_ivf build_kernel_methods build_svm; do
    build "$b"
done
sha256sum python/mojolearn/identical/*.so >> "$G" 2>/dev/null
say "vendor_readback=$(env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python pixi run python -c 'import mojolearn; print(mojolearn.vendor())' 2>&1 | tail -1)"

# THE DELIVERABLE. Default fixtures, default size, two repeats in one process,
# one device. No sabotage switch is set anywhere in this run.
run column-a timeout 900 env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/identity_break.py --lanes "$LANES_A" --repeats 2 \
    --vendor "$LABEL" --json "$OUT/$LABEL.classical.json"
say "column_a_exit=$(awk -F'	' '$1=="column-a"{print $2}' "$OUT/status.tsv")"
grep -E '^cells=|^train:|^infer:|^model:|^batch:|MOVED|DIVERGENT|REFUSED|RELOAD' "$OUT/logs/column-a.log" | head -60 >> "$G"
cp "$OUT/logs/column-a.log" "$OUT/column-a.log" 2>/dev/null

# --------------------------------------------------- PHASE B builds, then the column
# The weight-format lanes: ml.lowbit.pack goes through _mojolearn_linalg on a
# GPU column, the blocks through _mojolearn_mamba and _mojolearn_transformer,
# SmallMLPTrainer and SambaStack through _mojolearn_training, and the infer
# part of each through the CPU-only _mojolearn_neural_host. `env -u
# MOJOLEARN_GPU_ARCHS` comes FIRST and not as decoration: this file exports
# that variable above and bindings/build_host_family.sh refuses a CPU build
# that carries one; `env FOO=1 -u BAR` does NOT unset BAR, because env stops
# parsing options at the first assignment.
for b in build_linalg build_mamba build_transformer build_training; do
    build "$b"
done
run build_neural_host timeout 420 env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu \
    MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS="$JOBS" sh bindings/build_neural_host.sh
sha256sum python/mojolearn/identical/*.so >> "$G" 2>/dev/null

run column-b timeout 900 env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/identity_break.py --lanes "$LANES_B" --repeats 2 \
    --vendor "$LABEL" --json "$OUT/$LABEL.lowbit.json"
say "column_b_exit=$(awk -F'	' '$1=="column-b"{print $2}' "$OUT/status.tsv")"
grep -E '^cells=|^train:|^infer:|^model:|^batch:|MOVED|DIVERGENT|REFUSED|RELOAD' "$OUT/logs/column-b.log" | head -60 >> "$G"
cp "$OUT/logs/column-b.log" "$OUT/column-b.log" 2>/dev/null

# THE CROSS-COLUMN DIFF IS NOT RUN HERE. `git archive` ships the source only,
# so bench/results/ does not exist on this box and the Apple and AMD columns
# are not here to diff against. That diff costs nothing at home, against the
# fetched column, and that is where it runs.

# --------------------------------------------------------------- bring it home
cat "$OUT/status.tsv" >> "$G" 2>/dev/null
ls -l "$OUT" >> "$G" 2>/dev/null
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
