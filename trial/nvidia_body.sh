#!/bin/sh
# Mojo 1.2 nightly trial, NVIDIA leg body (runs on the pod from the pinned
# archive of trial/mojo-1-2-at-0819, via a MOJOLEARN_GEMM_LEG_EXTRA wrapper).
# Builds every identical GPU binding and every host binding with the nightly
# toolchain in pixi.lock, one at a time, timing each, then runs the 0.8.19
# NVIDIA release selection (trial/selection-cuda.json) as a --gpu-pass.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/trial
mkdir -p "$OUT/logs"
G="$OUT/gate.txt"
say() { echo "$@" >> "$G"; }
T0=$(date +%s)
JOBS=$(nproc 2>/dev/null || echo 8)
[ "$JOBS" -gt 16 ] && JOBS=16
export MOJOLEARN_COMMIT=69a519c1522d245f197d7eb1d2d06e1b0cb6ad0e
export MOJOLEARN_BINCACHE=0
PATH="$HOME/.pixi/bin:$PATH"; export PATH
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) jobs=$JOBS"
pixi run mojo --version >> "$G" 2>&1
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader >> "$G" 2>&1
_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
case "$_cc" in
    9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;;
    *)   MOJOLEARN_GPU_ARCHS="sm_$(echo "$_cc" | tr -d .)" ;;
esac
export MOJOLEARN_GPU_ARCHS
say "gpu_archs=$MOJOLEARN_GPU_ARCHS"

env PYTHONPATH=/root/mojolearn/packaging/portable_math pixi run python -c \
    "import pathlib, stage; stage.build(pathlib.Path('/root/mojolearn/python/mojolearn/.libs/libMojolearnMath.so'))" \
    > "$OUT/logs/portable_math.log" 2>&1
say "portable_math_exit=$?"

# One binding at a time, 900 s bound each (a compiler deadlock shows as 124).
for s in bindings/build*.sh; do
    n=$(basename "$s" .sh)
    [ "$n" = build_host_family ] && continue
    t=$(date +%s)
    case "$n" in
        *_host)
            fam=${n#build_}; fam=${fam%_host}; FAM=$(printf '%s' "$fam" | tr a-z A-Z)
            timeout -k 10 900 env -u MOJOLEARN_GPU_ARCHS -u MOJOLEARN_HOST_OUTDIR -u "MOJOLEARN_${FAM}_HOST_OUTDIR" \
                MOJOLEARN_TARGET_COLUMN=cpu MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
                MOJOLEARN_COMPILE_JOBS="$JOBS" MOJOLEARN_BUILD_JOBS="$JOBS" sh "$s" > "$OUT/logs/$n.log" 2>&1 ;;
        *)
            timeout -k 10 900 env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
                MOJOLEARN_COMPILE_JOBS="$JOBS" sh "$s" > "$OUT/logs/$n.log" 2>&1 ;;
    esac
    e=$?
    printf '%s\t%s\t%s\n' "$n" "$e" "$(( $(date +%s) - t ))" >> "$OUT/builds.tsv"
done
say "builds_failed=$(awk -F'	' '$2!=0{printf "%s(%s) ", $1, $2}' "$OUT/builds.tsv")"
say "elapsed_after_builds=$(( $(date +%s) - T0 ))"
sha256sum python/mojolearn/identical/*.so python/mojolearn/host/*.so > "$OUT/so_sha256.txt" 2>&1

timeout -k 10 2700 pixi run python tools/verify_lanes.py --gpu-pass cuda \
    --selection trial/selection-cuda.json --out "$OUT/cuda" --budget 2600 > "$OUT/logs/column.log" 2>&1
say "column_exit=$?"
tail -5 "$OUT/logs/column.log" >> "$G"
say "elapsed_total=$(( $(date +%s) - T0 ))"
