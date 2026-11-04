#!/bin/bash
# tools/two_device_checks.sh -- THE TWO-DEVICE GATE (owner, 2026-11-03), run ON a
# box with two or more GPUs through tools/nvidia_central.sh submit --gpus 2
# (so it persists to R2 from the box under tools/box_persist_run.sh).
#
# Multi-GPU VERIFICATION was retired 2026-10-31 (docs/TOMBSTONES.md,
# MOJOLEARN_PAR_VERIFY): the reference table is one-device only. This gate is
# the comparison that remains: every training/checks/*_parallel_check.mojo
# fits on one device and on two and compares the bits, then
# `python -m mojolearn verify-distributed --devices 0,1` compares
# one-device, two-device and reversed-device results of the Python-sharded
# drivers. Run it after any change under a family's multi_gpu.mojo, the
# device plan (core/device_plan.mojo) or a shard owner.
#
# Output: $MOJOLEARN_RESULTS_DIR (box_persist_run sets it; default
# box-out/two-device): one log per phase, status.tsv (phase<TAB>rc) appended
# as each phase ends so an interrupted run keeps every finished phase, and
# summary.txt at the end.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 2
OUT=${MOJOLEARN_RESULTS_DIR:-$ROOT/box-out/two-device}
mkdir -p "$OUT"
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=nvidia PYTHONUNBUFFERED=1
export MOJOLEARN_SKIP_BUILD_GATE=1 PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-2}
# The checks refuse to run off a RunPod box (`RUNPOD_POD_ID` empty); the queue
# job's environment does not carry the pod's variable, so name the pod here.
export RUNPOD_POD_ID=${RUNPOD_POD_ID:-$(hostname)}
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv > "$OUT/gpu.txt" 2>&1 || { echo "no nvidia-smi" >&2; exit 2; }
n_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
[ "$n_gpu" -ge 2 ] || { echo "two-device gate needs 2 GPUs, found $n_gpu" | tee "$OUT/summary.txt" >&2; exit 2; }
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')
case "$cap" in 9.0) ARCH=sm_90a ;; *) ARCH=sm_$(printf '%s' "$cap" | tr -d '.') ;; esac
export MOJOLEARN_GPU_ARCHS=$ARCH
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null
[ -s "$OUT/status.tsv" ] || : > "$OUT/status.tsv"
done_phase() { grep -q "^$1	0$" "$OUT/status.tsv" 2>/dev/null; }
phase() {  # <name> <command...>: skip a phase that already passed (resume after an interruption)
    name=$1; shift
    if done_phase "$name"; then echo "skip $name (passed earlier)"; return 0; fi
    echo "$(date -u +%FT%TZ) start $name"
    "$@" > "$OUT/$name.log" 2>&1
    code=$?
    printf '%s\t%s\n' "$name" "$code" >> "$OUT/status.tsv"
    echo "$(date -u +%FT%TZ) end $name rc=$code"
    return "$code"
}
phase install pixi install || exit $?

# 1. The Mojo two-device checks: one device vs two, same bits.
for c in training/checks/*_parallel_check.mojo; do
    name=$(basename "$c" .mojo)
    phase "$name" pixi run mojo run -j 1 --target-accelerator "$ARCH" -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . "$c"
done

# 2. verify-distributed: the Python-sharded drivers (ARIMA, Holt-Winters, GP
# classification, disjoint IVF search). The bindings it loads
# (python/mojolearn/_native_parallel_shard.py) are built here, once.
# GPU bindings the drivers call, then THE COMPLETE HOST LAYOUT (`import
# mojolearn` and the classical constructors refuse a partial one), i.e.
# every bindings/build_*_host.sh.
for b in main kernel_methods gp x_neighbors ivf tsa arima x_sequence estimators \
         $(ls bindings/build_*_host.sh | sed 's#bindings/build_##; s#\.sh$##'); do
    s=bindings/build_$b.sh; [ "$b" = main ] && s=bindings/build.sh
    [ -f "$s" ] || continue
    case "$b" in
        *_host) phase "build-$b" env -u MOJOLEARN_TARGET_COLUMN -u MOJOLEARN_GPU_ARCHS sh "$s" ;;  # host bindings compile the CPU column only, no GPU arch
        *) phase "build-$b" sh "$s" ;;
    esac
done
rm -f "$OUT/distributed.json"
phase verify-distributed pixi run python -m mojolearn verify-distributed --devices 0,1 --out "$OUT/distributed.json"

# 3. Summary.
{
    echo "two-device gate at $(cat "$OUT/commit.txt") on $(head -n 2 "$OUT/gpu.txt" | tail -n 1)"
    # the LAST row of each phase counts (a resumed run appends a new row)
    awk -F'\t' '{ rc[$1] = $2; if (!($1 in seen)) { seen[$1] = 1; order[++n] = $1 } }
        END { for (i = 1; i <= n; i++) { if (rc[order[i]] == 0) p++; else { f++; bad = bad " " order[i] } }
              printf "phases %d passed, %d failed:%s\n", p, f, bad }' "$OUT/status.tsv"
} | tee "$OUT/summary.txt"
grep -q ' 0 failed:' "$OUT/summary.txt" && exit 0
exit 1
