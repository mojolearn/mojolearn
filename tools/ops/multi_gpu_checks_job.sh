#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# tools/ops/multi_gpu_checks_job.sh -- ONE optional job on a box with two or more GPUs of one vendor (lane
# fx-multigpu-optional, 2026-11-20): tools/dev/multi_gpu_checks.py against the installed wheels.
#
# Optional. Not required for any merge or release. Not shipped to users. Nothing reads its result: no release
# gate, ledger row, admit step, CI trigger, merge gate or push hook (docs/dev/MULTI_GPU_CHECKS.md). It is started
# by hand, through the supported wrappers only, which run it under tools/box_persist_run.sh (results to R2 from
# the box):
#   tools/nvidia_central.sh submit <lane> --gpus 2 --results <res> --env MGC_WHEELS=<dir-on-box> \
#       <tree-on-box>/tools/ops/multi_gpu_checks_job.sh
#   tools/do_amd_steward.sh run <tag> <res> env MGC_WHEELS=<dir-on-box> bash tools/ops/multi_gpu_checks_job.sh
# (an `lq add <box> CMD` line works only on a slot with two GPUs and only if that queue sets MOJOLEARN_RESULTS_DIR.)
#
# Inputs (environment; it takes no arguments):
#   MGC_PYTHON   an interpreter that already has the wheels installed, or
#   MGC_WHEELS   a directory of the wheels to install into a fresh venv under the results dir (the core wheel and
#                the vendor wheel of one build); NumPy is added for the harness's fixtures (never the runtime)
#   MGC_UNITS    comma-separated unit names (default: every unit)
#   MGC_DEVICES  comma-separated device indices (default: every GPU found)
#   MGC_ARGS     extra arguments for the tool (for example "--strict-spread", "--no-fault", "--jobs 8",
#                "--host-gb-per-job 3", "--gpu-gb-per-job 2")
#   MGC_MOJO_CHECKS=1  also run tools/two_device_checks.sh afterwards (the Mojo-level *_parallel_check programs;
#                it builds from source on the box and takes far longer). Default 0.
#
# PARALLEL PLAN: the tool runs every (unit, arm) as its own process, as many at once as the allocation's cores and
# the host and device memory bounds allow, the one-GPU arms spread over the GPUs round robin; this script samples
# busy cores for the length of the run and prints "busy cores: mean X, max Y of N". The two per-arm memory figures
# are estimates until a first run: read "largest worker RSS" and the nvidia-smi peak in gpu-peak.txt, then pass
# measured values through MGC_ARGS.
#
# Output in $MOJOLEARN_RESULTS_DIR: job/job.log, job/box.txt, job/busy.txt, job/gpu-peak.txt, job/status.json and
# out/multi_gpu_checks.json with out/arms/<unit>.<arm>.{json,log}. Exit: the tool's (0 every unit OK, 1 not,
# 2 cannot run).
set -euo pipefail

[ $# -eq 0 ] || { echo "usage: MGC_PYTHON=... | MGC_WHEELS=... $0   (no arguments)" >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RES="${MOJOLEARN_RESULTS_DIR:?multi_gpu_checks_job.sh runs under tools/box_persist_run.sh (nvidia_central.sh submit or do_amd_steward.sh run)}"
[ -n "${MGC_PYTHON:-}" ] || [ -n "${MGC_WHEELS:-}" ] || { echo "multi-gpu-checks-job: set MGC_PYTHON or MGC_WHEELS" >&2; exit 2; }
[ ! -e "$RES/out" ] || { echo "multi-gpu-checks-job: $RES/out exists; a new job uses a new tag" >&2; exit 2; }
J="$RES/job"
mkdir -p "$J" "$RES/out"
exec > "$J/job.log" 2>&1
PHASE=inputs
export PATH="/root/.pixi/bin:/usr/local/cuda/bin:/opt/rocm/bin:$PATH"
# shellcheck source=tools/ops/box_cores_lib.sh
. "$ROOT/tools/ops/box_cores_lib.sh"
say() { echo "[$(date -u +%H:%M:%S) multi-gpu-checks-job] $*"; }
SAMPLER=""
GPU_SAMPLER=""
finish() {
    local code=$?
    trap - EXIT
    [ -z "$SAMPLER" ] || kill "$SAMPLER" 2>/dev/null || true
    [ -z "$GPU_SAMPLER" ] || kill "$GPU_SAMPLER" 2>/dev/null || true
    printf '{"job":"multi-gpu-checks","optional":true,"phase":"%s","exit":%d,"commit":"%s"}\n' \
        "$PHASE" "$code" "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)" > "$J/status.json"
    say "exit $code in phase $PHASE"
    exit "$code"
}
trap finish EXIT
trap 'exit 124' TERM INT

CORES=$(alloc_cores)
{
    uname -a
    echo "cores (allocation): $CORES; host cpus: $(nproc --all 2>/dev/null || nproc)"
    echo "memory available kB: $(alloc_mem_kb)"
    if command -v nvidia-smi > /dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total,memory.free --format=csv
    elif command -v rocm-smi > /dev/null 2>&1; then
        rocm-smi --showproductname --showmeminfo vram
    else
        echo "neither nvidia-smi nor rocm-smi found"
    fi
} > "$J/box.txt" 2>&1 || true
say "box: $(sed -n 2p "$J/box.txt")"

PHASE=install
PY=${MGC_PYTHON:-}
if [ -z "$PY" ]; then
    shopt -s nullglob
    wheels=("$MGC_WHEELS"/*.whl)
    shopt -u nullglob
    [ ${#wheels[@]} -gt 0 ] || { say "no wheel in $MGC_WHEELS"; exit 2; }
    PY="$RES/venv/bin/python"
    { python3 -m venv "$RES/venv" \
      && "$PY" -m pip install -q --no-cache-dir "${wheels[@]}" \
      && "$PY" -m pip install -q --no-cache-dir 'numpy>=1.26.4'; } > "$J/venv.log" 2>&1 \
        || { say "installing the wheels failed (see venv.log)"; tail -n 5 "$J/venv.log"; exit 2; }
    say "installed ${#wheels[@]} wheel(s) from $MGC_WHEELS"
fi
[ -x "$PY" ] || { say "no interpreter at $PY"; exit 2; }

PHASE=checks
args=(--python "$PY" --out "$RES/out")
[ -z "${MGC_UNITS:-}" ] || args+=(--units "$MGC_UNITS")
[ -z "${MGC_DEVICES:-}" ] || args+=(--devices "$MGC_DEVICES")
# MGC_ARGS is a short list of plain flags; word splitting is intended
# shellcheck disable=SC2206
extra=(${MGC_ARGS:-})
SAMPLER=$(busy_sampler "$J/busy.txt" 5)
if command -v nvidia-smi > /dev/null 2>&1; then
    # the device memory in use, every 5 s, for the per-arm device memory bound (part of this job, not a watcher)
    ( while sleep 5; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '; echo; done ) \
        > "$J/gpu-used.txt" 2>/dev/null &
    GPU_SAMPLER=$!
fi
rc=0
python3 "$ROOT/tools/dev/multi_gpu_checks.py" "${args[@]}" "${extra[@]}" > "$J/checks.log" 2>&1 || rc=$?
kill "$SAMPLER" 2>/dev/null || true; SAMPLER=""
if [ -n "$GPU_SAMPLER" ]; then
    kill "$GPU_SAMPLER" 2>/dev/null || true; GPU_SAMPLER=""
    awk '{for (i = 1; i <= NF; i++) if ($i > m[i]) m[i] = $i; if (NF > n) n = NF}
         END {printf "peak device memory used MiB:"; for (i = 1; i <= n; i++) printf " gpu%d=%d", i - 1, m[i]; print ""}' \
        "$J/gpu-used.txt" > "$J/gpu-peak.txt" 2>/dev/null || true
    say "$(cat "$J/gpu-peak.txt" 2>/dev/null)"
fi
grep -E '^MULTIGPU (plan|summary)' "$J/checks.log" || true
grep -E '^MULTIGPU ' "$J/checks.log" | grep -Ev ' OK$| OK |^MULTIGPU (plan|summary)' | head -n 40 || true
say "busy cores: $(busy_of "$J/busy.txt") of $CORES"
say "multi_gpu_checks exited $rc"

if [ "${MGC_MOJO_CHECKS:-0}" = 1 ]; then
    PHASE=mojo-checks
    mrc=0
    MOJOLEARN_RESULTS_DIR="$RES/out/two-device" MOJOLEARN_COMPILE_JOBS="$CORES" \
        bash "$ROOT/tools/two_device_checks.sh" > "$J/two-device.log" 2>&1 || mrc=$?
    say "two_device_checks exited $mrc: $(tail -n 1 "$RES/out/two-device/summary.txt" 2>/dev/null)"
    [ "$rc" != 0 ] || rc=$mrc
fi
PHASE=done
exit "$rc"
