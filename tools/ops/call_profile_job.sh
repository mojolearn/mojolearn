#!/bin/bash
# tools/ops/call_profile_job.sh -- WHERE THE WALL TIME OF ONE SMALL PUBLIC CALL GOES, at function level, host side and
# GPU side, on OUR wheel alone (lane fb9-call-profile, 2026-11-17; function level without perf events: fb15-profiler,
# 2026-11-18; no prior wheel: Andrew's direction). Diagnostic tooling, not a board and not a measurement. For each row
# of a rows file (CP_ROWS_FILE, default tools/ops/call_profile_rows.tsv: race id, forced phase, name) it runs the
# row's race on the candidate venv once per pass, each a forced scored run whose extra calls sit inside the harness's
# profile window (tools/bench_board_small_rows.py range_open/range_close: calls 2..n, nothing of the data load, the fit
# before an inference, the warmup or the first call) and whose n fills the pass's seconds at the first call's time.
# The phase `fit+infer` (lane fx-harness, 2026-11-18) profiles in BOARD ORDER: the window holds pairs, each fit then the
# FIRST predict right after it, as the board scores a one-call infer row (n sized by the first pair's time; the report's
# H2D column gives the copies and MB per pair), since consecutive forced predicts cannot show a cost that only the first
# predict after a large fit pays:
#
#   tools/nvidia_central.sh submit measure --cap 60 --results box-out/callprof-<tag> tools/ops/call_profile_job.sh
#
#   time       no tool: wall per call (the window's clock over its calls)                       CP_TIME_SECONDS (3)
#   pyspy      py-spy record --native --threads, attached BY THE HARNESS to the race process for the window only (the
#              process allows its child to trace it: prctl PR_SET_PTRACER_ANY; no perf events, no CAP_SYS_PTRACE).
#              Main-thread stacks: top 40 self and inclusive frames, and the leaf-category shares of the host split
#                                                                                                 CP_PYSPY_SECONDS (10)
#   nsys       nsys profile -t cuda,nvtx --capture-range=nvtx (the window's NVTX range): per call launches, syncs,
#              allocations, H2D/D2H copies and bytes, CUDA API ms, GPU kernel ms (tools/ops/attribution_report.py nsys)
#                                                                                                 CP_NSYS_SECONDS (3)
#   shim       LD_PRELOAD tools/ops/call_profile_shim.c (built here): count, ns and bytes per call of the CUDA driver
#              entries (cuLaunchKernel(Ex), cuStreamSynchronize, cuStreamIsCapturing, cuMemcpy*, cuMemAlloc*,
#              cuMemFree*; caught through dlsym and cuGetProcAddress, the way the Mojo runtime finds them), libc
#              allocations, mmap/munmap, memcpy/memset of 64 KiB or more                          CP_SHIM_SECONDS (3)
#   callgrind  valgrind --tool=callgrind --instr-atstart=no --collect-atstart=no --trace-children=yes; the harness
#              turns instrumentation and collection on for the window only (tools/ops/call_profile_cg.c, ctypes):
#              exact instructions per call by function, self and inclusive, top 40, and by object (bindings, CPython,
#              Mojo runtime, CUDA driver, libc). At most CP_CALLGRIND_CALLS (300) calls        CP_CALLGRIND_SECONDS (10)
#   perf       perf record -e cpu-clock --call-graph dwarf, enabled through the harness's FIFO: only where the
#              container allows perf events (RunPod: perf_event_paranoid=4, refused; skipped)  CP_PERF_SECONDS (10)
#   strace     strace -f -ttt -T cut to the window by the harness's marker stats: syscalls per call (opt-in)
#                                                                                                 CP_STRACE_SECONDS (3)
# Report: CALL-PROFILE-nvidia.md and .json (tools/ops/call_profile.py report): per row wall ms per call, the nsys split
# (GPU kernel / CUDA API), and the host microseconds per call split into our binding code (by function), CPython, the
# Mojo runtime, the CUDA driver entries (shim, by entry), libc allocations and copies (shim) and the unexplained rest;
# the top frames (py-spy, callgrind, perf) and one line LARGEST HOST COSTS naming functions. Raw data kept:
# rows/<row>/pyspy/pyspy.raw, rows/<row>/callgrind/callgrind.out.*.gz, the nsys report and CSVs, the window files.
# job/symbols.txt says which .so files carry a symbol table.
#
# PARALLEL PLAN (HARD RULES 2026-11-05/07): the timed passes (time, pyspy, nsys, shim, perf, strace) are SERIAL BY
# NATURE: one race process on the one GPU and host at a time (a timed run must not share either). The callgrind passes
# count instructions, which do not depend on what else runs: they run after the timed passes, every row at once, up
# to min(rows, allocated cores, MemAvailable / CP_CALLGRIND_GB (6)). The tool installs run at once (pip beside one apt
# chain: two apt-gets at once fail on the dpkg lock, which is what lost strace on the 2026-11-17 run); the
# post-processing of every row and tool runs at once over the allocated cores. job/cores.log: busy cores of total.
#
# LONG ROWS (lane fx-profile-big-rows, 2026-11-20): a rows-file line with `ms=<stored board ms>` of at least 10 s
# (tools/ops/call_profile_rows_big.tsv: lle, isomap, dbscan, ... at 16-1,050 s a call) is profiled in ONE call per
# pass: MOJOLEARN_BENCH_PROFILE_ONE_CALL=1 (tools/bench_board_small_rows.py one_call: no warmup round, the window
# around the first scored call), py-spy at CP_PYSPY_LONG_RATE (20 Hz), no time pass (the shim pass's wall stands for
# it) and no callgrind. Each pass runs under the row's cap (`cap=`, else 2 x its stored seconds + 300 s): at the cap
# the job SIGINTs py-spy and nsys so they write what they collected, then kills the pass's whole process tree (the
# race drivers run in their own sessions) and writes run.capped; the report marks the row CAPPED (exit 8), never a
# silent timeout. The driver's own round ceiling is set above the cap (--race-round-seconds), so the job's cap fires
# first. The plan prints the estimate (stored ms x passes, and the sum of the caps) before any row runs:
#
#   tools/nvidia_central.sh submit measure --cap 210 --results box-out/callprof-big \
#       --env CP_ROWS_FILE=tools/ops/call_profile_rows_big.tsv --env CP_PASSES=pyspy,nsys,shim \
#       --env CP_REPORT_TOP=15 --env CP_MANIFEST=<results dir>/job/wheel-manifest-candidate.json \
#       tools/ops/call_profile_job.sh
#
# Pod minutes: `python3 tools/ops/call_profile.py estimate [--rows-file F]`, printed at the plan phase (target < 30).
# Knobs: CP_ROWS_FILE, CP_ROWS (comma list of row names), CP_PASSES (default time,pyspy,nsys,shim,callgrind), CP_PARTS
# (default tools,plan,install,run,post,report; the report reads whatever the results dir holds), CP_<PASS>_SECONDS,
# CP_PYSPY_RATE (250), CP_PYSPY_LONG_RATE (20, long rows), CP_CAP_GRACE_S (600: what py-spy and nsys get to write
# after a cap), CP_REPORT_TOP (40 frames per top table), CP_KERNEL_REGEX (`.`: the GPU kernels nsys names), CP_PERF_FREQ (999), CP_CALLGRIND_CALLS (300), CP_CALLGRIND_GB (6), CP_CALLGRIND_TIMEOUT (1800 s),
# CP_MANIFEST (the candidate's wheel-manifest json; default: the newest sibling results dir's
# job/wheel-manifest-candidate.json: used to install the venv when it is missing), PC_CACHE_ROOT, PC_DATA_ROOT,
# PC_GPU_TOKEN (L40S), PC_ALLOW_HW_DIFF. EXIT: 0 every row has its wall, its nsys counts and at least one
# function-level source (py-spy, callgrind, the shim or perf); 6 a row lacks one; 7 a row lacks a wall time; 8 a row
# hit its cap (CAPPED, reported with what was collected); 2 refused
# to start, 3 wrong box, 4 no venv, 5 no plan, 124 terminated. Tests: tools/test_call_profile.py.
set -uo pipefail
JOB_NAME=call-profile
# shellcheck source=tools/ops/box_job_lib.sh
source "$(dirname "$0")/box_job_lib.sh"

# >>> cp-preflight (tools/test_job_preflight.py runs this block)
# job-preflight: `call_profile_job.sh --preflight` checks this job's inputs at SUBMIT time, reading only (run on the
# box by tools/ops/job_preflight.sh from tools/nvidia_central.sh submit; lane fx-pod-job, 2026-11-18). It covers the
# three ways the job failed on 2026-11-18, each found 90 s into a run after hours in the queue: (1) the venv holds
# another wheel than the manifest names and no install part runs (251f16771a); (2) no wheel manifest, so no row can be
# timed (bf3af46910); (3) the (re)install cannot run: no staged wheels in CP_WHEEL_DIR or no source commit in the
# manifest (eed2394df2). The job keeps its own run-time checks of the same conditions (the box can change after submit).
cp_preflight() {   # results dir, tree: PREFLIGHT lines; 0 when the job would start, 4 otherwise
    local res=$1 tree=$2 man ver have wdir src rows nwhl fails=0
    local parts=",${CP_PARTS:-tools,plan,install,run,post,report},"
    local apy="${PC_CACHE_ROOT:-/root/pc-cache}/venv-candidate/bin/python" py=${CP_PYTHON:-python3}
    rows=${CP_ROWS_FILE:-$tree/tools/ops/call_profile_rows.tsv}
    [ -e "$rows" ] || [ "${rows#/}" != "$rows" ] || rows=$tree/$rows      # a relative CP_ROWS_FILE names a tree file
    if [ -s "$rows" ]; then echo "PREFLIGHT ok rows file $rows"
    else echo "PREFLIGHT FAIL no rows file $rows (CP_ROWS_FILE)"; fails=$(( fails + 1 )); fi
    man=${CP_MANIFEST:-}
    # shellcheck disable=SC2012
    [ -z "$man" ] && man=$(ls -t "$(dirname "$res")"/*/job/wheel-manifest-candidate.json 2> /dev/null | head -n 1)
    if [ ! -f "$man" ]; then
        if [[ "$parts" == *",run,"* ]]; then
            echo "PREFLIGHT FAIL no wheel manifest: pass CP_MANIFEST=<a results dir>/job/wheel-manifest-candidate.json (searched $(dirname "$res")/*/job/)"
            return 4
        fi
        echo "PREFLIGHT ok no manifest needed (no run part)"
        return $(( fails > 0 ? 4 : 0 ))
    fi
    ver=$($py -c 'import json, sys; print(json.load(open(sys.argv[1])).get("version", ""))' "$man" 2> /dev/null)
    have=$([ -x "$apy" ] && "$apy" -c 'import importlib.metadata as m
for n in ("mojolearn-core", "mojolearn"):
    try:
        print(m.version(n)); break
    except m.PackageNotFoundError:
        pass' 2> /dev/null)
    echo "PREFLIGHT ok manifest $man (version ${ver:-none}); venv ${have:-none}"
    if [ -x "$apy" ] && { [ -z "$ver" ] || [ "$have" = "$ver" ]; }; then
        echo "PREFLIGHT ok the venv holds the manifest's wheel"
    elif [[ "$parts" == *",install,"* ]]; then
        # the job (re)installs: the manifest's version and source commit and the staged wheels must be there
        wdir=${CP_WHEEL_DIR:-/root/pc-in/${ver##*+}/nvidia}
        nwhl=$(ls "$wdir"/*.whl 2> /dev/null | grep -c .)
        src=$($py -c 'import json, sys; print(json.load(open(sys.argv[1])).get("source_commit", ""))' "$man" 2> /dev/null)
        if [ -n "$ver" ] && [ "${nwhl:-0}" -gt 0 ] && [ -n "$src" ]; then
            echo "PREFLIGHT ok the install has $nwhl wheel(s) in $wdir and source ${src:0:12}"
        else
            echo "PREFLIGHT FAIL the install cannot run: ${nwhl:-0} wheel(s) in $wdir (CP_WHEEL_DIR), version '${ver}', source commit '${src}' in $man"
            fails=$(( fails + 1 ))
        fi
    else
        echo "PREFLIGHT FAIL the venv holds ${have:-nothing}, the manifest names $ver, and CP_PARTS has no install: nothing would be timed"
        fails=$(( fails + 1 ))
    fi
    return $(( fails > 0 ? 4 : 0 ))
}
# <<< cp-preflight
if [ "${1:-}" = --preflight ]; then
    cp_preflight "${MOJOLEARN_RESULTS_DIR:-}" "$(cd "$(dirname "$0")/../.." && pwd)"
    exit $?
fi

if [ "${MOJOLEARN_BOX_PERSIST:-}" != 1 ] && [ "${MOJOLEARN_ALLOW_UNPERSISTED:-}" != 1 ]; then
    echo "call-profile: runs under tools/box_persist_run.sh (tools/nvidia_central.sh submit); MOJOLEARN_ALLOW_UNPERSISTED=1 overrides" >&2
    exit 2
fi
V=$(detect_vendor) || { echo "call-profile: no nvidia-smi" >&2; exit 2; }
[ "$V" = nvidia ] || { echo "call-profile: nsys and the CUDA split are NVIDIA only (this box: $V)" >&2; exit 3; }
TREE="$(cd "$(dirname "$0")/../.." && pwd)"
RES="${MOJOLEARN_RESULTS_DIR:?call_profile_job.sh needs MOJOLEARN_RESULTS_DIR (set by tools/box_persist_run.sh)}"
mkdir -p "$RES" && RES="$(cd "$RES" && pwd)"
export MOJOLEARN_RESULTS_DIR="$RES"
J="$RES/job"
mkdir -p "$J" "$RES/plan" "$RES/logs" "$RES/rows"
exec >> "$J/job.log" 2>&1
export PATH="/root/.pixi/bin:/usr/local/cuda/bin:$PATH"
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS
export MOJOLEARN_NUMERIC_MODE=identical
PY=${CP_PYTHON:-python3}
cd "$TREE" || exit 2
DATA_ROOT=${PC_DATA_ROOT:-/root/datasets/gbm-bench}
CACHE_ROOT=${PC_CACHE_ROOT:-/root/pc-cache}
TOKEN=${PC_GPU_TOKEN:-L40S}
RECIPES="$TREE/bench/results/paper_compare/recipes-$V.json"
ROWS_FILE=${CP_ROWS_FILE:-$TREE/tools/ops/call_profile_rows.tsv}
[ "${ROWS_FILE#/}" != "$ROWS_FILE" ] || ROWS_FILE=$TREE/$ROWS_FILE     # a relative CP_ROWS_FILE names a tree file
IFS=',' read -r -a PASSES <<< "${CP_PASSES:-time,pyspy,nsys,shim,callgrind}"
PARTS=",${CP_PARTS:-tools,plan,install,run,post,report},"
part() { [[ "$PARTS" == *",$1,"* ]]; }
ROWS_WANT=",${CP_ROWS:-},"
row_wanted() { [ "$ROWS_WANT" = ",," ] || [[ "$ROWS_WANT" == *",$1,"* ]]; }
declare -A SECS=([time]=${CP_TIME_SECONDS:-3} [pyspy]=${CP_PYSPY_SECONDS:-10} [perf]=${CP_PERF_SECONDS:-10}
                 [nsys]=${CP_NSYS_SECONDS:-3} [shim]=${CP_SHIM_SECONDS:-3} [callgrind]=${CP_CALLGRIND_SECONDS:-10}
                 [strace]=${CP_STRACE_SECONDS:-3})
has_pass() { [[ " ${PASSES[*]} " == *" $1 "* ]]; }
FREQ=${CP_PERF_FREQ:-999}
RATE=${CP_PYSPY_RATE:-250}
LONG_RATE=${CP_PYSPY_LONG_RATE:-20}
CAP_GRACE=${CP_CAP_GRACE_S:-600}
CG_CALLS=${CP_CALLGRIND_CALLS:-300}
RANGE=attr-calls     # the NVTX range name: tools/ops/attribution_report.py nsys reads the window under this name
VENV="$CACHE_ROOT/venv-candidate"
APY="$VENV/bin/python"
WORST=0
REPORT_RC=
PHASE=start
P_T0=$(date +%s)

SAMPLER=
if [ -r /proc/stat ]; then cores_sampler & SAMPLER=$!; fi
finish() {
    local code=$?
    trap - EXIT
    [ -n "$SAMPLER" ] && kill "$SAMPLER" 2> /dev/null
    if [ -n "$REPORT_RC" ] && [ "$code" -eq 0 ]; then code=$REPORT_RC
    elif [ "$WORST" -gt "$code" ]; then code=$WORST; fi
    printf '{"job":"call-profile","vendor":"%s","rows":"%s","passes":"%s","parts":"%s","host_id":"%s","tree_head":"%s","phase":"%s","exit":%d,"job_script_sha256":"%s"}\n' \
        "$V" "${CP_ROWS:-all}" "${PASSES[*]}" "$PARTS" "${HOST_ID:-}" "$(git -C "$TREE" rev-parse HEAD 2>/dev/null)" "$PHASE" "$code" \
        "$(sha256sum "$TREE/tools/ops/call_profile_job.sh" | cut -d' ' -f1)" > "$J/status.json"
    say "exit $code (last phase $PHASE)"
    exit "$code"
}
trap finish EXIT
trap 'exit 124' TERM INT

say "call-profile $V rows ${CP_ROWS:-all} passes ${PASSES[*]} parts $PARTS at $(git rev-parse HEAD 2>/dev/null) results $RES"
{ uname -a; echo "nproc $(nproc)"; free -g 2> /dev/null || true; df -h /root 2> /dev/null || true; } > "$J/box.txt" 2>&1
NCORES=$(alloc_cores)
say "BUSY at start: nproc $(nproc), allocation $NCORES cores; the race processes below are serial by nature (one on the GPU)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$J/gpu.txt" 2>&1
NGPU=$(grep -c . "$J/gpu.txt" 2> /dev/null || echo 0)
if gpu_other; then
    say "HW-DIFF: the boards ran on $TOKEN; this box: $(head -c 300 "$J/gpu.txt" | tr '\n' ' ')"
    [ "${PC_ALLOW_HW_DIFF:-0}" = 1 ] || { say "refused: NVIDIA is profiled on an $TOKEN only; PC_ALLOW_HW_DIFF=1 overrides"; exit 3; }
fi
prune_persist_staging
host_probe_phase
arith_gate   # tools/ops/box_job_lib.sh: identical-path arithmetic on this GPU, before any timing

# ---- tools (CPU, all at once): py-spy beside ONE apt chain (nsight, then valgrind/gcc/strace/perf) ----------------
find_perf() {   # the real perf binary (Ubuntu's /usr/bin/perf is a wrapper that refuses a kernel it has no tools for)
    local c
    for c in $(ls -d /usr/lib/linux-tools/*/perf /usr/lib/linux-tools-*/perf 2> /dev/null | sort -V -r) "$(command -v perf 2> /dev/null)"; do
        [ -n "$c" ] && [ -x "$c" ] && "$c" --version > /dev/null 2>&1 && { echo "$c"; return 0; }
    done
    return 1
}
cg_header() {   # valgrind's client-request header (apt's valgrind package ships it; a pixi/conda valgrind beside its bin)
    local v
    for v in /usr/include/valgrind "$(dirname "$(dirname "$(command -v valgrind 2> /dev/null || echo /x/x)")")/include/valgrind"; do
        [ -f "$v/callgrind.h" ] && { dirname "$v"; return 0; }
    done
    return 1
}
install_apt_tools() {   # whichever is missing, one apt transaction; waits for any other apt on the pod (dpkg lock)
    local log="$RES/logs/apt-tools.log" want=() apt=(apt-get -o DPkg::Lock::Timeout=600 -y -qq)
    if has_pass callgrind; then
        command -v valgrind > /dev/null && cg_header > /dev/null || want+=(valgrind)
    fi
    has_pass perf && { find_perf > /dev/null || want+=(linux-tools-common linux-tools-generic); }
    has_pass strace && { command -v strace > /dev/null || want+=(strace); }
    command -v cc > /dev/null || command -v gcc > /dev/null || want+=(gcc libc6-dev)
    command -v readelf > /dev/null || want+=(binutils)
    [ ${#want[@]} -eq 0 ] && return 0
    DEBIAN_FRONTEND=noninteractive "${apt[@]}" update >> "$log" 2>&1
    DEBIAN_FRONTEND=noninteractive "${apt[@]}" install --no-install-recommends "${want[@]}" >> "$log" 2>&1
    echo "apt ${want[*]} rc=$?" >> "$J/tools-how.txt"
}
if part tools; then
    phase tools
    : > "$J/tools-how.txt"
    pids=()
    { install_nsight; install_apt_tools; } & pids+=($!)      # serial: both use apt (one dpkg lock)
    "$PY" -m pip install -q --disable-pip-version-check py-spy > "$RES/logs/pip-pyspy.log" 2>&1 & pids+=($!)
    "$APY" -m pip install -q --disable-pip-version-check --target "$CACHE_ROOT/attr-nvtx-candidate" nvtx \
        > "$RES/logs/pip-nvtx.log" 2>&1 & pids+=($!)
    for p in "${pids[@]}"; do wait "$p"; done
    phase_end 0
fi
NSYS=$(find_tool nsys) || NSYS=
PERF=$(find_perf) || PERF=
PYSPY=$(find_tool py-spy) || PYSPY=
STRACE=$(command -v strace) || STRACE=
VALGRIND=$(command -v valgrind) || VALGRIND=
CC=$(command -v cc || command -v gcc) || CC=
build_so() {   # out, source, extra cc args...: (re)build a shim on the pod; empty echo when it does not build
    local out=$1 src=$2; shift 2
    if [ -n "$CC" ] && { [ ! -f "$out" ] || [ "$src" -nt "$out" ]; }; then
        "$CC" -O2 -fno-builtin -shared -fPIC -o "$out" "$src" "$@" > "$RES/logs/cc-$(basename "$out" .so).log" 2>&1 \
            || { say "$(basename "$src") did not build: $(head -n 3 "$RES/logs/cc-$(basename "$out" .so).log" | tr '\n' ' ')"; rm -f "$out"; }
    fi
    [ -f "$out" ] && echo "$out"
}
SHIM_SO=$(build_so "$CACHE_ROOT/callprof_shim.so" "$TREE/tools/ops/call_profile_shim.c" -ldl)
CG_SO=
CG_INC=$(cg_header) && CG_SO=$(build_so "$CACHE_ROOT/callprof_cg.so" "$TREE/tools/ops/call_profile_cg.c" -I"$CG_INC")
PERF_OK=
if [ -n "$PERF" ]; then      # can this container open perf events at all (seccomp, perf_event_paranoid)?
    "$PERF" stat -e cpu-clock -- true > "$RES/logs/perf-probe.log" 2>&1 && PERF_OK=1
fi
{
    echo "perf ${PERF:-absent} $([ -n "$PERF" ] && "$PERF" --version 2>&1 | head -n 1) events $([ -n "$PERF_OK" ] && echo usable || echo "REFUSED: $(tail -n 2 "$RES/logs/perf-probe.log" 2> /dev/null | tr '\n' ' ')") perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2> /dev/null || echo '?')"
    echo "py-spy ${PYSPY:-absent} $([ -n "$PYSPY" ] && "$PYSPY" --version 2>&1 | head -n 1) ptrace_scope=$(cat /proc/sys/kernel/yama/ptrace_scope 2> /dev/null || echo 'no yama')"
    echo "valgrind ${VALGRIND:-absent} $([ -n "$VALGRIND" ] && "$VALGRIND" --version 2>&1 | head -n 1) callgrind window ${CG_SO:-absent (no valgrind/callgrind.h or no C compiler: logs/apt-tools.log, logs/cc-callprof_cg.log)}"
    echo "timing shim ${SHIM_SO:-absent (no C compiler or build failed: logs/cc-callprof_shim.log)}"
    echo "nsys ${NSYS:-absent} $([ -n "$NSYS" ] && "$NSYS" --version 2>&1 | head -n 1)"
    echo "strace ${STRACE:-absent} $([ -n "$STRACE" ] && strace -V 2>&1 | head -n 1)"
    echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
} > "$J/tools.txt" 2>&1
say "tools: $(tr '\n' ';' < "$J/tools.txt" | head -c 900)"

# ---- plan (CPU): the rows, their races' board arguments, the estimate --------------------------------------------
if part plan || [ ! -s "$RES/plan/board-args.txt" ] || [ ! -s "$RES/plan/runs.tsv" ]; then
    phase plan
    cp "$ROWS_FILE" "$RES/plan/rows.tsv" 2>> "$RES/logs/plan.log" || { say "no rows file $ROWS_FILE"; exit 5; }
    $PY tools/ops/call_profile.py rows --rows-file "$RES/plan/rows.tsv" --out-dir "$RES/plan" > "$RES/logs/plan.log" 2>&1 \
        || { say "rows file refused: $(tail -n 1 "$RES/logs/plan.log")"; exit 5; }
    $PY tools/ops/targeted_rows.py plan --rows "$RES/plan/ids.txt" --vendor "$V" --planner tools/bench_board.py \
        --recipes "$RECIPES" --out-dir "$RES/plan" --one-arm >> "$RES/logs/plan.log" 2>&1
    rc=$?
    while read -r rid; do
        grep -qxF "$rid" "$RES/plan/race-ids-full.txt" 2> /dev/null || say "NOTE: $rid is not planned by the board now (plan/not-planned.json)"
    done < "$RES/plan/ids.txt"
    ARGS_SECS=(); for p in "${PASSES[@]}"; do ARGS_SECS+=(--seconds "$p=${SECS[$p]}"); done
    $PY tools/ops/call_profile.py estimate --rows-file "$RES/plan/rows.tsv" --passes "$(IFS=,; echo "${PASSES[*]}")" \
        "${ARGS_SECS[@]}" 2>&1 | while read -r ln; do say "$ln"; done
    $PY tools/measure_owed.py data-keys --paper-recipes "$RES/plan/races.json" --planner tools/bench_board.py \
        --vendor "$V" --data-root "$DATA_ROOT" > "$RES/plan/data-keys.tsv" 2> "$RES/logs/data-keys.log"
    MISSING=$(while IFS=$'\t' read -r key path; do [ -f "$path" ] || printf '%s ' "$key"; done < "$RES/plan/data-keys.tsv")
    [ -n "$MISSING" ] && { say "missing datasets under $DATA_ROOT: $MISSING(stage them: tools/ops/measure_candidate.py stage --rows)"; rc=5; }
    phase_end $rc
fi
[ -s "$RES/plan/board-args.txt" ] && [ -s "$RES/plan/runs.tsv" ] || { say "no plan in $RES/plan (logs/plan.log)"; exit 5; }
mapfile -t BARGS < "$RES/plan/board-args.txt"
COMMON=("${BARGS[@]}" --data-root "$DATA_ROOT" --rounds 1 --evidence-local-only --fixtures-venv "$CACHE_ROOT/venv-nf")

# ---- install (CPU): the candidate venv, when the pod has none or it holds another wheel ---------------------------------------------
MAN=${CP_MANIFEST:-}
if [ -z "$MAN" ]; then
    # shellcheck disable=SC2012
    MAN=$(ls -t "$(dirname "$RES")"/*/job/wheel-manifest-candidate.json 2> /dev/null | head -n 1)
fi
# A venv that holds another wheel than the manifest's is reinstalled: on 2026-11-18 an A/B job left its base wheel in
# the shared venv, the newest manifest named the later candidate, and every row was refused (no wall time, exit 7).
installed_version() {
    [ -x "$APY" ] && "$APY" -c 'import importlib.metadata as m
for n in ("mojolearn-core", "mojolearn"):
    try:
        print(m.version(n)); break
    except m.PackageNotFoundError:
        pass' 2> /dev/null
}
# No manifest, no timing: bench_board refuses to time a wheel it cannot tie to release files, so every row would end
# "no wall time" (the 2026-11-18 rerun, whose lane tree had no sibling results dir). Stop here and say what to pass.
if part run && [ ! -f "$MAN" ]; then
    say "no wheel manifest: pass CP_MANIFEST=<a results dir>/job/wheel-manifest-candidate.json (searched $(dirname "$RES")/*/job/)"; exit 4
fi
MAN_VERSION=$([ -f "$MAN" ] && $PY -c 'import json, sys; print(json.load(open(sys.argv[1])).get("version", ""))' "$MAN" 2> /dev/null)
if part install && { [ ! -x "$APY" ] || { [ -n "$MAN_VERSION" ] && [ "$(installed_version)" != "$MAN_VERSION" ]; }; }; then
    phase install
    [ -f "$MAN" ] || { say "no venv at $VENV and no wheel manifest (CP_MANIFEST)"; exit 4; }
    # The same install the measurement job makes (paper_compare_job.sh add_arm candidate): the manifest's version and
    # source commit plus the staged wheel files. --wheel-manifest is the file the install WRITES, so it goes to this
    # job's own dir. (Before 2026-11-18 this call passed neither version nor wheels and could never install.)
    WDIR=${CP_WHEEL_DIR:-/root/pc-in/$(echo "$MAN_VERSION" | sed 's/.*+//')/nvidia}
    SRC=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1])).get("source_commit", ""))' "$MAN" 2> /dev/null)
    CW=(); for w in "$WDIR"/*.whl; do [ -f "$w" ] && CW+=(--candidate-wheel "$w"); done
    [ ${#CW[@]} -gt 0 ] && [ -n "$SRC" ] || { say "install: no wheels in $WDIR (CP_WHEEL_DIR) or no source_commit in $MAN"; exit 4; }
    $PY tools/bench_board.py "${COMMON[@]}" --mojolearn-version "$MAN_VERSION" --candidate-source-commit "$SRC" "${CW[@]}" \
        --ours-install-only --install-only --venv "$VENV" --cache "$CACHE_ROOT/i-candidate" \
        --wheel-manifest "$J/wheel-manifest-candidate.json" --race-ids @"$RES/plan/race-ids-full.txt" --out "$RES/install-candidate" \
        > "$RES/logs/install-candidate.log" 2>&1
    irc=$?; phase_end $irc
    [ $irc = 0 ] || { say "install failed: $(tail -n 1 "$RES/logs/install-candidate.log" | cut -c1-200)"; exit 4; }
    MAN="$J/wheel-manifest-candidate.json"
fi
[ -x "$APY" ] || { say "no venv for the candidate at $VENV"; exit 4; }
WHEEL=$(installed_version)
[ -z "$MAN_VERSION" ] || [ "$WHEEL" = "$MAN_VERSION" ] || { say "venv holds $WHEEL, the manifest names $MAN_VERSION: nothing would be timed"; exit 4; }
say "candidate $WHEEL venv $VENV manifest ${MAN:-none}"
# which shared objects of the installed wheel carry a symbol table (perf names their functions) and an unwind table
SITE=$("$APY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2> /dev/null)
{
    echo "# so  symtab(static symbols)  dynsym-defined  eh_frame  debuginfo  build-id"
    find "$SITE"/mojolearn* -name '*.so*' -type f 2> /dev/null | sort | while read -r so; do
        sec=$(readelf -S -W "$so" 2> /dev/null)
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${so#"$SITE"/}" \
            "$(grep -q ' .symtab ' <<< "$sec" && echo yes || echo STRIPPED)" \
            "$(nm -D --defined-only "$so" 2> /dev/null | grep -c .)" \
            "$(grep -q ' .eh_frame ' <<< "$sec" && echo yes || echo no)" \
            "$(grep -q ' .debug_info ' <<< "$sec" && echo yes || echo no)" \
            "$(readelf -n "$so" 2> /dev/null | awk '/Build ID/ {print $3}')"
    done
} > "$J/symbols.txt" 2>&1
say "symbols: $(awk -F'\t' 'NR>1 {n++; if ($2=="STRIPPED") s++} END {printf "%d shared objects, %d stripped (job/symbols.txt)", n, s}' "$J/symbols.txt")"

# ---- run (GPU, serial; then the callgrind passes of every row at once) ---------------------------------------------
# one pass of one row: the forced scored run of the row's phase with its window, under the pass's tool. A pass whose
# tool this pod cannot run writes run.skipped (not an error: the report needs one function-level source per row).
tree_pids() {   # pid: the pid and every descendant (by parent pid: the race drivers and py-spy run in own sessions)
    local all kids k
    all=$(ps -e -o pid=,ppid= 2> /dev/null)
    kids=$1
    local out=$1 next
    while [ -n "$kids" ]; do
        next=
        for k in $kids; do
            next="$next $(awk -v p="$k" '$2 == p {print $1}' <<< "$all")"
        done
        kids=$(echo $next)
        [ -n "$kids" ] && out="$out $kids"
    done
    echo "$out"
}
cap_stop() {   # pid, pass dir, pass, cap s: stop a pass at its cap and keep what its tools collected
    local pid=$1 d=$2 pass=$3 cap=$4 pids p n
    pids=$(tree_pids "$pid")
    # py-spy writes its raw file on SIGINT; nsys stops its capture, ends the app and writes its report on SIGINT
    for p in $pids; do
        [ "$(cat "/proc/$p/comm" 2> /dev/null)" = py-spy ] && kill -INT "$p" 2> /dev/null
    done
    [ "$pass" = nsys ] && kill -INT "$pid" 2> /dev/null
    n=0
    while [ $n -lt "$CAP_GRACE" ]; do
        local live=0
        for p in $pids; do
            case "$(cat "/proc/$p/comm" 2> /dev/null)" in py-spy|nsys) kill -0 "$p" 2> /dev/null && live=1 ;; esac
        done
        [ "$pass" = nsys ] && kill -0 "$pid" 2> /dev/null && live=1
        [ $live = 0 ] && break
        sleep 5; n=$(( n + 5 ))
    done
    pids="$pids $(tree_pids "$pid")"
    for p in $pids; do kill -TERM "$p" 2> /dev/null; done
    sleep 20
    for p in $pids; do kill -KILL "$p" 2> /dev/null; done
    printf '{"cap_s": %d, "elapsed_s": %d, "t_capped_unix": %s, "grace_s": %d, "killed": "%s"}\n' "$cap" \
        "$(( SECONDS - ${PASS_T0:-$SECONDS} ))" "$(date +%s)" "$n" "$(echo $pids | tr ' ' ',')" > "$d/run.capped"
}
run_pass() {   # row, race id, phase, pass, cap s (0: none), long (1: one call per pass)
    local row=$1 rid=$2 ph=$3 pass=$4 cap=${5:-0} long=${6:-0} d="$RES/rows/$1/$4" t0=$SECONDS rc err cache="$CACHE_ROOT/c1"
    rm -rf "$d"; mkdir -p "$d"
    echo "$rid" > "$d/ids"
    local envs=(MOJOLEARN_BENCH_FORCE_CALLS=20 MOJOLEARN_BENCH_FORCE_CALLS_PHASE="$ph"
                MOJOLEARN_BENCH_PROFILE_WINDOW="$d/window.jsonl" MOJOLEARN_BENCH_PROFILE_SECONDS="${SECS[$pass]}")
    local rate=$RATE extra=()
    if [ "$long" = 1 ]; then
        # one call per pass: no warmup, the window around the first scored call (bench_board_small_rows one_call)
        envs=(MOJOLEARN_BENCH_FORCE_CALLS=1 MOJOLEARN_BENCH_FORCE_CALLS_PHASE="$ph" MOJOLEARN_BENCH_PROFILE_ONE_CALL=1
              MOJOLEARN_BENCH_PROFILE_WINDOW="$d/window.jsonl")
        rate=$LONG_RATE
    fi
    if [ "$cap" -gt 0 ]; then
        # the driver's own round ceiling sits above the job's cap, so the cap (and its CAPPED record) fires first
        printf '{"round_seconds": {"%s": %d}}\n' "$rid" $(( cap + CAP_GRACE + 600 )) > "$d/round-seconds.json"
        extra=(--race-round-seconds "$d/round-seconds.json")
    fi
    local pre=()
    skip() { echo "$1" > "$d/run.skipped"; say "run $row $pass SKIPPED: $1"; }
    case $pass in
        time)
            [ "$long" = 1 ] && { skip "long row: one call per tool; the shim pass's wall stands for the time pass"; return 0; } ;;
        pyspy)     # the harness attaches py-spy to the race process at the window's open and stops it at the close
            [ -n "$PYSPY" ] || { skip "py-spy absent (logs/pip-pyspy.log)"; return 0; }
            envs+=(MOJOLEARN_BENCH_PYSPY="$PYSPY,$d/pyspy.raw,$rate")
            echo "$rate" > "$d/rate" ;;
        perf)      # perf outside, enabled and disabled by the harness through the FIFO: only the window's samples
            [ -n "$PERF_OK" ] || { skip "perf events refused here (job/tools.txt)"; return 0; }
            mkfifo "$d/ctl" "$d/ack"
            envs+=(MOJOLEARN_BENCH_PERF_CTL="$d/ctl,$d/ack" PYTHONPERFSUPPORT=1)
            pre=("$PERF" record -e cpu-clock -F "$FREQ" --call-graph dwarf,8192 -D -1 --control "fifo:$d/ctl,$d/ack"
                 -o "$d/perf.data" --) ;;
        nsys)
            [ -n "$NSYS" ] || { echo "nsys absent (logs/apt-nsight.log)" > "$d/run.error"; return 6; }
            envs+=(MOJOLEARN_BENCH_NVTX_RANGE="$RANGE" MOJOLEARN_BENCH_NVTX_PATH="$CACHE_ROOT/attr-nvtx-candidate"
                   NSYS_NVTX_PROFILER_REGISTER_ONLY=0)
            pre=(env NSYS_NVTX_PROFILER_REGISTER_ONLY=0 "$NSYS" profile -t cuda,nvtx --sample=none --cpuctxsw=none
                 --capture-range=nvtx --nvtx-capture="$RANGE" --capture-range-end=stop --force-overwrite=true -o "$d/nsys") ;;
        shim)
            [ -n "$SHIM_SO" ] || { skip "timing shim did not build (logs/cc-callprof_shim.log)"; return 0; }
            envs+=(MOJOLEARN_BENCH_CALLPROF_SHIM="$SHIM_SO" LD_PRELOAD="$SHIM_SO") ;;
        callgrind) # every process of the race under valgrind's core; instrumented only inside the window
            [ "$long" = 1 ] && { skip "long row: no callgrind (one call of minutes runs about 50x longer under it)"; return 0; }
            [ -n "$VALGRIND" ] && [ -n "$CG_SO" ] || { skip "valgrind or its callgrind window absent (job/tools.txt)"; return 0; }
            cache="$CACHE_ROOT/c-cg/$row"     # rows run at once: no shared cache directory
            envs+=(MOJOLEARN_BENCH_CALLGRIND="$CG_SO" MOJOLEARN_BENCH_PROFILE_MAX_CALLS="$CG_CALLS")
            pre=(timeout -k 30 "${CP_CALLGRIND_TIMEOUT:-1800}" "$VALGRIND" --tool=callgrind --instr-atstart=no
                 --collect-atstart=no --trace-children=yes --trace-children-skip='*/nvidia-smi,*/git,*/uname,*/lscpu,*/ldconfig*'
                 --trace-children-skip-by-arg=pip --callgrind-out-file="$d/callgrind.out.%p" --log-file="$d/valgrind.%p.log") ;;
        strace)
            [ -n "$STRACE" ] || { skip "strace absent (logs/apt-tools.log)"; return 0; }
            pre=("$STRACE" -f -ttt -T -qq -o "$d/strace.txt") ;;
        *) echo "unknown pass $pass (time, pyspy, perf, nsys, shim, callgrind, strace)" > "$d/run.error"; return 5 ;;
    esac
    local which=(--mojolearn-version "$WHEEL")
    [ -f "$MAN" ] && which=(--wheel-manifest "$MAN")
    local cmd=(${pre[@]+"${pre[@]}"} env "${envs[@]}" "$PY" tools/bench_board.py "${COMMON[@]}" --python-env "$APY"
               --skip-install "${which[@]}" --cache "$cache" ${extra[@]+"${extra[@]}"}
               --race-ids @"$d/ids" --no-smoke-gate
               --guarded-smoke-override-reason "call profile job: diagnostic run of a board row this pod's earlier job smoked"
               --out "$d/board")
    if [ "$cap" -gt 0 ]; then
        # under the row's cap, polled every 5 s, in its own session with SIGINT back at its default: bash starts a
        # background job with SIGINT ignored, which every child inherits, and py-spy and nsys write their output on
        # SIGINT (cap_stop)
        PASS_T0=$SECONDS
        "$PY" -c 'import os, signal, sys
signal.signal(signal.SIGINT, signal.SIG_DFL)
os.setsid()
os.execvp(sys.argv[1], sys.argv[1:])' "${cmd[@]}" > "$d/run.log" 2>&1 < /dev/null &
        local pid=$!
        while kill -0 "$pid" 2> /dev/null && [ $(( SECONDS - PASS_T0 )) -lt "$cap" ]; do sleep 5; done
        if kill -0 "$pid" 2> /dev/null; then
            say "run $row $pass CAPPED at $cap s: stopping its tools so they write what they collected"
            cap_stop "$pid" "$d" "$pass" "$cap"
        fi
        wait "$pid"; rc=$?
        [ -f "$d/run.capped" ] && rc=8
    else
        "${cmd[@]}" > "$d/run.log" 2>&1 < /dev/null
        rc=$?
    fi
    rm -rf "$d/board/work" "$d/board/evidence" "$d/ctl" "$d/ack"
    if [ -f "$d/run.capped" ]; then
        echo "CAPPED at $cap s (run.capped): the pass was stopped; its tools' output so far is kept" > "$d/run.error"
    elif [ $rc -ne 0 ]; then
        err=$($PY tools/ops/attribution_report.py error --out "$d/board" --log "$d/run.log" 2>&1 | head -n 1)
        [ "$pass" = callgrind ] && err="valgrind: $(grep -h -m 3 -E '^==[0-9]+== [A-Z]|valgrind:|Fatal|unhandled' "$d"/valgrind.*.log 2> /dev/null | tr '\n' ' ' | head -c 300) $err"
        echo "$err" > "$d/run.error"
    elif [ ! -s "$d/window.jsonl" ]; then
        echo "no profile window: the race opened none (the phase name, or a driver without range_open)" > "$d/run.error"; rc=6
    fi
    say "run $row $pass rc=$rc $(( SECONDS - t0 )) s window: $($PY tools/ops/call_profile.py window --window "$d/window.jsonl" | head -c 200)$([ -s "$d/run.error" ] && echo " ERROR $(head -c 300 "$d/run.error")")"
    return $rc
}
cg_workers() {   # rows at once for the callgrind passes: cores, then memory (CP_CALLGRIND_GB per valgrind'd race)
    local n=$1 avail_gb per=${CP_CALLGRIND_GB:-6}
    avail_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo 2> /dev/null || echo 0)
    [ "$NCORES" -lt "$n" ] && n=$NCORES
    [ "$avail_gb" -gt 0 ] && [ $(( avail_gb / per )) -lt "$n" ] && n=$(( avail_gb / per ))
    [ "$n" -lt 1 ] && n=1
    echo "$n"
}
if part run; then
    phase run
    rc=0
    CG_ROWS=()
    while IFS=$'\t' read -r row rid ph cap long; do
        row_wanted "$row" || continue
        for pass in "${PASSES[@]}"; do
            [ "$pass" = callgrind ] && [ "${long:-0}" != 1 ] && { CG_ROWS+=("$row"$'\t'"$rid"$'\t'"$ph"); continue; }
            run_pass "$row" "$rid" "$ph" "$pass" "${cap:-0}" "${long:-0}"; r=$?; [ $r -gt $rc ] && rc=$r
        done
    done < "$RES/plan/runs.tsv"
    if [ ${#CG_ROWS[@]} -gt 0 ]; then
        W=$(cg_workers ${#CG_ROWS[@]})
        say "callgrind: ${#CG_ROWS[@]} rows, $W at once (allocation $NCORES cores; instruction counts do not depend on what else runs)"
        cg_pids=()      # (the cores sampler is a background job too: count only these)
        for line in "${CG_ROWS[@]}"; do
            IFS=$'\t' read -r row rid ph <<< "$line"
            while :; do
                live=0
                for p in "${cg_pids[@]}"; do kill -0 "$p" 2> /dev/null && live=$(( live + 1 )); done
                [ "$live" -lt "$W" ] && break
                wait -n "${cg_pids[@]}" 2> /dev/null || sleep 1
            done
            run_pass "$row" "$rid" "$ph" callgrind & cg_pids+=($!)
        done
        for p in "${cg_pids[@]}"; do wait "$p"; done
        say "BUSY after callgrind: $(tail -n 3 "$J/cores.log" 2> /dev/null | tr '\n' ' ')"
    fi
    phase_end $rc      # a pass's failure is in its run.error; the report decides the exit (function-level sources)
fi

# ---- post (CPU, every row and tool at once over the allocated cores) ----------------------------------------------
post_one() {   # row dir, phase, pass: the parsed json of one pass (the raw data stays beside it)
    local d=$1 ph=$2 pass=$3 p="$1/$3"
    case $pass in
        pyspy)
            [ -f "$p/pyspy.raw" ] || return 0
            $PY tools/ops/call_profile.py pyspy --raw "$p/pyspy.raw" --window "$p/window.jsonl" --rate "$(cat "$p/rate" 2> /dev/null || echo "$RATE")" \
                --out "$p/pyspy.json" > "$p/post.log" 2>&1 ;;
        callgrind)
            ls "$p"/callgrind.out.* > /dev/null 2>&1 || return 0
            $PY tools/ops/call_profile.py callgrind --dir "$p" --window "$p/window.jsonl" --out "$p/callgrind.json" \
                > "$p/post.log" 2>&1
            gzip -f "$p"/callgrind.out.* ;;
        perf)
            [ -f "$p/perf.data" ] || return 0
            "$PERF" script -i "$p/perf.data" -F comm,tid,period,ip,sym,dso > "$p/perf.script" 2> "$p/perf-script.log"
            $PY tools/ops/call_profile.py perf --script "$p/perf.script" --window "$p/window.jsonl" --out "$p/perf.json" \
                --folded "$p/perf.folded" > "$p/post.log" 2>&1
            rm -f "$p/perf.script" ;;     # perf.folded keeps every stack; perf.data stays for re-reading
        nsys)
            local rep; rep=$(ls "$p"/nsys.nsys-rep "$p"/nsys.qdrep 2> /dev/null | head -n 1)
            [ -n "$rep" ] || return 0
            "$NSYS" stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum,nvtx_sum \
                --format csv --output "$p/nsys" --force-export=true "$rep" > "$p/stats.log" 2>&1
            local calls; calls=$($PY -c 'import json,sys; print(json.load(sys.stdin)["calls"])' \
                < <($PY tools/ops/call_profile.py window --window "$p/window.jsonl"))
            $PY tools/ops/attribution_report.py nsys --prefix "$p/nsys" --calls "${calls:-1}" --out "$p/nsys.json" \
                --kernel-regex "${CP_KERNEL_REGEX:-.}" \
                > "$p/post.log" 2>&1 ;;
        strace)
            [ -f "$p/strace.txt" ] || return 0
            $PY tools/ops/call_profile.py strace --trace "$p/strace.txt" --window "$p/window.jsonl" --out "$p/strace.json" \
                > "$p/post.log" 2>&1
            gzip -f "$p/strace.txt" ;;
    esac
}
if part post; then
    phase post
    : > "$RES/plan/post.todo"
    while IFS=$'\t' read -r row _ ph _; do
        row_wanted "$row" || continue
        for pass in pyspy callgrind perf nsys strace; do
            [ -d "$RES/rows/$row/$pass" ] && printf '%s %s %s\n' "$RES/rows/$row" "$ph" "$pass" >> "$RES/plan/post.todo"
        done
    done < "$RES/plan/runs.tsv"
    export -f post_one
    export NSYS PERF PY RATE CP_KERNEL_REGEX
    # one worker per (row, pass) up to the allocated cores: perf script's unwinding is the long one (one core each)
    xargs -P "$NCORES" -n 3 bash -c 'post_one "$0" "$1" "$2"' < "$RES/plan/post.todo"
    rc=$?
    say "post: $(cat "$RES"/rows/*/*/post*.log 2> /dev/null | grep -E '^(PERF|PYSPY|CALLGRIND|NSYS|STRACE) ' | cut -c1-60 | sort | uniq -c | head -n 12 | tr '\n' ';')"
    phase_end $rc
fi

# ---- report (CPU) ------------------------------------------------------------------------------------------------
if part report; then
    phase report
    $PY tools/ops/call_profile.py report --res "$RES" --rows-file "$RES/plan/rows.tsv" --vendor "$V" --wheel "$WHEEL" \
        --host "${HOST_ID:-?}" --tools "$(tr '\n' ';' < "$J/tools.txt" | head -c 400)" --top "${CP_REPORT_TOP:-40}" \
        > "$RES/logs/report.log" 2>&1
    rc=$?
    grep -E '^(CALL-PROFILE|SUMMARY|LARGEST) ' "$RES/logs/report.log" | while read -r ln; do say "$ln"; done
    say "report rc=$rc -> CALL-PROFILE-$V.md$( case $rc in 0|6|7|8) ;; *) echo "; report failed: $(tail -n 1 "$RES/logs/report.log")" ;; esac)"
    REPORT_RC=$rc
    phase_end $rc
fi
exit 0      # finish() turns this into the report's code (REPORT_RC) or the worst phase's
