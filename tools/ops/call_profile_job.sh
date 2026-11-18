#!/bin/bash
# tools/ops/call_profile_job.sh -- WHERE THE WALL TIME OF ONE SMALL PUBLIC CALL GOES, at function level, host side and
# GPU side, on OUR wheel alone (lane fb9-call-profile, 2026-11-17; no prior wheel: Andrew's direction). Diagnostic
# tooling, not a board and not a measurement. For each row of a rows file (CP_ROWS_FILE, default
# tools/ops/call_profile_rows.tsv: race id, forced phase, name) it runs the row's race on the candidate venv once per
# pass, each a forced scored run whose extra calls sit inside the harness's profile window
# (tools/bench_board_small_rows.py range_open/range_close: calls 2..n, nothing of the data load, the fit before an
# inference, the warmup or the first call) and whose n fills the pass's seconds at the first call's time:
#
#   tools/nvidia_central.sh submit measure --cap 60 --results box-out/callprof-<tag> tools/ops/call_profile_job.sh
#
#   time    no tool: wall per call (the window's clock over its calls)                       CP_TIME_SECONDS (3)
#   perf    perf record -e cpu-clock -F 999 --call-graph dwarf -D -1 --control fifo: samples only inside the window
#           (the harness enables and disables perf through the FIFO); perf script -> folded stacks -> top 40 self and
#           top 40 inclusive frames, percent and microseconds of CPU per call. In the same process, as the
#           cross-check, py-spy record --native --subprocesses: stacks under the forced phase's runner method (its
#           ptrace pauses are off-CPU time, so perf's on-CPU shares stand; the wall comes from the time pass). No
#           usable perf in the container: py-spy alone                                          CP_PERF_SECONDS (10)
#   nsys    nsys profile -t cuda,nvtx --capture-range=nvtx (the window's NVTX range): per call launches, syncs,
#           allocations, H2D/D2H copies and bytes, CUDA API ms, GPU kernel ms (tools/ops/attribution_report.py nsys)
#                                                                                                 CP_NSYS_SECONDS (3)
#   strace  strace -f -ttt -T, cut to the window by the harness's marker stats: syscalls per call (mmap, munmap,
#           madvise, futex, ioctl, ...), plus the LD_PRELOAD libc allocation counter (tools/ops/call_profile_malloc.c)
#           read at the window's open and close: mallocs, frees and bytes per call             CP_STRACE_SECONDS (3)
# Report: CALL-PROFILE-nvidia.md and .json (tools/ops/call_profile.py report): per row wall ms per call, the split GPU
# kernel / CUDA API / host outside CUDA, the top frames, syscalls and allocations per call, and one line LARGEST HOST
# COSTS naming functions. Raw data kept: rows/<row>/perf/perf.folded, rows/<row>/perf/pyspy.collapsed, the nsys
# report and CSVs, rows/<row>/strace/strace.txt.gz. job/symbols.txt says which .so files carry a symbol table.
#
# PARALLEL PLAN (HARD RULES 2026-11-05/07): the timed and profiled race processes are SERIAL BY NATURE: one process on
# the one GPU at a time (a profiled run must not share the GPU or the host with another). Everything else uses the
# cgroup's cores: the tool installs (apt perf/strace/gcc/nsight, pip py-spy) at once; the post-processing (perf script
# and folding, py-spy folding, nsys stats exports, the strace parse) of every row at once, alloc_cores workers. The
# race drivers use every core (no thread caps). job/cores.log: busy cores of total.
#
# Pod minutes: `python3 tools/ops/call_profile.py estimate [--rows-file F]`, printed at the plan phase (target < 30).
# Knobs: CP_ROWS_FILE, CP_ROWS (comma list of row names), CP_PASSES (default time,perf,nsys,strace), CP_PARTS
# (default tools,plan,install,run,post,report; the report reads whatever the results dir holds), CP_<PASS>_SECONDS,
# CP_PERF_FREQ (999), CP_MANIFEST (the candidate's wheel-manifest json; default: the newest sibling results dir's
# job/wheel-manifest-candidate.json: used to install the venv when it is missing), PC_CACHE_ROOT, PC_DATA_ROOT,
# PC_GPU_TOKEN (L40S), PC_ALLOW_HW_DIFF. EXIT: 0 every row has its wall, frames and nsys counts; 6 a row lacks a
# profile part; 7 a row lacks a wall time; 2 refused to start, 3 wrong box, 4 no venv, 5 no plan, 124 terminated.
# Tests: tools/test_call_profile.py.
set -uo pipefail
JOB_NAME=call-profile
# shellcheck source=tools/ops/box_job_lib.sh
source "$(dirname "$0")/box_job_lib.sh"

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
IFS=',' read -r -a PASSES <<< "${CP_PASSES:-time,perf,nsys,strace}"
PARTS=",${CP_PARTS:-tools,plan,install,run,post,report},"
part() { [[ "$PARTS" == *",$1,"* ]]; }
ROWS_WANT=",${CP_ROWS:-},"
row_wanted() { [ "$ROWS_WANT" = ",," ] || [[ "$ROWS_WANT" == *",$1,"* ]]; }
declare -A SECS=([time]=${CP_TIME_SECONDS:-3} [perf]=${CP_PERF_SECONDS:-10}
                 [nsys]=${CP_NSYS_SECONDS:-3} [strace]=${CP_STRACE_SECONDS:-3})
FREQ=${CP_PERF_FREQ:-999}
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

# ---- tools (CPU, all at once): perf, strace, a C compiler, nsys, py-spy -------------------------------------------
find_perf() {   # the real perf binary (Ubuntu's /usr/bin/perf is a wrapper that refuses a kernel it has no tools for)
    local c
    for c in $(ls -d /usr/lib/linux-tools/*/perf /usr/lib/linux-tools-*/perf 2> /dev/null | sort -V -r) "$(command -v perf 2> /dev/null)"; do
        [ -n "$c" ] && [ -x "$c" ] && "$c" --version > /dev/null 2>&1 && { echo "$c"; return 0; }
    done
    return 1
}
install_apt_tools() {   # perf (linux-tools), strace, gcc: whichever is missing, one apt transaction
    local log="$RES/logs/apt-tools.log" want=()
    find_perf > /dev/null || want+=(linux-tools-common linux-tools-generic)
    command -v strace > /dev/null || want+=(strace)
    command -v cc > /dev/null || command -v gcc > /dev/null || want+=(gcc libc6-dev)
    command -v readelf > /dev/null || want+=(binutils)
    [ ${#want[@]} -eq 0 ] && return 0
    DEBIAN_FRONTEND=noninteractive apt-get update -qq >> "$log" 2>&1
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "${want[@]}" >> "$log" 2>&1
    echo "apt ${want[*]} rc=$?" >> "$J/tools-how.txt"
}
if part tools; then
    phase tools
    : > "$J/tools-how.txt"
    pids=()
    install_nsight & pids+=($!)
    install_apt_tools & pids+=($!)
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
CC=$(command -v cc || command -v gcc) || CC=
MALLOC_SO="$CACHE_ROOT/callprof_malloc.so"
if [ -n "$CC" ] && { [ ! -f "$MALLOC_SO" ] || [ "$TREE/tools/ops/call_profile_malloc.c" -nt "$MALLOC_SO" ]; }; then
    "$CC" -O2 -shared -fPIC -o "$MALLOC_SO" "$TREE/tools/ops/call_profile_malloc.c" > "$RES/logs/cc-malloc.log" 2>&1 \
        || { say "allocation counter did not build: $(head -n 3 "$RES/logs/cc-malloc.log" | tr '\n' ' ')"; rm -f "$MALLOC_SO"; }
fi
[ -f "$MALLOC_SO" ] || MALLOC_SO=
PERF_OK=
if [ -n "$PERF" ]; then      # can this container open perf events at all (seccomp, perf_event_paranoid)?
    "$PERF" stat -e cpu-clock -- true > "$RES/logs/perf-probe.log" 2>&1 && PERF_OK=1
fi
{
    echo "perf ${PERF:-absent} $([ -n "$PERF" ] && "$PERF" --version 2>&1 | head -n 1) events $([ -n "$PERF_OK" ] && echo usable || echo "REFUSED: $(tail -n 2 "$RES/logs/perf-probe.log" 2> /dev/null | tr '\n' ' ')") perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2> /dev/null || echo '?')"
    echo "py-spy ${PYSPY:-absent} $([ -n "$PYSPY" ] && "$PYSPY" --version 2>&1 | head -n 1)"
    echo "nsys ${NSYS:-absent} $([ -n "$NSYS" ] && "$NSYS" --version 2>&1 | head -n 1)"
    echo "strace ${STRACE:-absent} $([ -n "$STRACE" ] && strace -V 2>&1 | head -n 1)"
    echo "allocation counter ${MALLOC_SO:-absent (no C compiler or build failed: logs/cc-malloc.log)}"
    echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
} > "$J/tools.txt" 2>&1
say "tools: $(tr '\n' ';' < "$J/tools.txt" | head -c 800)"

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

# ---- install (CPU): the candidate venv, only when the pod has none ---------------------------------------------
MAN=${CP_MANIFEST:-}
if [ -z "$MAN" ]; then
    # shellcheck disable=SC2012
    MAN=$(ls -t "$(dirname "$RES")"/*/job/wheel-manifest-candidate.json 2> /dev/null | head -n 1)
fi
if part install && [ ! -x "$APY" ]; then
    phase install
    [ -f "$MAN" ] || { say "no venv at $VENV and no wheel manifest (CP_MANIFEST)"; exit 4; }
    $PY tools/bench_board.py "${COMMON[@]}" --ours-install-only --install-only --venv "$VENV" --cache "$CACHE_ROOT/i-candidate" \
        --wheel-manifest "$MAN" --race-ids @"$RES/plan/race-ids-full.txt" --out "$RES/install-candidate" \
        > "$RES/logs/install-candidate.log" 2>&1
    phase_end $?
fi
[ -x "$APY" ] || { say "no venv for the candidate at $VENV"; exit 4; }
WHEEL=$("$APY" -c 'import importlib.metadata as m
for n in ("mojolearn-core", "mojolearn"):
    try:
        print(m.version(n)); break
    except m.PackageNotFoundError:
        pass' 2> /dev/null)
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

# ---- run (GPU, serial): every pass of every row, one race process at a time --------------------------------------
# one pass of one row: the forced scored run of the row's phase with its window, under the pass's tool
run_pass() {   # row, race id, phase, pass
    local row=$1 rid=$2 ph=$3 pass=$4 d="$RES/rows/$1/$4" t0=$SECONDS rc err
    rm -rf "$d"; mkdir -p "$d"
    echo "$rid" > "$d/ids"
    local envs=(MOJOLEARN_BENCH_FORCE_CALLS=20 MOJOLEARN_BENCH_FORCE_CALLS_PHASE="$ph"
                MOJOLEARN_BENCH_PROFILE_WINDOW="$d/window.jsonl" MOJOLEARN_BENCH_PROFILE_SECONDS="${SECS[$pass]}")
    local pre=()
    case $pass in
        time) ;;
        perf)      # perf outside (only the window's samples), py-spy inside (the whole run, cut by its frames)
            [ -n "$PERF_OK$PYSPY" ] || { echo "neither perf (job/tools.txt) nor py-spy" > "$d/run.error"; return 6; }
            if [ -n "$PERF_OK" ]; then
                mkfifo "$d/ctl" "$d/ack"
                envs+=(MOJOLEARN_BENCH_PERF_CTL="$d/ctl,$d/ack" PYTHONPERFSUPPORT=1)
                pre=("$PERF" record -e cpu-clock -F "$FREQ" --call-graph dwarf,8192 -D -1 --control "fifo:$d/ctl,$d/ack"
                     -o "$d/perf.data" --)
            fi
            [ -n "$PYSPY" ] && pre+=("$PYSPY" record --native --subprocesses --rate 250 --format raw -o "$d/pyspy.collapsed" --) ;;
        nsys)
            [ -n "$NSYS" ] || { echo "nsys absent (logs/apt-nsight.log)" > "$d/run.error"; return 6; }
            envs+=(MOJOLEARN_BENCH_NVTX_RANGE="$RANGE" MOJOLEARN_BENCH_NVTX_PATH="$CACHE_ROOT/attr-nvtx-candidate"
                   NSYS_NVTX_PROFILER_REGISTER_ONLY=0)
            pre=(env NSYS_NVTX_PROFILER_REGISTER_ONLY=0 "$NSYS" profile -t cuda,nvtx --sample=none --cpuctxsw=none
                 --capture-range=nvtx --nvtx-capture="$RANGE" --capture-range-end=stop --force-overwrite=true -o "$d/nsys") ;;
        strace)
            [ -n "$STRACE" ] || { echo "strace absent (logs/apt-tools.log)" > "$d/run.error"; return 6; }
            [ -n "$MALLOC_SO" ] && envs+=(MOJOLEARN_BENCH_MALLOC_COUNTER="$MALLOC_SO" LD_PRELOAD="$MALLOC_SO")
            pre=("$STRACE" -f -ttt -T -qq -o "$d/strace.txt") ;;
        *) echo "unknown pass $pass (time, perf, nsys, strace)" > "$d/run.error"; return 5 ;;
    esac
    local which=(--mojolearn-version "$WHEEL")
    [ -f "$MAN" ] && which=(--wheel-manifest "$MAN")
    ${pre[@]+"${pre[@]}"} env "${envs[@]}" "$PY" tools/bench_board.py "${COMMON[@]}" --python-env "$APY" --skip-install \
        "${which[@]}" --cache "$CACHE_ROOT/c1" \
        --race-ids @"$d/ids" --no-smoke-gate \
        --guarded-smoke-override-reason "call profile job: diagnostic run of a board row this pod's earlier job smoked" \
        --out "$d/board" > "$d/run.log" 2>&1 < /dev/null
    rc=$?
    rm -rf "$d/board/work" "$d/board/evidence" "$d/ctl" "$d/ack"
    if [ $rc -ne 0 ]; then
        err=$($PY tools/ops/attribution_report.py error --out "$d/board" --log "$d/run.log" 2>&1 | head -n 1)
        echo "$err" > "$d/run.error"
    elif [ ! -s "$d/window.jsonl" ]; then
        echo "no profile window: the race opened none (the phase name, or a driver without range_open)" > "$d/run.error"; rc=6
    fi
    say "run $row $pass rc=$rc $(( SECONDS - t0 )) s window: $($PY tools/ops/call_profile.py window --window "$d/window.jsonl" | head -c 200)$([ -s "$d/run.error" ] && echo " ERROR $(head -c 300 "$d/run.error")")"
    return $rc
}
if part run; then
    phase run
    rc=0
    while IFS=$'\t' read -r row rid ph; do
        row_wanted "$row" || continue
        for pass in "${PASSES[@]}"; do
            run_pass "$row" "$rid" "$ph" "$pass"; r=$?; [ $r -gt $rc ] && rc=$r
        done
    done < "$RES/plan/runs.tsv"
    phase_end $rc
fi

# ---- post (CPU, every row and tool at once over the allocated cores) ----------------------------------------------
post_one() {   # row dir, phase, pass: the parsed json of one pass (the raw data stays beside it)
    local d=$1 ph=$2 pass=$3 p="$1/$3"
    case $pass in
        perf)
            if [ -f "$p/pyspy.collapsed" ]; then
                $PY tools/ops/call_profile.py pyspy --collapsed "$p/pyspy.collapsed" --window "$p/window.jsonl" --phase "$ph" \
                    --out "$p/pyspy.json" > "$p/post-pyspy.log" 2>&1
            fi
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
    while IFS=$'\t' read -r row _ ph; do
        row_wanted "$row" || continue
        for pass in perf nsys strace; do
            [ -d "$RES/rows/$row/$pass" ] && printf '%s %s %s\n' "$RES/rows/$row" "$ph" "$pass" >> "$RES/plan/post.todo"
        done
    done < "$RES/plan/runs.tsv"
    export -f post_one
    export NSYS PERF PY
    # one worker per (row, pass) up to the allocated cores: perf script's unwinding is the long one (one core each)
    xargs -P "$NCORES" -n 3 bash -c 'post_one "$0" "$1" "$2"' < "$RES/plan/post.todo"
    rc=$?
    say "post: $(cat "$RES"/rows/*/*/post*.log 2> /dev/null | grep -E '^(PERF|PYSPY|NSYS|STRACE) ' | cut -c1-60 | sort | uniq -c | head -n 12 | tr '\n' ';')"
    phase_end $rc
fi

# ---- report (CPU) ------------------------------------------------------------------------------------------------
if part report; then
    phase report
    $PY tools/ops/call_profile.py report --res "$RES" --rows-file "$RES/plan/rows.tsv" --vendor "$V" --wheel "$WHEEL" \
        --host "${HOST_ID:-?}" --tools "$(tr '\n' ';' < "$J/tools.txt" | head -c 400)" \
        > "$RES/logs/report.log" 2>&1
    rc=$?
    grep -E '^(CALL-PROFILE|SUMMARY) ' "$RES/logs/report.log" | while read -r ln; do say "$ln"; done
    say "report rc=$rc -> CALL-PROFILE-$V.md$( case $rc in 0|6|7) ;; *) echo "; report failed: $(tail -n 1 "$RES/logs/report.log")" ;; esac)"
    REPORT_RC=$rc
    phase_end $rc
fi
exit 0      # finish() turns this into the report's code (REPORT_RC) or the worst phase's
