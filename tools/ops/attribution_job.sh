#!/bin/bash
# tools/ops/attribution_job.sh -- WHERE THE TIME GOES in the regressed paper rows, on both wheels, same box, same job
# (lane attribution-job, 2026-11-16). Diagnostic tooling, not a board: it times and profiles OUR arms only, on the
# L40S pod whose earlier paper job left the candidate venv ($PC_CACHE_ROOT/venv-candidate), the 0.8.55 venv
# ($PC_CACHE_ROOT/venv-prior-0.8.55) and the datasets ($PC_DATA_ROOT). The rows, their forced phase and their arm runs
# come from a rows file: AT_ROWS_FILE, default tools/ops/attribution_rows.tsv (the nine regressed paper rows; the one
# list). Any board row can be profiled by naming its race id, forced phase and arm in such a file.
#
#   tools/nvidia_central.sh submit measure --cap 240 --results box-out/attr-<tag> tools/ops/attribution_job.sh
#
# Per row and per wheel (AT_ARMS, default candidate,prior-0.8.55; the reading compares the first against the second):
#   A. time     bench_board on the arm's venv with MOJOLEARN_BENCH_FORCE_CALLS=20 on the row's phase: the scored run
#               is 20 consecutive calls after one excluded warmup (median, min, max). gbdt-depthwise istella infer:
#               the prebinned arm ALONE, then both arms in each order (MOJOLEARN_BENCH_OWN_ARMS: the arm-order test).
#   B. profile  the same race under `nsys profile -t cuda,nvtx --capture-range=nvtx`: the harness opens the NVTX range
#               `attr-calls` around calls 2..20 (tools/bench_board_small_rows.py range_open), so the capture holds
#               exactly 19 calls and nothing of the data load, the fit before an inference, the warmup or the first
#               call's hashing. `nsys stats` -> per call: kernel launches, synchronizations, device allocations and
#               frees, H2D/D2H copies and bytes, GPU kernel time, CUDA API time, host time outside the API.
#               No nsys on the pod and none installable: py-spy --native over the run (frames by sample count,
#               tools/ops/profile_fixed_cost_summary.py) and, for the neural rows, tools/neural_stage_timing.py's
#               session stage ticks.
#   C. kernels  the distance kernels (pinned_distance_register_tile_kernel, smem_distance*) of the classical rows:
#               duration, registers, block and shared memory from the B trace (cuda_gpu_trace) with the theoretical
#               sm_89 occupancy; plus `ncu` (LaunchStats, Occupancy, SpeedOfLight) when the pod grants GPU counters
#               (ERR_NVGPUCTRPERM is recorded, never fatal).
# nsys and ncu: the pod's own (PATH, /usr/local/cuda/bin, /opt/nvidia/nsight-*), else apt (nsight-systems-cli /
# nsight-compute-<v> from the CUDA or devtools repository, the cuda-keyring added when absent). pip has no nsys. The
# NVTX range uses the `nvtx` package installed with `pip --target` beside each venv (the measured venvs stay as they
# are). job/tools.txt records what was used, its version and how it came.
#
# PARALLEL PLAN (HARD RULES 2026-11-05/07): everything that runs on the GPU is SERIAL BY NATURE: one race process at a
# time with nothing else on the GPU (timing must not share the GPU; a profiled run must not overlap a timed one), row
# by row, the two wheels back to back (same box, minutes apart). Everything else uses the cgroup's cores: the tool
# installs (apt, pip for each arm) at once, the `nsys stats` exports of every capture at once (alloc_cores workers),
# the report. The race drivers themselves use every core (no thread caps). job/cores.log: busy cores of total.
#
# AT_PARTS (default tools,plan,time,profile,stats,kernels,report) and AT_ROWS (comma list of row names, default all)
# split the job over submits on the same results dir; the report reads whatever the dir holds. Pod minutes per row
# and in all: `python3 tools/ops/attribution_report.py estimate --recipes bench/results/paper_compare/recipes-nvidia.json
# [--rows-file F]` (printed in job.log at the plan phase as ESTIMATE row=... pod_min=...).
# A failed run prints its first error line into job.log (`run ... FAILED: <line>`, also <run dir>.error). EXIT CODE
# (when the report part runs): 0 every row has its timing runs and a captured profile on every wheel; 6 a row lacks
# a profile; 7 a row lacks a time; the per-row failures are the SUMMARY lines in job.log and ATTRIBUTION-<v>.md. ncu
# (part C) never fails the job. Earlier exits: 2 refused to start, 3 wrong box, 4 no venv, 5 no plan, 124 terminated.
# AT_MEM_NODE (near|far|<node>) and AT_MEM_THP (off) time the part-A runs under a chosen host-memory placement through
# tools/ops/mem_state.py exec: the CPUs of the GPU's NUMA node, of another node, or of the named node (memory follows by
# first touch), and/or no transparent huge pages. Unset: the runs are untouched. Part A logs the box's memory state
# (mem_state.py snapshot: huge-page fault counters, free 2 MB blocks per node) before and after each timing run as
# `MEM` lines. The run-to-run drift test is three submits of AT_PARTS=tools,plan,time,report with
# AT_ROWS_FILE=tools/ops/drift_rows.tsv on three results dirs: neither knob, AT_MEM_NODE=far, AT_MEM_THP=off.
# Knobs: AT_ROWS_FILE, AT_ARMS, AT_CALLS (20), AT_MANIFEST_DIR (the earlier paper job's job/ holding wheel-manifest-<arm>.json;
# default: the newest sibling results dir that has one), AT_NCU_LAUNCHES (6), PC_CACHE_ROOT, PC_DATA_ROOT,
# PC_GPU_TOKEN (L40S), PC_ALLOW_HW_DIFF. Results: time-<arm>/<row>/<run>/ (bench_board out), profile-<arm>/<row>.*
# (nsys report, stats CSVs, parsed json), ncu-<arm>/<row>.*, ATTRIBUTION-nvidia.md and .json, job/ (job.log,
# phases.tsv, cores.log, host-probe.json, tools.txt, version-<arm>.txt, status.json).
# Tests: tools/test_attribution_job.py.
set -uo pipefail
JOB_NAME=attribution-job
# shellcheck source=tools/ops/box_job_lib.sh
source "$(dirname "$0")/box_job_lib.sh"

if [ "${MOJOLEARN_BOX_PERSIST:-}" != 1 ] && [ "${MOJOLEARN_ALLOW_UNPERSISTED:-}" != 1 ]; then
    echo "attribution-job: runs under tools/box_persist_run.sh (tools/nvidia_central.sh submit); MOJOLEARN_ALLOW_UNPERSISTED=1 overrides" >&2
    exit 2
fi
V=$(detect_vendor) || { echo "attribution-job: no nvidia-smi" >&2; exit 2; }
[ "$V" = nvidia ] || { echo "attribution-job: nsys/ncu attribution is NVIDIA only (this box: $V)" >&2; exit 3; }
TREE="$(cd "$(dirname "$0")/../.." && pwd)"
RES="${MOJOLEARN_RESULTS_DIR:?attribution_job.sh needs MOJOLEARN_RESULTS_DIR (set by tools/box_persist_run.sh)}"
mkdir -p "$RES" && RES="$(cd "$RES" && pwd)"
export MOJOLEARN_RESULTS_DIR="$RES"
J="$RES/job"
mkdir -p "$J" "$RES/plan" "$RES/logs"
exec >> "$J/job.log" 2>&1
export PATH="/root/.pixi/bin:/usr/local/cuda/bin:$PATH"
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS
export MOJOLEARN_NUMERIC_MODE=identical
PY=${AT_PYTHON:-python3}
cd "$TREE" || exit 2
DATA_ROOT=${PC_DATA_ROOT:-/root/datasets/gbm-bench}
CACHE_ROOT=${PC_CACHE_ROOT:-/root/pc-cache}
TOKEN=${PC_GPU_TOKEN:-L40S}
CALLS=${AT_CALLS:-20}
RECIPES="$TREE/bench/results/paper_compare/recipes-$V.json"
ROWS_FILE=${AT_ROWS_FILE:-$TREE/tools/ops/attribution_rows.tsv}
TIME_PREFIX=()      # part A under a chosen memory placement (AT_MEM_NODE, AT_MEM_THP); empty: untouched
if [ -n "${AT_MEM_NODE:-}${AT_MEM_THP:-}" ]; then
    TIME_PREFIX=("$PY" tools/ops/mem_state.py exec --node "${AT_MEM_NODE:-any}" --thp "${AT_MEM_THP:-on}" --)
fi
IFS=',' read -r -a ARMS <<< "${AT_ARMS:-candidate,prior-0.8.55}"
PARTS=",${AT_PARTS:-tools,plan,time,profile,stats,kernels,report},"
part() { [[ "$PARTS" == *",$1,"* ]]; }
ROWS_WANT=",${AT_ROWS:-},"
row_wanted() { [ "$ROWS_WANT" = ",," ] || [[ "$ROWS_WANT" == *",$1,"* ]]; }
WORST=0
REPORT_RC=      # the report's exit code (EXIT_* of attribution_report.py) once the report part ran: the job's exit
PHASE=start
P_T0=$(date +%s)
RANGE=attr-calls     # the NVTX range the harness opens (MOJOLEARN_BENCH_NVTX_RANGE) and nsys waits for

SAMPLER=
if [ -r /proc/stat ]; then cores_sampler & SAMPLER=$!; fi
finish() {
    local code=$?
    trap - EXIT
    [ -n "$SAMPLER" ] && kill "$SAMPLER" 2> /dev/null
    # a finished report decides the exit (per-row timing and profile); a TERM or an earlier refusal keeps its own code
    if [ -n "$REPORT_RC" ] && [ "$code" -eq 0 ]; then code=$REPORT_RC
    elif [ "$WORST" -gt "$code" ]; then code=$WORST; fi
    printf '{"job":"attribution","vendor":"%s","arms":"%s","rows":"%s","parts":"%s","host_id":"%s","tree_head":"%s","phase":"%s","exit":%d,"job_script_sha256":"%s"}\n' \
        "$V" "${ARMS[*]}" "${AT_ROWS:-all}" "$PARTS" "${HOST_ID:-}" "$(git -C "$TREE" rev-parse HEAD 2>/dev/null)" "$PHASE" "$code" \
        "$(sha256sum "$TREE/tools/ops/attribution_job.sh" | cut -d' ' -f1)" > "$J/status.json"
    say "exit $code (last phase $PHASE)"
    exit "$code"
}
trap finish EXIT
trap 'exit 124' TERM INT

say "attribution $V arms ${ARMS[*]} rows ${AT_ROWS:-all} parts $PARTS calls $CALLS at $(git rev-parse HEAD 2>/dev/null) results $RES"
{ uname -a; echo "nproc $(nproc)"; free -g 2> /dev/null || true; df -h /root 2> /dev/null || true; } > "$J/box.txt" 2>&1
NCORES=$(alloc_cores)
say "BUSY at start: nproc $(nproc), allocation $NCORES cores; GPU work below is serial by nature (one process on the GPU)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$J/gpu.txt" 2>&1
NGPU=$(grep -c . "$J/gpu.txt" 2> /dev/null || echo 0)
if gpu_other; then
    say "HW-DIFF: the paper ran on $TOKEN; this box: $(head -c 300 "$J/gpu.txt" | tr '\n' ' ')"
    [ "${PC_ALLOW_HW_DIFF:-0}" = 1 ] || { say "refused: NVIDIA is measured on an $TOKEN only; PC_ALLOW_HW_DIFF=1 overrides"; exit 3; }
fi
prune_persist_staging
host_probe_phase
arith_gate   # tools/ops/box_job_lib.sh: identical-path arithmetic on this GPU, before any timing

# ---- the arms: the venvs the paper job left, their versions and wheel manifests ------------------------------------
declare -A ARM_PY ARM_MAN ARM_VER
for arm in "${ARMS[@]}"; do
    ARM_PY[$arm]="$CACHE_ROOT/venv-$arm/bin/python"
    [ -x "${ARM_PY[$arm]}" ] || { say "no venv for $arm at ${ARM_PY[$arm]} (run the paper job's install part first)"; exit 4; }
    ARM_VER[$arm]=$("${ARM_PY[$arm]}" -c 'import importlib.metadata as m
for n in ("mojolearn-core", "mojolearn"):
    try:
        print(m.version(n)); break
    except m.PackageNotFoundError:
        pass' 2> /dev/null)
    man=
    if [ -n "${AT_MANIFEST_DIR:-}" ]; then man="$AT_MANIFEST_DIR/wheel-manifest-$arm.json"
    else
        # shellcheck disable=SC2012
        man=$(ls -t "$(dirname "$RES")"/*/job/wheel-manifest-"$arm".json 2> /dev/null | head -n 1)
    fi
    [ -f "$man" ] && ARM_MAN[$arm]=$man || ARM_MAN[$arm]=
    echo "${ARM_VER[$arm]:-?} venv ${ARM_PY[$arm]} manifest ${ARM_MAN[$arm]:-none}${ARM_MAN[$arm]:+ sha256 $(sha256sum "${ARM_MAN[$arm]}" | cut -c1-16)}" > "$J/version-$arm.txt"
    say "arm $arm: $(cat "$J/version-$arm.txt")"
done

# ---- tools (CPU, all at once): nsys, ncu, the nvtx package per arm, py-spy --------------------------------------
# find_tool, apt_install_first, install_nsight: tools/ops/box_job_lib.sh (shared with tools/ops/call_profile_job.sh)
NSYS=; NCU=; PYSPY=
if part tools; then
    phase tools
    : > "$J/tools-how.txt"
    pids=()
    install_nsight & pids+=($!)
    for arm in "${ARMS[@]}"; do      # nvtx beside each venv (pip --target: the measured venv is not changed)
        "${ARM_PY[$arm]}" -m pip install -q --disable-pip-version-check --target "$CACHE_ROOT/attr-nvtx-$arm" nvtx \
            > "$RES/logs/pip-nvtx-$arm.log" 2>&1 & pids+=($!)
    done
    "$PY" -m pip install -q --disable-pip-version-check py-spy > "$RES/logs/pip-pyspy.log" 2>&1 & pids+=($!)
    for p in "${pids[@]}"; do wait "$p"; done
    phase_end 0
fi
NSYS=$(find_tool nsys) || NSYS=
NCU=$(find_tool ncu) || NCU=
PYSPY=$(find_tool py-spy) || PYSPY=
{
    if [ -n "$NSYS" ]; then echo "nsys $("$NSYS" --version 2>&1 | head -n 1) at $NSYS ($(grep '^nsys' "$J/tools-how.txt" 2>/dev/null || echo 'on the pod'))"
    else echo "nsys absent (apt found none: logs/apt-nsight.log); B falls back to py-spy --native and the neural stage ticks"; fi
    if [ -n "$NCU" ]; then echo "ncu $("$NCU" --version 2>&1 | tail -n 1) at $NCU ($(grep '^ncu' "$J/tools-how.txt" 2>/dev/null || echo 'on the pod'))"
    else echo "ncu absent; C uses the nsys trace (registers, block, shared memory) and the theoretical sm_89 occupancy"; fi
    for arm in "${ARMS[@]}"; do
        echo "nvtx for $arm: $(PYTHONPATH="$CACHE_ROOT/attr-nvtx-$arm" "${ARM_PY[$arm]}" -c 'import importlib.metadata as m; print(m.version("nvtx"))' 2>&1 | tail -n 1) (pip --target $CACHE_ROOT/attr-nvtx-$arm)"
    done
    [ -n "$PYSPY" ] && echo "py-spy $("$PYSPY" --version 2>&1 | head -n 1) at $PYSPY (pip)"
    echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
} > "$J/tools.txt" 2>&1
say "tools: $(tr '\n' ';' < "$J/tools.txt" | head -c 600)"

# ---- plan (CPU): the board arguments of the paper recipes and the runs -------------------------------------------
if part plan || [ ! -s "$RES/plan/board-args.txt" ] || [ ! -s "$RES/plan/rows.tsv" ]; then
    phase plan
    cp "$ROWS_FILE" "$RES/plan/rows.tsv" 2>> "$RES/logs/plan.log" || { say "no rows file $ROWS_FILE"; exit 5; }
    $PY tools/paper_compare.py ids --recipes "$RECIPES" --planner tools/bench_board.py --out-dir "$RES/plan" \
        > "$RES/logs/plan.log" 2>&1
    rc=$?
    $PY tools/ops/attribution_report.py plan --rows-file "$RES/plan/rows.tsv" > "$RES/plan/runs.tsv" \
        2>> "$RES/logs/plan.log" || { rc=5; say "rows file refused: $(tail -n 1 "$RES/logs/plan.log")"; }
    while IFS=$'\t' read -r _ row rid _; do
        grep -qxF "$rid" "$RES/plan/race-ids-full.txt" 2> /dev/null || say "NOTE: $row ($rid) is not in the planned paper races"
    done < <(awk -F'\t' '!seen[$2]++' "$RES/plan/runs.tsv")
    $PY tools/ops/attribution_report.py estimate --recipes "$RECIPES" --rows-file "$RES/plan/rows.tsv" \
        --wheels "${#ARMS[@]}" 2>&1 | while read -r ln; do say "$ln"; done
    phase_end $rc
fi
[ -s "$RES/plan/board-args.txt" ] && [ -s "$RES/plan/runs.tsv" ] || { say "no plan in $RES/plan (part plan failed: logs/plan.log)"; exit 5; }
mapfile -t BARGS < "$RES/plan/board-args.txt"
COMMON=("${BARGS[@]}" --data-root "$DATA_ROOT" --rounds 1 --evidence-local-only --fixtures-venv "$CACHE_ROOT/venv-nf")

# one race on one arm's venv: forced calls on the row's phase, the own arms when given, under an optional prefix
# command (nsys, ncu, py-spy). Serial on the GPU: the caller runs one at a time.
run_board() {   # arm, race id, phase, own arms (`-` = planned), out dir, extra env (K=V,...; `-` none), prefix...
    local arm=$1 rid=$2 ph=$3 own=$4 out=$5 extra=$6; shift 6
    local envs=(MOJOLEARN_BENCH_FORCE_CALLS="$CALLS" MOJOLEARN_BENCH_FORCE_CALLS_PHASE="$ph") man=() ver=() kv
    [ "$own" != - ] && envs+=(MOJOLEARN_BENCH_OWN_ARMS="$own")
    if [ "$extra" != - ]; then IFS=',' read -r -a kv <<< "$extra"; envs+=("${kv[@]}"); fi
    if [ -n "${ARM_MAN[$arm]}" ]; then man=(--wheel-manifest "${ARM_MAN[$arm]}"); else ver=(--mojolearn-version "${ARM_VER[$arm]}"); fi
    mkdir -p "$(dirname "$out")"
    echo "$rid" > "$out.ids"
    local t0=$SECONDS
    "$@" env "${envs[@]}" "$PY" tools/bench_board.py "${COMMON[@]}" --python-env "${ARM_PY[$arm]}" --skip-install \
        ${man[@]+"${man[@]}"} ${ver[@]+"${ver[@]}"} --cache "$CACHE_ROOT/c1" --race-ids @"$out.ids" --no-smoke-gate \
        --guarded-smoke-override-reason "attribution job: diagnostic re-run of paper rows this pod's paper job smoked" \
        --out "$out" > "$out.log" 2>&1
    local rc=$? err
    rm -rf "$out/work"
    rm -f "$out.error"
    if [ $rc -eq 0 ]; then
        say "run $arm $rid ($ph, arms $own) rc=0 $(( SECONDS - t0 )) s: $(tail -n 1 "$out.log" | head -c 160)"
    else    # the failing command's first error line (a preflight refusal, a failed race, a driver error), not its last
        err=$($PY tools/ops/attribution_report.py error --out "$out" --log "$out.log" 2>&1 | head -n 1)
        echo "$err" > "$out.error"
        say "run $arm $rid ($ph, arms $own) rc=$rc $(( SECONDS - t0 )) s FAILED: ${err:0:400}"
    fi
    return $rc
}
runs_of() {   # kind: the plan lines of that kind for the wanted rows
    awk -F'\t' -v k="$1" '$1 == k' "$RES/plan/runs.tsv" | while IFS=$'\t' read -r kind row rid ph own tag; do
        row_wanted "$row" && printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$kind" "$row" "$rid" "$ph" "$own" "$tag"
    done
}

# ---- A: time (GPU, serial) --------------------------------------------------------------------------------------
if part time; then
    phase time
    say "time runs under: node ${AT_MEM_NODE:-any}, huge pages ${AT_MEM_THP:-on}"
    n=$(runs_of time | grep -c .)
    disk_gate time $(( n * ${#ARMS[@]} )) $(( n * ${#ARMS[@]} )) "${PC_EVIDENCE_MB_PER_RACE:-40}"
    rc=0
    while IFS=$'\t' read -r _ row rid ph own tag; do
        for arm in "${ARMS[@]}"; do      # the wheels back to back: same box, minutes apart
            say "MEM before $arm $row: $($PY tools/ops/mem_state.py snapshot 2>&1 < /dev/null | head -c 900)"
            run_board "$arm" "$rid" "$ph" "$own" "$RES/time-$arm/$row/$tag" - ${TIME_PREFIX[@]+"${TIME_PREFIX[@]}"} < /dev/null
            r=$?; [ $r -gt $rc ] && rc=$r
            say "MEM after $arm $row: $($PY tools/ops/mem_state.py snapshot 2>&1 < /dev/null | head -c 900)"
        done
    done < <(runs_of time)
    phase_end $rc
fi

# ---- B: profile (GPU, serial): nsys capture of calls 2..N, else py-spy -------------------------------------------
if part profile; then
    phase profile
    rc=0
    while IFS=$'\t' read -r _ row rid ph own _; do
        for arm in "${ARMS[@]}"; do
            d="$RES/profile-$arm"; mkdir -p "$d"
            if [ -n "$NSYS" ]; then
                run_board "$arm" "$rid" "$ph" "$own" "$d/$row-board" \
                    "MOJOLEARN_BENCH_NVTX_RANGE=$RANGE,MOJOLEARN_BENCH_NVTX_PATH=$CACHE_ROOT/attr-nvtx-$arm,NSYS_NVTX_PROFILER_REGISTER_ONLY=0" \
                    env NSYS_NVTX_PROFILER_REGISTER_ONLY=0 "$NSYS" profile -t cuda,nvtx --sample=none --cpuctxsw=none \
                    --capture-range=nvtx --nvtx-capture="$RANGE" --capture-range-end=stop --force-overwrite=true \
                    -o "$d/$row" < /dev/null
                r=$?
                if [ ! -f "$d/$row.nsys-rep" ] && [ ! -f "$d/$row.qdrep" ]; then
                    [ -s "$d/$row-board.error" ] || echo "no nsys report: the run opened no NVTX range $RANGE (one scored call?)" > "$d/$row-board.error"
                    say "profile $arm $row: no nsys report: $(head -n 1 "$d/$row-board.error" | head -c 300)"
                    [ $r -eq 0 ] && r=6
                fi
            elif [ -n "$PYSPY" ]; then
                run_board "$arm" "$rid" "$ph" "$own" "$d/$row-board" - \
                    "$PYSPY" record --native --subprocesses --rate 250 --format raw -o "$d/$row.collapsed" -- < /dev/null
                r=$?
                $PY tools/ops/profile_fixed_cost_summary.py --race "$rid" --collapsed "$d/$row.collapsed" \
                    --out "$d/$row.summary.txt" > "$RES/logs/pyspy-summary-$arm-$row.log" 2>&1
                case "$rid" in neural/*)
                    lane=${rid#neural/}; lane=${lane%%/*}
                    env MOJOLEARN_BENCH_INSTALLED=1 GBM_BENCH_DATA="$DATA_ROOT" "${ARM_PY[$arm]}" tools/neural_stage_timing.py \
                        --lane "$lane" --calls "$CALLS" > "$d/$row.stages.log" 2>&1 < /dev/null ;;
                esac
            else
                say "profile $arm $row: neither nsys nor py-spy"; r=6
            fi
            rm -rf "$d/$row-board/evidence"      # a profiled run is no measurement: its board, logs and report stay
            [ $r -gt $rc ] && rc=$r
        done
    done < <(runs_of profile)
    phase_end $rc
fi

# ---- nsys stats (CPU, every capture at once over the allocated cores) --------------------------------------------
stats_one() {   # report file, row: the CSVs and the per-call json beside it
    local rep=$1 row=$2 base reports="cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum,nvtx_sum"
    base=${rep%.*}
    case ",${KERNEL_ROWS}," in *",$row,"*) reports="$reports,cuda_gpu_trace" ;; esac
    "$NSYS" stats --report "$reports" --format csv --output "$base" --force-export=true "$rep" > "$base.stats.log" 2>&1
    $PY tools/ops/attribution_report.py nsys --prefix "$base" --calls $(( CALLS - 1 )) --out "$base.json" >> "$base.stats.log" 2>&1
}
KERNEL_ROWS=$(awk -F'\t' '$1 == "kernels" {print $2}' "$RES/plan/runs.tsv" | paste -sd, -)
if part stats && [ -n "$NSYS" ]; then
    phase stats
    : > "$RES/plan/stats.todo"
    for arm in "${ARMS[@]}"; do
        for rep in "$RES/profile-$arm"/*.nsys-rep "$RES/profile-$arm"/*.qdrep; do
            [ -f "$rep" ] || continue
            row=$(basename "${rep%.*}"); row_wanted "$row" && printf '%s %s\n' "$rep" "$row" >> "$RES/plan/stats.todo"
        done
    done
    export -f stats_one
    export NSYS PY CALLS KERNEL_ROWS
    # one exporter per capture, up to the allocated cores at once (each is one sqlite export, a few hundred MB)
    xargs -P "$NCORES" -n 2 bash -c 'stats_one "$0" "$1"' < "$RES/plan/stats.todo"
    rc=$?; [ $rc -ne 0 ] && rc=6
    say "stats: $(grep -h '^NSYS ' "$RES"/profile-*/*.stats.log 2> /dev/null | sort | uniq -c | head -n 20 | tr '\n' ';')"
    phase_end $rc
fi

# ---- C: kernels (GPU, serial): ncu on the distance-kernel rows ---------------------------------------------------
if part kernels && [ -n "$NCU" ]; then
    phase kernels
    rc=0
    while IFS=$'\t' read -r _ row rid ph own _; do
        for arm in "${ARMS[@]}"; do
            d="$RES/ncu-$arm"; mkdir -p "$d"
            # two forced calls: ncu reads static launch properties and replays the first matched launches only
            run_board "$arm" "$rid" "$ph" "$own" "$d/$row-board" MOJOLEARN_BENCH_FORCE_CALLS=2 \
                "$NCU" --target-processes all --kernel-name 'regex:.*[Dd]istan.*' --launch-count "${AT_NCU_LAUNCHES:-6}" \
                --section LaunchStats --section Occupancy --section SpeedOfLight --csv --page details \
                --log-file "$d/$row.csv" < /dev/null
            r=$?; [ $r -gt $rc ] && rc=$r
            rm -rf "$d/$row-board/evidence"
            $PY tools/ops/attribution_report.py ncu --csv "$d/$row.csv" --out "$d/$row.json" > "$RES/logs/ncu-$arm-$row.log" 2>&1
            say "ncu $arm $row: $(tail -n 1 "$RES/logs/ncu-$arm-$row.log")"
        done
    done < <(runs_of kernels)
    phase_end $rc
fi

# ---- report (CPU) ------------------------------------------------------------------------------------------------
if part report; then
    phase report
    arms_csv=$(IFS=,; echo "${ARMS[*]}")
    $PY tools/ops/attribution_report.py report --res "$RES" --vendor "$V" --arms "$arms_csv" \
        --rows-file "$RES/plan/rows.tsv" > "$RES/logs/report.log" 2>&1
    rc=$?
    grep -E '^(ATTRIBUTION|SUMMARY) ' "$RES/logs/report.log" | while read -r ln; do say "$ln"; done
    say "report rc=$rc -> ATTRIBUTION-$V.md$( case $rc in 0|6|7) ;; *) echo "; report failed: $(tail -n 1 "$RES/logs/report.log")" ;; esac)"
    REPORT_RC=$rc
    phase_end $rc
fi
exit 0      # finish() turns this into the report's code (REPORT_RC) or the worst phase's
