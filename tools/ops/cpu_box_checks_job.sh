#!/bin/bash
# tools/ops/cpu_box_checks_job.sh <sha> <suite> -- the native-checks.yml and tool-tests.yml checks of one commit on the
# rented CPU box (tools/cpu_central.sh check), every group at once across all the box's cores (lane fx-cpu-box-checks,
# 2026-11-19; Andrew: "why are we waiting on things? we use runpod?": a GitHub run waited 40 minutes for runners).
#
#   suite parse   every release host group and `identical` device group parsed and elaborated (host-parse, device-parse)
#         native  the CPU column's native checks: one build of run_cpu, then every check
#         tools   tools/run_tool_tests.sh, every tool test file
#         all     the three
# The sha and suite also come from CPU_CHECK_SHA / CPU_CHECK_SUITE (tools/cpu_central.sh check passes them as --env);
# NC_FILTER (`%` reads as `*`) and NC_TIMEOUT as in native-checks.yml's filter and timeout.
#
# NO SECOND COPY OF THE CHECK LOGIC. Each unit calls what the workflows call: tools/ops/native_checks_steps.sh plan,
# parse-group, build, shard, summary and verdict (native-checks.yml calls the same steps), and the ref's own
# tools/run_tool_tests.sh (tool-tests.yml). The steps script is read from THIS script's tree (the box's tools tree,
# /root/mojolearn), as the workflow reads it from its own commit, so an older sha is checked the same way.
#
# PARALLEL PLAN (AGENTS.md hard rules: 100 percent of rented CPU; parallel by default). The units are independent: every
# parse group, the checks build, the check slices (dealt from the plan once the build passes) and the tool tests. They
# run at once, admitted by MEMORY: a unit starts when the box-wide reservations (every concurrent check job on the box
# shares one ledger, $CPU_CHECKS_HOME/ledger) plus its estimate fit in 85 percent of the allocation's memory; a unit
# larger than that runs when nothing else is reserved. Each unit is itself multi-threaded on every core (the compiler at
# -j nproc, run_tool_tests.sh workers = cores), so memory, not cores, bounds the count. Estimates are MEASURED: each unit
# runs under /usr/bin/time -v and its peak RSS goes to $CPU_CHECKS_HOME/peaks.tsv, which the next job on the box reads;
# until a group has a measurement: host parse 8 GiB, device parse 12 GiB (the release recorded 10-12.5), checks build
# 40 GiB (the GPU column's build peaked at 43 GB), a check 2 GiB (largest 1.64 GB), a tool test file 0.5 GiB x cores.
# Order: the checks build first (it gates every check), then device groups, host groups, tool tests; the check slices
# go to the front when the build passes. Busy cores of total are printed every minute and in SUMMARY.md.
#
# Results ($MOJOLEARN_RESULTS_DIR, persisted to R2 by tools/box_persist_run.sh): SUMMARY.md (line 3: checks: N passed: N
# failed: N, then each part and the first error per failing group, as watch_checks.sh prints for a GitHub run),
# parse/<host|device>-parse-<key>/ (the parse step's log, result and memory files), native/ (results, logs, SUMMARY.md,
# REPORT.txt), tools/ (run_tool_tests.sh's logs and SUMMARY.txt), logs/<unit>.log, time/<unit>.txt, busy_cores.txt.
# Exit 0 when everything passed, 1 when anything failed, 2 on a setup failure.
#
# job-preflight: the sha is a full commit the box's source repo holds, the suite is known, the steps script is here
set -u
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
TOOLS="$(cd "$(dirname "$SELF")/../.." && pwd)"
STEPS="$TOOLS/tools/ops/native_checks_steps.sh"
BASE=${CPU_CHECKS_HOME:-/root/cpu-checks}
SRC=$BASE/src
PRE=0; [ "${1:-}" = --preflight ] && { PRE=1; shift; }
SHA=${1:-${CPU_CHECK_SHA:-}}; SUITE=${2:-${CPU_CHECK_SUITE:-all}}
usage() { echo "usage: cpu_box_checks_job.sh <40-hex sha> [parse|native|tools|all]" >&2; exit 2; }
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || usage
case "$SUITE" in parse|native|tools|all) ;; *) usage ;; esac
export PATH="/root/.pixi/bin:$PATH"

if [ "$PRE" = 1 ]; then
    f=0
    if git -C "$SRC" cat-file -e "$SHA^{commit}" 2> /dev/null; then echo "PREFLIGHT ok $SHA is in $SRC"
    else echo "PREFLIGHT FAIL $SHA is not in $SRC (tools/cpu_central.sh check seeds it first)"; f=1; fi
    if [ -f "$STEPS" ]; then echo "PREFLIGHT ok steps script $STEPS"
    else echo "PREFLIGHT FAIL no $STEPS in the tools tree (tools/cpu_central.sh tools <sha>)"; f=1; fi
    command -v pixi > /dev/null && echo "PREFLIGHT ok pixi" || { echo "PREFLIGHT FAIL no pixi on the box"; f=1; }
    exit $f
fi

. "$TOOLS/tools/ops/box_cores_lib.sh"
RES=${MOJOLEARN_RESULTS_DIR:?cpu_box_checks_job.sh needs MOJOLEARN_RESULTS_DIR (set by tools/box_persist_run.sh)}
mkdir -p "$RES" && RES="$(cd "$RES" && pwd)"
mkdir -p "$RES/logs" "$RES/time" "$RES/rc" "$RES/parse" "$RES/native" "$BASE/ledger" "$BASE/wt"
TAG=$(basename "$RES")
say() { echo "[$(date -u +%H:%M:%S) cpu-box-checks] $*"; }
FILTER=${NC_FILTER:-*}; FILTER=${FILTER//%/*}
CORES=$(alloc_cores)
TOTAL_KB=$(alloc_mem_total_kb)
BUDGET_KB=$(( TOTAL_KB * 85 / 100 ))
START=$(date +%s)
say "sha $SHA suite $SUITE on $(hostname): $CORES cores, $(( TOTAL_KB / 1048576 )) GiB, memory budget $(( BUDGET_KB / 1048576 )) GiB (85%), tools tree $TOOLS at $(git -C "$TOOLS" rev-parse --short=10 HEAD 2> /dev/null)"
[ -x /usr/bin/time ] || { apt-get install -y -q time > "$RES/logs/apt-time.log" 2>&1 || say "/usr/bin/time unavailable: peaks not measured, estimates stay at their defaults"; }

# the commit's own worktree (tools/cpu_central.sh check seeded the commit into $SRC)
WT=$BASE/wt/$TAG
( flock 9; git -C "$SRC" worktree prune; git -C "$SRC" worktree add -q --detach --force "$WT" "$SHA" ) 9> "$BASE/src.lock" \
    > "$RES/logs/worktree.log" 2>&1 || { say "worktree of $SHA failed: $(tail -n 3 "$RES/logs/worktree.log")"; exit 2; }
cd "$WT" || exit 2
say "worktree $WT; pixi install (default)"
pixi install --locked -e default > "$RES/logs/pixi-install.log" 2>&1 \
    || { say "pixi install failed: $(tail -n 5 "$RES/logs/pixi-install.log")"; exit 2; }
# The workflows' python3 is the runner's 3.12; the box image's is older, so the pixi env's python runs the steps.
[ -x "$WT/.pixi/envs/default/bin/python3" ] && export PATH="$WT/.pixi/envs/default/bin:$PATH"

# ---- plan (the workflow's plan step) ----
case "$SUITE" in parse) IN_SUITE="host-parse" ;; native) IN_SUITE=checks ;; *) IN_SUITE=all ;; esac
if [ "$SUITE" != tools ]; then
    IN_SUITE=$IN_SUITE IN_FILTER=$FILTER IN_COLUMN=cpu bash "$STEPS" plan > "$RES/logs/plan.log" 2>&1 \
        || { say "plan failed: $(tail -n 5 "$RES/logs/plan.log")"; exit 2; }
    mkdir -p "$RES/plan"; cp plan.json planned.txt host.json device.json "$RES/plan/"
    [ "$SUITE" = all ] || [ "$SUITE" = native ] || : > "$RES/plan/planned.txt"
fi
DO_NATIVE=0; DO_PARSE=0; DO_TOOLS=0
case "$SUITE" in native) DO_NATIVE=1 ;; parse) DO_PARSE=1 ;; tools) DO_TOOLS=1 ;; all) DO_NATIVE=1; DO_PARSE=1; DO_TOOLS=1 ;; esac

# ---- the units: name<TAB>class<TAB>estimate kB; the command is built by run_unit from the name ----
GIB=1048576
peak_of() {  # <class> <default kB>: the measured peak of that class on this box, else the default
    local v; v=$(awk -F'\t' -v c="$1" '$1 == c {v = $2} END {print v}' "$BASE/peaks.tsv" 2> /dev/null)
    [[ "$v" =~ ^[0-9]+$ ]] && [ "$v" -gt 0 ] && echo "$v" || echo "$2"
}
QUEUE=()      # pending unit lines, in start order
front() { QUEUE=("$@" "${QUEUE[@]}"); }
[ "$DO_NATIVE" = 1 ] && [ -s "$RES/plan/planned.txt" ] && QUEUE+=("build-cpu	build-cpu	$(peak_of build-cpu $(( 40 * GIB )))")
if [ "$DO_PARSE" = 1 ]; then
    while IFS=$'\t' read -r key; do
        [ -n "$key" ] && QUEUE+=("device-parse-$key	device-parse-$key	$(peak_of "device-parse-$key" $(( 12 * GIB )))")
    done < <(python3 -c 'import json;[print(e["key"]) for e in json.load(open("device.json"))]')
    while IFS=$'\t' read -r key; do
        [ -n "$key" ] && QUEUE+=("host-parse-$key	host-parse-$key	$(peak_of "host-parse-$key" $(( 8 * GIB )))")
    done < <(python3 -c 'import json;[print(e["key"]) for e in json.load(open("host.json"))]')
fi
TT_PEAK=$(peak_of tool-test-file $(( GIB / 2 )))
[ "$DO_TOOLS" = 1 ] && QUEUE+=("tools	tools	$(( TT_PEAK * CORES ))")

matrix_field() {  # <host|device> <key> <field>
    python3 -c 'import json,sys;print(next(e[sys.argv[3]] for e in json.load(open(sys.argv[1] + ".json")) if e["key"] == sys.argv[2]))' "$@"
}
# run_unit <name>: starts the unit in the background FROM THIS SHELL (UNIT_PID), so this shell reaps it (a container's
# pid 1 may not reap an orphan, and kill -0 answers for a zombie); the unit is over when $RES/rc/<name> exists
run_unit() {
    local name=$1 cmd
    case "$name" in
        build-cpu)
            cmd=(env NC_OUT="$RES/native" NC_BIN="$WT/bin" IN_COLUMN=cpu bash "$STEPS" build) ;;
        shard-*)
            cmd=(env TASKS="$(cat "$RES/plan/$name.tasks")" NC_OUT="$RES/native" NC_BIN="$WT/bin" IN_COLUMN=cpu
                 IN_TIMEOUT="${NC_TIMEOUT:-900}" bash "$STEPS" shard) ;;
        host-parse-*|device-parse-*)
            local kind=${name%%-parse-*} key=${name#*-parse-}
            cmd=(env KIND="$kind" KEY="$key" STEM="$(matrix_field "$kind" "$key" stem)" MEMBERS="$(matrix_field "$kind" "$key" members)"
                 PARSE_OUT="$RES/parse/$name" PARSE_LIMIT_SECONDS="${PARSE_LIMIT_SECONDS:-1320}" bash "$STEPS" parse-group) ;;
        tools)
            # tool-tests.yml checks out the whole history (fetch-depth 0) and installs the test env; then the ref's own
            # tools/run_tool_tests.sh with one worker per core
            cmd=(bash -c '
                if [ -f "$0/.git/shallow" ] || [ -f "$(git rev-parse --git-common-dir)/shallow" ]; then
                    ( flock 9; git -C "$0" fetch -q --unshallow origin || echo "TOOLS: the box could not unshallow its repo; tests that read git history may fail where GitHub passes" ) 9> "$1/src.lock"
                fi
                pixi install --locked -e test || exit 2
                MOJOLEARN_TOOL_TESTS_OUT="$2/tools" MOJOLEARN_TEST_PY="$PWD/.pixi/envs/test/bin/python" MOJOLEARN_TEST_WORKERS="$3" \
                    bash tools/run_tool_tests.sh' "$SRC" "$BASE" "$RES" "$CORES") ;;
        *) echo "unknown unit $name" >&2; UNIT_PID=''; return 2 ;;
    esac
    rm -f "$RES/rc/$name"
    (
        if [ -x /usr/bin/time ]; then /usr/bin/time -v -o "$RES/time/$name.txt" "${cmd[@]}"; else "${cmd[@]}"; fi
        echo $? > "$RES/rc/$name"
    ) > "$RES/logs/$name.log" 2>&1 < /dev/null 8>&- 9>&- &   # never inherit the ledger lock
    UNIT_PID=$!
}

# ---- the box-wide memory ledger (shared by every check job on this box) ----
ledger_sum() {
    local s=0 f p kb
    for f in "$BASE"/ledger/*; do
        [ -f "$f" ] || continue
        read -r p kb < "$f" 2> /dev/null || continue
        if kill -0 "$p" 2> /dev/null; then s=$(( s + kb )); else rm -f "$f"; fi
    done
    echo $s
}

# ---- the scheduler ----
declare -A PID
SAMPLER=$(busy_sampler "$RES/busy_cores.txt" 60)
last_report=$(date +%s)
say "plan: ${#QUEUE[@]} units to start (more when the build passes), memory budget $(( BUDGET_KB / GIB )) GiB shared by every job on the box"
while [ ${#QUEUE[@]} -gt 0 ] || [ ${#PID[@]} -gt 0 ]; do
    # reap
    for name in "${!PID[@]}"; do
        if [ -f "$RES/rc/$name" ] || ! kill -0 "${PID[$name]}" 2> /dev/null; then
            wait "${PID[$name]}" 2> /dev/null
            rm -f "$BASE/ledger/$TAG-$name"
            rc=$(cat "$RES/rc/$name" 2> /dev/null || echo none)
            say "done $name rc=$rc"
            unset "PID[$name]"
            if [ "$name" = build-cpu ]; then
                if [ "$rc" = 0 ]; then
                    # the checks dealt into one slice per core (each slice runs its checks one by one, like a shard)
                    n=$(grep -c . "$RES/plan/planned.txt")
                    s=$(( n < CORES ? n : CORES ))
                    est=$(peak_of check $(( 2 * GIB )))
                    lines=()
                    for i in $(seq 0 $(( s - 1 ))); do
                        awk -v s="$s" -v i="$i" 'NF && (NR - 1) % s == i' "$RES/plan/planned.txt" | tr '\n' ' ' > "$RES/plan/shard-$i.tasks"
                        lines+=("shard-$i	check	$est")
                    done
                    front "${lines[@]}"
                    say "build passed: $n checks in $s slices queued first"
                else
                    say "build failed (rc $rc): every planned check reads MISSING"
                fi
            fi
        fi
    done
    # start every queued unit that fits (backfill: a smaller unit may pass a larger one that does not fit yet)
    rest=()
    for line in "${QUEUE[@]}"; do
        IFS=$'\t' read -r name class est <<< "$line"
        started=0
        exec 8> "$BASE/ledger.lock"; flock 8
        used=$(ledger_sum)
        if [ $(( used + est )) -le "$BUDGET_KB" ] || [ "$used" -eq 0 ]; then
            run_unit "$name" || true
            if [ -n "$UNIT_PID" ]; then
                echo "$UNIT_PID $est" > "$BASE/ledger/$TAG-$name"
                PID[$name]=$UNIT_PID; started=1
                say "start $name (estimate $(( est / GIB )) GiB, reserved now $(( (used + est) / GIB )) of $(( BUDGET_KB / GIB )) GiB)"
            fi
        fi
        flock -u 8; exec 8>&-
        [ "$started" = 1 ] || rest+=("$line")
    done
    QUEUE=("${rest[@]}")
    now=$(date +%s)
    if [ $(( now - last_report )) -ge 60 ]; then
        last_report=$now
        say "BUSY $(tail -n 1 "$RES/busy_cores.txt" 2> /dev/null || echo '?') of $CORES cores; running ${#PID[@]} (${!PID[*]}); queued ${#QUEUE[@]}; reserved $(( $(ledger_sum) / GIB )) of $(( BUDGET_KB / GIB )) GiB"
    fi
    sleep 2
done
kill "$SAMPLER" 2> /dev/null
WALL=$(( $(date +%s) - START ))
BUSY=$(busy_of "$RES/busy_cores.txt")

# ---- measured peaks for the next job on this box (largest process of each unit, kB) ----
for t in "$RES"/time/*.txt; do
    [ -f "$t" ] || continue
    name=$(basename "$t" .txt); kb=$(awk -F': *' '/Maximum resident set size/ {print $2}' "$t")
    [[ "$kb" =~ ^[0-9]+$ ]] || continue
    case "$name" in shard-*) class=check ;; tools) class=tool-test-file ;; *) class=$name ;; esac
    printf '%s\t%s\n' "$class" "$kb"
done > "$RES/peaks.tsv"
( flock 9; cat "$RES/peaks.tsv" >> "$BASE/peaks.tsv" ) 9> "$BASE/peaks.lock"

# ---- verdicts, through the same steps the workflow runs ----
TOTAL=0; PASSED=0; FAILED=0; PARTS=(); FIRST=()
if [ "$DO_NATIVE" = 1 ]; then
    cp "$RES/plan/planned.txt" "$RES/native/planned.txt"
    { echo "box: $(hostname), $CORES cores, $(( TOTAL_KB / GIB )) GiB; busy cores of $CORES: $BUSY"; } > "$RES/notes.txt"
    MERGED="$RES/native" SOURCE_SHA=$SHA SUMMARY_NOTES="$RES/notes.txt" bash "$STEPS" summary > "$RES/logs/native-summary.log" 2>&1
    mv -f SUMMARY.md REPORT.txt notes.txt "$RES/native/" 2> /dev/null
    line=$(sed -n 3p "$RES/native/SUMMARY.md" 2> /dev/null)
    read -r n p f m < <(echo "$line" | sed -E 's/checks: ([0-9]+) +passed: ([0-9]+) +failed: ([0-9]+) +missing: ([0-9]+).*/\1 \2 \3 \4/')
    if [[ "${n:-}" =~ ^[0-9]+$ ]]; then
        TOTAL=$(( TOTAL + n )); PASSED=$(( PASSED + p )); FAILED=$(( FAILED + f + m ))
        PARTS+=("native checks: $line")
    else
        TOTAL=$(( TOTAL + 1 )); FAILED=$(( FAILED + 1 )); PARTS+=("native checks: no summary (see logs/native-summary.log)")
    fi
    if [ "$(cat "$RES/rc/build-cpu" 2> /dev/null)" != 0 ]; then
        FIRST+=("- build-cpu: $(grep -m 1 -E 'error' "$RES/native/logs/build-cpu.log" 2> /dev/null | cut -c1-240)")
    fi
    while IFS= read -r g; do FIRST+=("- native $g"); done < <(grep '^## ' "$RES/native/REPORT.txt" 2> /dev/null | head -n 15 | cut -c4-240)
fi
if [ "$DO_PARSE" = 1 ]; then
    for kind in host device; do
        n=0; p=0
        while IFS= read -r key; do
            [ -n "$key" ] || continue
            n=$(( n + 1 ))
            r="$RES/parse/$kind-parse-$key/result-$key.txt"
            k=$(sed -n 's/.*kind=\([a-z]*\).*/\1/p' "$r" 2> /dev/null)
            if [ "$k" = pass ]; then p=$(( p + 1 ))
            else FIRST+=("- $kind-parse $key (${k:-no result}): $(grep -m 1 'error:' "$RES/parse/$kind-parse-$key/$key.log" 2> /dev/null | cut -c1-220)"); fi
        done < <(python3 -c 'import json,sys;[print(e["key"]) for e in json.load(open(sys.argv[1]))]' "$RES/plan/$kind.json")
        TOTAL=$(( TOTAL + n )); PASSED=$(( PASSED + p )); FAILED=$(( FAILED + n - p ))
        PARTS+=("$kind-parse: $n groups, $p passed, $(( n - p )) failed")
    done
    DEVICE_MATRIX=$(cat "$RES/plan/device.json") RETRY_MATRIX='[]' PARSE_DIR="$RES/parse" bash "$STEPS" verdict \
        > "$RES/parse/device-verdict.md" 2>&1
fi
if [ "$DO_TOOLS" = 1 ]; then
    tt=$(grep -m 1 '^TOOL-TESTS ' "$RES/logs/tools.log" 2> /dev/null)
    if [ -n "$tt" ]; then
        tp=$(echo "$tt" | sed -n 's/.*passed=\([0-9]*\).*/\1/p'); tf=$(echo "$tt" | sed -n 's/.* failed=\([0-9]*\).*/\1/p')
        te=$(echo "$tt" | sed -n 's/.*errors=\([0-9]*\).*/\1/p')
        TOTAL=$(( TOTAL + tp + tf + te )); PASSED=$(( PASSED + tp )); FAILED=$(( FAILED + tf + te ))
        PARTS+=("tools: $tt")
        while IFS= read -r l; do FIRST+=("- tools $l"); done < <(grep -E '^  FAILED ' "$RES/logs/tools.log" | head -n 20 | sed 's/^  //' | cut -c1-240)
    else
        TOTAL=$(( TOTAL + 1 )); FAILED=$(( FAILED + 1 ))
        PARTS+=("tools: no result (rc $(cat "$RES/rc/tools" 2> /dev/null || echo none); see logs/tools.log)")
        FIRST+=("- tools: $(tail -n 2 "$RES/logs/tools.log" 2> /dev/null | tr '\n' ' ' | cut -c1-240)")
    fi
fi
grep -h '^TOOLS: ' "$RES/logs/tools.log" 2> /dev/null | head -n 1 | sed 's/^/- /' > "$RES/tool-notes.txt"
{
    echo "# CPU box checks ${SHA:0:10} suite $SUITE"
    echo
    echo "checks: $TOTAL  passed: $PASSED  failed: $FAILED"
    echo
    for l in "${PARTS[@]}"; do echo "- $l"; done
    cat "$RES/tool-notes.txt"
    echo "- busy cores of $CORES: $BUSY (sampled every 60 s); wall ${WALL}s"
    echo "- parallel plan: units admitted by memory (budget $(( BUDGET_KB / GIB )) GiB of $(( TOTAL_KB / GIB )), shared by every job on the box), estimates from measured peaks ($BASE/peaks.tsv; this job's in peaks.tsv)"
    echo "- source $SHA; tools tree $(git -C "$TOOLS" rev-parse --short=10 HEAD 2> /dev/null); box $(hostname)"
    echo
    echo "## first error per failing group"
    if [ ${#FIRST[@]} -eq 0 ]; then echo "none"; else for l in "${FIRST[@]}"; do echo "$l"; done; fi
} > "$RES/SUMMARY.md"
cat "$RES/SUMMARY.md"
# the worktree goes (its results are in $RES); the shared repo and the measured peaks stay
cd / || true
( flock 9; git -C "$SRC" worktree remove --force "$WT" ) 9> "$BASE/src.lock" > /dev/null 2>&1 || true
[ "$FAILED" -eq 0 ]
