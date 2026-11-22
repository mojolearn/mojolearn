#!/bin/bash
# tools/ops/cpu_box_checks_job.sh <sha> <suite> -- the native-checks.yml and tool-tests.yml checks of one commit on the
# rented CPU box (tools/cpu_central.sh check), every group at once across all the box's cores (lane fx-cpu-box-checks,
# 2026-11-19; Andrew: "why are we waiting on things? we use runpod?": a GitHub run waited 40 minutes for runners).
#
#   suite parse   every release host group and `identical` device group parsed and elaborated (host-parse, device-parse)
#         native  the CPU column's native checks: one build of the column's TestSuite group programs, then every check
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
# run at once, admitted by MEMORY through tools/ops/box_mem_sched.sh: a unit starts when the box-wide reservations
# (every concurrent check job on the box shares one ledger, $CPU_CHECKS_HOME/ledger) plus its estimate fit in 85 percent
# of the allocation's memory; a unit that does not fit WAITS until memory frees (2026-11-20: six checks queued behind
# cpc1-0001 each ended in a minute with every unit "no result"); a unit larger than the budget runs when nothing else is
# reserved, and only one larger than the whole allocation is "cannot fit: needs N GiB of M". Each unit is itself
# multi-threaded on every core (the compiler at
# -j nproc, run_tool_tests.sh workers = cores), so memory, not cores, bounds the count. Estimates are MEASURED: each unit
# runs under /usr/bin/time -v and its peak RSS goes to $CPU_CHECKS_HOME/peaks.tsv; the next job takes the largest of the
# last three of each class;
# until a group has a measurement: host parse 8 GiB, device parse 12 GiB (the release recorded 10-12.5), checks build
# 40 GiB (the GPU column's build peaked at 43 GB), a check 2 GiB (largest 1.64 GB), a tool test file 0.5 GiB x cores.
# Order: the checks build first (it gates every check), then device groups, host groups, tool tests; the check slices
# go to the front when the build passes. Busy cores of total are printed every minute and in SUMMARY.md.
#
# Results ($MOJOLEARN_RESULTS_DIR, persisted to R2 by tools/box_persist_run.sh): SUMMARY.md (line 3: checks: N passed: N
# failed: N not run: N; line 4: result: green | red: ...; lines 5-7: verdict compile|native|tools: green (N) | red: ...
# | not in this suite, so a compile result reads apart from red tool tests; then each part, the first error per failed
# group and the reason per unit not run),
# parse/<host|device>-parse-<key>/ (the parse step's log, result and memory files), native/ (results, logs, SUMMARY.md,
# REPORT.txt), tools/ (run_tool_tests.sh's logs and SUMMARY.txt), logs/<unit>.log, time/<unit>.txt, busy_cores.txt.
# Exit 0 when everything passed, 1 when anything failed, 3 when nothing failed but something was not run (or nothing was
# checked), 2 on a setup failure.
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
# Fixture cases are tracked as one archive each (tools/fixture_packs.py): recreate the case directories.
[ -f tools/fixture_packs.py ] && python3 tools/fixture_packs.py unpack
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
peak_of() {  # <class> <default kB>: the largest of the last 3 measured peaks of that class on this box, else the default
    local v; v=$(awk -F'\t' -v c="$1" '$1 == c && $2 ~ /^[0-9]+$/ {a = b; b = d; d = $2} END {m = a; if (b > m) m = b; if (d > m) m = d; if (m > 0) print m}' "$BASE/peaks.tsv" 2> /dev/null)
    [[ "$v" =~ ^[0-9]+$ ]] && [ "$v" -gt 0 ] && echo "$v" || echo "$2"
}
peak_src() {  # <class>: where its estimate comes from
    local n; n=$(awk -F'\t' -v c="$1" '$1 == c && $2 ~ /^[0-9]+$/ {n++} END {print n + 0}' "$BASE/peaks.tsv" 2> /dev/null)
    [ "${n:-0}" -gt 0 ] && echo "measured on this box, largest of the last $(( n < 3 ? n : 3 )) of $n in peaks.tsv" || echo "default, not yet measured"
}
QUEUE=()      # pending unit lines, in start order
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
[ "$DO_TOOLS" = 1 ] && QUEUE+=("tools	tool-test-file	$(( TT_PEAK * CORES ))")

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
                # --filter (NC_FILTER, comma globs) selects tool test files by basename; it was not passed
                # here, so a filtered tools check ran every file (fx-tooltests, 2026-11-21).
                files=()
                if [ "$4" != "*" ]; then
                    IFS=, read -ra globs <<< "$4"
                    for p in tools/test_*.py tools/tests/test_*.py; do
                        for g in "${globs[@]}"; do
                            case "${p##*/}" in $g) files+=("$p"); break ;; esac
                        done
                    done
                    [ "${#files[@]}" -gt 0 ] || { echo "TOOLS: --filter $4 matches no tool test file" >&2; exit 2; }
                fi
                MOJOLEARN_TOOL_TESTS_OUT="$2/tools" MOJOLEARN_TEST_PY="$PWD/.pixi/envs/test/bin/python" MOJOLEARN_TEST_WORKERS="$3" \
                    bash tools/run_tool_tests.sh "${files[@]}"' "$SRC" "$BASE" "$RES" "$CORES" "$FILTER") ;;
        *) echo "unknown unit $name" >&2; UNIT_PID=''; return 2 ;;
    esac
    rm -f "$RES/rc/$name"
    (
        if [ -x /usr/bin/time ]; then /usr/bin/time -v -o "$RES/time/$name.txt" "${cmd[@]}"; else "${cmd[@]}"; fi
        echo $? > "$RES/rc/$name"
    ) > "$RES/logs/$name.log" 2>&1 < /dev/null 8>&- 9>&- &   # never inherit the ledger lock
    UNIT_PID=$!
}

# ---- the scheduler: tools/ops/box_mem_sched.sh (one ledger shared by every check job on this box; a unit that does
# not fit WAITS for memory, it never ends the job unrun) ----
. "$TOOLS/tools/ops/box_mem_sched.sh"
SCHED_HOME=$BASE; SCHED_TAG=$TAG; BUSY_FILE="$RES/busy_cores.txt"
sched_on_done() {  # <name> <rc>: the check slices are queued first once the build passes
    local name=$1 rc=$2 n s est i lines
    [ "$name" = build-cpu ] || return 0
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
        [ ${#lines[@]} -eq 0 ] || sched_front "${lines[@]}"
        say "build passed: $n checks in $s slices queued first (estimate $(sched_gib "$est") GiB each)"
    else
        say "build failed (rc $rc): the planned checks are not run; the summary names the build's first error"
    fi
}
sched_pending_note() {
    if [ -n "${SCHED_PID[build-cpu]:-}" ]; then echo "the check slices are queued when build-cpu passes"; fi
}
SAMPLER=$(busy_sampler "$BUSY_FILE" 60)
# the plan: every unit with its estimate (measured on this box, else the default) against the shared budget
sum=0
for line in ${QUEUE[@]+"${QUEUE[@]}"}; do
    IFS=$'\t' read -r name class est <<< "$line"; sum=$(( sum + est ))
    say "plan unit $name: estimate $(sched_gib "$est") GiB ($(peak_src "$class"))"
done
say "PARALLEL PLAN: ${#QUEUE[@]} units now$([ "$DO_NATIVE" = 1 ] && echo " + up to $CORES check slices after the build"), estimates sum $(sched_gib "$sum") GiB;" \
    "memory budget $(sched_gib "$BUDGET_KB") GiB of $(sched_gib "$TOTAL_KB") shared by every job on the box; $CORES cores." \
    "Units start as memory frees (waiting, never skipped); busy cores of total every minute with the reason when below 80%"
sched_loop
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
# Three outcomes, never folded: PASSED, FAILED (it ran and failed; its first error is listed) and NOT RUN (it never ran
# or left no result: cannot fit, the build it needs failed, or the job was cut short; its reason is listed).
TOTAL=0; PASSED=0; FAILED=0; NOTRUN=0; PARTS=(); FIRST=(); NOTRUN_WHY=()
# Per-part counts for the three verdict lines (lane fx-box-tooltests, 2026-11-20): compile = build-cpu and the host
# and device parse groups; native = the native checks; tools = the tool tests. A lane's compile result stays readable
# when only the tool tests are red. "" = the part is not in this suite.
CT=""; CF=0; CN=0; NT=""; NF=0; NN=0; TT=""; TF=0; TN=0
why_not_run() {  # <unit>: the reason a unit has no result
    if [ -s "$RES/notrun/$1" ]; then cat "$RES/notrun/$1"
    elif [ ! -f "$RES/rc/$1" ]; then echo "never started (the job ended or was cancelled first)"
    else echo "ran (rc $(cat "$RES/rc/$1")) but left no result"; fi
}
first_error() {  # <log> [<log>...]: the first error line, else the last two lines of the first log that exists
    local f l
    for f in "$@"; do [ -s "$f" ] || continue; l=$(grep -m 1 -iE 'error' "$f" | cut -c1-220); [ -z "$l" ] || { echo "$l"; return; }; done
    for f in "$@"; do [ -s "$f" ] || continue; tail -n 2 "$f" | tr '\n' ' ' | cut -c1-220; return; done
    echo "no log"
}
if [ "$DO_NATIVE" = 1 ]; then
    cp "$RES/plan/planned.txt" "$RES/native/planned.txt"
    { echo "box: $(hostname), $CORES cores, $(( TOTAL_KB / GIB )) GiB; busy cores of $CORES: $BUSY"; } > "$RES/notes.txt"
    MERGED="$RES/native" SOURCE_SHA=$SHA SUMMARY_NOTES="$RES/notes.txt" bash "$STEPS" summary > "$RES/logs/native-summary.log" 2>&1
    mv -f SUMMARY.md REPORT.txt notes.txt "$RES/native/" 2> /dev/null
    planned=$(grep -c . "$RES/plan/planned.txt" 2> /dev/null || echo 0)
    brc=$(cat "$RES/rc/build-cpu" 2> /dev/null || echo none)
    # the build is a unit of its own: passed, failed (first error) or not run
    if [ "$planned" -gt 0 ]; then
        TOTAL=$(( TOTAL + 1 )); CT=$(( ${CT:-0} + 1 ))
        case "$brc" in
            0) PASSED=$(( PASSED + 1 )) ;;
            none) NOTRUN=$(( NOTRUN + 1 )); CN=$(( CN + 1 )); NOTRUN_WHY+=("- build-cpu: $(why_not_run build-cpu)") ;;
            *) FAILED=$(( FAILED + 1 )); CF=$(( CF + 1 )); FIRST+=("- build-cpu (rc $brc): $(first_error "$RES/native/logs/build-cpu.log" "$RES/logs/build-cpu.log")") ;;
        esac
    fi
    line=$(sed -n 3p "$RES/native/SUMMARY.md" 2> /dev/null)
    read -r n p f m < <(echo "$line" | sed -E 's/checks: ([0-9]+) +passed: ([0-9]+) +failed: ([0-9]+) +missing: ([0-9]+).*/\1 \2 \3 \4/')
    if [[ "${n:-}" =~ ^[0-9]+$ ]] && [[ "${m:-}" =~ ^[0-9]+$ ]]; then
        # a planned check without a result was not run (the build failed or was not run, or its slice never started)
        TOTAL=$(( TOTAL + n )); PASSED=$(( PASSED + p )); FAILED=$(( FAILED + f )); NOTRUN=$(( NOTRUN + m ))
        NT=$n; NF=$f; NN=$m
        PARTS+=("native checks: $n planned, $p passed, $f failed, $m not run")
        if [ "$m" -gt 0 ]; then
            if [ "$brc" != 0 ]; then NOTRUN_WHY+=("- $m native checks: build-cpu $([ "$brc" = none ] && echo 'was not run' || echo "failed (rc $brc)")")
            else
                for t in "$RES"/plan/shard-*.tasks; do
                    [ -f "$t" ] || continue; u=$(basename "$t" .tasks)
                    [ -f "$RES/rc/$u" ] || NOTRUN_WHY+=("- native $u ($(wc -w < "$t" | tr -d ' ') checks): $(why_not_run "$u")")
                done
                NOTRUN_WHY+=("- $m native checks have no result file in total (native/REPORT.txt lists them)")
            fi
        fi
    else
        TOTAL=$(( TOTAL + planned )); NOTRUN=$(( NOTRUN + planned )); NT=$planned; NN=$planned
        PARTS+=("native checks: $planned planned, no summary (logs/native-summary.log): counted not run")
        NOTRUN_WHY+=("- native checks: the summary step left no counts ($(first_error "$RES/logs/native-summary.log"))")
    fi
    while IFS= read -r g; do FIRST+=("- native $g"); done < <(grep '^## ' "$RES/native/REPORT.txt" 2> /dev/null | head -n 15 | cut -c4-240)
fi
if [ "$DO_PARSE" = 1 ]; then
    for kind in host device; do
        n=0; p=0; f=0; r0=0
        while IFS= read -r key; do
            [ -n "$key" ] || continue
            n=$(( n + 1 )); u="$kind-parse-$key"
            r="$RES/parse/$u/result-$key.txt"
            k=$(sed -n 's/.*kind=\([a-z]*\).*/\1/p' "$r" 2> /dev/null)
            if [ "$k" = pass ]; then p=$(( p + 1 ))
            elif [ -n "$k" ] || [ -f "$RES/rc/$u" ]; then   # it ran: a non-pass result, or no result at all
                f=$(( f + 1 )); FIRST+=("- $u (${k:-no result, rc $(cat "$RES/rc/$u" 2> /dev/null)}): $(first_error "$RES/parse/$u/$key.log" "$RES/logs/$u.log")")
            else
                r0=$(( r0 + 1 )); NOTRUN_WHY+=("- $u: $(why_not_run "$u")")
            fi
        done < <(python3 -c 'import json,sys;[print(e["key"]) for e in json.load(open(sys.argv[1]))]' "$RES/plan/$kind.json")
        TOTAL=$(( TOTAL + n )); PASSED=$(( PASSED + p )); FAILED=$(( FAILED + f )); NOTRUN=$(( NOTRUN + r0 ))
        CT=$(( ${CT:-0} + n )); CF=$(( CF + f )); CN=$(( CN + r0 ))
        PARTS+=("$kind-parse: $n groups, $p passed, $f failed, $r0 not run")
    done
    DEVICE_MATRIX=$(cat "$RES/plan/device.json") RETRY_MATRIX='[]' PARSE_DIR="$RES/parse" bash "$STEPS" verdict \
        > "$RES/parse/device-verdict.md" 2>&1
fi
if [ "$DO_TOOLS" = 1 ]; then
    tt=$(grep -m 1 '^TOOL-TESTS ' "$RES/logs/tools.log" 2> /dev/null)
    if [ -n "$tt" ]; then
        tp=$(echo "$tt" | sed -n 's/.*passed=\([0-9]*\).*/\1/p'); tf=$(echo "$tt" | sed -n 's/.* failed=\([0-9]*\).*/\1/p')
        te=$(echo "$tt" | sed -n 's/.*errors=\([0-9]*\).*/\1/p')
        tp=${tp:-0}; tf=${tf:-0}; te=${te:-0}
        TOTAL=$(( TOTAL + tp + tf + te )); PASSED=$(( PASSED + tp )); FAILED=$(( FAILED + tf + te ))
        TT=$(( tp + tf + te )); TF=$(( tf + te ))
        PARTS+=("tools: $tt")
        while IFS= read -r l; do FIRST+=("- tools $l"); done < <(grep -E '^  FAILED ' "$RES/logs/tools.log" | head -n 20 | sed 's/^  //' | cut -c1-240)
    elif [ -f "$RES/rc/tools" ] && [ ! -s "$RES/notrun/tools" ]; then
        TOTAL=$(( TOTAL + 1 )); FAILED=$(( FAILED + 1 )); TT=1; TF=1
        PARTS+=("tools: ran (rc $(cat "$RES/rc/tools")) with no TOOL-TESTS line: counted failed")
        FIRST+=("- tools (rc $(cat "$RES/rc/tools")): $(first_error "$RES/logs/tools.log")")
    else
        TOTAL=$(( TOTAL + 1 )); NOTRUN=$(( NOTRUN + 1 )); TT=1; TN=1
        PARTS+=("tools: not run")
        NOTRUN_WHY+=("- tools: $(why_not_run tools)")
    fi
fi
grep -h '^TOOLS: ' "$RES/logs/tools.log" 2> /dev/null | head -n 1 | sed 's/^/- /' > "$RES/tool-notes.txt"
if [ "$FAILED" -eq 0 ] && [ "$NOTRUN" -eq 0 ] && [ "$TOTAL" -gt 0 ]; then RESULT="green"; EXIT=0
elif [ "$FAILED" -gt 0 ]; then RESULT="red: $FAILED failed, $NOTRUN not run"; EXIT=1
elif [ "$TOTAL" -eq 0 ]; then RESULT="red: nothing was checked"; EXIT=3
else RESULT="red: $NOTRUN not run"; EXIT=3; fi
part_verdict() {  # <total or ""> <failed> <not run>: green | red: ... | not in this suite
    if [ -z "$1" ]; then echo "not in this suite"
    elif [ "$1" -eq 0 ]; then echo "red: nothing was checked"
    elif [ "$2" -gt 0 ]; then echo "red: $2 of $1 failed, $3 not run"
    elif [ "$3" -gt 0 ]; then echo "red: $3 of $1 not run"
    else echo "green ($1)"; fi
}
{
    echo "# CPU box checks ${SHA:0:10} suite $SUITE"
    echo
    echo "checks: $TOTAL  passed: $PASSED  failed: $FAILED  not run: $NOTRUN"
    echo "result: $RESULT"
    echo "verdict compile: $(part_verdict "$CT" "$CF" "$CN")"
    echo "verdict native: $(part_verdict "$NT" "$NF" "$NN")"
    echo "verdict tools: $(part_verdict "$TT" "$TF" "$TN")"
    echo
    for l in "${PARTS[@]}"; do echo "- $l"; done
    cat "$RES/tool-notes.txt"
    echo "- busy cores of $CORES: $BUSY (sampled every 60 s); wall ${WALL}s"
    echo "- parallel plan: units admitted by memory (budget $(( BUDGET_KB / GIB )) GiB of $(( TOTAL_KB / GIB )), one ledger shared by every job on the box; a unit waits for memory, never skipped), estimates from measured peaks ($BASE/peaks.tsv; this job's in peaks.tsv)"
    echo "- source $SHA; tools tree $(git -C "$TOOLS" rev-parse --short=10 HEAD 2> /dev/null); box $(hostname)"
    echo
    echo "## first error per failed group"
    if [ ${#FIRST[@]} -eq 0 ]; then echo "none"; else for l in "${FIRST[@]}"; do echo "$l"; done; fi
    echo
    echo "## not run"
    if [ ${#NOTRUN_WHY[@]} -eq 0 ]; then echo "none"; else for l in "${NOTRUN_WHY[@]}"; do echo "$l"; done; fi
} > "$RES/SUMMARY.md"
cat "$RES/SUMMARY.md"
# the worktree goes (its results are in $RES); the shared repo and the measured peaks stay
cd / || true
( flock 9; git -C "$SRC" worktree remove --force "$WT" ) 9> "$BASE/src.lock" > /dev/null 2>&1 || true
exit "$EXIT"
