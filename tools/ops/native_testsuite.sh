#!/bin/bash
# tools/ops/native_testsuite.sh build|run -- THE TESTSUITE CHECK FILES of checks/native_tests/ (lane ts-pilot,
# 2026-11-22; docs/plans/TESTSUITE_CONVERSION.md). A file whose second line is `# native-check: <columns>` is one check:
# its `test_*` functions run by the standard library's TestSuite from its own main(). No wrapper, no registry row.
# Run with the checkout under test as the working directory (as tools/ops/native_checks_steps.sh is).
#
#   build <column> <bindir> <outdir>   compile every TestSuite file of the column to <bindir>/ts/<name>, in parallel:
#                                      workers = min(cores, memory available x 0.8 / NC_TS_PEAK_MB) (the measured peak
#                                      of one file's compile, default 1200 MB: 0.2-1.1 GB measured on cpc1, 2026-11-22),
#                                      NC_TS_JOBS pins it. A file that fails to compile gets its result now
#                                      (<outdir>/results/<name>.txt, the compile's first errors) and no binary; the
#                                      others build on. Flags: NC_DEFINES and NC_TARGET (the column's, from the caller).
#                                      Exit 0 even when a file failed (its own row is red); 2 on a setup failure.
#   run <column> <bindir> <outdir> <name>
#                                      run one TestSuite check as one process (timeout IN_TIMEOUT, default 900 s) and
#                                      write <outdir>/results/<name>.txt from its Summary line. A CRASH (signal, no
#                                      summary: the binary's buffered output is lost, checked 2026-11-22) is localised:
#                                      the tests are listed with `--skip-all` and each runs alone with `--only <test>`;
#                                      a test that crashes alone is reported `CRASHED <test> rc=<rc>` (the result's first
#                                      error). When the file never compiled (its build result exists, no binary) the
#                                      build's result stands.
#   is <name>                          exit 0 when <name> is a TestSuite check file
#
# Parallel plan (AGENTS.md hard rules): one compile per file, every file at once up to the worker count; the runs are
# one process per file, dealt across the cores by the caller (cpu_box_checks_job.sh slices, native-checks.yml shards,
# native_checks_gpu_job.sh xargs). Inside a file TestSuite runs its tests in order (the library's loop); a case table
# inside one test runs across the cores (checks/native_tests/testsuite_cases.mojo check_cases).
set -u
cmd=${1:-}
PY=python3
python3 -c 'import sys' 2> /dev/null || PY="pixi run python3"

case "$cmd" in
is)
    # checks/native_tests/<name>.mojo, or a standalone check at the top of checks/ (lane ts-b9, 2026-11-22)
    for f in "checks/native_tests/${2:?is <name>}.mojo" "checks/$2.mojo"; do
        head -n 2 "$f" 2> /dev/null | sed -n 2p | grep -qE '^# native-check:( (cpu|gpu))+[[:space:]]*$' && exit 0
    done
    exit 1 ;;
build)
    column=${2:?build <column> <bindir> <outdir>}; bindir=${3:?}; out=${4:?}
    mkdir -p "$bindir/ts" "$out/results" "$out/logs"
    list="$out/logs/testsuite-$column.txt"
    $PY tools/ops/native_checks_report.py testsuite --column "$column" --filter "${IN_FILTER:-*}" > "$list" || exit 2
    n=$(grep -c . "$list")
    [ "$n" -gt 0 ] || { echo "testsuite: no TestSuite file on the $column column"; exit 0; }
    cores=$(nproc 2> /dev/null || getconf _NPROCESSORS_ONLN)
    avail_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2> /dev/null)
    [ -n "$avail_kb" ] || avail_kb=$(( 16 * 1048576 ))
    bymem=$(( avail_kb * 8 / 10 / (${NC_TS_PEAK_MB:-1200} * 1024) ))
    jobs=${NC_TS_JOBS:-$(( bymem < cores ? bymem : cores ))}
    [ "$jobs" -ge 1 ] || jobs=1
    echo "testsuite: building $n files on the $column column, $jobs at once ($cores cores; memory allows $bymem at ${NC_TS_PEAK_MB:-1200} MB)"
    export TS_BIN="$bindir/ts" TS_OUT="$out" TS_PY="$PY" TS_DEFINES="${NC_DEFINES:-}" TS_TARGET="${NC_TARGET:-}"
    start=$(date +%s)
    awk '{print $1, $2}' "$list" | xargs -P "$jobs" -n 2 bash -c '
        name=$0; src=$1; log="$TS_OUT/logs/build-ts-$name.log"; t0=$(date +%s)
        pixi run mojo build $TS_DEFINES $TS_TARGET -I . -o "$TS_BIN/$name" "$src" > "$log" 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then
            rm -f "$TS_BIN/$name"
            $TS_PY tools/ops/native_checks_report.py result "$name" --log "$log" --rc "$rc" \
                --seconds "$(( $(date +%s) - t0 ))" --out-dir "$TS_OUT/results"
        fi
        echo "testsuite build $name rc=$rc $(( $(date +%s) - t0 ))s"'
    failed=$(awk '{print $1}' "$list" | while read -r x; do [ -x "$bindir/ts/$x" ] || printf '%s ' "$x"; done)
    echo "testsuite: $n files built in $(( $(date +%s) - start ))s; failed to compile: ${failed:-none}"
    exit 0 ;;
run)
    column=${2:?run <column> <bindir> <outdir> <name>}; bindir=${3:?}; out=${4:?}; name=${5:?}
    mkdir -p "$out/results" "$out/logs"
    bin="$bindir/ts/$name"
    case "$bin" in /*) ;; *) bin="./$bin" ;; esac
    log="$out/logs/$name.log"
    if [ ! -x "$bin" ]; then
        # the build wrote this check's result (its compile errors); without one the check reads MISSING
        [ -f "$out/results/$name.txt" ] && echo "$name: no binary (compile failed, result kept)" && exit 0
        echo "$name: no binary and no build result" > "$log"
        $PY tools/ops/native_checks_report.py result "$name" --log "$log" --rc 2 --seconds 0 --out-dir "$out/results"
        exit 0
    fi
    limit=${IN_TIMEOUT:-900}
    start=$(date +%s)
    timeout -k 30 "$limit" pixi run "$bin" > "$log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ] && ! grep -qE '^Summary \[' "$log"; then
        # a crash before the summary: name the test that crashes alone
        echo "--- crash localisation (rc $rc, no TestSuite summary): each test alone with --only ---" >> "$log"
        pixi run "$bin" --skip-all 2> /dev/null | sed -n 's/^ *SKIP \[[^]]*\] *//p' > "$log.tests"
        # each test alone, every test at once up to the cores (independent processes)
        export TS_RUN="$bin" TS_LOG="$log" TS_LIMIT="$limit"
        xargs -a "$log.tests" -P "$(nproc 2> /dev/null || echo 4)" -I{} bash -c '
            timeout -k 30 "$TS_LIMIT" pixi run "$TS_RUN" --only "$0" > "$TS_LOG.$0" 2>&1
            trc=$?
            if [ "$trc" -ne 0 ] && ! grep -qE "^Summary \[" "$TS_LOG.$0"; then
                echo "CRASHED $0 rc=$trc (alone, no summary)" > "$TS_LOG.$0.res"
            else
                echo "alone $0 rc=$trc: $(grep -E "^ *(PASS|FAIL|SKIP) \[" "$TS_LOG.$0" | head -n 1 | sed "s/^ *//")" > "$TS_LOG.$0.res"
            fi' {}
        while read -r t; do cat "$log.$t.res" >> "$log"; rm -f "$log.$t" "$log.$t.res"; done < "$log.tests"
        rm -f "$log.tests"
        grep -q '^CRASHED' "$log" || echo "CRASHED (rc $rc) with every test passing alone: an order dependence between tests" >> "$log"
    fi
    $PY tools/ops/native_checks_report.py result "$name" --log "$log" --rc "$rc" \
        --seconds "$(( $(date +%s) - start ))" --out-dir "$out/results"
    echo "$name rc=$rc"
    exit 0 ;;
*)
    sed -n 2,27p "$0"; exit 2 ;;
esac
