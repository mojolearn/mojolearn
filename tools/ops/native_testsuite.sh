#!/bin/bash
# tools/ops/native_testsuite.sh build|run -- THE TESTSUITE CHECK FILES of checks/native_tests/ (lane ts-pilot,
# 2026-11-22; docs/plans/TESTSUITE_CONVERSION.md). A file whose second line is `# native-check: <columns>` is one check:
# its `test_*` functions run by the standard library's TestSuite from its own main(). No wrapper and no registry.
# Run with the checkout under test as the working directory (as tools/ops/native_checks_steps.sh is).
#
#   build <column> <bindir> <outdir>   compile every TestSuite file of the column, GROUPED (lane ts-build, 2026-11-22;
#                                      tools/ops/native_testsuite_groups.py): the files are dealt into G group programs
#                                      (families together, balanced by tools/ops/native_testsuite_costs.tsv), each one
#                                      `mojo build`, so the code its members share compiles once per group, not once
#                                      per file. <bindir>/ts/<name> is a two-line wrapper that runs its group's binary
#                                      (<bindir>/tsg/group_<k>) with MOJOLEARN_TS_FILE=<name>: same command line
#                                      (--only, --skip, --skip-all), same output, one process per file. A group that
#                                      fails to compile is rebuilt one file at a time, so a file that fails gets its
#                                      own result (<outdir>/results/<name>.txt, the compile's first errors) and no
#                                      binary, and its group's other files build on. The plan's singles (a file that
#                                      cannot share a program) compile alone, as before.
#                                      G and the compiles at once = min(cores, memory available x 0.8 /
#                                      NC_TS_GROUP_PEAK_MB) (the measured peak of one group compile: 7.0 GB host column,
#                                      13.2 GB device column on cpc2; defaults 7500 / 13500 MB);
#                                      NC_TS_GROUPS / NC_TS_JOBS pin them; a single file's compile is budgeted at
#                                      NC_TS_PEAK_MB (default 1200: 0.2-1.1 GB measured on cpc1). The step prepares
#                                      what the files need itself: the generated sources (tools/generated_sources.py),
#                                      the fixture packs (tools/fixture_packs.py unpack), the column's defines
#                                      (NC_DEFINES, default: identical, plus the host column's MOJOLEARN_COLUMN_CPU=1 or
#                                      the device column's MOJOLEARN_RESIDENT_UPLOAD_COUNT=1) and the wheel's x86-64-v3
#                                      target (NC_TARGET, default that baseline on x86_64; set empty for none).
#                                      Exit 0 even when a file failed (its own row is red); 2 on a setup failure.
#                                      NC_TS_CACHE=<dir> (the box jobs set it) reuses a group program or single
#                                      whose inputs did not change since an earlier build on the box, and
#                                      NC_TS_URLS (a map the Mac staged at submit) since a build on any box, through
#                                      R2 (tools/ops/native_testsuite_cache.py); unset, everything compiles.
#   one <column> [-D DEFINE ...] <name> ...
#                                      build each named file alone (the build step's defines and target, plus the given
#                                      defines) and run it, its output on stdout; exit 0 when every one passed (the
#                                      pixi tasks check-native-cpu / check-native-gpu, the e1 scripts)
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
# Parallel plan (AGENTS.md hard rules): G group compiles (G about the allocation's cores, fewer when memory says so) at
# once, each a multi-threaded `mojo build` of 1/G of the files; the runs are
# one process per file, dealt across the cores by the caller (cpu_box_checks_job.sh slices, native-checks.yml shards,
# native_checks_gpu_job.sh xargs). Inside a file TestSuite runs its tests in order (the library's loop); a case table
# inside one test runs across the cores (checks/native_tests/testsuite_cases.mojo check_cases).
set -u
cmd=${1:-}
PY=python3
python3 -c 'import sys' 2> /dev/null || PY="pixi run python3"

# What every TestSuite file needs before it compiles: the gitignored generated modules (the archive Unicode table, the
# tokenizer table) and the fixture directories tracked as packs. Logs to <dir>.
prepare() {
    $PY tools/generated_sources.py prepare --repo "$PWD" < /dev/null > "$1/generated-sources.log" 2>&1 \
        || { echo "testsuite: generated sources failed ($1/generated-sources.log)"; return 1; }
    if [ -f tools/fixture_packs.py ]; then
        $PY tools/fixture_packs.py unpack > "$1/fixture-unpack.log" 2>&1 \
            || { echo "testsuite: fixture unpack failed ($1/fixture-unpack.log)"; return 1; }
    fi
}

# The column's defines and the target: the caller's NC_DEFINES / NC_TARGET, else the ones the column is checked with.
column_flags() {
    local d="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
    if [ "$1" = cpu ]; then d="$d -D MOJOLEARN_COLUMN_CPU=1"; else d="$d -D MOJOLEARN_RESIDENT_UPLOAD_COUNT=1"; fi
    # The wheel's Linux x86-64 baseline (bindings/build_mamba.sh TARGET_FLAGS): without it the binary targets the
    # build machine's chip and traps on an older one (native_checks_steps.sh build, run 43722561124).
    local t="--target-cpu x86-64-v3 --target-features +avx,+avx2,+bmi,+bmi2,+cx16,+f16c,+fma,+lzcnt,+movbe,+popcnt,+sahf,+sse3,+sse4.1,+sse4.2,+ssse3,+xsave"
    [ "$(uname -m)" = x86_64 ] || t=""
    TS_DEFINES=${NC_DEFINES-$d}; TS_TARGET=${NC_TARGET-$t}
}

# One compile of the build: `group <k>` (the generated driver of group k) or `single <name>` (the file alone). A
# single that fails writes its result row now. Run by xargs, so exported.
compile_unit() {
    local kind=$1 id=$2 log src dst t0 rc tm key miss=
    t0=$(date +%s)
    if [ "$kind" = group ]; then
        log="$TS_OUT/logs/build-ts-group-$id.log"; dst="$TS_BIN/tsg/group_$id"; src="$TS_GEN/ts_group_$id.mojo"
    else
        log="$TS_OUT/logs/build-ts-$id.log"; dst="$TS_BIN/ts/$id"
        src="checks/native_tests/$id.mojo"; [ -f "$src" ] || src="checks/$id.mojo"
    fi
    tm="$log.time"
    # a program whose inputs did not change is copied from NC_TS_CACHE (tools/ops/native_testsuite_cache.py: the key
    # is every source file it imports, the flags and the toolchain); TS_KEYS is empty when there is no cache
    key=""
    [ -s "${TS_KEYS:-}" ] && key=$(awk -v k="$kind" -v i="$id" '$1 == k && $2 == i {print $3; exit}' "$TS_KEYS")
    if [ -n "$key" ] && $TS_PY tools/ops/native_testsuite_cache.py get "$key" "$dst" --fields "$TS_GEN/keys" > "$log.get" 2>&1; then
        mv "$log.get" "$log"
        local from=local; grep -q '^native_testsuite_cache: R2 hit' "$log" && from=r2
        echo "reused from NC_TS_CACHE ($from), key $key" >> "$log"
        echo "testsuite build $kind $id reused ${key:0:12} from $from $(( $(date +%s) - t0 ))s"
        return 0
    fi
    if [ -x /usr/bin/time ]; then
        /usr/bin/time -v -o "$tm" pixi run mojo build $TS_FLAGS -I . -I "$TS_GEN" -o "$dst" "$src" > "$log" 2>&1
    else
        pixi run mojo build $TS_FLAGS -I . -I "$TS_GEN" -o "$dst" "$src" > "$log" 2>&1
    fi
    rc=$?
    # the lookup's lines (the miss report: the nearest stored key and the first differing field) stay in the log
    if [ -f "$log.get" ]; then cat "$log.get" >> "$log"; miss=$(grep -m 1 '^native_testsuite_cache: R2 miss report:' "$log.get"); rm -f "$log.get"; fi
    [ -n "${miss:-}" ] && echo "testsuite build $kind $id ${miss#native_testsuite_cache: }"
    [ $rc -eq 0 ] && [ -n "$key" ] && $TS_PY tools/ops/native_testsuite_cache.py put "$key" "$dst" --fields "$TS_GEN/keys" >> "$log" 2>&1
    if [ $rc -ne 0 ]; then
        rm -f "$dst"
        [ "$kind" = single ] && $TS_PY tools/ops/native_checks_report.py result "$id" --log "$log" --rc "$rc" \
            --seconds "$(( $(date +%s) - t0 ))" --out-dir "$TS_OUT/results"
    fi
    echo "testsuite build $kind $id rc=$rc $(( $(date +%s) - t0 ))s $(awk -F': *' '/Maximum resident/ {printf "peak %d MB", $2/1024}' "$tm" 2> /dev/null)"
}
export -f compile_unit

case "$cmd" in
is)
    # checks/native_tests/<name>.mojo, or a standalone check at the top of checks/ (lane ts-b9, 2026-11-22)
    for f in "checks/native_tests/${2:?is <name>}.mojo" "checks/$2.mojo"; do
        head -n 2 "$f" 2> /dev/null | sed -n 2p | grep -qE '^# native-check:( (cpu|gpu))+[[:space:]]*$' && exit 0
    done
    exit 1 ;;
build)
    column=${2:?build <column> <bindir> <outdir>}; bindir=${3:?}; out=${4:?}
    mkdir -p "$bindir/ts" "$bindir/tsg" "$out/results" "$out/logs"
    case "$bindir" in /*) ;; *) bindir="$PWD/$bindir" ;; esac
    prepare "$out/logs" || exit 2
    column_flags "$column"
    list="$out/logs/testsuite-$column.txt"
    $PY tools/ops/native_checks_report.py testsuite --column "$column" --filter "${IN_FILTER:-*}" > "$list" || exit 2
    n=$(grep -c . "$list")
    [ "$n" -gt 0 ] || { echo "testsuite: no TestSuite file on the $column column"; exit 0; }
    # Cores are the ALLOCATION (cgroup quota), never the host's visible CPUs: on the L40S nproc says 128 on a 14-core
    # pod, 44 compiles started at once, forks failed ("Resource temporarily unavailable") and every pixi hung
    # (nvc1-0003, 2026-11-22). Same helper the other box jobs use.
    . "$(dirname "$0")/box_cores_lib.sh"
    cores=$(alloc_cores)
    avail_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2> /dev/null)
    [ -n "$avail_kb" ] || avail_kb=$(( 16 * 1048576 ))
    # one group compile's measured peak (cpc2, 2026-11-22, 16 groups: host column 7.0 GB, device column 13.2 GB)
    gdef=7500; [ "$column" = cpu ] || gdef=13500
    gpeak=${NC_TS_GROUP_PEAK_MB:-$gdef}
    bymem=$(( avail_kb * 8 / 10 / (gpeak * 1024) ))
    jobs=${NC_TS_JOBS:-$(( bymem < cores ? bymem : cores ))}
    [ "$jobs" -ge 1 ] || jobs=1
    groups=${NC_TS_GROUPS:-$jobs}
    gen="$bindir/tsgen"
    $PY tools/ops/native_testsuite_groups.py plan --column "$column" --list "$list" --groups "$groups" --gen "$gen" \
        > "$out/logs/testsuite-plan-$column.txt" || { echo "testsuite: group plan failed"; exit 2; }
    ng=$(grep -c '^group ' "$gen/plan.txt"); ns=$(grep -c '^single ' "$gen/plan.txt")
    echo "testsuite: building $n files on the $column column as $ng group programs and $ns single files, $jobs compiles at once ($cores cores; memory allows $bymem at $gpeak MB a group)"
    export TS_BIN="$bindir" TS_OUT="$out" TS_PY="$PY" TS_FLAGS="$TS_DEFINES $TS_TARGET" TS_GEN="$gen"
    # NC_TS_CACHE (a directory on the box; unset = every program compiles): one key per planned unit. A failure to
    # key is not a build failure: the units compile.
    export TS_KEYS=""
    if [ -n "${NC_TS_CACHE:-}" ]; then
        if $PY tools/ops/native_testsuite_cache.py keys --column "$column" --gen "$gen" --flags "$TS_FLAGS" \
                > "$gen/keys.txt" 2> "$out/logs/testsuite-keys-$column.log"; then
            TS_KEYS="$gen/keys.txt"
            echo "testsuite: reuse from $NC_TS_CACHE ($(grep -c . "$gen/keys.txt") of $(( ng + ns )) units keyed)"
        else
            echo "testsuite: no reuse, the keys failed (logs/testsuite-keys-$column.log): $(tail -n 1 "$out/logs/testsuite-keys-$column.log")"
        fi
    fi
    start=$(date +%s)
    # every group and every planned single at once, up to the worker count (independent compiles)
    { grep '^group ' "$gen/plan.txt" | awk '{print "group", $2}'; grep '^single ' "$gen/plan.txt" | awk '{print "single", $2}'; } \
        | xargs -P "$jobs" -n 2 bash -c 'compile_unit "$0" "$1"' | tee "$out/logs/testsuite-units-$column.txt"
    # a group that did not compile: its files one at a time, so each failing file has its own red row
    : > "$out/logs/testsuite-regroup-$column.txt"
    while read -r _ k members; do
        [ -x "$bindir/tsg/group_$k" ] && continue
        echo "testsuite: group $k did not compile (logs/build-ts-group-$k.log); its $(wc -w <<< "$members") files one at a time"
        printf 'single %s\n' $members >> "$out/logs/testsuite-regroup-$column.txt"
    done < <(grep '^group ' "$gen/plan.txt")
    # (a file alone is smaller than a group: its own memory budget, NC_TS_PEAK_MB)
    bymem1=$(( avail_kb * 8 / 10 / (${NC_TS_PEAK_MB:-1200} * 1024) ))
    jobs1=$(( bymem1 < cores ? bymem1 : cores )); [ "$jobs1" -ge 1 ] || jobs1=1
    : > "$out/logs/testsuite-units-regroup-$column.txt"
    [ -s "$out/logs/testsuite-regroup-$column.txt" ] && \
        xargs -P "$jobs1" -n 2 bash -c 'compile_unit "$0" "$1"' < "$out/logs/testsuite-regroup-$column.txt" \
        | tee "$out/logs/testsuite-units-regroup-$column.txt"
    # bin/ts/<name>: the group members' wrappers (singles are their own binary there already)
    while read -r _ k members; do
        [ -x "$bindir/tsg/group_$k" ] || continue
        for m in $members; do
            printf '#!/bin/sh\n# %s is a member of group %s (tools/ops/native_testsuite_groups.py)\nexport MOJOLEARN_TS_FILE=%s\nexec "$(dirname "$0")/../tsg/group_%s" "$@"\n' \
                "$m" "$k" "$m" "$k" > "$bindir/ts/$m"
            chmod +x "$bindir/ts/$m"
        done
    done < <(grep '^group ' "$gen/plan.txt")
    failed=$(awk '{print $1}' "$list" | while read -r x; do [ -x "$bindir/ts/$x" ] || printf '%s ' "$x"; done)
    regrouped=$(awk '{print $2}' "$out/logs/testsuite-regroup-$column.txt" | wc -l)
    echo "testsuite: $n files built in $(( $(date +%s) - start ))s ($ng groups, $ns planned singles, $regrouped files of failed groups rebuilt alone, $(grep -c ' reused ' "$out/logs/testsuite-units-$column.txt") units reused from the cache); failed to compile: ${failed:-none}"
    # THE CACHE IN ONE LINE (rel-fix-1, 2026-11-26), the last line of the build: units planned, keyed, hits (from the
    # box's directory, from R2), misses (keyed units that compiled), compiles (every `mojo build` this build ran,
    # failed groups' files included) and R2 uploads. A second box proving the R2 path reads hits=N r2=N compiles=0.
    units_f="$out/logs/testsuite-units-$column.txt"; regroup_f="$out/logs/testsuite-units-regroup-$column.txt"
    keyed=0; [ -s "${TS_KEYS:-}" ] && keyed=$(grep -c . "$TS_KEYS")
    hl=$(grep -c ' reused .* from local ' "$units_f"); hr=$(grep -c ' reused .* from r2 ' "$units_f")
    compiles=$(cat "$units_f" "$regroup_f" | grep -c ' rc=')
    uploads=$(cat "$out"/logs/build-ts-*.log 2> /dev/null | grep -c '^native_testsuite_cache: R2 uploaded')
    echo "testsuite cache: column=$column units=$(( ng + ns )) keyed=$keyed hits=$(( hl + hr )) local=$hl r2=$hr misses=$(( keyed - hl - hr )) compiles=$compiles uploads=$uploads"
    exit 0 ;;
one)
    column=${2:?one <column> [-D DEFINE ...] <name> [<name> ...]}; shift 2
    extra=""; names=()
    while [ $# -gt 0 ]; do
        case "$1" in -D) extra="$extra -D ${2:?-D needs a define}"; shift 2 ;; *) names+=("$1"); shift ;; esac
    done
    [ "${#names[@]}" -gt 0 ] || { echo "one: name at least one TestSuite file"; exit 2; }
    work=${NC_ONE_DIR:-build/native-one}; mkdir -p "$work"
    prepare "$work" || exit 2
    column_flags "$column"
    worst=0
    for name in "${names[@]}"; do
        src=""
        for f in "checks/native_tests/$name.mojo" "checks/$name.mojo"; do [ -f "$f" ] && src=$f && break; done
        [ -n "$src" ] || { echo "no TestSuite file named $name"; worst=2; continue; }
        if ! pixi run mojo build $TS_DEFINES $extra $TS_TARGET -I . -o "$work/$name" "$src" > "$work/build-$name.log" 2>&1; then
            echo "$name: compile failed ($work/build-$name.log)"; grep -m 5 -A 6 'error' "$work/build-$name.log"; worst=1; continue
        fi
        pixi run "$work/$name" || worst=1
    done
    exit "$worst" ;;
run)
    column=${2:?run <column> <bindir> <outdir> <name>}; bindir=${3:?}; out=${4:?}; name=${5:?}
    mkdir -p "$out/results" "$out/logs"
    bin="$bindir/ts/$name"
    case "$bin" in /*) ;; *) bin="./$bin" ;; esac
    log="$out/logs/$name.log"
    # an artifact download drops the exec bits (GitHub's upload-artifact keeps none): restore them, the wrapper's group too
    if [ -f "$bin" ] && [ ! -x "$bin" ]; then
        chmod +x "$bin" "$(dirname "$bin")"/../tsg/group_* 2> /dev/null
    fi
    if [ ! -x "$bin" ]; then
        # the build wrote this check's result (its compile errors); without one the check reads MISSING
        [ -f "$out/results/$name.txt" ] && echo "$name: no binary (compile failed, result kept)" && exit 0
        echo "$name: no binary and no build result" > "$log"
        $PY tools/ops/native_checks_report.py result "$name" --log "$log" --rc 2 --seconds 0 --out-dir "$out/results"
        exit 0
    fi
    limit=${IN_TIMEOUT:-900}
    start=$(date +%s)
    # a program reused from NC_TS_CACHE was linked in another checkout and its runpath names that checkout's
    # toolchain: this checkout's comes first (the same toolchain, it is in the key; no effect on a fresh build)
    [ -d "$PWD/.pixi/envs/default/lib" ] && export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    # THAT HOLDS ONLY FOR DT_RUNPATH (rel-fix-1, review of 3423ae229e..8f0d9a9642 item 6): the loader searches a
    # DT_RPATH BEFORE LD_LIBRARY_PATH, so a program linked with an RPATH would load the libraries of the checkout that
    # built it. `mojo build` writes a RUNPATH (measured for the bindings, tools/development_build.py group_fields), and
    # the program's link flags are only TS_FLAGS (defines and target): nothing asks for old dtags. Still, a program
    # (a member's wrapper is resolved to its group program) with an RPATH outside this checkout is refused here
    # rather than run against another checkout's libraries.
    prog=$bin
    if head -c 2 "$prog" 2> /dev/null | grep -q '^#!'; then
        g=$(grep -o 'tsg/group_[0-9]*' "$prog" | head -n 1); [ -n "$g" ] && prog="$(dirname "$bin")/../$g"
    fi
    if command -v readelf > /dev/null 2>&1; then
        rpath=$(readelf -d "$prog" 2> /dev/null | sed -n 's/.*(RPATH).*\[\(.*\)\]/\1/p')
        outside=$(tr ':' '\n' <<< "$rpath" | grep -v '^$' | grep -v -e "^$PWD/" -e '^\$ORIGIN' | head -n 1)
        if [ -n "$outside" ]; then
            echo "$name: refused: $prog has a DT_RPATH outside this checkout ($outside); LD_LIBRARY_PATH cannot override it" > "$log"
            $PY tools/ops/native_checks_report.py result "$name" --log "$log" --rc 2 --seconds 0 --out-dir "$out/results"
            echo "$name rc=2 (refused: DT_RPATH)"
            exit 0
        fi
    fi
    # under /usr/bin/time when the box has it: the check's peak RSS goes into its result (the GPU job's K and summary)
    if [ -x /usr/bin/time ]; then
        /usr/bin/time -v -o "$log.time" timeout -k 30 "$limit" pixi run "$bin" > "$log" 2>&1
    else
        timeout -k 30 "$limit" pixi run "$bin" > "$log" 2>&1
    fi
    rc=$?
    rss=$(awk -F': *' '/Maximum resident set size/ {print $2}' "$log.time" 2> /dev/null); rm -f "$log.time"
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ] && ! grep -qE '^Summary \[' "$log"; then
        # a crash before the summary: name the test that crashes alone
        echo "--- crash localisation (rc $rc, no TestSuite summary): each test alone with --only ---" >> "$log"
        pixi run "$bin" --skip-all 2> /dev/null | sed -n 's/^ *SKIP \[[^]]*\] *//p' > "$log.tests"
        # each test alone, every test at once up to the cores (independent processes)
        . "$(dirname "$0")/box_cores_lib.sh"; export TS_CORES=$(alloc_cores)
        export TS_RUN="$bin" TS_LOG="$log" TS_LIMIT="$limit"
        xargs -a "$log.tests" -P "${TS_CORES:-4}" -I{} bash -c '
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
        --seconds "$(( $(date +%s) - start ))" ${rss:+--maxrss-kb "$rss"} --out-dir "$out/results"
    echo "$name rc=$rc"
    exit 0 ;;
*)
    sed -n "2,/^# Parallel plan/p" "$0"; exit 2 ;;
esac
