#!/bin/bash
# tools/cpu_box_dev_build_and_tests.sh -- the development build + the whole Python suite on a RunPod CPU pod, every core.
#   bash tools/runpod_cpu_leg.sh --lane devtests-<ref> --worktree <checkout of ref> --cmd-file <this file> --envs default,test --vcpu 32 --flavors cpu5m,cpu3m --disk 80 --lease 150 --rent
#   comm -13 <main leg out>/fail.txt <branch leg out>/fail.txt   # failures the branch has and main does not
#
# Shipped with --cmd-file, so the same script runs against ANY ref (main included,
# even before this file exists there). Run it once on main and once on the branch;
# the two fail.txt files are the comparison. It replaces the Mac for this work:
# the Mac is for Apple only, and its build (79 groups, one compiler, 93 min) and
# suite (17,739 tests in one process, 55 min) were serial.
#
# What it does, in order (cwd = the unpacked repo, $LEG_OUT comes back to the Mac):
#   1. box_resources.txt: nproc, MemTotal/MemAvailable, the cgroup limits.
#      load.tsv: a sampler every 30 s (busy cores from /proc/stat, MemAvailable),
#      so the report can state busy cores of total and per-worker memory is measured.
#   2. The development build, groups in parallel at the box's core count limited by
#      each group's measured memory: `development_build.py build --sets identical,host
#      --parallel auto`. A tree whose development_build.py has no --parallel (main
#      before the scheduler lands) gets `--jobs <nproc/2> --memory-per-job-gib
#      $DEV_FALLBACK_GIB` (default 18: the largest Linux group peak recorded, 12.3 GB
#      when stopped by the 12 GiB limit, times the tool's 1.5 headroom for a limited
#      peak); the tool itself lowers the count to what memory allows.
#      Linux identical groups need one accelerator target: DEV_GPU_ARCH (default sm_89).
#      Logs dev_build.log, evidence dev/, report dev.json; the recorded per-group peaks
#      are copied to dev-peaks/ so the Linux RSS table can be filled from them.
#   3. The suite: the test-python task (`tools/run_python_tests.py -- --continue-on-collection-errors`
#      in the test env; pytest-xdist, every core, then the serial pass).
#      Workers: MOJOLEARN_TEST_WORKERS if set; else nproc, lowered to
#      0.8*MemAvailable / TEST_WORKER_GIB when TEST_WORKER_GIB (a measured per-worker
#      peak from an earlier load.tsv) is given. A test env without pytest-xdist (a lock
#      from before it was added) gets pytest-xdist 3.8.0 + execnet 2.1.2 from pip,
#      noted in xdist_install.log; the run stays parallel and says so.
#   4. tests.log (complete), tests.rc, fail.txt (sorted unique `FAILED` ids),
#      errors.txt (sorted unique `ERROR` ids), summary.txt (the pytest totals line,
#      the rcs, workers, busy cores of total).
# Exit status: the suite's, or the build's when the build failed (then no suite runs:
# tests on a partial index would only measure the missing bindings).
set -u

[ -n "${LEG_OUT:-}" ] || { echo "LEG_OUT is not set (run through tools/runpod_cpu_leg.sh)" >&2; exit 2; }
mkdir -p "$LEG_OUT"
[ -f tools/development_build.py ] || cd /root/mojolearn || exit 2
R=$(pwd)
DEFAULT_PY="$R/.pixi/envs/default/bin/python3"
TEST_PY="$R/.pixi/envs/test/bin/python3"
CORES=$(nproc)
note() { printf '%s\t%s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$LEG_OUT/summary.txt"; }

# 1. resources and the load sampler -------------------------------------------
{
    echo "nproc	$CORES"
    grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
    for f in /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max; do
        [ -r "$f" ] && echo "$f	$(cat "$f")"
    done
    echo "commit	${MOJOLEARN_COMMIT:-unknown}"
} > "$LEG_OUT/box_resources.txt" 2>&1

sample_load() {
    printf 'utc\tbusy_cores\tcores\tmem_available_kib\tphase\n' > "$LEG_OUT/load.tsv"
    read -r _ a1 b1 c1 d1 e1 f1 g1 h1 _ < /proc/stat
    while :; do
        sleep 30
        read -r _ a2 b2 c2 d2 e2 f2 g2 h2 _ < /proc/stat
        busy=$(( (a2 + b2 + c2 + f2 + g2 + h2) - (a1 + b1 + c1 + f1 + g1 + h1) ))
        total=$(( busy + (d2 + e2) - (d1 + e1) ))
        avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        phase=$(cat "$LEG_OUT/.phase" 2>/dev/null || echo -)
        awk -v b="$busy" -v t="$total" -v n="$CORES" -v m="$avail" -v p="$phase" -v u="$(date -u +%H:%M:%SZ)" \
            'BEGIN { printf "%s\t%.1f\t%d\t%s\t%s\n", u, (t > 0 ? n * b / t : 0), n, m, p }' >> "$LEG_OUT/load.tsv"
        a1=$a2 b1=$b2 c1=$c2 d1=$d2 e1=$e2 f1=$f2 g1=$g2 h1=$h2
    done
}
sample_load &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null' EXIT

busy_summary() {  # mean and max busy cores while phase $1 ran
    awk -F'\t' -v p="$1" 'NR > 1 && $5 == p { s += $2; n++; if ($2 > m) m = $2 }
        END { if (n) printf "mean %.1f max %.1f of %d cores (%d samples)", s / n, m, $3, n; else print "no samples" }' \
        "$LEG_OUT/load.tsv"
}

# 2. the development build -----------------------------------------------------
echo build > "$LEG_OUT/.phase"
export MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS="${DEV_GPU_ARCH:-sm_89}"
if "$DEFAULT_PY" tools/development_build.py --help 2>/dev/null | grep -q -- '--parallel'; then
    BUILD_PAR=(--parallel auto)
else
    BUILD_PAR=(--jobs $(( CORES / 2 > 0 ? CORES / 2 : 1 )) --memory-per-job-gib "${DEV_FALLBACK_GIB:-18}")
fi
note "build: sets identical,host GPU_ARCHS=$MOJOLEARN_GPU_ARCHS ${BUILD_PAR[*]}"
t0=$(date +%s)
"$DEFAULT_PY" tools/development_build.py build --sets identical,host "${BUILD_PAR[@]}" \
    --evidence "$LEG_OUT/dev" --json "$LEG_OUT/dev.json" > "$LEG_OUT/dev_build.log" 2>&1
BUILD_RC=$?
echo "$BUILD_RC" > "$LEG_OUT/dev_build.rc"
[ -d "$HOME/.cache/mojolearn-development/peaks" ] && cp -r "$HOME/.cache/mojolearn-development/peaks" "$LEG_OUT/dev-peaks"
note "build: rc=$BUILD_RC seconds=$(( $(date +%s) - t0 )) busy $(busy_summary build)"
if [ "$BUILD_RC" != 0 ]; then
    grep -m 5 -E 'error|development_build:' "$LEG_OUT/dev_build.log" >> "$LEG_OUT/summary.txt"
    note "suite: NOT RUN (the development build failed; see dev_build.log)"
    exit "$BUILD_RC"
fi

# 3. the suite -----------------------------------------------------------------
echo tests > "$LEG_OUT/.phase"
if ! "$TEST_PY" -c 'import xdist' > /dev/null 2>&1; then
    "$TEST_PY" -m pip install --no-deps 'pytest-xdist==3.8.0' 'execnet==2.1.2' > "$LEG_OUT/xdist_install.log" 2>&1
    note "suite: the test env had no pytest-xdist; pip install rc=$? (xdist_install.log)"
fi
if [ -z "${MOJOLEARN_TEST_WORKERS:-}" ]; then
    WORKERS=$CORES
    if [ -n "${TEST_WORKER_GIB:-}" ]; then
        BY_MEM=$(awk -v g="$TEST_WORKER_GIB" '/^MemAvailable:/ { n = int(0.8 * $2 / (g * 1048576)); print (n < 1 ? 1 : n) }' /proc/meminfo)
        [ "$BY_MEM" -lt "$WORKERS" ] && { WORKERS=$BY_MEM; note "suite: workers lowered to $WORKERS by TEST_WORKER_GIB=$TEST_WORKER_GIB"; }
    fi
    export MOJOLEARN_TEST_WORKERS=$WORKERS
fi
# A tree whose runner predates the parallel default gets the same flags by hand.
if grep -q MOJOLEARN_TEST_WORKERS tools/run_python_tests.py; then
    SUITE_PAR=()
else
    SUITE_PAR=(-n "$MOJOLEARN_TEST_WORKERS" --dist loadfile)
fi
note "suite: workers=$MOJOLEARN_TEST_WORKERS ${SUITE_PAR[*]}"
t0=$(date +%s)
# The test-python task's own command, in the test env, without `pixi run`: a run could
# re-sync the env and drop a pip-added pytest-xdist (the task defines no env vars).
PATH="$R/.pixi/envs/test/bin:$PATH" CONDA_PREFIX="$R/.pixi/envs/test" PIXI_ENVIRONMENT_NAME=test \
    MOJOLEARN_ORCHESTRATOR=1 "$TEST_PY" tools/run_python_tests.py -- --continue-on-collection-errors "${SUITE_PAR[@]}" \
    > "$LEG_OUT/tests.log" 2>&1
TESTS_RC=$?
echo "$TESTS_RC" > "$LEG_OUT/tests.rc"
echo finished > "$LEG_OUT/.phase"

# 4. the comparison files ------------------------------------------------------
grep -E '^FAILED ' "$LEG_OUT/tests.log" | sed -E 's/ - .*//' | sort -u > "$LEG_OUT/fail.txt"
grep -E '^ERROR ' "$LEG_OUT/tests.log" | sed -E 's/ - .*//' | sort -u > "$LEG_OUT/errors.txt"
note "suite: rc=$TESTS_RC seconds=$(( $(date +%s) - t0 )) busy $(busy_summary tests)"
# One totals line per pass (parallel, then serial when any test is marked serial).
grep -E '(passed|failed).* in [0-9.]+s' "$LEG_OUT/tests.log" | while IFS= read -r line; do note "suite: $line"; done
note "suite: failed ids $(wc -l < "$LEG_OUT/fail.txt"), error ids $(wc -l < "$LEG_OUT/errors.txt")"
exit "$TESTS_RC"
