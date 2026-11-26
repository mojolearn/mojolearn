#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# tools/ops/gpu_smoke_job.sh -- THE GPU VERIFY SMOKE of a measurement build on the L40S pod (lane gpu-smoke,
# 2026-11-25; Andrew: the wheel verify smoke runs on the GPU side too, cheap and fast, after every build, as part of
# the flow). The CPU smoke (.github/workflows/verify-smoke.yml) installs the core wheel alone on a CPU runner and so
# never reaches device="gpu"; the first GPU run of main 0becfc8f6b found 527 refused lanes there
# (~/mojolearn-evidence/briefs-20261122/gpu-diagnosis-RESULT.md). This job is tools/ops/wheel_verify_smoke.sh in GPU
# mode (SMOKE_SIDE=gpu) on the staged NVIDIA wheels: install (core + nvidia, vendor() must be cuda), imports, verify
# --self-test, every lane on BOTH sides, the test dependencies, the Python tests against the GPU install. It is NOT the
# identity verification: it compares nothing across vendors, records nothing and writes no reference.
#
#   tools/nvidia_central.sh submit <lane> --cap 45 --results box-out/gsmoke-<sha12> \
#       --env GS_WHEEL_DIR=/root/pc-in/<sha12>/nvidia tools/ops/gpu_smoke_job.sh
# GS_WHEEL_DIR holds exactly the build's two wheels (one mojolearn_core-*.whl, one mojolearn_nvidia-*.whl), as
# `tools/ops/measure_candidate.py stage nvidia` leaves them; the lane tree must be the build's commit (the tests and
# the harness provenance come from it): `tools/release.py candidate` prints the stage, sync and submit lines.
# Optional: GS_GPU_MB_PER_WORKER (device MB a worker; default the smoke's GPU_MEM_PER_WORKER_MB), GS_STEPS (the
# smoke's STEPS), GS_MEMORY_FROM (an earlier GPU smoke's results dir: plan each pool from its measured seconds and RSS).
#
# PARALLEL PLAN (AGENTS.md hard rules: 100 percent of rented CPU, parallel by default): one process per unit (public
# module, lane, test file), workers = the core allocation (cgroup quota, not the 128 host CPUs) capped by the GPU's
# free memory at GS_GPU_MB_PER_WORKER a worker (46 GB / 2 GB = 20 on an L40S, so the 14-core pod's cores are the
# limit); each unit starts when its measured RSS fits. The header line `# parallel plan:` in SUMMARY.txt states it.
# This job prints busy cores of the allocation (15 s sampler) and the GPU's peak memory used and mean utilisation
# (nvidia-smi every 15 s, logs/gpu.log), so the next run can set GS_GPU_MB_PER_WORKER from a measurement.
#
# ESTIMATE: about 12 to 18 minutes on one L40S, 14 cores, well under the 30-minute target. From the CPU smoke's
# committed plan (tools/ops/wheel_verify_smoke_plan/, a 4-core runner): lanes 1410 s of unit time for 510 lanes, tests
# 2760 s (the one 1084 s file is split one unit per test), imports 33 s. Both sides roughly double the lanes and each
# GPU process adds a device context (assume 3 to 5 s a lane): about 1410 x 2 + 530 x 4 = 4900 s / 14 = 6 min; the tests
# with their GPU tests now running (assume 1.5x) about 4100 s / 14 = 5 min; install (two wheels, about 1 GB) and the
# test dependencies about 3 min, the torch download overlapping the earlier steps; selftest 1 min. Unmeasured on a GPU:
# the first run's SUMMARY.txt (`seconds=` per pool) replaces this estimate. --cap 45 leaves room for one slow pool and
# the crash reruns (each capped at 600 s by the smoke).
#
# Output: $MOJOLEARN_RESULTS_DIR/smoke/ (SUMMARY.txt, per-pool results.tsv, logs, lanes/SIDES.tsv, crashes/), and one
# line per smoke step `GSMOKE <step> PASS|FAIL ...` on stdout. Exit: the smoke's (0 only when every step passed); 4 when
# the inputs are refused before anything ran.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
. "$HERE/tree_commit.sh"

# job-preflight: `gpu_smoke_job.sh --preflight` checks the job's inputs at SUBMIT time, reading only (run on the box by
# tools/ops/job_preflight.sh from tools/nvidia_central.sh submit): GS_WHEEL_DIR holds one core and one nvidia wheel and
# nothing else, their local version names this tree's commit (the synced one, tools/ops/tree_commit.sh; the tests and the harness run from this tree), the smoke
# script is in the tree, nvidia-smi answers, and MOJOLEARN_RESULTS_DIR is set. The job keeps the same checks at run time.
gs_wheels() { ls "${GS_WHEEL_DIR:-/nonexistent}"/mojolearn_core-*.whl "${GS_WHEEL_DIR:-/nonexistent}"/mojolearn_nvidia-*.whl 2> /dev/null; }
gs_preflight() {   # PREFLIGHT lines; 0 when the job would start, 4 otherwise
    local fails=0 n head w12
    pf_fail() { echo "PREFLIGHT FAIL $*"; fails=$(( fails + 1 )); }
    if [ -z "${GS_WHEEL_DIR:-}" ]; then pf_fail "GS_WHEEL_DIR is not set (the staged wheel dir of the build, /root/pc-in/<sha12>/nvidia)"
    elif [ ! -d "$GS_WHEEL_DIR" ]; then pf_fail "GS_WHEEL_DIR $GS_WHEEL_DIR: no such directory on the box (stage the build first)"
    else
        n=$(gs_wheels | grep -c .)
        if [ "$n" -eq 2 ] && [ "$(ls "$GS_WHEEL_DIR"/*.whl 2> /dev/null | grep -c .)" -eq 2 ]; then
            echo "PREFLIGHT ok wheels in $GS_WHEEL_DIR: $(gs_wheels | xargs -n 1 basename | tr '\n' ' ')"
        else
            pf_fail "GS_WHEEL_DIR $GS_WHEEL_DIR holds $n of the 2 wheels (one mojolearn_core, one mojolearn_nvidia, nothing else)"
        fi
        # the tree's commit is the synced one (.git/devpod_synced), not the box HEAD, which a sync leaves at the merge
        # base with main (tools/ops/tree_commit.sh)
        while read -r verdict rest; do
            if [ "$verdict" = MATCH ]; then echo "PREFLIGHT ok wheel $rest"
            else pf_fail "$rest (sync the lane tree to the build's commit)"; fi
        done < <(wheels_match_tree "$REPO" $(gs_wheels))
    fi
    [ -f "$REPO/tools/ops/wheel_verify_smoke.sh" ] && echo "PREFLIGHT ok smoke $REPO/tools/ops/wheel_verify_smoke.sh" \
        || pf_fail "no tools/ops/wheel_verify_smoke.sh in $REPO"
    if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
        echo "PREFLIGHT ok gpu $(nvidia-smi -L 2> /dev/null | head -n 1 | cut -c1-120)"
    else
        pf_fail "nvidia-smi does not answer: no GPU for a GPU smoke"
    fi
    [ -n "${MOJOLEARN_RESULTS_DIR:-}" ] || pf_fail "MOJOLEARN_RESULTS_DIR is not set"
    [ "$fails" -eq 0 ] && return 0 || return 4
}
if [ "${1:-}" = --preflight ]; then gs_preflight; exit $?; fi

RES="${MOJOLEARN_RESULTS_DIR:?gpu_smoke_job.sh needs MOJOLEARN_RESULTS_DIR (set by tools/box_persist_run.sh)}"
mkdir -p "$RES/logs" && RES="$(cd "$RES" && pwd)"
pf=$(gs_preflight); prc=$?
printf '%s\n' "$pf" | sed 's/^PREFLIGHT/GSMOKE preflight/'
[ "$prc" -eq 0 ] || { echo "GSMOKE inputs refused; nothing ran"; exit 4; }
. "$HERE/box_cores_lib.sh"
echo "GSMOKE allocation: $(alloc_cores) core(s) of $(nproc) visible, $(( $(alloc_mem_kb) / 1048576 )) GiB available"
echo "GSMOKE gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2> /dev/null | head -n 1)"

# The venv and the downloaded extras live outside the results dir (it is uploaded to R2) and go when the job ends.
SCRATCH="$(mktemp -d "${TMPDIR:-/root}/gsmoke-XXXXXX")"
gpu_sampler() {   # logs/gpu.log: "<memory.used MB>, <utilization.gpu %>" every 15 s; part of this job, not a watcher
    ( while sleep 15; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2> /dev/null | head -n 1; done ) \
        > "$RES/logs/gpu.log" 2> /dev/null &
    echo $!
}
sp=$(busy_sampler "$RES/logs/smoke.busy" 15)
gp=$(gpu_sampler)
trap 'kill "$sp" "$gp" 2> /dev/null; rm -rf "$SCRATCH"' EXIT

t0=$SECONDS
[ -n "${GS_STEPS:-}" ] && export STEPS="$GS_STEPS"
SMOKE_SIDE=gpu VENV="$SCRATCH/venv" SOURCE_COMMIT="$(git -C "$REPO" rev-parse HEAD 2> /dev/null || true)" \
    SMOKE_GPU_MB_PER_WORKER="${GS_GPU_MB_PER_WORKER:-}" MEMORY_FROM="${GS_MEMORY_FROM:-}" TMPDIR="$SCRATCH" \
    bash "$HERE/wheel_verify_smoke.sh" "$GS_WHEEL_DIR" "$RES/smoke" > "$RES/logs/smoke.log" 2>&1
rc=$?
kill "$sp" "$gp" 2> /dev/null; wait "$sp" "$gp" 2> /dev/null

S="$RES/smoke/SUMMARY.txt"
grep -E '^# (gpu|parallel plan|allocation)' "$S" 2> /dev/null | sed 's/^# /GSMOKE /'
# one line per step: its verdict and the first 200 characters of its summary line
grep -E '^[a-z-]+ rc=' "$S" 2> /dev/null | while read -r step rest; do
    case "$rest" in rc=0\ *) v=PASS ;; *) v=FAIL ;; esac
    echo "GSMOKE $step $v $(printf '%s' "$rest" | cut -c1-200)"
done
grep -E '^  lanes sides:' "$S" 2> /dev/null | sed 's/^  /GSMOKE /'
echo "GSMOKE busy cores $(busy_of "$RES/logs/smoke.busy") of $(alloc_cores); $((SECONDS - t0))s"
awk -F', *' '{if ($1 > m) m = $1; u += $2; n++} END {if (n) printf "GSMOKE gpu memory.used max %d MB, utilisation mean %.0f%% (%d samples)\n", m, u / n, n; else print "GSMOKE gpu not sampled"}' "$RES/logs/gpu.log"
printf 'smoke\t%s\t%s\t\t\tbusy cores %s of %s\n' "$rc" "$((SECONDS - t0))" "$(busy_of "$RES/logs/smoke.busy")" "$(alloc_cores)" > "$RES/steps.tsv"
# The in-repo evidence (verify-complete, Andrew 2026-11-26): SUMMARY.txt and the failed lanes and tests of every pool,
# staged under evidence/nvidia/smoke; the Mac's fetch (`tools/box_persist.py fetch --tag <tag> --extract --evidence`)
# writes it into verification/latest/nvidia/smoke/.
python3 "$REPO/tools/verification_evidence.py" stage --results "$RES" --column nvidia --part smoke \
    --out "$RES/evidence/nvidia/smoke" || echo "GSMOKE evidence not staged"
echo "GSMOKE done rc=$rc summary=$S"
exit "$rc"
