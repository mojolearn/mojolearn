#!/bin/bash
# tools/ops/identity_record_job.sh -- the NVIDIA identity record of a measurement build on the L40S pod (lane
# identity-record-job, 2026-11-25; night-decisions m5: `nvidia_central.sh submit` takes one script and no arguments,
# and before this file the only caller of tools/identity_wheel_record.py was the release gate job). Three steps on the
# staged wheels of one build, each printing `IDREC <name> PASS|FAIL rc=<n> <s>s` on one line, full output in
# logs/<name>.log. A failed step never stops the others; the job exits 1 when any failed.
#   (i)  verify: `python -I -m mojolearn verify --all --json-out $MOJOLEARN_RESULTS_DIR/verify-all.json`, from an empty
#        directory, on a venv holding the GPU install (the v2 report holds the CPU and GPU side of every cell from one
#        process);
#   (i b) verify-neural: the same with `--neural-training` (`--json-out .../verify-neural.json`): the neural-training
#        lanes `verify --all` leaves out by profile, so the two runs cover every lane;
#   (ii) record: `python3 tools/identity_wheel_record.py --columns nvidia --label-device l40s --wheel ... --sha256 ...
#        --gpus N --workers-per-gpu auto --out <scratch>/identity` (no reuse flag: every lane is recorded), copied to
#        $MOJOLEARN_RESULTS_DIR/identity (the recorder refuses an outdir inside the tree, and the results dir is in it).
#        The recorder requires the wheels' source commit to equal this tree's HEAD (it runs the harness of this tree).
#
# PARALLEL PLAN (AGENTS.md hard rules: 100 percent of rented CPU, parallel by default). Both steps use the one GPU, so
# they run one after the other; inside each, the work spreads over the allocation:
#   verify: its cell workers default to the process's core allocation (affinity and cgroup cpu.max), bounded by
#           available memory at 4 GiB a worker (mojolearn._verify_all.verify_cell_workers); no --cell-workers pin.
#   record: --workers-per-gpu auto: cores / 3 CPU threads per worker, bounded by the GPU's free memory at 12 GiB a
#           worker; it prints "busy plan: W worker(s) of T" (grepped into this job's output).
# This job prints the allocation (cores, memory) first and the busy cores (mean, max of a 15 s sampler) per step.
#
#   tools/nvidia_central.sh submit <lane> --cap 240 --results box-out/idrec-<sha12> \
#       --env IR_WHEEL_DIR=/root/pc-in/<sha12>/nvidia tools/ops/identity_record_job.sh
# IR_WHEEL_DIR holds exactly the build's two wheels: one mojolearn_core-*.whl and one mojolearn_nvidia-*.whl.
# Optional: IR_GPUS (devices; default nvidia-smi's count), IDENTITY_RECORD_BOX (default: the RunPod pod id).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"

# job-preflight: `identity_record_job.sh --preflight` checks the job's inputs at SUBMIT time, reading only (run on the
# box by tools/ops/job_preflight.sh from tools/nvidia_central.sh submit): IR_WHEEL_DIR holds one core and one nvidia
# wheel, their local version names this tree's HEAD (the recorder refuses any other), the recorder is in the tree, and
# MOJOLEARN_RESULTS_DIR is set. The job keeps the same checks at run time.
ir_wheels() { ls "${IR_WHEEL_DIR:-/nonexistent}"/mojolearn_core-*.whl "${IR_WHEEL_DIR:-/nonexistent}"/mojolearn_nvidia-*.whl 2> /dev/null; }
ir_preflight() {   # PREFLIGHT lines; 0 when the job would start, 4 otherwise
    local fails=0 n head w12
    pf_fail() { echo "PREFLIGHT FAIL $*"; fails=$(( fails + 1 )); }
    if [ -z "${IR_WHEEL_DIR:-}" ]; then pf_fail "IR_WHEEL_DIR is not set (the staged wheel dir of the build)"
    elif [ ! -d "$IR_WHEEL_DIR" ]; then pf_fail "IR_WHEEL_DIR $IR_WHEEL_DIR: no such directory on the box"
    else
        n=$(ir_wheels | grep -c .)
        if [ "$n" -eq 2 ] && [ "$(ls "$IR_WHEEL_DIR"/*.whl | grep -c .)" -eq 2 ]; then
            echo "PREFLIGHT ok wheels in $IR_WHEEL_DIR: $(ir_wheels | xargs -n 1 basename | tr '\n' ' ')"
        else
            pf_fail "IR_WHEEL_DIR $IR_WHEEL_DIR holds $n of the 2 wheels (one mojolearn_core, one mojolearn_nvidia, nothing else)"
        fi
        head=$(git -C "$REPO" rev-parse HEAD 2> /dev/null || true)
        for w in $(ir_wheels); do
            w12=$(basename "$w" | sed -n 's/^[^-]*-[^-]*+\([0-9a-f]\{7,40\}\)-.*/\1/p')
            if [ -z "$head" ]; then echo "PREFLIGHT ok (no git HEAD in $REPO: commit match left to the recorder)"; break
            elif [ -z "$w12" ]; then pf_fail "$(basename "$w"): no +<sha> local version to match HEAD ${head:0:12}"
            elif [ "${head:0:${#w12}}" != "$w12" ]; then pf_fail "$(basename "$w") is commit $w12, this tree's HEAD is ${head:0:12} (sync the tree to the build's commit)"
            fi
        done
    fi
    [ -f "$REPO/tools/identity_wheel_record.py" ] && echo "PREFLIGHT ok recorder $REPO/tools/identity_wheel_record.py" \
        || pf_fail "no tools/identity_wheel_record.py in $REPO"
    [ -n "${MOJOLEARN_RESULTS_DIR:-}" ] || pf_fail "MOJOLEARN_RESULTS_DIR is not set"
    [ "$fails" -eq 0 ] && return 0 || return 4
}
if [ "${1:-}" = --preflight ]; then ir_preflight; exit $?; fi

RES="${MOJOLEARN_RESULTS_DIR:?identity_record_job.sh needs MOJOLEARN_RESULTS_DIR (set by tools/box_persist_run.sh)}"
mkdir -p "$RES/logs" && RES="$(cd "$RES" && pwd)"
pf=$(ir_preflight); prc=$?
printf '%s\n' "$pf" | sed 's/^PREFLIGHT/IDREC preflight/'
[ "$prc" -eq 0 ] || { echo "IDREC inputs refused; nothing ran"; exit 4; }
. "$HERE/box_cores_lib.sh"
echo "IDREC allocation: $(alloc_cores) core(s) of $(nproc) visible, $(( $(alloc_mem_kb) / 1048576 )) GiB available"

# The venvs live outside the results dir (it is uploaded to R2) and go when the job ends.
SCRATCH="$(mktemp -d "${TMPDIR:-/root}/idrec-XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT
fails=0
step() { # name, command...
    local n=$1; shift; local t0=$SECONDS sp
    sp=$(busy_sampler "$RES/logs/$n.busy" 15)
    "$@" > "$RES/logs/$n.log" 2>&1; local rc=$?
    kill "$sp" 2> /dev/null; wait "$sp" 2> /dev/null
    local busy; busy="busy cores $(busy_of "$RES/logs/$n.busy") of $(alloc_cores)"
    if [ $rc = 0 ]; then echo "IDREC $n PASS rc=0 $((SECONDS - t0))s $busy"
    else echo "IDREC $n FAIL rc=$rc $((SECONDS - t0))s $busy: $(grep -v '^\s*$' "$RES/logs/$n.log" | tail -n 2 | tr '\n' ' ' | cut -c1-240)"; fails=$((fails + 1)); fi
}

mapfile -t WHEELS < <(ir_wheels)
whl=(); shas=()
for w in "${WHEELS[@]}"; do
    s=$(sha256sum "$w" | cut -d' ' -f1); whl+=(--wheel "$w"); shas+=(--sha256 "$(basename "$w")=$s")
    echo "IDREC wheel $(basename "$w") $s"
done
GPUS="${IR_GPUS:-$(nvidia-smi -L 2> /dev/null | grep -c '^GPU ')}"

# (i) verify, from an empty directory on the GPU install
VENV="$SCRATCH/verify-venv"; EMPTY="$SCRATCH/empty"; mkdir -p "$EMPTY"
verify_all() {
    python3 -m venv "$VENV" && "$VENV/bin/python" -m pip install -q --no-cache-dir "${WHEELS[@]}" "numpy>=1.26.4" &&
        "$VENV/bin/python" -I -c "import mojolearn as ml; print('IDREC verify wheel', ml.__version__)" &&
        ( cd "$EMPTY" && "$VENV/bin/python" -I -m mojolearn verify --all --json-out "$RES/verify-all.json" )
}
step verify verify_all
grep -m 1 '^IDREC verify wheel' "$RES/logs/verify.log" || true
# (i b) the neural-training profile, the rest of every lane (gd-verify-view, 2026-11-25). `verify --all` is the routine
# profile (python/mojolearn/_verification_profiles.py): it excludes the neural-training lanes (mlp, mamba*, samba*,
# transformer*, byte-lm, embedding, sequence-*, x-cnn-*, optim-*, training-*: 85 on 0becfc8f6b), and its report reads
# them "exposed, but this run did not select it" (83 on nvc-ab-20261125T115541Z-2b94b1). They are selected only by
# `--neural-training`, so this second run on the same venv covers them; the two reports together cover every lane.
verify_neural() {
    ( cd "$EMPTY" && "$VENV/bin/python" -I -m mojolearn verify --all --neural-training --json-out "$RES/verify-neural.json" )
}
step verify-neural verify_neural

# (ii) the NVIDIA identity record. The box each record names (as tools/ops/release_gate_job.sh): the RunPod pod id
# (container env, or /etc/rp_environment for an ssh session); a launcher's own IDENTITY_RECORD_BOX wins.
if [ -z "${IDENTITY_RECORD_BOX:-}" ]; then
    pod="${RUNPOD_POD_ID:-}"
    if [ -z "$pod" ] && [ -r /etc/rp_environment ]; then
        pod=$( { sed -n 's/^export RUNPOD_POD_ID=//p' /etc/rp_environment | head -n 1 | tr -d '"'; } 2> /dev/null || true)
    fi
    [ -n "$pod" ] && export IDENTITY_RECORD_BOX="runpod:$pod"
fi
echo "IDREC record: column nvidia on $GPUS GPU(s), box ${IDENTITY_RECORD_BOX:-unnamed}"
mkdir -p "$RES/identity"
# The recorder's chunks run tools/record_identity_column.sh, which refuses an <outdir> inside the tree; the results dir
# is box-out/ inside /root/mojolearn-ab (first run, nvc-ab-20261125T115541Z-2b94b1: every chunk rc 2). Record into
# the scratch dir, outside the tree, and copy it into the results dir (uploaded to R2) whatever the step's rc.
step record bash -c 'cd "$1" && shift && exec python3 tools/identity_wheel_record.py "$@"' _ "$REPO" \
    --columns nvidia --label-device l40s "${whl[@]}" "${shas[@]}" --gpus "$GPUS" --workers-per-gpu auto \
    --out "$SCRATCH/identity" --venv "$SCRATCH/identity-venv"
[ -d "$SCRATCH/identity" ] && cp -a "$SCRATCH/identity/." "$RES/identity/"
grep -E '^identity-wheels: (busy plan|DONE|FAILED)' "$RES/logs/record.log" | head -n 6 || true

echo "IDREC done fails=$fails"
[ $fails = 0 ]
