#!/bin/bash
# tools/ops/identity_record_job.sh -- the NVIDIA identity record of a measurement build on the L40S pod (lane
# identity-record-job, 2026-11-25; night-decisions m5: `nvidia_central.sh submit` takes one script and no arguments,
# and before this file the only caller of tools/identity_wheel_record.py was the release gate job). Three steps on the
# staged wheels of one build, each printing `IDREC <name> PASS|FAIL rc=<n> <s>s` on one line, full output in
# logs/<name>.log. A failed step never stops the others; the job exits 1 when any failed.
#   (i)  verify: `python -I -m mojolearn verify --all --json-out $MOJOLEARN_RESULTS_DIR/verify-all.json`, from an empty
#        directory, on a venv holding the GPU install: every lane of both profiles (classic and neural) in one process
#        and one v2 report (the CPU and GPU side of every cell, per-profile counts under `profiles`, one verdict);
#   (ii) record: `python3 tools/identity_wheel_record.py --columns nvidia --label-device l40s --wheel ... --sha256 ...
#        --gpus N --workers-per-gpu auto --out <scratch>/identity` (no reuse flag: every lane is recorded), copied to
#        $MOJOLEARN_RESULTS_DIR/identity (the recorder refuses an outdir inside the tree, and the results dir is in it).
#        The recorder requires the wheels' source commit to equal this tree's HEAD (it runs the harness of this tree).
#   (iii) evidence: the verify part staged under evidence/nvidia/verify (tools/verification_evidence.py stage); the
#        recorder staged the record part; the Mac's fetch writes both into verification/latest/nvidia/.
#
# PARALLEL PLAN (AGENTS.md hard rules: 100 percent of rented CPU, parallel by default). Both steps use the one GPU, so
# they run one after the other; inside each, the work spreads over the allocation:
#   verify: its cell workers default to the process's core allocation (affinity and cgroup cpu.max), bounded by
#           available memory at 4 GiB a worker (mojolearn._verify_all.verify_cell_workers); no --cell-workers pin.
#   record: --workers-per-gpu auto: one worker per core (a GPU record worker keeps 0.83 of a core busy, measured),
#           bounded by the GPU's free memory at 3 GiB a worker and the host's at 4 GiB; lanes packed into chunks of
#           equal estimated cost, longest first. It prints "busy plan: W worker(s) of T" (grepped into this job's
#           output). On the 14-core L40S: 13 workers, expected about 250-350 s (was 3 workers, 1030 s).
# This job prints the allocation (cores, memory) first and, per step, the wall seconds and busy cores (mean, max of a
# 15 s sampler), then each step's own timing line: verify's "# sweep: ... slowest units: ..." and the record's
# "timing: ... slowest lanes: ..." and "gpu memory: ..." (rel-fix-2).
#
#   tools/nvidia_central.sh submit <lane> --cap 240 --results box-out/idrec-<sha12> \
#       --env IR_WHEEL_DIR=/root/pc-in/<sha12>/nvidia tools/ops/identity_record_job.sh
# IR_WHEEL_DIR holds exactly the build's two wheels: one mojolearn_core-*.whl and one mojolearn_nvidia-*.whl.
# Optional: IR_GPUS (devices; default nvidia-smi's count), IDENTITY_RECORD_BOX (default: the RunPod pod id).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
. "$HERE/tree_commit.sh"

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
        # the tree's commit is the synced one (tools/ops/tree_commit.sh), not the box HEAD a patch sync leaves at the
        # merge base with main (rel-fix-1, 2026-11-26)
        while read -r line; do
            case $line in MATCH*) echo "PREFLIGHT ok $line" ;; *) pf_fail "$line (sync the tree to the build's commit)" ;; esac
        done < <(wheels_match_tree "$REPO" $(ir_wheels))
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
trap 'git -C "$REPO" worktree remove --force "$SCRATCH/tree" > /dev/null 2>&1; rm -rf "$SCRATCH"; git -C "$REPO" worktree prune > /dev/null 2>&1' EXIT

# THE JOB'S OWN TREE (rel-fix-1, 2026-11-26). The box tree is shared: nvc-ab-20261126T153033Z-865421 passed its
# preflight at 9bdb1a9e8c, the orchestrator synced the tree to main 3423ae229e while verify ran (27 min), and every
# record chunk then refused the wheel ("installed wheel source 9bdb1a9e8c34, harness tree 3423ae229e0b", rc 2 in 15 s).
# So the job fixes its harness at the start: a detached worktree at the wheels' commit when the box has that commit,
# else a copy of the synced tree as it is now (tracked and new files, its HEAD and its sync record). Every later step
# runs the harness from $TREE; a sync of $REPO during the job no longer changes what runs.
TREE="$SCRATCH/tree"
WSHA=$(git -C "$REPO" rev-parse -q --verify "$(wheel_commit "$(ir_wheels | head -n 1)")^{commit}" 2> /dev/null || true)
if [ -n "$WSHA" ] && git -C "$REPO" worktree add -q --detach "$TREE" "$WSHA" > "$RES/logs/tree.log" 2>&1; then
    echo "IDREC tree: worktree at ${WSHA:0:12} ($TREE)"
elif git -C "$REPO" worktree add -q --detach --no-checkout "$TREE" HEAD >> "$RES/logs/tree.log" 2>&1 \
        && git -C "$TREE" read-tree HEAD \
        && ( cd "$REPO" && git ls-files -z -c -o --exclude-standard | tar --null -cf - -T - ) | ( cd "$TREE" && tar -xf - ) \
        && { [ ! -f "$(git -C "$REPO" rev-parse --absolute-git-dir)/devpod_synced" ] \
             || cp "$(git -C "$REPO" rev-parse --absolute-git-dir)/devpod_synced" "$(git -C "$TREE" rev-parse --absolute-git-dir)/devpod_synced"; }; then
    echo "IDREC tree: copy of the synced tree, commit $(tree_commit "$TREE" | cut -c1-12) ($TREE)"
else
    echo "IDREC tree FAIL: could not make the job's own tree ($(tail -n 2 "$RES/logs/tree.log" | tr '\n' ' '))"; exit 4
fi
wheels_match_tree "$TREE" $(ir_wheels) | sed 's/^/IDREC tree /' || { echo "IDREC tree refused: it does not hold the wheels' commit"; exit 4; }
fails=0
step() { # name, command...
    local n=$1; shift; local t0=$SECONDS sp st
    st=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    sp=$(busy_sampler "$RES/logs/$n.busy" 15)
    "$@" > "$RES/logs/$n.log" 2>&1; local rc=$?
    kill "$sp" 2> /dev/null; wait "$sp" 2> /dev/null
    local busy; busy="busy cores $(busy_of "$RES/logs/$n.busy") of $(alloc_cores)"
    # one row per step for the in-repo evidence MANIFEST (tools/verification_evidence.py)
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$rc" "$((SECONDS - t0))" "$st" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$busy" >> "$RES/steps.tsv"
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
grep -m 1 -E '^# sweep: ' "$RES/logs/verify.log" | sed 's/^# /IDREC verify /' | cut -c1-400 || true
grep -m 1 -E '^# TIMED OUT: ' "$RES/logs/verify.log" | sed 's/^# /IDREC verify /' | cut -c1-400 || true
# The separate `verify --all --neural-training` step (verify-neural.json) is gone (verify-profiles, 2026-11-26): `verify
# --all` is now both profiles, so the neural training lanes' cells are in verify-all.json beside the classic ones, the
# cell workers run both profiles' cells in one parallel sweep, and the one exit status covers both.

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
# The recorder's chunks run tools/record_identity_column.sh, which refuses an <outdir> inside ITS tree. The harness
# runs from $TREE (in the scratch dir), so the results dir is outside it: the recorder writes straight into
# $RES/identity, and every chunk's full log (identity/<label>/w<i>/record.log), the selection log and the summary are in
# the uploaded results as they are written, whatever the step's rc, never in a scratch dir only.
step record bash -c 'cd "$1" && shift && exec python3 tools/identity_wheel_record.py "$@"' _ "$TREE" \
    --columns nvidia --label-device l40s "${whl[@]}" "${shas[@]}" --gpus "$GPUS" --workers-per-gpu auto \
    --out "$RES/identity" --venv "$SCRATCH/identity-venv"
grep -E '^identity-wheels: (busy plan|timing|gpu memory|DONE|FAILED)' "$RES/logs/record.log" | head -n 8 | cut -c1-400 || true

# (iii) the in-repo evidence (verify-complete, Andrew 2026-11-26): the verify part here; the record part was staged by
# the recorder under identity/evidence/nvidia/record. On the Mac, `tools/box_persist.py fetch --tag <tag> --extract
# --evidence` (or `tools/verification_evidence.py install --from <fetched dir> --commit`) writes both into
# verification/latest/nvidia/ and commits them.
python3 "$TREE/tools/verification_evidence.py" stage --results "$RES" --column nvidia --part verify \
    --out "$RES/evidence/nvidia/verify" || { echo "IDREC evidence FAIL (staging the verify part)"; fails=$((fails + 1)); }
echo "IDREC done fails=$fails"
[ $fails = 0 ]
