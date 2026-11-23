#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# tools/ops/sabotage_gpu_job.sh -- the GPU column of the verify fault-injection arms on the rented L40S pod (lane
# sabotage-gpu, 2026-11-23). The arms of tools/identity_lanes/coverage_sabotage.tsv were proven on Metal and on the
# CPU column only; their device patches were never compiled for NVIDIA. This job builds the clean development set of
# this tree once (identical sm_89 + host, tools/development_build.py, content-addressed cache: a later job of the
# same tree reuses every group), then runs the named batches of the batch plan one after another through
# tools/coverage_sabotage_batch.sh (clean -> patched build -> restored, then coverage_sabotage.py check per arm and
# column). Batches run one after another because each one patches this tree and owns the one GPU.
#
# Launch (results persist to R2 from the pod through tools/box_persist_run.sh):
#   tools/nvidia_central.sh sync sab ~/mojolearn-wt/sabotage-gpu
#   tools/nvidia_central.sh submit sab --cap 120 --note sab-<batches> --env SAB_BATCHES=<id>[,<id>...] \
#       tools/ops/sabotage_gpu_job.sh
#   [--env SAB_PLAN=<plan in the tree>] (default tools/identity_lanes/coverage_sabotage_gpu_plan.json)
#   [--env SAB_BUDGET_MIN=<minutes>]    no batch starts after this many minutes of the job (default 100)
# Read: tools/box_persist.py fetch --tag <tag>; summary.txt holds one line per arm and column:
#   SABOTAGE batch=<id> arm=<arm> column=<device|cpu> verdict=CAUGHT|MISSED|BUILD-FAIL
#
# PARALLEL PLAN (AGENTS.md hard rules): the base build packs development groups over the allocation's cores by their
# recorded peak memory (development_build.py --parallel auto, cgroup-aware on Linux); inside a batch every binding
# build starts at once with cores / builds compile jobs each, and the device and CPU columns run at once. Busy cores
# of total are sampled for the base build (busy_cores_base.txt) and for each batch's builds. Serial by nature: the
# batches themselves (one patched tree, one GPU), stated above.
#
# Exit status: 0 every arm CAUGHT, 1 any MISSED / BUILD-FAIL / batch error, 2 setup failure.
set -u
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
TREE=$(cd "$SELF_DIR/../.." && pwd)
RES=${MOJOLEARN_RESULTS_DIR:?sabotage_gpu_job.sh needs MOJOLEARN_RESULTS_DIR (set by tools/box_persist_run.sh)}
mkdir -p "$RES" && RES="$(cd "$RES" && pwd)"
PLAN=${SAB_PLAN:-tools/identity_lanes/coverage_sabotage_gpu_plan.json}
BATCHES=${SAB_BATCHES:?set SAB_BATCHES=<batch id>[,<batch id>...] (ids from the plan)}
BUDGET_MIN=${SAB_BUDGET_MIN:-100}
START=$(date +%s)
cd "$TREE" || exit 2
export PATH="/root/.pixi/bin:$PATH"
. "$SELF_DIR/box_cores_lib.sh"
say() { echo "[$(date -u +%H:%M:%S) sabotage-gpu] $*" | tee -a "$RES/summary.txt"; }
: > "$RES/summary.txt"
[ -f "$PLAN" ] || { say "no plan $PLAN in $TREE"; exit 2; }
[ -x /usr/local/cuda/bin/ptxas ] && export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}
export MOJOLEARN_GPU_ARCHS=${MOJOLEARN_GPU_ARCHS:-sm_89}
CORES=$(alloc_cores)
export SAB_CORES=$CORES

# The synced commit (tools/dev_pod.sh sync applies the worktree's diff over its merge base with main, so the tree is
# dirty). coverage_sabotage_batch.sh needs a clean tracked tree to attribute a patch, so the synced state is committed
# on the pod for the run and the HEAD is put back (soft, the files untouched) on every exit path.
SYNC_REC="$(git rev-parse --git-dir)/devpod_synced"
SOURCE=$(sed -n 's/^commit=//p' "$SYNC_REC" 2>/dev/null | head -n 1); SOURCE=${SOURCE:-$(git rev-parse HEAD)}
ORIG=$(git rev-parse HEAD)
git -c user.name=sabotage-gpu -c user.email=sabotage-gpu@localhost commit -q -a -m "sabotage-gpu: synced $SOURCE" \
    > /dev/null 2>&1 || true
trap 'git reset -q --soft "$ORIG"' EXIT
say "source $SOURCE (pod HEAD $(git rev-parse --short=10 "$ORIG") + sync patch), cores $CORES of $(nproc) visible," \
    "GPU $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n 1), batches $BATCHES"
echo "$SOURCE" > "$RES/source-commit.txt"

[ -f tools/fixture_packs.py ] && python3 tools/fixture_packs.py unpack > "$RES/fixtures.log" 2>&1
pixi install -e default > "$RES/pixi-install.log" 2>&1 || { say "pixi install failed"; tail -n 20 "$RES/pixi-install.log"; exit 2; }
pixi run -e default mojo --version > "$RES/compiler-version.log" 2>&1
pixi run -e default python tools/generated_sources.py prepare --repo "$PWD" < /dev/null > "$RES/generated-sources.log" 2>&1 \
    || { say "generated sources failed"; exit 2; }

# 1. the clean set: every development group of identical (sm_89) and host, cached by content
s=$(busy_sampler "$RES/busy_cores_base.txt"); t=$(date +%s)
pixi run -e default python tools/development_build.py build --sets identical,host \
    --cache "${SAB_DEV_CACHE:-/root/.cache/mojolearn-development-sab}" --evidence "$RES/devbuild" \
    > "$RES/devbuild.log" 2>&1
brc=$?
kill "$s" 2>/dev/null
say "BASE development build rc=$brc secs=$(( $(date +%s) - t )) busy $(busy_of "$RES/busy_cores_base.txt") of $CORES cores"
if [ $brc -ne 0 ]; then
    grep -m 5 -B 2 -A 8 -i 'error' "$RES/devbuild.log" | cut -c1-240 | head -40 >> "$RES/summary.txt"
    exit 2
fi

# 2. the batches, one after another (one patched tree, one GPU)
rc=0
for b in ${BATCHES//,/ }; do  # driver-loop: over batches, each owns the tree and the GPU
    el=$(( ($(date +%s) - START) / 60 ))
    if [ "$el" -ge "$BUDGET_MIN" ]; then say "NOT STARTED batch=$b (job at ${el} min, budget $BUDGET_MIN)"; rc=1; continue; fi
    t=$(date +%s)
    OUT="$RES/$b" bash tools/coverage_sabotage_batch.sh "$PLAN" "$b" > "$RES/batch_$b.log" 2>&1
    brc=$?
    grep '^SABOTAGE \|BUSY\|BUILD \|APPLY FAILED\|TREE NOT CLEAN\|RESTORE FAILED\|REVERT FAILED' "$RES/$b/summary.txt" 2>/dev/null \
        | sed "s/^/  /" >> "$RES/summary.txt"
    say "BATCH $b rc=$brc secs=$(( $(date +%s) - t ))"
    [ $brc -eq 0 ] || rc=1
done
# 3. the proven-table rows of this job (tools/identity_lanes/coverage_sabotage_proven.tsv columns: arm, builds,
# verdict, lanes, reference, reverted, seconds), to paste after reading the checks; MISSED and BUILD-FAIL rows too.
for b in ${BATCHES//,/ }; do  # driver-loop: over batches
    [ -f "$RES/$b/summary.txt" ] || continue
    secs=$(awk '/BUILD .* secs=/ {sub(/.*secs=/, ""); if ($1+0 > m) m = $1+0} END {print m+0}' "$RES/$b/summary.txt")
    reverted=$(grep -q 'NATIVE RESTORED' "$RES/$b/summary.txt" && echo restored || echo NOT-restored)
    grep '^SABOTAGE ' "$RES/$b/summary.txt" | while read -r _ bid arm col verdict; do
        arm=${arm#arm=}; col=${col#column=}; verdict=${verdict#verdict=}
        row=$(python3 tools/coverage_sabotage.py list | awk -F'\t' -v a="$arm" '$1==a')
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$(echo "$row" | cut -f3) ($col column, batch ${bid#batch=})" \
            "$verdict" "$(echo "$row" | cut -f4)" "L40S sm_89 $col column, source $SOURCE, $RES" "$reverted" "$secs"
    done
done > "$RES/proven_rows.tsv"
say "DONE rc=$rc ($(grep -c . "$RES/proven_rows.tsv") arm-column rows in proven_rows.tsv)"
exit $rc
