#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# tools/ops/sabotage_gpu_job.sh -- the GPU column of the verify fault-injection arms on the rented L40S pod (lane
# sabotage-gpu, 2026-11-23). The arms of tools/identity_lanes/coverage_sabotage.tsv were proven on Metal and on the
# CPU column only; their device patches were never compiled for NVIDIA. This job activates the clean development set
# of this tree (identical sm_89 + host, tools/development_build.py) from the pod's content-addressed cache, then runs
# the named batches of the batch plan one after another through tools/coverage_sabotage_batch.sh (clean -> patched
# build -> restored, then coverage_sabotage.py check per arm and column). Batches run one after another because each
# one patches this tree and owns the one GPU.
#
# Launch (results persist to R2 from the pod through tools/box_persist_run.sh):
#   tools/nvidia_central.sh sync sab ~/mojolearn-wt/sabotage-gpu
#   tools/nvidia_central.sh submit sab --cap 120 --note sab-<part> --env SAB_BATCHES=all tools/ops/sabotage_gpu_job.sh
#   SAB_BATCHES=all | from:<id> | <id>[,<id>...]   (ids from the plan; `from:<id>` = that batch and every later one,
#                                    in plan order: the resume point a capped job prints as RESUME)
#   [--env SAB_PLAN=<plan in the tree>] (default tools/identity_lanes/coverage_sabotage_gpu_plan.json)
#   [--env SAB_BUDGET_MIN=<minutes>]    no batch starts after this many minutes of the job (default 100)
#   [--env SAB_BATCH_MIN=<minutes>]     one batch's time limit (default 12); past it the batch is stopped, restored,
#                                       and its arms read TIMEOUT
#   [--env SAB_DEV_CACHE=<dir>]         the development cache (default /root/.cache/mojolearn-development-sab)
# Read: tools/box_persist.py fetch --tag <tag>; summary.txt holds one line per arm and column:
#   SABOTAGE batch=<id> arm=<arm> column=<device|cpu|all> verdict=CAUGHT|MISSED|BUILD-FAIL|TIMEOUT
# and proven_rows.tsv the proven-table rows. Both are written and persisted (a `hard` ring upload) after EVERY batch,
# so a cap or a kill loses at most the batch in flight (nvc1-0003, 2026-11-23: 46 batches finished, but the job's
# final upload of a 5.5 GB results dir never landed before the 120-minute cap, and no verdict was read).
#
# WHERE THE TIME GOES (nvc1-0003, 14 cores of the L40S pod, 57 batches asked):
#   setup (fixtures, pixi, generated sources)             about 1 min warm
#   base build, cold cache: 90 groups compiled            1516 s; one group alone (_mojolearn_dev_runtime_trees_host)
#                                                         took 1274 s, so a cold base is never under about 21 min
#   base build, warm cache: every group a hit, activation about 1 min (the plan line says how many are cold)
#   one batch = its slowest binding build + about 30 s (clean, sabotaged and restored runs, device and CPU at once;
#               measured median 30 s, max 35 s over 24 batches); slowest builds measured 37 to 245 s (device) and,
#               from the base build's records, 34 to 323 s (host). Median batch about 100 s, worst about 355 s.
#   the 57 batches: about 6400 s of slowest builds + 57 x 30 s = about 135 min. That is more than one 120-minute
#   cap, so the plan takes two submissions: the first (SAB_BATCHES=all) stops starting batches at SAB_BUDGET_MIN and
#   prints `RESUME SAB_BATCHES=from:<id>`; the second runs from there on the warm cache.
#   The first batch starts at about 3 min on a warm cache; on a cold one (the first job of a changed tree) at about
#   26 min, the critical path of one host group, which no core count shortens.
#
# PARALLEL PLAN (AGENTS.md hard rules): the base build packs development groups over the allocation's cores by their
# recorded peak memory (development_build.py --parallel auto, cgroup-aware on Linux); inside a batch every binding
# build starts at once with -j = the allocation's cores each, and the device and CPU columns run at once. Busy cores of
# total are sampled for the base build (busy_cores_base.txt) and for each batch's builds. Serial by nature: the
# batches themselves (one patched tree, one GPU), stated above.
#
# Exit status: 0 every arm CAUGHT, 1 any MISSED / BUILD-FAIL / TIMEOUT / batch error / batch not started, 2 setup
# failure.
set -u
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
TREE=$(cd "$SELF_DIR/../.." && pwd)
RES=${MOJOLEARN_RESULTS_DIR:?sabotage_gpu_job.sh needs MOJOLEARN_RESULTS_DIR (set by tools/box_persist_run.sh)}
mkdir -p "$RES" && RES="$(cd "$RES" && pwd)"
PLAN=${SAB_PLAN:-tools/identity_lanes/coverage_sabotage_gpu_plan.json}
BATCHES=${SAB_BATCHES:?set SAB_BATCHES=all | from:<batch id> | <batch id>[,<batch id>...] (ids from the plan)}
BUDGET_MIN=${SAB_BUDGET_MIN:-100}
BATCH_MIN=${SAB_BATCH_MIN:-12}
START=$(date +%s)
cd "$TREE" || exit 2
export PATH="/root/.pixi/bin:$PATH"
. "$SELF_DIR/box_cores_lib.sh"
say() { echo "[$(date -u +%H:%M:%S) sabotage-gpu] $*" | tee -a "$RES/summary.txt"; }
: > "$RES/summary.txt"
: > "$RES/proven_rows.tsv"
[ -f "$PLAN" ] || { say "no plan $PLAN in $TREE"; exit 2; }
# the batch ids to run, in plan order
IDS=$(python3 - "$PLAN" "$BATCHES" <<'EOF'
import json, sys
plan, want = sys.argv[1:3]
ids = [b["id"] for b in json.load(open(plan))["batches"]]
if want == "all":
    print(" ".join(ids))
elif want.startswith("from:"):
    start = want[len("from:"):]
    if start not in ids:
        sys.exit(f"no batch {start} in {plan}")
    print(" ".join(ids[ids.index(start):]))
else:
    asked = [x for x in want.split(",") if x]
    bad = [x for x in asked if x not in ids]
    if bad:
        sys.exit(f"no batch {','.join(bad)} in {plan}")
    print(" ".join(asked))
EOF
) || { say "bad SAB_BATCHES=$BATCHES"; exit 2; }
[ -x /usr/local/cuda/bin/ptxas ] && export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}
# The identical set's target. coverage_sabotage_batch.sh unsets it for the host builds, which refuse it.
export MOJOLEARN_GPU_ARCHS=${MOJOLEARN_GPU_ARCHS:-sm_89}
CORES=$(alloc_cores)
export SAB_CORES=$CORES
# the clean natives' tar of each batch lives here, outside the persisted results dir
export SAB_SCRATCH=${SAB_SCRATCH:-/root/mojolearn-sabotage-scratch}

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
    "GPU $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n 1)," \
    "$(echo "$IDS" | wc -w) batches ($BATCHES), batch limit $BATCH_MIN min, budget $BUDGET_MIN min"
echo "$SOURCE" > "$RES/source-commit.txt"

# One upload of the results dir now (a `hard` ring slot; the job runs on). Bounded: a busy uploader is skipped, the
# next batch's upload carries everything.
persist_now() {
    [ -n "${MOJOLEARN_PERSIST_RUNNER:-}" ] && [ -n "${MOJOLEARN_PERSIST_JSON:-}" ] || return 0
    timeout 300 bash "$MOJOLEARN_PERSIST_RUNNER" --snapshot "$MOJOLEARN_PERSIST_JSON" "$RES" hard \
        >> "$RES/persist-hard.log" 2>&1 || echo "$(date -u +%FT%TZ) hard upload rc=$? (skipped)" >> "$RES/persist-hard.log"
}

[ -f tools/fixture_packs.py ] && python3 tools/fixture_packs.py unpack > "$RES/fixtures.log" 2>&1
pixi install -e default > "$RES/pixi-install.log" 2>&1 || { say "pixi install failed"; tail -n 20 "$RES/pixi-install.log"; exit 2; }
pixi run -e default mojo --version > "$RES/compiler-version.log" 2>&1
pixi run -e default python tools/generated_sources.py prepare --repo "$PWD" < /dev/null > "$RES/generated-sources.log" 2>&1 \
    || { say "generated sources failed"; exit 2; }
# the arm table once (arm, ..., bindings, lanes), read by the proven rows of every batch
pixi run -e default python tools/coverage_sabotage.py list > "$RES/arms.tsv" 2> "$RES/arms.err" \
    || { say "coverage_sabotage.py list failed"; exit 2; }

# 1. the clean set: every development group of identical (sm_89) and host, from the pod's content-addressed cache.
# A group is compiled only when its source closure changed since the last job (the first job of a changed tree).
DEV_CACHE=${SAB_DEV_CACHE:-/root/.cache/mojolearn-development-sab}
pixi run -e default python tools/development_build.py plan --sets identical,host --cache "$DEV_CACHE" \
    --json "$RES/devbuild-plan.json" > "$RES/devbuild-plan.log" 2>&1
say "BASE plan: $(python3 - "$RES/devbuild-plan.json" <<'EOF'
import json, sys
try:
    rows = json.load(open(sys.argv[1]))["groups"]
except Exception as e:
    print(f"unread ({e})"); sys.exit(0)
cold = [r for r in rows if r.get("action") != "reuse"]
print(f"{len(rows)} groups, {len(rows) - len(cold)} cached, {len(cold)} to compile")
EOF
)"
s=$(busy_sampler "$RES/busy_cores_base.txt"); t=$(date +%s)
pixi run -e default python tools/development_build.py build --sets identical,host --cache "$DEV_CACHE" \
    --evidence "$RES/devbuild" > "$RES/devbuild.log" 2>&1
brc=$?
kill "$s" 2>/dev/null
say "BASE development build rc=$brc secs=$(( $(date +%s) - t )) busy $(busy_of "$RES/busy_cores_base.txt") of $CORES cores"
if [ $brc -ne 0 ]; then
    grep -m 5 -B 2 -A 8 -i 'error' "$RES/devbuild.log" | cut -c1-240 | head -40 >> "$RES/summary.txt"
    persist_now
    exit 2
fi
persist_now

# the proven-table rows of one batch (tools/identity_lanes/coverage_sabotage_proven.tsv columns: arm, builds,
# verdict, lanes, reference, reverted, seconds), to paste after reading the checks; MISSED, BUILD-FAIL and TIMEOUT
# rows too.
proven_rows() {  # <batch id>
    local b=$1 secs reverted
    [ -f "$RES/$b/summary.txt" ] || return 0
    secs=$(awk '/BUILD .* secs=/ {sub(/.*secs=/, ""); if ($1+0 > m) m = $1+0} END {print m+0}' "$RES/$b/summary.txt")
    reverted=$(grep -q 'NATIVE RESTORED' "$RES/$b/summary.txt" && echo restored || echo NOT-restored)
    grep '^SABOTAGE ' "$RES/$b/summary.txt" | while read -r _ bid arm col verdict; do
        # coverage_sabotage_batch.sh prints side=<cpu|gpu|cpu+gpu> (one check per arm, both sides, 2026-11-24)
        arm=${arm#arm=}; col=${col#column=}; col=${col#side=}; verdict=${verdict#verdict=}
        row=$(awk -F'\t' -v a="$arm" '$1==a' "$RES/arms.tsv")
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$(echo "$row" | cut -f3) ($col side, batch ${bid#batch=})" \
            "$verdict" "$(echo "$row" | cut -f4)" "L40S sm_89 $col side (both sides recorded in one process), source $SOURCE, $RES" "$reverted" "$secs"
    done >> "$RES/proven_rows.tsv"
}

# 2. the batches, one after another (one patched tree, one GPU), each under its time limit
rc=0
left=""
for b in $IDS; do  # driver-loop: over batches, each owns the tree and the GPU
    el=$(( ($(date +%s) - START) / 60 ))
    if [ -n "$left" ] || [ "$el" -ge "$BUDGET_MIN" ]; then
        [ -n "$left" ] || { left=$b; say "RESUME SAB_BATCHES=from:$b (job at ${el} min, budget $BUDGET_MIN)"; }
        say "NOT STARTED batch=$b"; rc=1; continue
    fi
    t=$(date +%s)
    OUT="$RES/$b" timeout -s TERM -k 120 "$(( BATCH_MIN * 60 ))" bash tools/coverage_sabotage_batch.sh "$PLAN" "$b" \
        > "$RES/batch_$b.log" 2>&1
    brc=$?
    if [ $brc -eq 124 ] || [ $brc -eq 137 ]; then
        # stopped at its limit: the batch shell restored the tree on TERM (137: it did not, the check below decides)
        for arm in $(python3 -c 'import json,sys; print(" ".join(next(x["arms"] for x in json.load(open(sys.argv[1]))["batches"] if x["id"] == sys.argv[2])))' "$PLAN" "$b"); do
            echo "SABOTAGE batch=$b arm=$arm side=all verdict=TIMEOUT" >> "$RES/$b/summary.txt"
        done
    fi
    grep '^SABOTAGE \|BUSY\|BUILD \|APPLY FAILED\|PLAN STALE\|TREE NOT CLEAN\|RESTORE FAILED\|REVERT FAILED\|STOPPED' \
        "$RES/$b/summary.txt" 2>/dev/null | sed "s/^/  /" >> "$RES/summary.txt"
    say "BATCH $b rc=$brc secs=$(( $(date +%s) - t ))$([ $brc -eq 124 ] || [ $brc -eq 137 ] && echo " TIMEOUT (limit $BATCH_MIN min)")"
    [ $brc -eq 0 ] || rc=1
    proven_rows "$b"
    persist_now
    # A batch that could not put the tree back leaves every later batch unattributable: stop starting batches.
    if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
        say "TREE NOT CLEAN after batch $b: no further batch starts ($(git status --porcelain --untracked-files=no | head -n 3 | tr '\n' ' '))"
        left=STOPPED; rc=1
    fi
done
say "VERDICTS $(grep -o 'SABOTAGE batch=.*verdict=[A-Z-]*' "$RES/summary.txt" | sed 's/.*verdict=//' | sort | uniq -c | tr -s ' ' | tr '\n' ' ')"
say "DONE rc=$rc ($(grep -c . "$RES/proven_rows.tsv") arm-column rows in proven_rows.tsv)"
exit $rc
