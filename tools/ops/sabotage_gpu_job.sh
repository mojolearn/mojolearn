#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Run the current fault manifest on the native GPU and CPU bindings. The job retains
# its source reach audit, per-arm plan, clean/patched/restored records and final receipt.
# Batches take turns because each patches the same source tree and owns the GPU.
#
# Launch (results persist to R2 from the pod through tools/box_persist_run.sh):
#   tools/nvidia_central.sh sync sab ~/mojolearn-wt/sabotage-gpu
#   tools/nvidia_central.sh submit sab --cap 120 --note sab-<part> --env SAB_BATCHES=all tools/ops/sabotage_gpu_job.sh
#   SAB_BATCHES=all | from:<id> | <id>[,<id>...]   (default all; ids from the plan; `from:<id>` = that batch and every later one,
#                                    in plan order: the resume point a capped job prints as RESUME)
#   [--env SAB_FULL_PLAN=<CPU fault-plan.json> | --env SAB_NATIVE_AUDIT=<native-fault-reach.json>]
#      The default reads fault-plan.json alongside SAB_WHEEL_DIR. With no CPU artifact, distinct patches run separately.
#   [--env SAB_PLAN=<plan in the tree>] (optional historical arm selection; the default uses every current arm)
#   [--env SAB_BUDGET_MIN=<minutes>]    no batch starts after this many minutes of the job (default 100)
#   [--env SAB_BATCH_MIN=<minutes>]     one batch's time limit (default 12); past it the batch is stopped, restored,
#                                       and its arms read TIMEOUT
#   [--env SAB_DEV_CACHE=<dir>]         the development cache (default /root/.cache/mojolearn-development-sab)
#   [--env SAB_WHEEL_DIR=<dir>]         the staged candidate wheels of this commit (core + nvidia): the clean set, no base build
#   [--env SAB_BUILD_MIN=<minutes>]     one prebuilt binding build's limit (default 40)
# Read: tools/box_persist.py fetch --tag <tag>; summary.txt holds one line per arm and column:
#   SABOTAGE batch=<id> arm=<arm> column=<device|cpu|all> verdict=CAUGHT|MISSED|BUILD-FAIL|TIMEOUT
# and proven_rows.tsv the proven-table rows. Both are written and persisted (a `hard` ring upload) after EVERY batch,
# so a cap or a kill loses at most the batch in flight (nvc1-0003, 2026-11-23: 46 batches finished, but the job's
# final upload of a 5.5 GB results dir never landed before the 120-minute cap, and no verdict was read).
#
# WHERE THE TIME GOES (nvc-ab-20261126T153027Z-a3757d, 9bdb1a9e8c, 14 cores, all 155 batches asked: 9 ran in 100 min).
#   setup (fixtures, pixi, generated sources)             about 1 min warm
#   base, SAB_WHEEL_DIR: the staged wheels' natives       seconds (no compile)
#   base, development build, cold cache                   1907 s (90 groups; runtime_trees_host 920 s stopped by its
#                                                         memory limit, then 844 s); warm: every group a hit, ~1 min
#   one binding build is ONE busy core (elapsed = cpu seconds in every base record), 30 to 874 s; the plan's 258
#   builds are about 88,600 s = 1,476 core-minutes. Compiled inside each batch, one batch after another, the pod kept
#   2 to 5 of 14 cores busy and the 155 batches needed about 1,250 min.
#   The prebuild now compiles ahead within the pod's core and memory budget. The current plan and its
#   receipts determine the total duration. SAB_BATCH_MIN bounds one batch's runs and SAB_BUILD_MIN bounds
#   one binding build. On a budget stop keep the original selection and use the printed SAB_RESUME_FROM.
#
# PARALLEL PLAN (AGENTS.md hard rules): the base build packs development groups over the allocation's cores by their
# recorded peak memory (development_build.py --parallel auto, cgroup-aware on Linux); the prebuild packs every batch's
# binding builds over the allocation's cores (one compile per core, the sum of their recorded peaks under 80 percent of
# the allocation's memory), in run order; inside a batch the device and CPU columns run at once. Busy cores of total are
# sampled for the base build (busy_cores_base.txt), the prebuild (busy_cores_prebuild.txt) and a batch that compiles
# in place. Serial by nature: the batches' runs (one patched tree, one GPU), stated above.
#
# Exit status: 0 every arm CAUGHT, 1 any MISSED / BUILD-FAIL / TIMEOUT / batch error / batch not started, 2 setup
# failure.
set -u
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
TREE=$(cd "$SELF_DIR/../.." && pwd)
RES=${MOJOLEARN_RESULTS_DIR:?sabotage_gpu_job.sh needs MOJOLEARN_RESULTS_DIR (set by tools/box_persist_run.sh)}
mkdir -p "$RES" && RES="$(cd "$RES" && pwd)"
PERSIST_ROOT=$RES
# The upload root can be reused for a resume. Keep the original attempts,
# including failed receipts and their raw records, under that root.
if [ -e "$RES/summary.txt" ] || [ -e "$RES/proven_rows.tsv" ] || [ -e "$RES/fault-plan.json" ]; then
    RES=$(mktemp -d "$RES/fault-run.XXXXXXXX") || exit 2
fi
PLAN="$RES/fault-plan.json"
BATCHES=${SAB_BATCHES:-all}
BUDGET_MIN=${SAB_BUDGET_MIN:-100}
BATCH_MIN=${SAB_BATCH_MIN:-12}
START=$(date +%s)
cd "$TREE" || exit 2
export PATH="/root/.pixi/bin:$PATH"
. "$SELF_DIR/box_cores_lib.sh"
say() { echo "[$(date -u +%H:%M:%S) sabotage-gpu] $*" | tee -a "$RES/summary.txt"; }
: > "$RES/summary.txt"
: > "$RES/proven_rows.tsv"
# Read batch IDs or arm names from the requested slice of a plan.
select_from_plan() {
    python3 - "$1" "$2" "$3" "${4:-}" <<'EOF'
import json, sys
path, want, what, resume = sys.argv[1:]
batches = json.load(open(path))["batches"]
ids = [b["id"] for b in batches]
if not ids or len(set(ids)) != len(ids):
    sys.exit("empty plan or duplicate batch IDs")
if want == "all":
    chosen = ids
elif want.startswith("from:") and want[5:] in ids:
    chosen = ids[ids.index(want[5:]):]
else:
    chosen = want.split(",")
    if len(set(chosen)) != len(chosen) or any(i not in ids for i in chosen):
        sys.exit(f"invalid batch selection: {want}")
if resume:
    if resume not in chosen:
        sys.exit(f"resume batch outside the selected plan: {resume}")
    chosen = chosen[chosen.index(resume):]
if what == "ids":
    print(" ".join(chosen))
else:
    arms = [a for b in batches if b["id"] in chosen for a in b["arms"]]
    if not arms or len(set(arms)) != len(arms):
        sys.exit("empty selection or duplicate arms")
    print(",".join(arms))
EOF
}
[ -x /usr/local/cuda/bin/ptxas ] && export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}
# The identical set's target. coverage_sabotage_batch.sh unsets it for the host builds, which refuse it.
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' .')
if [[ "$cap" =~ ^[0-9]+$ ]]; then
    gpu_probe=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n 1)
    detected_arch="sm_$cap"
    export SAB_REQUIRED_BACKEND=cuda
else
    detected_arch=$(rocminfo 2>/dev/null | awk '/Name:[[:space:]]+gfx[0-9a-f]+$/ && !a {a=$NF} END {print a}')
    [[ "$detected_arch" =~ ^gfx[0-9a-f]+$ ]] || { say "could not identify a CUDA or HIP device"; exit 2; }
    gpu_probe=$(rocminfo 2>/dev/null | awk '/Marketing Name:/ && /AMD/ && !n {n=$0} END {print n}')
    export SAB_REQUIRED_BACKEND=hip
fi
export MOJOLEARN_GPU_ARCHS=${MOJOLEARN_GPU_ARCHS:-$detected_arch}
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
# dev_pod sync applies new files without staging them. The native reach audit
# uses git ls-files and prebuild checks out HEAD: both must see those additions.
index=$(git rev-parse --git-path index)
saved_index=$(mktemp) || exit 2
cp "$index" "$saved_index" || exit 2
clean_up() {
    local rc=$?
    trap - EXIT
    if [ -n "${PREPID:-}" ]; then
        kill "$PREPID" "$psampler" 2>/dev/null || true
        wait "$PREPID" 2>/dev/null || true
    fi
    if git reset -q --soft "$ORIG" && cp "$saved_index" "$index"; then
        rm -f "$saved_index"
    else
        say "Could not restore the job checkout; original index retained at $saved_index"
        rc=2
    fi
    exit "$rc"
}
trap clean_up EXIT
added=$(git rev-parse --git-path devpod_added)
if [ -s "$added" ]; then
    git add --pathspec-from-file="$added" || { say "could not stage the synced source additions"; exit 2; }
fi
if ! git diff --quiet HEAD --; then
    # The same synced tree must keep its commit and prebuild cache on resume.
    stamp=$(git show -s --format=%cI "$ORIG") || exit 2
    GIT_AUTHOR_DATE="$stamp" GIT_COMMITTER_DATE="$stamp" \
        git -c user.name=sabotage-gpu -c user.email=sabotage-gpu@localhost commit -q -a -m "sabotage-gpu: synced $SOURCE" \
        || { say "could not snapshot the synced source"; exit 2; }
fi
echo "$SOURCE" > "$RES/source-commit.txt"

# One upload of the results dir now (a `hard` ring slot; the job runs on). Bounded: a busy uploader is skipped, the
# next batch's upload carries everything.
persist_now() {
    [ -n "${MOJOLEARN_PERSIST_RUNNER:-}" ] && [ -n "${MOJOLEARN_PERSIST_JSON:-}" ] || return 0
    timeout 300 bash "$MOJOLEARN_PERSIST_RUNNER" --snapshot "$MOJOLEARN_PERSIST_JSON" "$PERSIST_ROOT" hard \
        >> "$RES/persist-hard.log" 2>&1 || echo "$(date -u +%FT%TZ) hard upload rc=$? (skipped)" >> "$RES/persist-hard.log"
}

[ -f tools/fixture_packs.py ] && python3 tools/fixture_packs.py unpack > "$RES/fixtures.log" 2>&1
pixi install -e default > "$RES/pixi-install.log" 2>&1 || { say "pixi install failed"; tail -n 20 "$RES/pixi-install.log"; exit 2; }
pixi run -e default mojo --version > "$RES/compiler-version.log" 2>&1 || { say "compiler version unavailable"; exit 2; }
pixi run -e default python tools/generated_sources.py prepare --repo "$PWD" < /dev/null > "$RES/generated-sources.log" 2>&1 \
    || { say "generated sources failed"; exit 2; }
# Always plan from this source and its manifest. A named campaign plan
# selects arm names only. Its old patches, lane maps and results cannot qualify this run.
scope=full
current_batches=$BATCHES
plan_arms=()
if [ -n "${SAB_PLAN:-}" ]; then
    selected=$(select_from_plan "$SAB_PLAN" "$BATCHES" arms) || { say "invalid campaign selection"; exit 2; }
    plan_arms=(--arms "$selected")
    if [ ! "$SAB_PLAN" -ef "$RES/requested-plan.json" ]; then cp "$SAB_PLAN" "$RES/requested-plan.json" || exit 2; fi
    current_batches=all
    scope=targeted
elif [ "$BATCHES" != all ]; then
    scope=targeted
fi
# Candidate packing writes a conflict-aware plan beside the candidate wheels.
# A pod without that artifact can still run every fault in isolated builds.
preplan=${SAB_FULL_PLAN:-}
reach=${SAB_NATIVE_AUDIT:-}
if [ -z "$preplan$reach" ] && [ -n "${SAB_WHEEL_DIR:-}" ]; then
    if [ -f "$SAB_WHEEL_DIR/fault-plan.json" ]; then preplan="$SAB_WHEEL_DIR/fault-plan.json"
    elif [ -f "$SAB_WHEEL_DIR/native-fault-reach.json" ]; then reach="$SAB_WHEEL_DIR/native-fault-reach.json"; fi
fi
plan_input=()
if [ -n "$preplan" ] && [ -n "$reach" ]; then say "select one full plan or native audit"; exit 2
elif [ -n "$preplan" ]; then
    cp "$preplan" "$RES/cpu-fault-plan.json" || exit 2
    plan_input=(--full-plan "$RES/cpu-fault-plan.json")
elif [ -n "$reach" ]; then
    cp "$reach" "$RES/native-fault-reach.json" || exit 2
    plan_input=(--audit "$RES/native-fault-reach.json")
else
    say "no CPU plan staged; each distinct patch runs separately"
fi
pixi run -e default python tools/coverage_sabotage.py batch-plan --box "$SAB_REQUIRED_BACKEND" \
    ${plan_input[@]+"${plan_input[@]}"} ${plan_arms[@]+"${plan_arms[@]}"} --json "$PLAN" > "$RES/plan.log" 2>&1 \
    || { say "current fault plan failed: $RES/plan.log"; persist_now; exit 2; }
IDS=$(select_from_plan "$PLAN" "$current_batches" ids "${SAB_RESUME_FROM:-}") || { say "invalid selection or resume position"; exit 2; }
[ -z "${SAB_RESUME_FROM:-}" ] || scope=targeted
plan_hash=$(sha256sum "$PLAN" | awk '{print $1}')
[[ "$plan_hash" =~ ^[0-9a-f]{64}$ ]] || { say "could not hash $PLAN"; exit 2; }
say "source $SOURCE (pod HEAD $(git rev-parse --short=10 HEAD)), cores $CORES of $(nproc) visible," \
    "GPU $gpu_probe $SAB_REQUIRED_BACKEND $detected_arch (build target $MOJOLEARN_GPU_ARCHS)," \
    "$scope plan $plan_hash, $(echo "$IDS" | wc -w) batches ($BATCHES), batch limit $BATCH_MIN min, budget $BUDGET_MIN min"
# the arm table once (arm, ..., bindings, lanes), read by the proven rows of every batch
pixi run -e default python tools/coverage_sabotage.py list > "$RES/arms.tsv" 2> "$RES/arms.err" \
    || { say "coverage_sabotage.py list failed"; exit 2; }

# 1. the clean set. SAB_WHEEL_DIR (the measurement build's staged candidate wheels of THIS commit, core + nvidia, as
# tools/ops/proving_job.sh takes them): their natives ARE the clean set, placed into python/ in the release layout
# (the merged files and their indexes; a per-binding .so beside them wins, python/mojolearn/_merged.py, which is how
# a batch's patched binding loads). Nothing compiles. Otherwise every development group of identical (sm_89) and host,
# from the pod's content-addressed cache: a group compiles only when its source closure changed since the last job
# (9bdb1a9e8c: 90 of 90 cold, 1907 s, the first attempt of runtime_trees_host stopped by its memory limit at 920 s
# and the retry took 844 s). Either way the clean cache is written only here, before any patch is applied: no batch
# reads or fills it (tools/sabotage_prebuild.py builds in scratch worktrees), so no fault-injected binary enters it.
DEV_CACHE=${SAB_DEV_CACHE:-/root/.cache/mojolearn-development-sab}
if [ -n "${SAB_WHEEL_DIR:-}" ]; then
    . "$SELF_DIR/tree_commit.sh"
    match=$(wheels_match_tree "$TREE" "$SAB_WHEEL_DIR"/*.whl); mrc=$?
    echo "$match" | sed 's/^/  wheel /' >> "$RES/summary.txt"
    [ "$mrc" = 0 ] || { say "BASE the staged wheels are not this tree's commit ($SAB_WHEEL_DIR)"; exit 2; }
    t=$(date +%s)
    python3 - "$TREE/python" "$SAB_WHEEL_DIR"/*.whl > "$RES/base-wheels.log" 2>&1 <<'EOF'
import os, re, sys, zipfile
root, wheels = sys.argv[1], sys.argv[2:]
native = re.compile(r"(\.so(\.\d+)*|\.dylib|/_mojolearn_merged\.json)$")
top = ("mojolearn/", "mojolearn.libs/")
removed = 0
for base in ("mojolearn", "mojolearn.libs"):  # driver-loop: over the two native trees
    for d, _, files in os.walk(os.path.join(root, base)):
        for f in files:  # driver-loop: over the tree's files (the old clean set goes, sources stay)
            p = os.path.join(d, f)
            if native.search(p):
                os.remove(p); removed += 1
placed = 0
for w in wheels:  # driver-loop: over the staged wheels
    with zipfile.ZipFile(w) as z:
        for m in z.namelist():  # driver-loop: over one wheel's members (natives only: the sources are the tree's)
            if m.startswith(top) and native.search(m):
                z.extract(m, root); placed += 1
                info = z.getinfo(m)
                mode = (info.external_attr >> 16) & 0o777
                if mode:
                    os.chmod(os.path.join(root, m), mode)
print(f"removed {removed} native files, placed {placed} from {len(wheels)} wheels")
if not placed:
    sys.exit("no native file in the staged wheels")
EOF
    brc=$?
    say "BASE from the staged wheels rc=$brc secs=$(( $(date +%s) - t )): $(tail -n 1 "$RES/base-wheels.log")"
    [ $brc -eq 0 ] || exit 2
    git status --porcelain --untracked-files=no | grep -q . && { say "BASE the wheel natives changed tracked files"; exit 2; }
else
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
fi
persist_now

# 1b. the batches' patched bindings, compiled AHEAD (tools/sabotage_prebuild.py): one scratch worktree per batch, the
# builds of every batch packed over the allocation's cores and memory in run order, beside the runs of the batches
# before them. Keyed by the source commit, so a RESUME submission on the same pod reuses the batches already built.
compiler_key=$(sha256sum "$RES/compiler-version.log" | cut -c1-16)
PREBUILT=${SAB_SCRATCH}/prebuilt-$(git rev-parse --short=12 HEAD)-${MOJOLEARN_GPU_ARCHS}-${compiler_key}-${plan_hash}
mkdir -p "$PREBUILT"
rm -f "$PREBUILT/_clean.done" || exit 2
export SAB_PREBUILT=$PREBUILT
psampler=$(busy_sampler "$RES/busy_cores_prebuild.txt")
pixi run -e default python tools/sabotage_prebuild.py --plan "$PLAN" --tree "$TREE" --out "$PREBUILT" \
    --batches "$IDS" --cores "$CORES" --mem-kb "$(alloc_mem_kb)" --cache "$DEV_CACHE" \
    --unit-seconds "$(( ${SAB_BUILD_MIN:-40} * 60 ))" > "$RES/prebuild.log" 2>&1 &
PREPID=$!
say "PREBUILD started (pid $PREPID, $(echo "$IDS" | wc -w) batches, $CORES cores): $PREBUILT"
# No batch touches the tree before the prebuild has its snapshot of the clean natives.
while [ ! -f "$PREBUILT/_clean.done" ] && kill -0 "$PREPID" 2>/dev/null; do sleep 2; done  # driver-loop: one wait
if [ ! -f "$PREBUILT/_clean.done" ]; then
    unset SAB_PREBUILT
    say "PREBUILD stopped before its clean snapshot: every batch compiles in place"
fi

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
            "$verdict" "$(echo "$row" | cut -f4)" "$gpu_probe $SAB_REQUIRED_BACKEND $detected_arch $col side (build target $MOJOLEARN_GPU_ARCHS), source $SOURCE, $RES" "$reverted" "$secs"
    done >> "$RES/proven_rows.tsv"
}

# 2. the batches, one after another (one patched tree, one GPU), each under its time limit
rc=0
left=""
for b in $IDS; do  # driver-loop: over batches, each owns the tree and the GPU
    el=$(( ($(date +%s) - START) / 60 ))
    if [ -n "$left" ] || [ "$el" -ge "$BUDGET_MIN" ]; then
        [ -n "$left" ] || { left=$b; say "RESUME SAB_RESUME_FROM=$b with the original SAB_BATCHES and plan inputs (job at ${el} min, budget $BUDGET_MIN)"; }
        say "NOT STARTED batch=$b"; rc=1; continue
    fi
    # The batch's prebuilt natives first (a controller wait, bounded by the unit limit: the builds of the batches
    # ahead are already running). A prebuild that died leaves no `ready`: the batch then compiles in place as before.
    w=$(date +%s)
    while [ ! -f "$PREBUILT/$b/ready" ] && kill -0 "$PREPID" 2>/dev/null \
            && [ $(( $(date +%s) - w )) -lt $(( ${SAB_BUILD_MIN:-40} * 60 )) ]; do  # driver-loop: wait on one batch
        sleep 10
    done
    [ -f "$PREBUILT/$b/ready" ] || say "PREBUILD not ready for $b after $(( $(date +%s) - w )) s: it compiles in the batch"
    [ $(( $(date +%s) - w )) -gt 30 ] && say "WAITED $(( $(date +%s) - w )) s for the prebuild of $b"
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
    rm -rf "${PREBUILT:?}/$b"   # placed and restored (a rerun of this batch compiles it again): the disk keeps only the batches still to run
    persist_now
    # A batch that could not put the tree back leaves every later batch unattributable: stop starting batches.
    if [ "$brc" -eq 7 ] || ! grep -q 'NATIVE RESTORED$' "$RES/$b/summary.txt" || [ -n "$(git status --porcelain --untracked-files=no)" ]; then
        say "RESTORATION UNCONFIRMED after batch $b: no further batch starts"
        left=STOPPED; rc=1
    fi
done
kill "$PREPID" 2>/dev/null   # batches past the budget are not built for nothing; their `ready` ones stay for RESUME
kill "$psampler" 2>/dev/null
say "PREBUILD busy $(busy_of "$RES/busy_cores_prebuild.txt") of $CORES cores; $(grep -c '^PREBUILD ready' "$RES/prebuild.log") batches built here"
say "VERDICTS $(grep -o 'SABOTAGE batch=.*verdict=[A-Z-]*' "$RES/summary.txt" | sed 's/.*verdict=//' | sort | uniq -c | tr -s ' ' | tr '\n' ' ')"
python3 tools/coverage_sabotage.py batch-receipt --plan "$PLAN" --batches "$IDS" --scope "$scope" \
    --rows "$RES/proven_rows.tsv" --job-status "$rc" --json "$RES/fault-receipt.json" >> "$RES/summary.txt" 2>&1 || rc=1
persist_now
say "DONE rc=$rc ($(grep -c . "$RES/proven_rows.tsv") arm-column rows in proven_rows.tsv)"
exit $rc
